#!/usr/bin/env python3
"""
Simple ASL Detection Server
Place this file in your asl_project/ directory and run: python app.py
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import time
from collections import deque, Counter
import json
import base64
import io
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading

# ASL Classes
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

class FingerSpellingEngine:
    def __init__(self):
        self.current_word = ""
        self.sentence = ""
        self.word_history = []
        self.last_prediction = ""
        self.last_prediction_time = 0
        self.prediction_hold_time = 1.5
        self.delete_hold_time = 2.0
        self.space_hold_time = 1.0
        
        # Common words for suggestions
        self.common_words = {
            'hello', 'world', 'how', 'are', 'you', 'what', 'when', 'where',
            'why', 'who', 'good', 'bad', 'yes', 'no', 'please', 'thank',
            'sorry', 'help', 'love', 'like', 'want', 'need', 'time', 'day',
            'night', 'morning', 'afternoon', 'evening', 'food', 'water',
            'home', 'work', 'school', 'friend', 'family', 'happy', 'sad'
        }
        
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.7
    
    def smooth_prediction(self, predicted_class, confidence):
        """Smooth predictions using a buffer"""
        self.prediction_buffer.append((predicted_class, confidence))
        
        high_conf_predictions = [pred for pred, conf in self.prediction_buffer if conf > self.confidence_threshold]
        
        if len(high_conf_predictions) < 3:
            return None, 0
        
        counter = Counter(high_conf_predictions)
        most_common_pred, count = counter.most_common(1)[0]
        
        avg_confidence = np.mean([conf for pred, conf in self.prediction_buffer if pred == most_common_pred])
        
        return most_common_pred, avg_confidence
    
    def process_prediction(self, predicted_class, confidence):
        """Process prediction and update word/sentence"""
        current_time = time.time()
        
        smoothed_pred, smoothed_conf = self.smooth_prediction(predicted_class, confidence)
        
        if smoothed_pred is None:
            return False
        
        if smoothed_pred == self.last_prediction:
            hold_time = current_time - self.last_prediction_time
            
            if smoothed_pred == 'del' and hold_time > self.delete_hold_time:
                self.handle_delete()
                self.last_prediction_time = current_time
                return True
                
            elif smoothed_pred == 'space' and hold_time > self.space_hold_time:
                self.handle_space()
                self.last_prediction_time = current_time
                return True
                
            elif smoothed_pred not in ['del', 'space', 'nothing'] and hold_time > self.prediction_hold_time:
                self.add_letter(smoothed_pred)
                self.last_prediction_time = current_time
                return True
        else:
            self.last_prediction = smoothed_pred
            self.last_prediction_time = current_time
            
        return False
    
    def add_letter(self, letter):
        """Add letter to current word"""
        if letter not in ['del', 'space', 'nothing']:
            self.current_word += letter.lower()
    
    def handle_delete(self):
        """Handle delete action"""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.sentence = self.sentence[:-1]
    
    def handle_space(self):
        """Handle space action"""
        if self.current_word:
            self.word_history.append(self.current_word)
            
            if self.sentence:
                self.sentence += " " + self.current_word
            else:
                self.sentence = self.current_word
            
            self.current_word = ""
    
    def get_word_suggestions(self):
        """Get word suggestions"""
        if len(self.current_word) < 2:
            return []
        
        suggestions = []
        for word in self.common_words:
            if word.startswith(self.current_word.lower()):
                suggestions.append(word)
        
        return sorted(suggestions)[:3]
    
    def get_status(self):
        """Get current status"""
        return {
            'current_word': self.current_word,
            'sentence': self.sentence,
            'suggestions': self.get_word_suggestions(),
            'last_prediction': self.last_prediction,
            'word_count': len(self.word_history)
        }
    
    def clear_all(self):
        """Clear everything"""
        self.current_word = ""
        self.sentence = ""
        self.word_history = []
    
    def undo_last_word(self):
        """Undo last word"""
        if self.word_history:
            self.word_history.pop()
            self.sentence = " ".join(self.word_history)

class ASLModel:
    def __init__(self, model_path="models/asl_resnet_model.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.load_model()
    
    def create_resnet_model(self, num_classes=29):
        """Create ResNet18 model"""
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = self.create_resnet_model()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure your model file is at:", self.model_path)
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return "error", 0.0
        
        try:
            input_tensor = self.preprocess_image(image)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = ASL_CLASSES[predicted.item()]
                confidence_score = confidence.item()
                
            return predicted_class, confidence_score
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asl_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global instances
asl_model = None
spelling_engine = FingerSpellingEngine()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASL Detection Studio</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #fff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .video-section {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }
        
        .video-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            aspect-ratio: 4/3;
        }
        
        #video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1);
        }
        
        .roi-overlay {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 200px;
            height: 200px;
            border: 3px dashed #00ff88;
            border-radius: 10px;
            background: rgba(0, 255, 136, 0.1);
        }
        
        .prediction-display {
            position: absolute;
            top: 15px;
            left: 15px;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(5px);
        }
        
        .prediction-class {
            font-size: 2em;
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 5px;
        }
        
        .confidence-bar {
            width: 150px;
            height: 8px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff4444, #ffaa00, #00ff88);
            transition: width 0.3s ease;
        }
        
        .controls-section {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }
        
        .status-box {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .current-word {
            font-family: 'Courier New', monospace;
            font-size: 1.5em;
            color: #ffeb3b;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 5px;
            min-height: 40px;
        }
        
        .suggestions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        
        .suggestion {
            background: rgba(33, 150, 243, 0.3);
            color: #64b5f6;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            border: 1px solid rgba(33, 150, 243, 0.5);
        }
        
        .btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 10px;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #4caf50, #45a049);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #ff9800, #f57c00);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f44336, #d32f2f);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .sentence-section {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }
        
        .sentence-display {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 20px;
            min-height: 100px;
            font-family: 'Courier New', monospace;
            font-size: 1.2em;
            line-height: 1.5;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.7;
        }
        
        .instructions {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .instruction-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.9em;
        }
        
        .instruction-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .connected { background: #4caf50; }
        .disconnected { background: #f44336; }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .stats {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤟 ASL Detection Studio</h1>
            <p>Real-time American Sign Language Recognition</p>
        </div>
        
        <div class="main-grid">
            <div class="video-section">
                <div class="status-indicator">
                    <div class="status-dot" id="connectionStatus"></div>
                    <span id="statusText">Connecting...</span>
                    <span style="margin-left: auto;" id="fpsDisplay">FPS: 0</span>
                </div>
                
                <div class="video-container">
                    <video id="video" autoplay muted playsinline></video>
                    <div class="roi-overlay">
                        <div style="position: absolute; top: -25px; left: 0; font-size: 12px; background: #00ff88; color: black; padding: 2px 8px; border-radius: 3px;">
                            Detection Area
                        </div>
                    </div>
                    <div class="prediction-display">
                        <div class="prediction-class" id="predictionClass">NOTHING</div>
                        <div style="font-size: 0.9em; margin-bottom: 10px;">
                            Confidence: <span id="confidenceText">0%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 20px; text-align: center;">
                    <button class="btn btn-primary" id="startBtn" onclick="startCamera()">📹 Start Camera</button>
                    <button class="btn btn-danger" id="stopBtn" onclick="stopCamera()" style="display: none;">⏹️ Stop Camera</button>
                </div>
            </div>
            
            <div class="controls-section">
                <h3 style="margin-bottom: 20px;">📝 Current Status</h3>
                
                <div class="status-box">
                    <div style="margin-bottom: 10px; font-weight: bold;">Current Word:</div>
                    <div class="current-word" id="currentWord">_</div>
                    
                    <div id="suggestionsContainer" style="display: none;">
                        <div style="margin-top: 15px; margin-bottom: 5px; font-weight: bold;">Suggestions:</div>
                        <div class="suggestions" id="suggestions"></div>
                    </div>
                </div>
                
                <h3 style="margin-bottom: 15px;">🎮 Controls</h3>
                <button class="btn btn-secondary" onclick="undoWord()">↶ Undo Last Word</button>
                <button class="btn btn-danger" onclick="clearAll()">🗑️ Clear All</button>
                <button class="btn btn-primary" onclick="saveSentence()">💾 Save Sentence</button>
                
                <div class="instructions">
                    <h4 style="margin-bottom: 10px;">How to Use:</h4>
                    <div class="instruction-item">
                        <div class="instruction-dot" style="background: #4caf50;"></div>
                        Hold letter signs for 1.5 seconds
                    </div>
                    <div class="instruction-item">
                        <div class="instruction-dot" style="background: #2196f3;"></div>
                        Hold SPACE for 1 second to add word
                    </div>
                    <div class="instruction-item">
                        <div class="instruction-dot" style="background: #ff5722;"></div>
                        Hold DELETE for 2 seconds to delete
                    </div>
                    <div class="instruction-item">
                        <div class="instruction-dot" style="background: #ffc107;"></div>
                        Keep hand in green detection box
                    </div>
                </div>
            </div>
        </div>
        
        <div class="sentence-section">
            <h3 style="margin-bottom: 15px;">📄 Composed Sentence</h3>
            <div class="sentence-display" id="sentenceDisplay">
                <em style="color: #888;">Your sentence will appear here as you sign...</em>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" style="color: #4caf50;" id="fpsValue">0</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #2196f3;" id="confidenceValue">0%</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #ff9800;" id="wordCount">0</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #9c27b0;" id="letterCount">0</div>
                <div class="stat-label">Letters</div>
            </div>
        </div>
    </div>
    
    <canvas id="canvas" style="display: none;"></canvas>
    
    <script>
        const socket = io();
        let video = null;
        let canvas = null;
        let isStreaming = false;
        let streamInterval = null;
        let fpsCounter = 0;
        let lastFpsTime = Date.now();
        
        // Socket connection events
        socket.on('connect', () => {
            document.getElementById('connectionStatus').className = 'status-dot connected';
            document.getElementById('statusText').textContent = 'Connected';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('connectionStatus').className = 'status-dot disconnected';
            document.getElementById('statusText').textContent = 'Disconnected';
        });
        
        // Prediction response
        socket.on('prediction_response', (data) => {
            updateUI(data);
            updateFPS();
        });
        
        // Status update
        socket.on('status_update', (data) => {
            updateStatus(data);
        });
        
        async function startCamera() {
            try {
                video = document.getElementById('video');
                canvas = document.getElementById('canvas');
                
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'user' }
                });
                
                video.srcObject = stream;
                isStreaming = true;
                
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'inline-block';
                
                // Start prediction loop
                streamInterval = setInterval(() => {
                    if (isStreaming) {
                        captureAndPredict();
                    }
                }, 200);
                
            } catch (error) {
                alert('Could not access camera. Please check permissions.');
                console.error('Camera error:', error);
            }
        }
        
        function stopCamera() {
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            isStreaming = false;
            
            if (streamInterval) {
                clearInterval(streamInterval);
            }
            
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
        }
        
        function captureAndPredict() {
            if (!video || !canvas) return;
            
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            context.drawImage(video, 0, 0);
            
            // Get ROI (center square)
            const size = Math.min(canvas.width, canvas.height) * 0.6;
            const x = (canvas.width - size) / 2;
            const y = (canvas.height - size) / 2;
            
            const roiCanvas = document.createElement('canvas');
            roiCanvas.width = size;
            roiCanvas.height = size;
            const roiContext = roiCanvas.getContext('2d');
            roiContext.drawImage(canvas, x, y, size, size, 0, 0, size, size);
            
            const imageData = roiCanvas.toDataURL('image/jpeg', 0.8);
            
            socket.emit('predict', { image: imageData });
        }
        
        function updateUI(data) {
            document.getElementById('predictionClass').textContent = data.predicted_class.toUpperCase();
            document.getElementById('confidenceText').textContent = Math.round(data.confidence * 100) + '%';
            document.getElementById('confidenceFill').style.width = (data.confidence * 100) + '%';
            
            document.getElementById('currentWord').textContent = data.current_word || '_';
            
            if (data.sentence) {
                document.getElementById('sentenceDisplay').innerHTML = data.sentence + 
                    (data.current_word ? ' <span style="color: #ffeb3b;">' + data.current_word + '</span>' : '') + 
                    ' <span style="color: #2196f3;">|</span>';
            } else if (data.current_word) {
                document.getElementById('sentenceDisplay').innerHTML = '<span style="color: #ffeb3b;">' + data.current_word + '</span> <span style="color: #2196f3;">|</span>';
            } else {
                document.getElementById('sentenceDisplay').innerHTML = '<em style="color: #888;">Your sentence will appear here as you sign...</em>';
            }
            
            // Update suggestions
            if (data.suggestions && data.suggestions.length > 0) {
                const suggestionsHtml = data.suggestions.map(s => 
                    '<span class="suggestion">' + s + '</span>'
                ).join('');
                document.getElementById('suggestions').innerHTML = suggestionsHtml;
                document.getElementById('suggestionsContainer').style.display = 'block';
            } else {
                document.getElementById('suggestionsContainer').style.display = 'none';
            }
            
            // Update stats
            document.getElementById('confidenceValue').textContent = Math.round(data.confidence * 100) + '%';
            document.getElementById('wordCount').textContent = data.word_count || 0;
            document.getElementById('letterCount').textContent = (data.current_word || '').length;
        }
        
        function updateStatus(data) {
            document.getElementById('currentWord').textContent = data.current_word || '_';
            
            if (data.sentence) {
                document.getElementById('sentenceDisplay').innerHTML = data.sentence + 
                    (data.current_word ? ' <span style="color: #ffeb3b;">' + data.current_word + '</span>' : '') + 
                    ' <span style="color: #2196f3;">|</span>';
            }
            
            document.getElementById('wordCount').textContent = data.word_count || 0;
            document.getElementById('letterCount').textContent = (data.current_word || '').length;
        }
        
        function updateFPS() {
            fpsCounter++;
            const now = Date.now();
            
            if (now - lastFpsTime >= 1000) {
                document.getElementById('fpsDisplay').textContent = 'FPS: ' + fpsCounter;
                document.getElementById('fpsValue').textContent = fpsCounter;
                fpsCounter = 0;
                lastFpsTime = now;
            }
        }
        
        function clearAll() {
            socket.emit('action', { action: 'clear' });
        }
        
        function undoWord() {
            socket.emit('action', { action: 'undo' });
        }
        
        function saveSentence() {
            const sentence = document.getElementById('sentenceDisplay').textContent;
            if (sentence && sentence.trim()) {
                const blob = new Blob([sentence], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'asl_sentence_' + new Date().toISOString().slice(0, 19).replace(/:/g, '-') + '.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        }
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            stopCamera();
        });
    </script>
</body>
</html>
    '''

@socketio.on('predict')
def handle_prediction(data):
    """Handle prediction request"""
    global spelling_engine
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None and asl_model is not None:
            # Make prediction
            predicted_class, confidence = asl_model.predict(image)
            
            # Process with spelling engine
            spelling_engine.process_prediction(predicted_class, confidence)
            status = spelling_engine.get_status()
            
            # Send response
            response = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'current_word': status['current_word'],
                'sentence': status['sentence'],
                'suggestions': status['suggestions'],
                'word_count': status['word_count'],
                'timestamp': time.time()
            }
            
            emit('prediction_response', response)
        else:
            emit('prediction_response', {
                'predicted_class': 'error',
                'confidence': 0.0,
                'current_word': '',
                'sentence': '',
                'suggestions': [],
                'word_count': 0,
                'timestamp': time.time()
            })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        emit('prediction_response', {
            'predicted_class': 'error',
            'confidence': 0.0,
            'current_word': '',
            'sentence': '',
            'suggestions': [],
            'word_count': 0,
            'timestamp': time.time()
        })

@socketio.on('action')
def handle_action(data):
    """Handle user actions"""
    global spelling_engine
    
    action = data.get('action')
    if action == 'clear':
        spelling_engine.clear_all()
    elif action == 'undo':
        spelling_engine.undo_last_word()
    
    # Send updated status
    status = spelling_engine.get_status()
    emit('status_update', status)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': asl_model is not None and asl_model.model is not None,
        'timestamp': time.time()
    })

def initialize_model():
    """Initialize the ASL model"""
    global asl_model
    
    # Try different model paths
    model_paths = [
        "models/asl_resnet_model.pth",
        "asl_resnet_model.pth",
        "models/fast_asl_gnn_final.pth",  # Your existing model
        "models/asl_vit_model.pth"        # Your existing model
    ]
    
    for model_path in model_paths:
        try:
            asl_model = ASLModel(model_path)
            if asl_model.model is not None:
                print(f"✅ Successfully loaded model from: {model_path}")
                return True
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            continue
    
    print("❌ Could not load any model. Please check your model files.")
    return False

if __name__ == '__main__':
    print("🚀 Starting ASL Detection Server...")
    print("=" * 50)
    
    # Initialize model
    model_loaded = initialize_model()
    
    if not model_loaded:
        print("\n⚠️  WARNING: No model loaded! The server will start but predictions won't work.")
        print("Please ensure you have one of these model files:")
        print("  - models/asl_resnet_model.pth")
        print("  - models/fast_asl_gnn_final.pth")
        print("  - models/asl_vit_model.pth")
        print("\nIf your model has a different architecture, you may need to modify the ASLModel class.")
    
    print("\n🌐 Server starting...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask-SocketIO server
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    
    print("✅ Server shutdown complete")