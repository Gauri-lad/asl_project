import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import time
from collections import deque, Counter
import json
import os

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
        self.prediction_hold_time = 1.5  # seconds to hold a prediction
        self.delete_hold_time = 2.0  # seconds to hold delete
        self.space_hold_time = 1.0   # seconds to hold space
       
        # Load common words dictionary for auto-correct suggestions
        self.common_words = self.load_common_words()
       
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.75
       
    def load_common_words(self):
        """Load common English words for auto-correction"""
        # You can replace this with a file containing common words
        common_words = [
            'hello', 'world', 'how', 'are', 'you', 'what', 'when', 'where',
            'why', 'who', 'good', 'bad', 'yes', 'no', 'please', 'thank',
            'sorry', 'help', 'love', 'like', 'want', 'need', 'time', 'day',
            'night', 'morning', 'afternoon', 'evening', 'food', 'water',
            'home', 'work', 'school', 'friend', 'family', 'happy', 'sad',
            'the', 'and', 'for', 'with', 'this', 'that', 'have', 'will',
            'can', 'could', 'should', 'would', 'make', 'take', 'give',
            'get', 'go', 'come', 'see', 'know', 'think', 'feel', 'look'
        ]
        return set(common_words)
   
    def smooth_prediction(self, predicted_class, confidence):
        """Smooth predictions using a buffer"""
        self.prediction_buffer.append((predicted_class, confidence))
       
        # Only consider high confidence predictions
        high_conf_predictions = [pred for pred, conf in self.prediction_buffer if conf > self.confidence_threshold]
       
        if len(high_conf_predictions) < 3:  # Need at least 3 consistent predictions
            return None, 0
       
        # Get most common prediction
        counter = Counter(high_conf_predictions)
        most_common_pred, count = counter.most_common(1)[0]
       
        # Calculate average confidence for the most common prediction
        avg_confidence = np.mean([conf for pred, conf in self.prediction_buffer if pred == most_common_pred])
       
        return most_common_pred, avg_confidence
   
    def process_prediction(self, predicted_class, confidence):
        """Process the prediction and update word/sentence"""
        current_time = time.time()
       
        # Smooth the prediction
        smoothed_pred, smoothed_conf = self.smooth_prediction(predicted_class, confidence)
       
        if smoothed_pred is None:
            return
       
        # Check if prediction has been held long enough
        if smoothed_pred == self.last_prediction:
            hold_time = current_time - self.last_prediction_time
           
            if smoothed_pred == 'del' and hold_time > self.delete_hold_time:
                self.handle_delete()
                self.last_prediction_time = current_time  # Reset timer
               
            elif smoothed_pred == 'space' and hold_time > self.space_hold_time:
                self.handle_space()
                self.last_prediction_time = current_time  # Reset timer
               
            elif smoothed_pred not in ['del', 'space', 'nothing'] and hold_time > self.prediction_hold_time:
                self.add_letter(smoothed_pred)
                self.last_prediction_time = current_time  # Reset timer
        else:
            # New prediction
            self.last_prediction = smoothed_pred
            self.last_prediction_time = current_time
   
    def add_letter(self, letter):
        """Add letter to current word"""
        if letter not in ['del', 'space', 'nothing']:
            self.current_word += letter.lower()
   
    def handle_delete(self):
        """Handle delete action"""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            # If no current word, remove last character from sentence
            self.sentence = self.sentence[:-1]
   
    def handle_space(self):
        """Handle space action - finalize word and add to sentence"""
        if self.current_word:
            # Add word to history
            self.word_history.append(self.current_word)
           
            # Add to sentence
            if self.sentence:
                self.sentence += " " + self.current_word
            else:
                self.sentence = self.current_word
           
            # Clear current word
            self.current_word = ""
   
    def get_word_suggestions(self):
        """Get word suggestions based on current partial word"""
        if len(self.current_word) < 2:
            return []
       
        suggestions = []
        for word in self.common_words:
            if word.startswith(self.current_word.lower()):
                suggestions.append(word)
       
        return sorted(suggestions)[:3]  # Return top 3 suggestions
   
    def get_status(self):
        """Get current status for display"""
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
            removed_word = self.word_history.pop()
            # Rebuild sentence without last word
            self.sentence = " ".join(self.word_history)
   
    def save_session(self, filename="asl_session.json"):
        """Save current session"""
        session_data = {
            'sentence': self.sentence,
            'word_history': self.word_history,
            'timestamp': time.time()
        }
       
        with open(filename, 'w') as f:
            json.dump(session_data, f)
   
    def load_session(self, filename="asl_session.json"):
        """Load previous session"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                session_data = json.load(f)
                self.sentence = session_data.get('sentence', '')
                self.word_history = session_data.get('word_history', [])

def create_resnet_model(num_classes):
    """Create ResNet18 model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_resnet_model(model_path, num_classes=29):
    """Load trained ResNet model"""
    model = create_resnet_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_frame(frame, image_size=224):
    """Preprocess frame for ResNet"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
   
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
   
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

def draw_ui(frame, spelling_engine, predicted_class, confidence, fps):
    """Draw enhanced UI with finger spelling information"""
    h, w = frame.shape[:2]
    status = spelling_engine.get_status()
   
    # Draw semi-transparent background for text
    overlay = frame.copy()
   
    # Top panel for current prediction
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
   
    # Bottom panel for sentence building
    cv2.rectangle(overlay, (0, h-200), (w, h), (0, 0, 0), -1)
   
    # Blend overlay
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
   
    # Current prediction with confidence
    pred_text = f"Sign: {predicted_class} ({confidence:.2f})"
    cv2.putText(frame, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
   
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
   
    # Current word being spelled
    word_text = f"Current Word: {status['current_word']}_"
    cv2.putText(frame, word_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
   
    # Word suggestions
    suggestions = status['suggestions']
    if suggestions:
        sugg_text = f"Suggestions: {', '.join(suggestions)}"
        cv2.putText(frame, sugg_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
   
    # Sentence display (bottom panel)
    sentence_y_start = h - 180
   
    # Title
    cv2.putText(frame, "Sentence:", (10, sentence_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
   
    # Sentence text (wrap if too long)
    sentence = status['sentence']
    if len(sentence) > 80:  # Wrap long sentences
        lines = [sentence[i:i+80] for i in range(0, len(sentence), 80)]
        for i, line in enumerate(lines[-3:]):  # Show last 3 lines
            cv2.putText(frame, line, (10, sentence_y_start + 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(frame, sentence, (10, sentence_y_start + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
   
    # Instructions
    instructions = [
        "Hold gesture for 1.5s to add letter",
        "Hold SPACE for 1s to add word",
        "Hold DEL for 2s to delete",
        "Keys: C-clear, U-undo, S-save, Q-quit"
    ]
   
    start_y = sentence_y_start + 80
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (10, start_y + i*20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
   
    # Progress indicator for held gestures
    if status['last_prediction'] in ['del', 'space'] or (status['last_prediction'] not in ['nothing']):
        current_time = time.time()
        hold_time = current_time - spelling_engine.last_prediction_time
       
        if status['last_prediction'] == 'del':
            required_time = spelling_engine.delete_hold_time
            color = (0, 0, 255)  # Red
        elif status['last_prediction'] == 'space':
            required_time = spelling_engine.space_hold_time
            color = (255, 0, 0)  # Blue
        else:
            required_time = spelling_engine.prediction_hold_time
            color = (0, 255, 0)  # Green
       
        if hold_time < required_time:
            progress = hold_time / required_time
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (w-220, 50), (w-220 + bar_width, 70), color, -1)
            cv2.rectangle(frame, (w-220, 50), (w-20, 70), (255, 255, 255), 2)

def run_enhanced_asl_detection():
    """Run enhanced ASL detection with finger spelling"""
    # Load model
    model_path = "models/asl_resnet_model.pth"
   
    try:
        model = load_resnet_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists at: {model_path}")
        return
   
    # Initialize finger spelling engine
    spelling_engine = FingerSpellingEngine()
   
    # Try to load previous session
    spelling_engine.load_session()
   
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
   
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
   
    print("Enhanced ASL Detection with Finger Spelling Started!")
    print("Controls:")
    print("  Hold gestures to spell words")
    print("  C - Clear all")
    print("  U - Undo last word")
    print("  S - Save session")
    print("  Q - Quit")
   
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
   
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
       
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
       
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:  # Update FPS every 30 frames
            current_fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
       
        # Create ROI for hand detection
        h, w = frame.shape[:2]
        roi_size = 300
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2 - 50  # Move ROI up a bit
        x2 = x1 + roi_size
        y2 = y1 + roi_size
       
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
       
        # Preprocess and predict
        try:
            input_tensor = preprocess_frame(roi)
           
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
               
                predicted_class = ASL_CLASSES[predicted.item()]
                confidence_score = confidence.item()
           
            # Process prediction with spelling engine
            spelling_engine.process_prediction(predicted_class, confidence_score)
           
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_class = "error"
            confidence_score = 0
       
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
       
        # Draw enhanced UI
        draw_ui(frame, spelling_engine, predicted_class, confidence_score, current_fps)
       
        cv2.imshow('Enhanced ASL Detection - Finger Spelling', frame)
       
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
       
        if key == ord('q'):
            break
        elif key == ord('c'):
            spelling_engine.clear_all()
            print("Cleared all text")
        elif key == ord('u'):
            spelling_engine.undo_last_word()
            print("Undid last word")
        elif key == ord('s'):
            spelling_engine.save_session()
            print("Session saved")
        elif key == ord('l'):
            spelling_engine.load_session()
            print("Session loaded")
   
    # Save session before quitting
    spelling_engine.save_session()
    print(f"Final sentence: {spelling_engine.sentence}")
   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_enhanced_asl_detection()
