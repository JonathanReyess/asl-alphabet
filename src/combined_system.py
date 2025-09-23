"""
Step 7: Combined ASL Recognition System
======================================

This is the ultimate system that combines:
- CNN model for static letters (A, B, C, D)
- LSTM model for motion letters (J, Z)

A true multi-modal AI system for complete ASL recognition!
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path

# Import our components
from motion_model import HandLandmarkExtractor

class CombinedASLSystem:
    """
    Combined ASL recognition system
    
    What this does:
    - Uses CNN for static letters (A, B, C, D)
    - Uses LSTM for motion letters (J, Z)
    - Automatically detects motion vs static
    - Provides seamless recognition experience
    """
    
    def __init__(self, static_model_path="models/asl_model_simple.h5", 
                 motion_model_path="models/motion_model.h5"):
        
        print("Loading Combined ASL Recognition System...")
        
        # Load both models
        self.static_model = None
        self.motion_model = None
        
        if Path(static_model_path).exists():
            self.static_model = tf.keras.models.load_model(static_model_path)
            print("Static model (CNN) loaded!")
        else:
            print(f"Static model not found: {static_model_path}")
            
        if Path(motion_model_path).exists():
            self.motion_model = tf.keras.models.load_model(motion_model_path)
            print("Motion model (LSTM) loaded!")
        else:
            print(f"Motion model not found: {motion_model_path}")
        
        # Initialize hand tracking
        self.landmark_extractor = HandLandmarkExtractor()
        
        # Define letter mappings
        self.static_letters = ['A', 'B', 'C', 'D']
        self.motion_letters = ['J', 'Z']
        self.all_letters = self.static_letters + self.motion_letters
        
        # Motion detection settings
        self.sequence_length = 20
        self.landmark_history = deque(maxlen=self.sequence_length)
        self.motion_threshold = 0.05  # Threshold for detecting motion
        self.motion_frames_required = 15  # Frames needed to trigger motion detection
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        self.last_prediction = None
        self.last_prediction_time = time.time()
        self.prediction_timeout = 2.0  # Reset after 2 seconds
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # Detection state
        self.motion_detected = False
        self.motion_confidence = 0.0
        self.static_confidence = 0.0
        self.current_mode = "static"  # "static" or "motion"
        
        print(f"System ready! Can recognize: {self.all_letters}")
        print(f"Static letters (CNN): {self.static_letters}")
        print(f"Motion letters (LSTM): {self.motion_letters}")
    
    def detect_motion(self):
        """
        Detect if there's significant hand motion for J/Z detection
        
        Returns:
            is_motion: True if motion detected
            motion_score: How much motion (0-1)
        """
        
        if len(self.landmark_history) < 5:
            return False, 0.0
        
        # Calculate motion by comparing recent landmarks
        motion_scores = []
        recent_landmarks = list(self.landmark_history)[-5:]
        
        for i in range(1, len(recent_landmarks)):
            if recent_landmarks[i] is not None and recent_landmarks[i-1] is not None:
                # Calculate distance between consecutive frames
                diff = np.linalg.norm(recent_landmarks[i] - recent_landmarks[i-1])
                motion_scores.append(diff)
        
        if not motion_scores:
            return False, 0.0
        
        avg_motion = np.mean(motion_scores)
        is_motion = avg_motion > self.motion_threshold
        
        return is_motion, min(avg_motion / self.motion_threshold, 1.0)
    
    def predict_static_letter(self, frame, roi_coords):
        """
        Predict static letter using CNN
        
        Args:
            frame: Camera frame
            roi_coords: (x, y, width, height) of ROI
        
        Returns:
            letter: Predicted letter
            confidence: Prediction confidence
        """
        
        if self.static_model is None:
            return "?", 0.0
        
        try:
            x, y, w, h = roi_coords
            
            # Extract and preprocess ROI
            roi = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (224, 224))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_normalized = roi_rgb.astype(np.float32) / 255.0
            roi_batch = np.expand_dims(roi_normalized, axis=0)
            
            # Get prediction
            predictions = self.static_model.predict(roi_batch, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            letter = self.static_letters[predicted_idx]
            
            return letter, confidence
            
        except Exception as e:
            print(f"Error in static prediction: {e}")
            return "?", 0.0
    
    def predict_motion_letter(self):
        """
        Predict motion letter using LSTM
        
        Returns:
            letter: Predicted letter (J or Z)
            confidence: Prediction confidence
        """
        
        if self.motion_model is None or len(self.landmark_history) < self.sequence_length:
            return "?", 0.0
        
        try:
            # Prepare sequence for LSTM
            sequence = []
            recent_landmarks = list(self.landmark_history)
            
            for landmarks in recent_landmarks:
                if landmarks is not None:
                    sequence.append(landmarks)
                else:
                    # Fill missing with zeros
                    sequence.append(np.zeros(63))
            
            # Ensure we have exactly sequence_length frames
            while len(sequence) < self.sequence_length:
                sequence.append(np.zeros(63))
            
            sequence = np.array(sequence[:self.sequence_length])
            
            # Preprocess: normalize relative to wrist (first landmark)
            wrist_pos = sequence[:, :3]  # First landmark (wrist) x,y,z
            normalized_sequence = sequence.copy()
            
            for i in range(0, 63, 3):  # Every landmark (x,y,z)
                normalized_sequence[:, i:i+3] = normalized_sequence[:, i:i+3] - wrist_pos
            
            # Add batch dimension
            sequence_batch = np.expand_dims(normalized_sequence, axis=0)
            
            # Get prediction
            predictions = self.motion_model.predict(sequence_batch, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            letter = self.motion_letters[predicted_idx]
            
            return letter, confidence
            
        except Exception as e:
            print(f"Error in motion prediction: {e}")
            return "?", 0.0
    
    def get_combined_prediction(self, frame, roi_coords):
        """
        Get prediction using both static and motion models
        
        Args:
            frame: Camera frame
            roi_coords: ROI coordinates
        
        Returns:
            final_letter: Best prediction
            final_confidence: Confidence in prediction
            detection_type: "static" or "motion"
            debug_info: Additional info for display
        """
        
        # Extract landmarks for motion tracking
        landmarks = self.landmark_extractor.extract_landmarks(frame)
        self.landmark_history.append(landmarks)
        
        # Detect motion
        has_motion, motion_score = self.detect_motion()
        
        # Get predictions from both models
        static_letter, static_conf = self.predict_static_letter(frame, roi_coords)
        motion_letter, motion_conf = self.predict_motion_letter()
        
        # Decide which prediction to use
        debug_info = {
            'motion_score': motion_score,
            'has_motion': has_motion,
            'static_prediction': f"{static_letter} ({static_conf:.2%})",
            'motion_prediction': f"{motion_letter} ({motion_conf:.2%})",
            'landmarks_available': landmarks is not None
        }
        
        # Motion detection logic
        if has_motion and motion_conf > 0.6 and landmarks is not None:
            # Use motion prediction if confident and motion detected
            self.current_mode = "motion"
            return motion_letter, motion_conf, "motion", debug_info
        else:
            # Use static prediction
            self.current_mode = "static" 
            return static_letter, static_conf, "static", debug_info
    
    def smooth_prediction(self, current_letter, confidence):
        """
        Smooth predictions over time to reduce jitter
        """
        
        current_time = time.time()
        
        # Add to history
        self.prediction_history.append((current_letter, confidence))
        
        # Reset if too much time has passed
        if current_time - self.last_prediction_time > self.prediction_timeout:
            self.prediction_history.clear()
            self.prediction_history.append((current_letter, confidence))
        
        # Get most common recent prediction with high confidence
        if len(self.prediction_history) >= 3:
            # Count high-confidence predictions
            high_conf_predictions = [(letter, conf) for letter, conf in self.prediction_history if conf > 0.7]
            
            if high_conf_predictions:
                # Get most common high-confidence prediction
                letters = [letter for letter, _ in high_conf_predictions]
                most_common = max(set(letters), key=letters.count)
                avg_conf = np.mean([conf for letter, conf in high_conf_predictions if letter == most_common])
                
                self.last_prediction_time = current_time
                return most_common, avg_conf
        
        return current_letter, confidence
    
    def draw_advanced_ui(self, frame, letter, confidence, detection_type, debug_info):
        """
        Draw comprehensive UI showing both models' status
        """
        
        height, width = frame.shape[:2]
        
        # Draw ROI
        roi_size = 300
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        # Color based on detection type and confidence
        if detection_type == "motion":
            roi_color = (255, 0, 255)  # Magenta for motion
        else:
            if confidence > 0.8:
                roi_color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                roi_color = (0, 255, 255)  # Yellow
            else:
                roi_color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), roi_color, 3)
        
        # Main prediction display
        cv2.putText(frame, f"Letter: {letter}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {detection_type.upper()}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, roi_color, 2)
        
        # Model status
        y_start = 200
        cv2.putText(frame, "Model Status:", (50, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Static (CNN): {debug_info['static_prediction']}", (70, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Motion (LSTM): {debug_info['motion_prediction']}", (70, y_start + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Motion detection info
        motion_color = (0, 255, 0) if debug_info['has_motion'] else (100, 100, 100)
        cv2.putText(frame, f"Motion Score: {debug_info['motion_score']:.2f}", (70, y_start + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 1)
        
        # Landmarks status
        landmarks_color = (0, 255, 0) if debug_info['landmarks_available'] else (0, 0, 255)
        cv2.putText(frame, f"Hand Tracking: {'ON' if debug_info['landmarks_available'] else 'OFF'}", 
                   (70, y_start + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, landmarks_color, 1)
        
        # System info
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (width - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"All Letters: {', '.join(self.all_letters)}", (50, height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instructions
        instructions = [
            "Static: A, B, C, D (hold steady)",
            "Motion: J (hook), Z (trace)",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (50, height - 50 + (i * 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """
        Run the combined ASL recognition system
        """
        
        if self.static_model is None and self.motion_model is None:
            print("No models loaded! Train both models first.")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Combined ASL Recognition System Started!")
        print("Recognizing: A, B, C, D (static) + J, Z (motion)")
        print("Instructions:")
        print("   • Static letters: Hold hand steady in rectangle")
        print("   • Motion letters: Perform clear motions")
        print("   • J: Hook motion with pinky")
        print("   • Z: Trace letter Z with index finger")
        print("   • Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Define ROI
                height, width = frame.shape[:2]
                roi_size = 300
                roi_x = (width - roi_size) // 2
                roi_y = (height - roi_size) // 2
                roi_coords = (roi_x, roi_y, roi_size, roi_size)
                
                # Get combined prediction
                letter, confidence, detection_type, debug_info = self.get_combined_prediction(frame, roi_coords)
                
                # Smooth prediction
                smoothed_letter, smoothed_conf = self.smooth_prediction(letter, confidence)
                
                # Draw landmarks if available
                if debug_info['landmarks_available']:
                    frame = self.landmark_extractor.draw_landmarks(frame)
                
                # Draw UI
                frame = self.draw_advanced_ui(frame, smoothed_letter, smoothed_conf, detection_type, debug_info)
                
                # Update FPS
                self.fps_counter += 1
                if time.time() - self.fps_timer > 1.0:
                    self.current_fps = self.fps_counter / (time.time() - self.fps_timer)
                    self.fps_counter = 0
                    self.fps_timer = time.time()
                
                # Display
                cv2.imshow('🚀 Combined ASL Recognition - Your Custom AI System!', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Thanks for using the Combined ASL Recognition System!")

def main():
    """
    Main function
    """
    
    print("Combined ASL Recognition System")
    print("=" * 50)
    print("The Ultimate Multi-Modal AI System!")
    print("CNN + LSTM + Computer Vision")
    
    # Check if models exist
    static_path = "models/asl_model_simple.h5"
    motion_path = "models/motion_model.h5"
    
    if not Path(static_path).exists():
        print(f"Static model not found: {static_path}")
        print("Train it with: python src/train_model.py")
        
    if not Path(motion_path).exists():
        print(f"Motion model not found: {motion_path}")
        print("Train it with: python src/train_motion.py")
        
    if not Path(static_path).exists() and not Path(motion_path).exists():
        print("No models found! Train them first.")
        return
    
    # Create and run system
    system = CombinedASLSystem(static_path, motion_path)
    system.run()

if __name__ == "__main__":
    main()