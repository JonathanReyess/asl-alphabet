import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
# Force CPU-only to avoid M1 GPU conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pathlib import Path
from collections import deque
import time

# Configure TensorFlow for M1 compatibility
tf.config.set_visible_devices([], 'GPU')

class SimplifiedCombinedASL:
    def __init__(self):
        print("Loading Combined ASL System...")
        
        # Load models with explicit CPU usage
        self.static_model = None
        self.motion_model = None
        
        static_path = "models/asl_model_simple.keras"
        motion_path = "models/motion_model.keras"
        
        try:
            if Path(static_path).exists():
                with tf.device('/CPU:0'):  # Force CPU
                    self.static_model = tf.keras.models.load_model(static_path)
                print("✓ Static CNN model loaded")
            else:
                print("✗ Static model not found")
        except Exception as e:
            print(f"Error loading static model: {e}")
            
        try:    
            if Path(motion_path).exists():
                with tf.device('/CPU:0'):  # Force CPU
                    self.motion_model = tf.keras.models.load_model(motion_path)
                print("✓ Motion LSTM model loaded")
            else:
                print("✗ Motion model not found")
        except Exception as e:
            print(f"Error loading motion model: {e}")
        
        # Initialize MediaPipe AFTER models are loaded
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lower confidence to be more reliable
            min_tracking_confidence=0.3
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Letter mappings
        self.static_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        self.motion_letters = ['J', 'Z']
        
        # Motion tracking
        self.landmark_history = deque(maxlen=20)
        self.prediction_history = deque(maxlen=5)
        
        print(f"Ready! Static: {len(self.static_letters)} letters, Motion: {len(self.motion_letters)} letters")
    
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks), results.multi_hand_landmarks
        
        return None, None
    
    def predict_static(self, frame, roi_coords):
        """Predict static letter using CNN"""
        if self.static_model is None:
            return "?", 0.0
        
        try:
            x, y, w, h = roi_coords
            roi = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (224, 224))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_normalized = roi_rgb.astype(np.float32) / 255.0
            roi_batch = np.expand_dims(roi_normalized, axis=0)
            
            predictions = self.static_model.predict(roi_batch, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            letter = self.static_letters[predicted_idx]
            
            return letter, confidence
            
        except Exception as e:
            print(f"Static prediction error: {e}")
            return "?", 0.0
    
    def predict_motion(self):
        """Predict motion letter using LSTM"""
        if self.motion_model is None or len(self.landmark_history) < 20:
            return "?", 0.0
        
        try:
            # Prepare sequence
            sequence = []
            for landmarks in list(self.landmark_history):
                if landmarks is not None:
                    sequence.append(landmarks)
                else:
                    sequence.append(np.zeros(63))
            
            sequence = np.array(sequence[:20])
            
            # Normalize relative to wrist
            if len(sequence) == 20:
                wrist_pos = sequence[:, :3]
                normalized_sequence = sequence.copy()
                
                for i in range(0, 63, 3):
                    normalized_sequence[:, i:i+3] = normalized_sequence[:, i:i+3] - wrist_pos
                
                sequence_batch = np.expand_dims(normalized_sequence, axis=0)
                predictions = self.motion_model.predict(sequence_batch, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_idx]
                letter = self.motion_letters[predicted_idx]
                
                return letter, confidence
            
        except Exception as e:
            print(f"Motion prediction error: {e}")
        
        return "?", 0.0
    
    def detect_motion(self):
        """Simple motion detection"""
        if len(self.landmark_history) < 5:
            return False, 0.0
        
        recent = [lm for lm in list(self.landmark_history)[-5:] if lm is not None]
        if len(recent) < 2:
            return False, 0.0
        
        motions = []
        for i in range(1, len(recent)):
            diff = np.linalg.norm(recent[i] - recent[i-1])
            motions.append(diff)
        
        if motions:
            avg_motion = np.mean(motions)
            is_motion = avg_motion > 0.05  # Threshold
            return is_motion, min(avg_motion / 0.05, 1.0)
        
        return False, 0.0
    
    def get_prediction(self, frame, roi_coords):
        """Get combined prediction"""
        # Extract landmarks
        landmarks, hand_landmarks = self.extract_hand_landmarks(frame)
        self.landmark_history.append(landmarks)
        
        # Get both predictions
        static_letter, static_conf = self.predict_static(frame, roi_coords)
        motion_letter, motion_conf = self.predict_motion()
        
        # Detect motion
        has_motion, motion_score = self.detect_motion()
        
        # Choose prediction
        if has_motion and motion_conf > 0.6 and landmarks is not None:
            return motion_letter, motion_conf, "motion", hand_landmarks, {
                'static': f"{static_letter} ({static_conf:.2%})",
                'motion': f"{motion_letter} ({motion_conf:.2%})",
                'motion_score': motion_score
            }
        else:
            return static_letter, static_conf, "static", hand_landmarks, {
                'static': f"{static_letter} ({static_conf:.2%})",
                'motion': f"{motion_letter} ({motion_conf:.2%})",
                'motion_score': motion_score
            }
    
    def draw_ui(self, frame, letter, confidence, mode, hand_landmarks, debug_info):
        """Draw user interface"""
        height, width = frame.shape[:2]
        
        # ROI
        roi_size = 300
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        # Color based on mode
        color = (255, 0, 255) if mode == "motion" else (0, 255, 0)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), color, 3)
        
        # Draw hand landmarks if available
        if hand_landmarks:
            for hand_lms in hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        
        # Predictions
        cv2.putText(frame, f"Letter: {letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {mode.upper()}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Debug info
        cv2.putText(frame, f"Static: {debug_info['static']}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Motion: {debug_info['motion']}", (50, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Motion Score: {debug_info['motion_score']:.2f}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return frame
    
    def run(self):
        """Run the combined system"""
        if self.static_model is None and self.motion_model is None:
            print("No models loaded!")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera!")
            return
        
        print("Combined ASL System Running!")
        print("Static letters: A-Y (except J,Z)")
        print("Motion letters: J (hook), Z (trace)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                
                # ROI coordinates
                roi_size = 300
                roi_x = (width - roi_size) // 2
                roi_y = (height - roi_size) // 2
                roi_coords = (roi_x, roi_y, roi_size, roi_size)
                
                # Get prediction
                letter, confidence, mode, hand_landmarks, debug_info = self.get_prediction(frame, roi_coords)
                
                # Draw UI
                frame = self.draw_ui(frame, letter, confidence, mode, hand_landmarks, debug_info)
                
                cv2.imshow('Combined ASL Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = SimplifiedCombinedASL()
    system.run()