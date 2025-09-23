"""
Step 5: Motion Detection for ASL Letters J and Z
===============================================

This adds motion detection to handle the dynamic letters J and Z.
We'll use MediaPipe to track hand landmarks and LSTM to learn sequences.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from collections import deque
import time
from pathlib import Path

class HandLandmarkExtractor:
    """
    Extract hand landmarks using MediaPipe for motion detection
    
    What this does:
    - Uses MediaPipe to find 21 hand landmarks (fingertips, joints, etc.)
    - Tracks hand movement over time
    - Provides normalized landmark coordinates
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from a frame
        
        Args:
            frame: Camera frame (BGR format)
        
        Returns:
            landmarks: Array of 63 values (21 landmarks × 3 coordinates) or None
        """
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
            landmarks = []
            
            # Extract x, y, z coordinates for each of 21 landmarks
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                
            return np.array(landmarks)  # 63 values total
        
        return None
    
    def draw_landmarks(self, frame):
        """
        Draw hand landmarks on frame for visualization
        
        Args:
            frame: Camera frame
        
        Returns:
            frame_with_landmarks: Frame with landmarks drawn
        """
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
        return frame_bgr

class MotionASLModel:
    """
    LSTM model for recognizing motion-based ASL letters (J, Z)
    
    What this does:
    - Takes sequences of hand landmarks over time
    - Uses LSTM to learn motion patterns
    - Predicts J or Z based on hand movement
    """
    
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length  # How many frames to look at
        self.num_landmarks = 63  # 21 landmarks × 3 coordinates
        self.num_classes = 2  # J and Z
        self.model = None
        
        # Letter mappings
        self.letters = ['J', 'Z']
        self.letter_to_index = {'J': 0, 'Z': 1}
        self.index_to_letter = {0: 'J', 1: 'Z'}
        
        print(f"Motion model will recognize: {self.letters}")
        print(f"Sequence length: {sequence_length} frames")
    
    def build_model(self):
        """
        Build LSTM model for motion recognition
        
        Model architecture:
        - Input: Sequences of hand landmarks
        - LSTM layers: Learn temporal patterns
        - Dense layers: Final classification
        - Output: J or Z prediction
        """
        
        # Input: sequence of landmark coordinates
        inputs = layers.Input(
            shape=(self.sequence_length, self.num_landmarks), 
            name='landmark_sequence'
        )
        
        # Normalize the input data
        x = layers.LayerNormalization(name='input_norm')(inputs)
        
        # LSTM layers to learn motion patterns
        x = layers.LSTM(
            64, 
            return_sequences=True,  # Return full sequence
            dropout=0.2,
            name='lstm_1'
        )(x)
        
        x = layers.LSTM(
            32, 
            return_sequences=False,  # Only return final output
            dropout=0.2,
            name='lstm_2'
        )(x)
        
        # Dense layers for final decision
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.3, name='dropout')(x)
        
        # Output layer - predict J or Z
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='predictions'
        )(x)
        
        self.model = Model(inputs, outputs, name='Motion_ASL_Model')
        
        print("Motion model architecture created!")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the motion model for training"""
        
        if self.model is None:
            print("Build the model first!")
            return
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Motion model compiled!")
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            print("Build the model first!")
            return
        
        print("\nMotion Model Architecture:")
        print("=" * 50)
        self.model.summary()

class MotionDataCollector:
    """
    Collect motion data for J and Z using hand landmarks
    
    What this does:
    - Captures sequences of hand movements
    - Records landmark coordinates over time
    - Saves sequences for training the motion model
    """
    
    def __init__(self, data_dir="data/motion"):
        self.data_dir = Path(data_dir)
        self.landmark_extractor = HandLandmarkExtractor()
        self.sequence_length = 20
        
        # Create directories
        for letter in ['J', 'Z']:
            (self.data_dir / letter).mkdir(parents=True, exist_ok=True)
    
    def collect_letter_sequences(self, letter, num_sequences=50):
        """
        Collect motion sequences for a specific letter
        
        Args:
            letter: 'J' or 'Z'
            num_sequences: How many sequences to collect
        """
        
        if letter not in ['J', 'Z']:
            print(f"This is for motion letters J and Z only!")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera!")
            return
        
        print(f"\n🎯 Collecting motion data for letter: {letter}")
        print(f"Target: {num_sequences} sequences")
        
        if letter == 'J':
            print("Instructions for J:")
            print("   - Start with pinky finger extended")
            print("   - Make a small hook motion with your pinky")
            print("   - Keep the motion clear and consistent")
        else:  # Z
            print("Instructions for Z:")
            print("   - Point with your index finger")
            print("   - Trace the letter 'Z' in the air")
            print("   - Make the motion smooth and deliberate")
        
        print("\n🎮 Controls:")
        print("   - Press SPACE to start recording a sequence")
        print("   - Perform the motion while recording")
        print("   - Sequence will auto-stop after recording")
        print("   - Press 'q' to quit")
        
        count = 0
        recording = False
        current_sequence = []
        
        while count < num_sequences:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Extract landmarks
            landmarks = self.landmark_extractor.extract_landmarks(frame)
            
            # Draw landmarks for visualization
            frame_with_landmarks = self.landmark_extractor.draw_landmarks(frame)
            
            # Recording logic
            if recording:
                if landmarks is not None:
                    current_sequence.append(landmarks)
                else:
                    # Fill missing landmarks with zeros
                    current_sequence.append(np.zeros(63))
                
                # Check if sequence is complete
                if len(current_sequence) >= self.sequence_length:
                    # Save the sequence
                    sequence_array = np.array(current_sequence[:self.sequence_length])
                    filename = self.data_dir / letter / f"{letter}_{count:04d}.npy"
                    np.save(str(filename), sequence_array)
                    
                    count += 1
                    print(f"Recorded sequence {count}/{num_sequences}")
                    
                    # Reset for next sequence
                    recording = False
                    current_sequence = []
                    
                    # Brief pause
                    time.sleep(0.5)
            
            # Draw UI
            status_color = (0, 0, 255) if recording else (0, 255, 0)
            status_text = f"RECORDING {len(current_sequence)}/{self.sequence_length}" if recording else "READY"
            
            cv2.putText(frame_with_landmarks, f"Letter: {letter}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame_with_landmarks, f"Sequences: {count}/{num_sequences}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_with_landmarks, status_text, (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(frame_with_landmarks, "Press SPACE to record", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Motion Data Collection', frame_with_landmarks)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and not recording:
                # Start recording
                recording = True
                current_sequence = []
                print(f"Recording sequence {count + 1}...")
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Completed collecting {count} sequences for letter {letter}")

def test_landmark_extraction():
    """
    Quick test to make sure MediaPipe hand tracking works
    """
    
    print("Testing MediaPipe hand landmark extraction...")
    
    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera!")
        return False
    
    print("Instructions:")
    print("   - Show your hand to the camera")
    print("   - You should see blue dots and lines on your hand")
    print("   - Press 'q' to quit test")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks = extractor.extract_landmarks(frame)
            
            # Draw landmarks
            frame_with_landmarks = extractor.draw_landmarks(frame)
            
            # Show landmark count
            landmark_status = f"Landmarks: {len(landmarks) if landmarks is not None else 0}/63"
            color = (0, 255, 0) if landmarks is not None else (0, 0, 255)
            
            cv2.putText(frame_with_landmarks, landmark_status, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame_with_landmarks, "Press 'q' to quit", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('MediaPipe Hand Tracking Test', frame_with_landmarks)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("MediaPipe test completed!")
    return True

def main():
    """
    Main function for motion detection setup
    """
    
    print("ASL Motion Detection Setup")
    print("=" * 40)
    print("Adding J and Z recognition to your system!")
    
    print("\nChoose an option:")
    print("1. Test MediaPipe hand tracking")
    print("2. Build motion model architecture")
    print("3. Collect motion data for J")
    print("4. Collect motion data for Z")
    print("5. Show motion model info")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        test_landmark_extraction()
        
    elif choice == "2":
        motion_model = MotionASLModel()
        motion_model.build_model()
        motion_model.compile_model()
        motion_model.get_model_summary()
        
    elif choice == "3":
        collector = MotionDataCollector()
        sequences = int(input("How many J sequences to collect (default 30): ") or "30")
        collector.collect_letter_sequences('J', sequences)
        
    elif choice == "4":
        collector = MotionDataCollector()
        sequences = int(input("How many Z sequences to collect (default 30): ") or "30")
        collector.collect_letter_sequences('Z', sequences)
        
    elif choice == "5":
        print("\nMotion Detection Info:")
        print("=" * 30)
        print("• Uses MediaPipe for hand landmark tracking")
        print("• Tracks 21 hand landmarks in 3D space")
        print("• LSTM model learns motion patterns")
        print("• Sequence length: 20 frames")
        print("• Letters: J (hook motion), Z (tracing motion)")
        
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()