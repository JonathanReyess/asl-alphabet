"""
Simple Motion Test
=================

Basic test to isolate motion model issues
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

import cv2
import numpy as np
from collections import deque
from pathlib import Path

def simple_motion_test():
    """Simple test focusing on just getting motion working"""
    
    print("Simple Motion Test Starting...")
    
    # Load motion model
    model_path = "models/motion_model.keras"
    if not Path(model_path).exists():
        print(f" Model not found: {model_path}")
        return
    
    try:
        motion_model = tf.keras.models.load_model(model_path)
        print("Motion model loaded!")
    except Exception as e:
        print(f"Model loading error: {e}")
        return
    
    # Load MediaPipe
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        print("MediaPipe loaded!")
    except Exception as e:
        print(f"MediaPipe error: {e}")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed!")
        return
    
    print("Camera opened!")
    print("Instructions:")
    print("   - Show your hand to see blue dots")
    print("   - Make J or Z motions")
    print("   - Press 'q' to quit")
    
    # Tracking variables
    landmark_history = deque(maxlen=20)
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Get hand landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            landmarks = None
            if results.multi_hand_landmarks:
                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks = np.array(landmarks)
            
            # Add to history
            landmark_history.append(landmarks)
            
            # Draw simple status
            cv2.putText(frame, f"Frame: {frame_count}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            landmarks_status = f"Hand: {'ON' if landmarks is not None else 'OFF'}"
            cv2.putText(frame, landmarks_status, (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if landmarks is not None else (0, 0, 255), 2)
            
            sequence_status = f"Sequence: {len(landmark_history)}/20"
            cv2.putText(frame, sequence_status, (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Try prediction if we have enough frames
            if len(landmark_history) == 20:
                try:
                    # Prepare sequence
                    sequence = []
                    for lm in landmark_history:
                        if lm is not None:
                            sequence.append(lm)
                        else:
                            sequence.append(np.zeros(63))
                    
                    sequence = np.array(sequence)
                    
                    # Simple normalization (subtract first frame)
                    if not np.all(sequence[0] == 0):
                        normalized_sequence = sequence - sequence[0]
                    else:
                        normalized_sequence = sequence
                    
                    # Add batch dimension
                    sequence_batch = np.expand_dims(normalized_sequence, axis=0)
                    
                    # Predict
                    predictions = motion_model.predict(sequence_batch, verbose=0)
                    predicted_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_idx]
                    
                    letters = ['J', 'Z']
                    predicted_letter = letters[predicted_idx]
                    
                    # Display prediction
                    pred_text = f"Prediction: {predicted_letter} ({confidence:.1%})"
                    color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
                    cv2.putText(frame, pred_text, (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                except Exception as e:
                    cv2.putText(frame, f"Prediction Error: {str(e)[:30]}", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Collecting sequence...", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Show frame
            cv2.imshow('Simple Motion Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Simple motion test completed!")

if __name__ == "__main__":
    simple_motion_test()