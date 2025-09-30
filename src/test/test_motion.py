"""
Motion Model Test
=================

Test the trained motion LSTM model for J and Z gesture recognition
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

import cv2
import numpy as np
from collections import deque
from pathlib import Path
import time

def motion_model_test():
    """Test motion model with improved preprocessing and error handling"""
    
    print("Motion Model Test Starting...")
    print("Testing J and Z gesture recognition")
    
    # Load motion model
    model_path = "models/motion_model.keras"
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train motion model first: python src/train/train_motion.py")
        return
    
    try:
        motion_model = tf.keras.models.load_model(model_path)
        print("Motion model loaded successfully!")
        print(f"Model expects input shape: {motion_model.input_shape}")
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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        mp_drawing = mp.solutions.drawing_utils
        print("MediaPipe loaded successfully!")
    except Exception as e:
        print(f"MediaPipe error: {e}")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera opened successfully!")
    print("\nInstructions:")
    print("   - Show your hand to see landmark tracking")
    print("   - Make clear J or Z motions")
    print("   - J: Hook motion with pinky finger")
    print("   - Z: Trace the letter Z shape")
    print("   - Wait for 20 frames to collect sequence")
    print("   - Press 'q' to quit")
    print("\nStarting motion recognition...")
    
    # Tracking variables
    landmark_history = deque(maxlen=20)
    frame_count = 0
    prediction_history = deque(maxlen=5)
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            height, width = frame.shape[:2]
            
            # Get hand landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            landmarks = None
            if results.multi_hand_landmarks:
                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates (63 values: 21 landmarks * 3 coordinates)
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks = np.array(landmarks)
                
                # Verify correct size
                if len(landmarks) != 63:
                    print(f"Warning: Expected 63 landmarks, got {len(landmarks)}")
            
            # Add to history
            landmark_history.append(landmarks)
            
            # Draw UI
            # Frame info
            cv2.putText(frame, f"Frame: {frame_count}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Hand detection status
            hand_color = (0, 255, 0) if landmarks is not None else (0, 0, 255)
            landmarks_status = f"Hand Detected: {'YES' if landmarks is not None else 'NO'}"
            cv2.putText(frame, landmarks_status, (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
            
            # Sequence collection status
            sequence_progress = len(landmark_history)
            sequence_status = f"Sequence: {sequence_progress}/20"
            sequence_color = (0, 255, 0) if sequence_progress == 20 else (255, 255, 0)
            cv2.putText(frame, sequence_status, (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, sequence_color, 2)
            
            # Progress bar for sequence collection
            bar_width = 300
            bar_height = 20
            bar_x = 50
            bar_y = 170
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Progress bar
            progress_width = int((sequence_progress / 20) * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), sequence_color, -1)
            
            # Try prediction if we have enough frames
            if len(landmark_history) == 20:
                try:
                    # Prepare sequence (same preprocessing as training)
                    sequence = []
                    for lm in landmark_history:
                        if lm is not None and len(lm) == 63:
                            sequence.append(lm)
                        else:
                            sequence.append(np.zeros(63))
                    
                    sequence = np.array(sequence)
                    
                    # Normalize relative to wrist position (same as training)
                    wrist_pos = sequence[:, :3]  # First landmark (wrist) x,y,z
                    normalized_sequence = sequence.copy()
                    
                    for i in range(0, 63, 3):  # Every landmark (x,y,z)
                        normalized_sequence[:, i:i+3] = normalized_sequence[:, i:i+3] - wrist_pos
                    
                    # Add batch dimension
                    sequence_batch = np.expand_dims(normalized_sequence, axis=0)
                    
                    # Predict
                    predictions = motion_model.predict(sequence_batch, verbose=0)
                    predicted_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_idx]
                    
                    letters = ['J', 'Z']
                    predicted_letter = letters[predicted_idx]
                    
                    # Add to prediction history for smoothing
                    if confidence > 0.5:
                        prediction_history.append(predicted_letter)
                    
                    # Get smoothed prediction
                    if len(prediction_history) >= 3:
                        # Most common recent prediction
                        counts = {letter: list(prediction_history).count(letter) for letter in ['J', 'Z']}
                        smoothed_letter = max(counts, key=counts.get)
                    else:
                        smoothed_letter = predicted_letter
                    
                    # Display prediction
                    pred_text = f"Prediction: {predicted_letter}"
                    smoothed_text = f"Smoothed: {smoothed_letter}"
                    conf_text = f"Confidence: {confidence:.1%}"
                    
                    # Color based on confidence
                    if confidence > 0.8:
                        pred_color = (0, 255, 0)  # Green - very confident
                    elif confidence > 0.6:
                        pred_color = (0, 255, 255)  # Yellow - confident
                    else:
                        pred_color = (0, 100, 255)  # Orange - uncertain
                    
                    cv2.putText(frame, pred_text, (50, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, pred_color, 2)
                    cv2.putText(frame, conf_text, (50, 290), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, smoothed_text, (50, 330), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    # Show both class probabilities
                    j_prob = predictions[0][0]
                    z_prob = predictions[0][1]
                    cv2.putText(frame, f"J: {j_prob:.1%}", (400, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Z: {z_prob:.1%}", (400, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                except Exception as e:
                    error_text = f"Prediction Error: {str(e)[:40]}"
                    cv2.putText(frame, error_text, (50, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print(f"Prediction error: {e}")
            else:
                remaining = 20 - len(landmark_history)
                cv2.putText(frame, f"Collecting sequence... {remaining} frames left", (50, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # FPS calculation
            fps_counter += 1
            if time.time() - fps_timer > 1.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
            
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (width - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Make J (hook) or Z (trace) motions", (50, height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, "Press 'q' to quit", (50, height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Model info
            cv2.putText(frame, "Motion LSTM - 100% Test Accuracy", (50, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show frame
            cv2.imshow('Motion Model Test - J and Z Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Motion model test completed!")
        print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    motion_model_test()