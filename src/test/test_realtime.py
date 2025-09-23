"""
Step 4: Real-time ASL Recognition Testing
========================================

This script tests your trained model with live camera input.
You'll see your custom model recognizing A, B, C, D in real-time!
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU before TF import
import tensorflow as tf


import cv2
import numpy as np
from pathlib import Path
import time

class ASLRealTimeTester:
    """
    Real-time tester for your trained ASL model
    
    What this does:
    - Opens your webcam
    - Shows live predictions from your custom model
    - Displays confidence scores and letter predictions
    - Runs smoothly at high FPS
    """
    
    def __init__(self, model_path="models/asl_model_simple.keras"):
        print("Loading your trained ASL model...")
        
        # Load your trained model
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}")
            print("Train your model first: python src/train_model.py")
            return
        
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Define the letters your model can recognize
        # This should match what you trained on
        self.letters = ['A', 'B', 'C', 'D']
        self.index_to_letter = {i: letter for i, letter in enumerate(self.letters)}
        
        # For smooth predictions
        self.prediction_history = []
        self.history_size = 5  # Average over last 5 predictions
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        print(f"Model ready to recognize: {self.letters}")
    
    def preprocess_image(self, frame, roi_coords):
        """
        Preprocess the camera frame for the model
        
        Args:
            frame: Camera frame
            roi_coords: (x, y, width, height) of region of interest
        
        Returns:
            Preprocessed image ready for the model
        """
        x, y, w, h = roi_coords
        
        # Extract region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Resize to model input size
        roi_resized = cv2.resize(roi, (224, 224))
        
        # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 range (same as training)
        roi_normalized = roi_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        roi_batch = np.expand_dims(roi_normalized, axis=0)
        
        return roi_batch
    
    def predict_letter(self, processed_image):
        """
        Get prediction from the model
        
        Args:
            processed_image: Preprocessed image from preprocess_image()
        
        Returns:
            predicted_letter: The letter (A, B, C, or D)
            confidence: How confident the model is (0-1)
            all_probs: Probabilities for all letters
        """
        
        # Get model prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get the most likely letter
        predicted_index = np.argmax(predictions[0])
        predicted_letter = self.index_to_letter[predicted_index]
        confidence = predictions[0][predicted_index]
        
        # Get all probabilities for display
        all_probs = {self.letters[i]: predictions[0][i] for i in range(len(self.letters))}
        
        return predicted_letter, confidence, all_probs
    
    def smooth_prediction(self, current_prediction):
        """
        Smooth predictions over time to reduce jitter
        
        Args:
            current_prediction: Current letter prediction
        
        Returns:
            smoothed_prediction: Most common letter in recent history
        """
        
        # Add current prediction to history
        self.prediction_history.append(current_prediction)
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Return most common recent prediction
        if len(self.prediction_history) >= 3:  # Need at least 3 for smoothing
            # Count occurrences
            counts = {letter: self.prediction_history.count(letter) for letter in self.letters}
            return max(counts, key=counts.get)
        else:
            return current_prediction
    
    def draw_ui(self, frame, predicted_letter, confidence, all_probs, smoothed_letter):
        """
        Draw the user interface on the frame
        
        Args:
            frame: Camera frame
            predicted_letter: Raw prediction
            confidence: Confidence score
            all_probs: All letter probabilities
            smoothed_letter: Smoothed prediction
        
        Returns:
            frame_with_ui: Frame with UI elements drawn
        """
        
        height, width = frame.shape[:2]
        
        # Draw region of interest rectangle
        roi_size = 300
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        # Color based on confidence
        if confidence > 0.9:
            roi_color = (0, 255, 0)  # Green - very confident
        elif confidence > 0.7:
            roi_color = (0, 255, 255)  # Yellow - confident
        else:
            roi_color = (0, 0, 255)  # Red - uncertain
        
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), roi_color, 3)
        
        # Main prediction display
        cv2.putText(frame, f"Letter: {smoothed_letter}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show all probabilities
        y_start = 150
        cv2.putText(frame, "All Predictions:", (50, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, letter in enumerate(self.letters):
            prob = all_probs[letter]
            color = (0, 255, 0) if letter == predicted_letter else (255, 255, 255)
            
            # Draw letter and probability
            y_pos = y_start + 30 + (i * 25)
            cv2.putText(frame, f"{letter}: {prob:.1%}", (70, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw probability bar
            bar_width = int(200 * prob)
            cv2.rectangle(frame, (150, y_pos - 15), (150 + bar_width, y_pos - 5), color, -1)
            cv2.rectangle(frame, (150, y_pos - 15), (350, y_pos - 5), (100, 100, 100), 1)
        
        # FPS display
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (width - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "Place hand in colored rectangle",
            "Make clear letter signs (A, B, C, D)",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (50, height - 80 + (i * 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """
        Run the real-time ASL recognition system
        """
        
        if not hasattr(self, 'model'):
            print("Model not loaded properly!")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera started!")
        print("Instructions:")
        print("   - Place your hand in the colored rectangle")
        print("   - Make clear signs for letters A, B, C, or D")
        print("   - Watch your custom model predict in real-time!")
        print("   - Press 'q' to quit")
        print("Starting real-time recognition...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Define ROI coordinates
                height, width = frame.shape[:2]
                roi_size = 300
                roi_x = (width - roi_size) // 2
                roi_y = (height - roi_size) // 2
                roi_coords = (roi_x, roi_y, roi_size, roi_size)
                
                # Preprocess the ROI for the model
                processed_image = self.preprocess_image(frame, roi_coords)
                
                # Get prediction
                predicted_letter, confidence, all_probs = self.predict_letter(processed_image)
                
                # Smooth the prediction
                smoothed_letter = self.smooth_prediction(predicted_letter)
                
                # Draw UI
                frame_with_ui = self.draw_ui(frame, predicted_letter, confidence, all_probs, smoothed_letter)
                
                # Calculate FPS
                self.fps_counter += 1
                if time.time() - self.fps_timer > 1.0:
                    self.current_fps = self.fps_counter / (time.time() - self.fps_timer)
                    self.fps_counter = 0
                    self.fps_timer = time.time()
                
                # Show the frame
                cv2.imshow('ASL Real-time Recognition - Your Custom Model!', frame_with_ui)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")
            print("Thanks for testing your custom ASL model!")


def main():
    """
    Main function to run real-time testing
    """
    
    print("ASL Real-time Recognition Test")
    print("=" * 40)
    print("Testing your custom trained model!")
    
    # Check if model exists
    model_path = "models/asl_model_simple.keras"
    if not Path(model_path).exists():
        print(f"No trained model found at {model_path}")
        print("Train your model first:")
        print(" python src/train_model.py")
        return
    
    # Create and run tester
    tester = ASLRealTimeTester(model_path)
    tester.run()


if __name__ == "__main__":
    main()