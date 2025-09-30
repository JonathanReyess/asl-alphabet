"""
Real-time ASL Recognition Testing
=================================

Tests the trained ASL model with live camera input for all 24 static letters.
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
    Real-time tester for trained ASL model
    
    Tests the model with live camera input for all 24 static letters
    """
    
    def __init__(self, model_path="models/asl_model_simple.keras"):
        print("Loading trained ASL model...")
        
        # Load trained model
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}")
            print("Train your model first: python src/train/train_model.py")
            return
        
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Updated to match your actual 24-letter model
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        self.index_to_letter = {i: letter for i, letter in enumerate(self.letters)}
        
        # For smooth predictions
        self.prediction_history = []
        self.history_size = 5
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        print(f"Model ready to recognize: {len(self.letters)} letters")
        print(f"Letters: {', '.join(self.letters)}")
    
    def preprocess_image(self, frame, roi_coords):
        """
        Preprocess camera frame for the model
        
        Args:
            frame: Camera frame
            roi_coords: (x, y, width, height) of region of interest
        
        Returns:
            Preprocessed image ready for the model
        """
        x, y, w, h = roi_coords
        
        # Extract region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Resize to model input size (224x224)
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
            processed_image: Preprocessed image
        
        Returns:
            predicted_letter: The predicted letter
            confidence: Confidence score (0-1)
            top_predictions: Top 3 predictions with probabilities
        """
        
        # Get model prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get the most likely letter
        predicted_index = np.argmax(predictions[0])
        predicted_letter = self.index_to_letter[predicted_index]
        confidence = predictions[0][predicted_index]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [(self.letters[i], predictions[0][i]) for i in top_indices]
        
        return predicted_letter, confidence, top_predictions
    
    def smooth_prediction(self, current_prediction, confidence):
        """
        Smooth predictions over time to reduce jitter
        
        Args:
            current_prediction: Current letter prediction
            confidence: Current confidence score
        
        Returns:
            smoothed_prediction: Most common letter in recent history
        """
        
        # Only add to history if confidence is reasonable
        if confidence > 0.3:
            self.prediction_history.append(current_prediction)
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Return most common recent prediction
        if len(self.prediction_history) >= 3:
            # Count occurrences
            counts = {}
            for letter in self.prediction_history:
                counts[letter] = counts.get(letter, 0) + 1
            return max(counts, key=counts.get)
        else:
            return current_prediction
    
    def draw_ui(self, frame, predicted_letter, confidence, top_predictions, smoothed_letter):
        """
        Draw the user interface on the frame
        
        Args:
            frame: Camera frame
            predicted_letter: Raw prediction
            confidence: Confidence score
            top_predictions: Top 3 predictions
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
        if confidence > 0.8:
            roi_color = (0, 255, 0)  # Green - very confident
        elif confidence > 0.6:
            roi_color = (0, 255, 255)  # Yellow - confident
        else:
            roi_color = (0, 0, 255)  # Red - uncertain
        
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), roi_color, 3)
        
        # Main prediction display
        cv2.putText(frame, f"Letter: {smoothed_letter}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show top 3 predictions
        y_start = 150
        cv2.putText(frame, "Top Predictions:", (50, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, (letter, prob) in enumerate(top_predictions):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            y_pos = y_start + 30 + (i * 25)
            cv2.putText(frame, f"{i+1}. {letter}: {prob:.1%}", (70, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # FPS display
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (width - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Model info
        cv2.putText(frame, f"Model: 24 static letters", (50, height - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Accuracy: 98.3%", (50, height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instructions
        instructions = [
            "Place hand in colored rectangle",
            "Make clear letter signs (A-Y except J,Z)", 
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (50, height - 50 + (i * 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
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
        print("   - Make clear signs for any of the 24 letters")
        print("   - Green rectangle = high confidence")
        print("   - Yellow rectangle = medium confidence") 
        print("   - Red rectangle = low confidence")
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
                predicted_letter, confidence, top_predictions = self.predict_letter(processed_image)
                
                # Smooth the prediction
                smoothed_letter = self.smooth_prediction(predicted_letter, confidence)
                
                # Draw UI
                frame_with_ui = self.draw_ui(frame, predicted_letter, confidence, top_predictions, smoothed_letter)
                
                # Calculate FPS
                self.fps_counter += 1
                if time.time() - self.fps_timer > 1.0:
                    self.current_fps = self.fps_counter / (time.time() - self.fps_timer)
                    self.fps_counter = 0
                    self.fps_timer = time.time()
                
                # Show the frame
                cv2.imshow('ASL Real-time Recognition - 24 Letters', frame_with_ui)
                
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
            print("Thanks for testing your ASL model!")


def main():
    """
    Main function to run real-time testing
    """
    
    print("ASL Real-time Recognition Test")
    print("=" * 40)
    print("Testing trained model on 24 static letters!")
    
    # Check if model exists
    model_path = "models/asl_model_simple.keras"
    if not Path(model_path).exists():
        print(f"No trained model found at {model_path}")
        print("Train your model first:")
        print("  python src/train/train_model.py")
        return
    
    # Create and run tester
    tester = ASLRealTimeTester(model_path)
    tester.run()


if __name__ == "__main__":
    main()