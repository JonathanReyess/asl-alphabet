"""
Step 2: Simple Data Collection for ASL Letters
==============================================

This script helps collect training images for each ASL letter using a webcam one letter at a time.
"""

import cv2
import os
import numpy as np
from pathlib import Path

class ASLDataCollector:
    """
    Simple data collector for ASL letters
    
    What this does:
    - Opens webcam
    - Shows a region where hand should be placed
    - Captures images when spacebar is pressed
    - Saves images organized by letter
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                       'T', 'U', 'V', 'W', 'X', 'Y']
        
        # Create directories for each letter
        self.setup_directories()
    
    def setup_directories(self):
        """Create folders to store images for each letter"""
        for letter in self.letters:
            letter_dir = self.data_dir / letter
            letter_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Created directories in {self.data_dir}")
    
    def collect_letter_data(self, letter, target_images=100):
        """
        Collect images for a specific letter
        
        Args:
            letter: The letter to collect (e.g., 'A')
            target_images: How many images to collect
        """
        
        if letter not in self.letters:
            print(f"Letter {letter} not in our list!")
            return
        
        # Setup camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera!")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        letter_dir = self.data_dir / letter
        existing_images = len(list(letter_dir.glob("*.jpg")))
        
        print(f"\nCollecting data for letter: {letter}")
        print(f"Existing images: {existing_images}")
        print(f"Target: {target_images} total images")
        print(f"Need to capture: {max(0, target_images - existing_images)} more")
        print("\nInstructions:")
        print("- Position your hand in the blue rectangle")
        print("- Make the letter sign clearly")
        print("- Press SPACE to capture an image")
        print("- Press 'q' to quit")
        print("- Press 'n' to move to next letter")
        
        count = existing_images
        
        while count < target_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect (more natural)
            frame = cv2.flip(frame, 1)
            
            # Define region of interest (ROI) - where hand should be
            roi_size = 300
            start_x = (frame.shape[1] - roi_size) // 2
            start_y = (frame.shape[0] - roi_size) // 2
            end_x = start_x + roi_size
            end_y = start_y + roi_size
            
            # Draw the ROI rectangle
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3)
            
            # Add text information
            cv2.putText(frame, f"Letter: {letter}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Images: {count}/{target_images}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 'n' for next", (50, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('ASL Data Collection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar - capture image
                # Extract the ROI
                roi = frame[start_y:end_y, start_x:end_x]
                
                # Resize to our model's expected size (224x224)
                roi_resized = cv2.resize(roi, (224, 224))
                
                # Save the image
                filename = letter_dir / f"{letter}_{count:04d}.jpg"
                cv2.imwrite(str(filename), roi_resized)
                
                count += 1
                print(f"Captured image {count}/{target_images}")
                
                # Brief pause so you can reposition
                cv2.waitKey(200)
                
            elif key == ord('q'):  # Quit
                print("Quitting data collection")
                break
                
            elif key == ord('n'):  # Next letter
                print(f"Moving to next letter (collected {count} images for {letter})")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Finished collecting data for letter {letter}")
        print(f"Total images collected: {count}")
    
    def collect_all_letters(self, images_per_letter=100):
        """
        Collect data for all letters one by one
        
        Args:
            images_per_letter: How many images to collect for each letter
        """
        
        print("Starting data collection for ALL letters!")
        print(f"Target: {images_per_letter} images per letter")
        print(f"Total images to collect: {len(self.letters) * images_per_letter}")
        
        for i, letter in enumerate(self.letters):
            print(f"\n{'='*50}")
            print(f"Letter {i+1}/{len(self.letters)}: {letter}")
            print(f"{'='*50}")
            
            input(f"Get ready to show letter '{letter}' and press Enter...")
            
            self.collect_letter_data(letter, images_per_letter)
            
            if i < len(self.letters) - 1:  # Not the last letter
                print(f"\nCompleted letter {letter}!")
                choice = input("Continue to next letter? (y/n): ").lower()
                if choice != 'y':
                    print("Stopping data collection")
                    break
        
        print("\nData collection complete!")
        self.show_collection_summary()
    
    def show_collection_summary(self):
        """Show a summary of collected data"""
        print("\nData Collection Summary:")
        print("="*40)
        
        total_images = 0
        for letter in self.letters:
            letter_dir = self.data_dir / letter
            count = len(list(letter_dir.glob("*.jpg")))
            total_images += count
            
            status = "Good" if count >= 50 else "!" if count >= 20 else "Bad"
            print(f"{status} Letter {letter}: {count} images")
        
        print(f"\nTotal images: {total_images}")
        print(f"Average per letter: {total_images/len(self.letters):.1f}")
        
        if total_images >= 1000:
            print("Great! You have enough data to start training!")
        elif total_images >= 500:
            print("Good start! More data will improve accuracy.")
        else:
            print("You might need more data for good results.")


# Example usage and main function
if __name__ == "__main__":
    print("ASL Data Collection Tool")
    print("="*40)
    
    collector = ASLDataCollector()
    
    print("\nChoose an option:")
    print("1. Collect data for a specific letter")
    print("2. Collect data for all letters")
    print("3. Show current collection summary")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        letter = input("Enter letter to collect (A-Y, except J,Z): ").upper()
        images = int(input("How many images to collect (default 100): ") or "100")
        collector.collect_letter_data(letter, images)
        
    elif choice == "2":
        images = int(input("Images per letter (default 100): ") or "100")
        collector.collect_all_letters(images)
        
    elif choice == "3":
        collector.show_collection_summary()
        
    else:
        print("Invalid choice!")