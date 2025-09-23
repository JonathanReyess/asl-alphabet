"""
Step 3: Simple Model Training
============================

This script trains your ASL model on the images you collected.
We'll start with just A, B, C, D to see it working quickly.
"""

import numpy as np
import cv2
import os
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Import our model
from asl_model import SimpleASLModel

from tensorflow.keras import mixed_precision # type: ignore

# Force float32 (avoid buggy float16 on Metal)
mixed_precision.set_global_policy("float32")

# Allow dynamic GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

print("Devices:", gpus)

class ASLTrainer:
    """
    Simple trainer for the ASL model
    
    What this does:
    - Loads your collected images
    - Splits them into training and testing sets
    - Trains the model
    - Shows you the results
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.model_trainer = SimpleASLModel()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, letters=['A', 'B', 'C', 'D']):
        """
        Load images from the data directory
        
        Args:
            letters: List of letters to include in training
        """
        
        print(f"Loading data for letters: {letters}")
        
        images = []
        labels = []
        
        for letter in letters:
            letter_dir = self.data_dir / letter
            
            if not letter_dir.exists():
                print(f"⚠️  No data found for letter {letter}")
                continue
            
            # Get all jpg files for this letter
            image_files = list(letter_dir.glob("*.jpg"))
            print(f"📸 Found {len(image_files)} images for letter {letter}")
            
            for img_path in image_files:
                # Load image
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Convert BGR to RGB (OpenCV loads as BGR, but models expect RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Make sure it's the right size
                    img = cv2.resize(img, (224, 224))
                    
                    # Convert to float and normalize to 0-1 range
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(letter)
        
        print(f"Loaded {len(images)} total images")
        print(f"Letters distribution:")
        for letter in letters:
            count = labels.count(letter)
            print(f"   {letter}: {count} images")
        
        return np.array(images), np.array(labels)
    
    def prepare_data(self, letters=['A', 'B', 'C', 'D']):
        """
        Load and prepare data for training
        """
        
        print("Preparing training data...")
        
        # Load images and labels
        X, y = self.load_data(letters)
        
        if len(X) == 0:
            print(" No data loaded! Make sure you collected images first.")
            return False
        
        # Convert letters to numbers (A=0, B=1, C=2, D=3)
        # Update our model to only use the letters we have
        self.model_trainer.letter_to_index = {letter: i for i, letter in enumerate(letters)}
        self.model_trainer.index_to_letter = {i: letter for letter, i in self.model_trainer.letter_to_index.items()}
        self.model_trainer.num_classes = len(letters)
        
        # Convert string labels to numbers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Convert to one-hot encoding (required for categorical crossentropy)
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(letters))
        
        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_categorical, 
            test_size=0.2,  # 20% for testing
            random_state=42,  # For reproducible results
            stratify=y_encoded  # Keep same proportion of each letter in train/test
        )
        
        print(f"Data prepared!")
        print(f"Training set: {len(self.X_train)} images")
        print(f"Test set: {len(self.X_test)} images")
        
        return True
    
    def train_model(self, epochs=20, batch_size=1):
        """
        Train the model on your data
        
        Args:
            epochs: How many times to go through all the data
            batch_size: How many images to process at once
        """
        
        if self.X_train is None:
            print("Prepare data first!")
            return
        
        print(f"Starting training...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Build and compile the model for our specific number of letters
        model = self.model_trainer.build_model()
        self.model_trainer.compile_model(learning_rate=0.001)
        
        # Train the model
        print("Training in progress...")
        history = model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            verbose=1  # Show progress
        )
        
        print("Training completed!")
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Final test accuracy: {test_accuracy:.2%}")
        
        # Save the model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "asl_model_simple.keras"
        model.save(str(model_path))
        print(f"Model saved to: {model_path}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def plot_training_history(self, history):
        """
        Create plots showing how training went
        """
        
        print("Creating training plots...")
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Save the plot
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        plot_path = logs_dir / "training_history.png"
        plt.savefig(str(plot_path))
        print(f"Training plots saved to: {plot_path}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def test_predictions(self, num_samples=8):
        """
        Test the model on some random images and show predictions
        """
        
        if self.X_test is None:
            print("No test data available!")
            return
        
        print(f"Testing model on {num_samples} random images...")
        
        # Get random samples from test set
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        # Make predictions
        model = tf.keras.models.load_model("models/asl_model_simple.keras")
        
        for i, idx in enumerate(indices):
            # Get image and true label
            image = self.X_test[idx]
            true_label_idx = np.argmax(self.y_test[idx])
            true_label = self.model_trainer.index_to_letter[true_label_idx]
            
            # Make prediction
            prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
            predicted_idx = np.argmax(prediction[0])
            predicted_label = self.model_trainer.index_to_letter[predicted_idx]
            confidence = prediction[0][predicted_idx]
            
            # Show result
            status = "Good" if predicted_label == true_label else "Bad"
            print(f"{status} Image {i+1}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.2%}")


def main():
    """
    Main training function
    """
    
    print("ASL Model Training")
    print("=" * 40)
    
    # Create trainer
    trainer = ASLTrainer()
    
    # Which letters to train on (you can add more as you collect data)
    letters_to_train = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y'
]

    
    print(f"Training model to recognize: {letters_to_train}")
    
    # Check if we have data
    total_images = 0
    for letter in letters_to_train:
        letter_dir = Path("data") / letter
        if letter_dir.exists():
            count = len(list(letter_dir.glob("*.jpg")))
            total_images += count
            print(f"📸 Letter {letter}: {count} images")
        else:
            print(f"⚠️  No data found for letter {letter}")
    
    if total_images < 50:
        print("Not enough data! Collect more images first.")
        print("Run: python src/collect_data.py")
        return
    
    # Prepare data
    if not trainer.prepare_data(letters_to_train):
        return
    
    # Train the model
    print("\n" + "=" * 40)
    print("Starting Training!")
    print("=" * 40)
    
    model, history = trainer.train_model(epochs=20, batch_size=16)
    
    # Test some predictions
    print("\n" + "=" * 40)
    print("Testing Predictions")
    print("=" * 40)
    
    trainer.test_predictions(num_samples=8)
    
    print("\nTraining complete!")
    print(" Next: Test your model with real-time camera input!")


if __name__ == "__main__":
    main()