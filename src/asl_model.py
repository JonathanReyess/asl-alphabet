"""
Step 1: Basic ASL Model Architecture
===================================

This file creates a simple custom CNN model to replace Teachable Machine.
We'll start with just the static letters (A-I, K-Y) - no motion yet.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import numpy as np

class SimpleASLModel:
    """
    A simple CNN model for recognizing static ASL letters
    
    What this does:
    - Takes in 224x224 images of hand signs
    - Uses a basic CNN architecture 
    - Outputs predictions for 24 letters (A-I, K-Y, excluding J and Z)
    """
    
    def __init__(self):
        self.model = None
        self.num_classes = 24  # A-I, K-Y (24 letters total, excluding J and Z)
        self.input_shape = (224, 224, 3)  # Standard image size: 224x224 pixels, 3 color channels
        
        # Create mapping from letters to numbers
        # We skip J and Z since they require motion
        letters = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in ['J', 'Z']]
        self.letter_to_index = {letter: i for i, letter in enumerate(letters)}
        self.index_to_letter = {i: letter for letter, i in self.letter_to_index.items()}
        
        print(f"Model will recognize these {self.num_classes} letters:")
        print(letters)
    
    def build_model(self):
        """
        Build the CNN model architecture
        
        What this creates:
        - Input layer for images
        - Several convolutional layers to detect features
        - Pooling layers to reduce image size
        - Dense layers for final classification
        - Output layer with 24 neurons (one for each letter)
        """
        
        # Input layer - where we feed in our 224x224 images
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # First set of conv + pooling layers
        # Conv layers detect edges, shapes, patterns in the image
        x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        
        # Second set - detects more complex patterns
        x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        
        # Third set - even more complex patterns
        x = layers.Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        
        # Flatten the 2D feature maps into 1D for the dense layers
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layers for final decision making
        x = layers.Dense(512, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)  # Prevents overfitting
        x = layers.Dense(256, activation='relu', name='dense2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        
        # Output layer - 24 neurons, one for each letter
        # Softmax gives us probabilities that sum to 1
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs, name='Simple_ASL_Model')
        
        print("✅ Model architecture created!")
        print(f"Input shape: {self.input_shape}")
        print(f"Output classes: {self.num_classes}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model for training
        
        What this does:
        - Sets up the optimizer (how the model learns)
        - Sets up the loss function (how we measure errors)
        - Sets up metrics to track (accuracy)
        """
        
        if self.model is None:
            print("❌ You need to build the model first!")
            return
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',  # Good for multi-class classification
            metrics=['accuracy']
        )
        
        print("✅ Model compiled and ready for training!")
        print(f"Learning rate: {learning_rate}")
    
    def get_model_summary(self):
        """Print a summary of the model architecture"""
        if self.model is None:
            print("❌ Build the model first!")
            return
        
        print("\n📋 Model Architecture Summary:")
        print("=" * 50)
        self.model.summary()
    
    def predict_letter(self, image):
        """
        Predict what letter is shown in an image
        
        Args:
            image: numpy array of shape (224, 224, 3)
        
        Returns:
            letter: predicted letter (e.g., 'A')
            confidence: how confident the model is (0-1)
        """
        
        if self.model is None:
            print("❌ Build and compile the model first!")
            return None, 0
        
        # Make sure image is the right shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Get prediction
        predictions = self.model.predict(image, verbose=0)
        
        # Find the most likely letter
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        predicted_letter = self.index_to_letter[predicted_index]
        
        return predicted_letter, confidence


# Example usage
if __name__ == "__main__":
    print("🚀 Creating Simple ASL Model")
    print("=" * 40)
    
    # Create the model
    asl_model = SimpleASLModel()
    
    # Build the architecture
    model = asl_model.build_model()
    
    # Compile for training
    asl_model.compile_model()
    
    # Show model details
    asl_model.get_model_summary()
    
    print("\n✅ Basic model is ready!")
    print("\nNext steps:")
    print("1. Collect training data for each letter")
    print("2. Train the model on your data")
    print("3. Test it with real-time camera input")