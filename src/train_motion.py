"""
Step 6: Train the Motion Model for J and Z
==========================================

This script trains the LSTM model on your collected motion sequences.
It will learn to distinguish between J (hook motion) and Z (tracing motion).
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json

# Import our motion model
from motion_model import MotionASLModel

class MotionTrainer:
    """
    Trainer for the motion-based ASL model
    
    What this does:
    - Loads your collected motion sequences
    - Preprocesses the landmark data
    - Trains the LSTM model
    - Evaluates performance
    """
    
    def __init__(self, data_dir="data/motion"):
        self.data_dir = Path(data_dir)
        self.model_trainer = MotionASLModel(sequence_length=20)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_motion_data(self):
        """
        Load motion sequences from the data directory
        
        Returns:
            X: Array of motion sequences
            y: Array of labels (J or Z)
        """
        
        print("Loading motion sequences...")
        
        sequences = []
        labels = []
        
        for letter in ['J', 'Z']:
            letter_dir = self.data_dir / letter
            
            if not letter_dir.exists():
                print(f"⚠️  No data found for letter {letter}")
                continue
            
            # Get all .npy files for this letter
            sequence_files = list(letter_dir.glob("*.npy"))
            print(f"Found {len(sequence_files)} sequences for letter {letter}")
            
            for seq_path in sequence_files:
                try:
                    # Load the sequence
                    sequence = np.load(str(seq_path))
                    
                    # Check if sequence has the right shape
                    if sequence.shape[0] >= 20 and sequence.shape[1] == 63:
                        # Take first 20 frames and ensure 63 landmarks
                        sequence = sequence[:20, :63]
                        sequences.append(sequence)
                        labels.append(letter)
                    else:
                        print(f"Skipping {seq_path}: wrong shape {sequence.shape}")
                        
                except Exception as e:
                    print(f"Error loading {seq_path}: {e}")
        
        print(f" Loaded {len(sequences)} total sequences")
        print(f"Motion data distribution:")
        for letter in ['J', 'Z']:
            count = labels.count(letter)
            print(f"   {letter}: {count} sequences")
        
        return np.array(sequences), np.array(labels)
    
    def preprocess_sequences(self, sequences):
        """
        Preprocess motion sequences for training
        
        Args:
            sequences: Raw motion sequences
        
        Returns:
            preprocessed_sequences: Normalized and cleaned sequences
        """
        
        print("Preprocessing motion sequences...")
        
        # Handle missing landmarks (replace with interpolation)
        processed_sequences = []
        
        for seq in sequences:
            processed_seq = seq.copy()
            
            # Replace zero landmarks with interpolation
            for frame_idx in range(len(processed_seq)):
                frame = processed_seq[frame_idx]
                
                # Check if frame has all zeros (missing landmarks)
                if np.all(frame == 0) and frame_idx > 0:
                    # Use previous frame's landmarks
                    processed_seq[frame_idx] = processed_seq[frame_idx - 1]
            
            # Normalize landmarks to reduce variation
            # Subtract wrist position (first landmark) to make relative
            wrist_pos = processed_seq[:, :3]  # First landmark (wrist) x,y,z
            
            for i in range(0, 63, 3):  # Every landmark (x,y,z)
                processed_seq[:, i:i+3] = processed_seq[:, i:i+3] - wrist_pos
            
            processed_sequences.append(processed_seq)
        
        return np.array(processed_sequences)
    
    def prepare_data(self):
        """
        Load and prepare data for training
        """
        
        print("Preparing motion training data...")
        
        # Load sequences and labels
        X, y = self.load_motion_data()
        
        if len(X) == 0:
            print("No motion data loaded! Collect sequences first.")
            return False
        
        # Preprocess sequences
        X_processed = self.preprocess_sequences(X)
        
        # Convert string labels to numbers (J=0, Z=1)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Convert to one-hot encoding
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=2)
        
        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y_categorical,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        
        print(f"Motion data prepared!")
        print(f"Training sequences: {len(self.X_train)}")
        print(f"Test sequences: {len(self.X_test)}")
        print(f"Sequence shape: {self.X_train.shape}")
        
        return True
    
    def train_model(self, epochs=50, batch_size=8):
        """
        Train the motion model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        
        if self.X_train is None:
            print("Prepare data first!")
            return
        
        print(f"Starting motion model training...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Build and compile the model
        model = self.model_trainer.build_model()
        self.model_trainer.compile_model(learning_rate=0.001)
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        print("Training motion model...")
        history = model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Motion training completed!")
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Final test accuracy: {test_accuracy:.2%}")
        
        # Save the model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "motion_model.keras"  # Change extension
        model.save(str(model_path))
        print(f" Motion model saved to: {model_path}")
        
        # Save training info
        training_info = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'epochs_trained': len(history.history['accuracy']),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1])
        }
        
        with open('logs/motion_training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Test some predictions
        self.test_motion_predictions()
        
        return model, history
    
    def plot_training_history(self, history):
        """
        Plot motion model training history
        """
        
        print("Creating motion training plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Motion Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Motion Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Save plot
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        plot_path = logs_dir / "motion_training_history.png"
        plt.savefig(str(plot_path))
        print(f"Motion training plots saved to: {plot_path}")
        
        plt.tight_layout()
        plt.show()
    
    def test_motion_predictions(self, num_samples=6):
        """
        Test the motion model on random sequences
        """
        
        if self.X_test is None:
            print("No test data available!")
            return
        
        print(f"Testing motion model on {num_samples} random sequences...")
        
        # Load the trained model
        model = tf.keras.models.load_model("models/motion_model.keras")
        
        # Get random samples
        indices = np.random.choice(len(self.X_test), min(num_samples, len(self.X_test)), replace=False)
        
        for i, idx in enumerate(indices):
            # Get sequence and true label
            sequence = self.X_test[idx:idx+1]  # Add batch dimension
            true_label_idx = np.argmax(self.y_test[idx])
            true_label = self.model_trainer.index_to_letter[true_label_idx]
            
            # Make prediction
            prediction = model.predict(sequence, verbose=0)
            predicted_idx = np.argmax(prediction[0])
            predicted_label = self.model_trainer.index_to_letter[predicted_idx]
            confidence = prediction[0][predicted_idx]
            
            # Show result
            status = "Good" if predicted_label == true_label else "Bad"
            print(f"{status} Sequence {i+1}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.2%}")

def check_motion_data():
    """
    Check if we have enough motion data to train
    """
    
    data_dir = Path("data/motion")
    
    if not data_dir.exists():
        print(" No motion data directory found!")
        print(" Run: python src/motion_model.py")
        print("   Then collect data for J and Z")
        return False
    
    total_sequences = 0
    for letter in ['J', 'Z']:
        letter_dir = data_dir / letter
        if letter_dir.exists():
            count = len(list(letter_dir.glob("*.npy")))
            total_sequences += count
            print(f"📸 Letter {letter}: {count} sequences")
        else:
            print(f"⚠️  No data for letter {letter}")
    
    if total_sequences < 20:
        print(" Not enough motion data! Need at least 20 sequences total.")
        print(" Collect more data using: python src/motion_model.py")
        return False
    
    print(f" Found {total_sequences} motion sequences total")
    return True

def main():
    """
    Main training function for motion model
    """
    
    print(" Motion Model Training")
    print("=" * 40)
    print("Training LSTM to recognize J and Z motions!")
    
    # Check if we have motion data
    if not check_motion_data():
        return
    
    # Create trainer
    trainer = MotionTrainer()
    
    # Prepare data
    print("\n" + "=" * 40)
    print(" Loading and Preprocessing Data")
    print("=" * 40)
    
    if not trainer.prepare_data():
        return
    
    # Train the model
    print("\n" + "=" * 40)
    print(" Training Motion Model")
    print("=" * 40)
    
    epochs = int(input("Number of epochs (default 50): ") or "50")
    batch_size = int(input("Batch size (default 8): ") or "8")
    
    model, history = trainer.train_model(epochs=epochs, batch_size=batch_size)
    
    print("\n" + "=" * 40)
    print(" Motion Training Complete!")
    print("=" * 40)
    print("Your motion model can now recognize J and Z!")
    print("Next: Create a combined system for all letters A-Z!")

if __name__ == "__main__":
    main()