# Complete Motion Model Training Script with Evaluation and Logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
from datetime import datetime

def create_motion_model(sequence_length=20, num_features=63):
    """Create LSTM model for motion recognition"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')  # J and Z
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_motion_data():
    """Load motion sequences from data directory"""
    data_dir = Path("data/motion")
    letters = ['J', 'Z']
    
    sequences = []
    labels = []
    
    print("Loading motion sequences...")
    
    for letter in letters:
        letter_dir = data_dir / letter
        
        if not letter_dir.exists():
            print(f"No data found for letter {letter}")
            continue
        
        sequence_files = list(letter_dir.glob("*.npy"))
        print(f"{letter}: {len(sequence_files)} sequences")
        
        for seq_path in sequence_files:
            try:
                sequence = np.load(str(seq_path))
                
                if sequence.shape[0] >= 20 and sequence.shape[1] == 63:
                    sequence = sequence[:20, :63]
                    sequences.append(sequence)
                    labels.append(letter)
                else:
                    print(f"Skipping {seq_path}: wrong shape {sequence.shape}")
                    
            except Exception as e:
                print(f"Error loading {seq_path}: {e}")
    
    print(f"Loaded {len(sequences)} total sequences")
    for letter in letters:
        count = labels.count(letter)
        print(f"   {letter}: {count} sequences")
    
    return np.array(sequences), np.array(labels), letters

def preprocess_sequences(sequences):
    """Preprocess motion sequences for training"""
    print("Preprocessing motion sequences...")
    
    processed_sequences = []
    
    for seq in sequences:
        processed_seq = seq.copy()
        
        # Handle missing landmarks
        for frame_idx in range(len(processed_seq)):
            frame = processed_seq[frame_idx]
            
            if np.all(frame == 0) and frame_idx > 0:
                processed_seq[frame_idx] = processed_seq[frame_idx - 1]
        
        # Normalize relative to wrist position
        wrist_pos = processed_seq[:, :3]
        
        for i in range(0, 63, 3):
            processed_seq[:, i:i+3] = processed_seq[:, i:i+3] - wrist_pos
        
        processed_sequences.append(processed_seq)
    
    return np.array(processed_sequences)

def comprehensive_motion_evaluation(model, X_test, y_test, letters, history):
    """Evaluate motion model and save detailed results"""
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    
    # Calculate overall accuracy
    accuracy = np.mean(pred_labels == true_labels)
    
    # Get detailed classification report
    report = classification_report(true_labels, pred_labels, 
                                 target_names=letters, 
                                 output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Per-class accuracy
    per_class_accuracy = {}
    for i, letter in enumerate(letters):
        mask = true_labels == i
        if np.sum(mask) > 0:
            class_accuracy = np.mean(pred_labels[mask] == true_labels[mask])
            per_class_accuracy[letter] = float(class_accuracy)
    
    # Training history metrics
    final_train_acc = float(history.history['accuracy'][-1])
    final_val_acc = float(history.history['val_accuracy'][-1])
    final_train_loss = float(history.history['loss'][-1])
    final_val_loss = float(history.history['val_loss'][-1])
    
    # Create comprehensive report
    evaluation_report = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "Motion LSTM",
        "total_parameters": int(model.count_params()),
        "training_epochs": len(history.history['accuracy']),
        "sequence_length": 20,
        "features_per_frame": 63,
        
        "overall_performance": {
            "test_accuracy": float(accuracy),
            "final_training_accuracy": final_train_acc,
            "final_validation_accuracy": final_val_acc,
            "final_training_loss": final_train_loss,
            "final_validation_loss": final_val_loss
        },
        
        "per_class_performance": per_class_accuracy,
        
        "classification_metrics": {
            "precision_macro": float(report['macro avg']['precision']),
            "recall_macro": float(report['macro avg']['recall']),
            "f1_macro": float(report['macro avg']['f1-score']),
            "precision_weighted": float(report['weighted avg']['precision']),
            "recall_weighted": float(report['weighted avg']['recall']),
            "f1_weighted": float(report['weighted avg']['f1-score'])
        },
        
        "detailed_classification_report": {
            letter: {
                "precision": float(report[letter]['precision']),
                "recall": float(report[letter]['recall']),
                "f1_score": float(report[letter]['f1-score']),
                "support": int(report[letter]['support'])
            }
            for letter in letters
        },
        
        "confusion_matrix": cm.tolist()
    }
    
    # Save evaluation report
    import os
    os.makedirs('logs', exist_ok=True)
    with open('logs/motion_evaluation_report.json', 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(f"Motion evaluation complete! Overall accuracy: {accuracy:.2%}")
    print(f"Report saved to logs/motion_evaluation_report.json")
    
    return evaluation_report

def save_motion_training_history(history):
    """Save motion training history plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.set_title('Motion Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax2.set_title('Motion Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('logs/motion_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Motion training history saved to logs/motion_training_history.png")

def test_motion_predictions(model, X_test, y_test, letters, num_samples=6):
    """Test motion model predictions"""
    if len(X_test) == 0:
        print("No test data available!")
        return
    
    print(f"Testing motion model on {num_samples} random sequences...")
    
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    for i, idx in enumerate(indices):
        sequence = X_test[idx:idx+1]
        true_label_idx = np.argmax(y_test[idx])
        true_label = letters[true_label_idx]
        
        prediction = model.predict(sequence, verbose=0)
        predicted_idx = np.argmax(prediction[0])
        predicted_label = letters[predicted_idx]
        confidence = prediction[0][predicted_idx]
        
        status = "✓" if predicted_label == true_label else "✗"
        print(f"{status} Sequence {i+1}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.2%}")

def train_motion_model():
    """Train the motion model with comprehensive evaluation"""
    print("Loading motion data...")
    X, y, letters = load_motion_data()
    
    if len(X) == 0:
        print("No motion data found! Make sure data/motion/J and data/motion/Z exist with .npy files.")
        return
    
    # Preprocess sequences
    X_processed = preprocess_sequences(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=2)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    print(f"Sequence shape: {X_train.shape}")
    
    # Create and train model
    model = create_motion_model()
    model.summary()
    
    # Callbacks
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
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Final test accuracy: {test_accuracy:.2%}")
    
    # Save model
    import os
    os.makedirs('models', exist_ok=True)
    model.save('models/motion_model.keras')
    print("Motion model saved!")
    
    # Comprehensive evaluation and logging
    print("\nRunning comprehensive evaluation...")
    evaluation_results = comprehensive_motion_evaluation(model, X_test, y_test, letters, history)
    save_motion_training_history(history)
    
    # Test predictions
    print("\nTesting predictions...")
    test_motion_predictions(model, X_test, y_test, letters)
    
    # Download files if in Colab
    try:
        from google.colab import files
        print("\nDownloading results...")
        files.download('models/motion_model.keras')
        files.download('logs/motion_evaluation_report.json')
        files.download('logs/motion_training_history.png')
    except:
        print("Files saved locally (not in Colab environment)")
    
    return model, history, evaluation_results

# Run the complete motion training pipeline
if __name__ == "__main__":
    print("Motion Model Training")
    print("=" * 40)
    print("Training LSTM to recognize J and Z motions!")
    
    model, history, results = train_motion_model()
    print("\nMotion training and evaluation complete!")