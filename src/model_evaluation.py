"""
Simple Parameter Check for ASL Models
====================================

Quick and reliable parameter counting for your models
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from pathlib import Path

def check_model_parameters():
    """
    Simple parameter check that works reliably
    """
    
    print("ASL Model Parameter Check")
    print("=" * 40)
    
    models = [
        ("Static CNN", "models/asl_model_simple.keras"),
        ("Motion LSTM", "models/motion_model.keras")
    ]
    
    results = {}
    
    for model_name, model_path in models:
        print(f"\n{model_name}")
        print("-" * 30)
        
        if not Path(model_path).exists():
            print(f"Not found: {model_path}")
            continue
        
        try:
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # File size
            file_size_bytes = Path(model_path).stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Parameter count
            total_params = model.count_params()
            
            # Calculate trainable parameters manually
            trainable_params = 0
            for layer in model.layers:
                if hasattr(layer, 'trainable_weights'):
                    for weight in layer.trainable_weights:
                        if hasattr(weight, 'numpy'):
                            trainable_params += weight.numpy().size
                        else:
                            # For newer TensorFlow versions
                            shape = weight.shape
                            if shape:
                                size = 1
                                for dim in shape:
                                    size *= dim
                                trainable_params += size
            
            non_trainable_params = total_params - trainable_params
            
            print(f"File Size: {file_size_mb:.1f} MB")
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable: {trainable_params:,}")
            print(f"Non-trainable: {non_trainable_params:,}")
            
            # Store results
            results[model_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'file_size_mb': file_size_mb
            }
            
            # Model info
            print(f"Input Shape: {model.input_shape}")
            print(f"Output Shape: {model.output_shape}")
            print(f"Layers: {len(model.layers)}")
            
            # Quick layer summary
            print(f"Key Layers:")
            for layer in model.layers:
                layer_params = layer.count_params()
                if layer_params > 1000:  # Only show layers with significant parameters
                    layer_type = type(layer).__name__
                    print(f"   • {layer.name} ({layer_type}): {layer_params:,} params")
        
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Summary
    if results:
        print(f"\nSystem Summary")
        print("=" * 40)
        
        total_params = sum(r['total_params'] for r in results.values())
        total_size = sum(r['file_size_mb'] for r in results.values())
        
        print(f"Combined System:")
        print(f"   • Total Parameters: {total_params:,}")
        print(f"   • Total Size: {total_size:.1f} MB")
        print(f"   • Models: {len(results)}")
        
        # Individual breakdowns
        for name, data in results.items():
            params = data['total_params']
            percentage = (params / total_params) * 100 if total_params > 0 else 0
            print(f"   • {name}: {params:,} ({percentage:.1f}%)")
        
        # Parameter classifications
        print(f"\n Parameter Scale:")
        for name, data in results.items():
            params = data['total_params']
            if params >= 1_000_000:
                scale = f"{params/1_000_000:.1f}M"
            elif params >= 1_000:
                scale = f"{params/1_000:.0f}K"
            else:
                scale = f"{params}"
            print(f"   • {name}: {scale} parameters")
        
        # Efficiency metrics
        static_letters = 24
        motion_letters = 2
        total_letters = 26
        
        print(f"\nEfficiency Metrics:")
        if 'Static CNN' in results:
            static_params = results['Static CNN']['total_params']
            static_efficiency = static_params / static_letters
            print(f"   • Static: {static_efficiency:,.0f} params per letter")
        
        if 'Motion LSTM' in results:
            motion_params = results['Motion LSTM']['total_params']
            motion_efficiency = motion_params / motion_letters
            print(f"   • Motion: {motion_efficiency:,.0f} params per letter")
        
        overall_efficiency = total_params / total_letters
        print(f"   • Overall: {overall_efficiency:,.0f} params per letter")
        
        if total_params >= 1_000_000:
            total_display = f"{total_params/1_000_000:.1f}M"
        else:
            total_display = f"{total_params/1_000:.0f}K"
        
        # Architecture description
        architectures = []
        if 'Static CNN' in results:
            static_params = results['Static CNN']['total_params']
            if static_params >= 1_000_000:
                static_display = f"{static_params/1_000_000:.1f}M"
            else:
                static_display = f"{static_params/1_000:.0f}K"
            architectures.append(f"{static_display}-parameter CNN")
        
        if 'Motion LSTM' in results:
            motion_params = results['Motion LSTM']['total_params']
            if motion_params >= 1_000_000:
                motion_display = f"{motion_params/1_000_000:.1f}M"
            else:
                motion_display = f"{motion_params/1_000:.0f}K"
            architectures.append(f"{motion_display}-parameter LSTM")


if __name__ == "__main__":
    check_model_parameters()