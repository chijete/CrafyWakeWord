import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import os
import json

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

wake_words = config_datos['wake_words']
sr = 16000

path_to_dataset = 'dataset'
path_to_dataset_w = path_to_dataset + '/'

# Auto-detect which ONNX model to use (improved or legacy)
onnx_model_paths = [
    (path_to_dataset_w + "onnx_model_trained_improved.onnx", "improved"),
    (path_to_dataset_w + "onnx_model_trained.onnx", "legacy")
]

onnx_model = None
model_type = None

for onnx_path, model_variant in onnx_model_paths:
    if os.path.exists(onnx_path):
        try:
            onnx_model = onnx.load(onnx_path)
            model_type = model_variant
            print(f"Successfully loaded {model_variant} ONNX model from {onnx_path}")
            break
        except Exception as e:
            print(f"Failed to load ONNX model from {onnx_path}: {e}")
            continue

if onnx_model is None:
    raise FileNotFoundError("Could not load any ONNX model. Please ensure either 'onnx_model_trained_improved.onnx' or 'onnx_model_trained.onnx' exists in the dataset folder.")

print(f"Converting {model_type} ONNX model to TensorFlow...")

# Prepare TensorFlow representation
try:
    tf_rep = prepare(onnx_model)
    print("Successfully prepared TensorFlow representation")
except Exception as e:
    print(f"Error preparing TensorFlow representation: {e}")
    raise

# Input nodes to the model
print("inputs:", tf_rep.inputs)

# Output nodes from the model
print("outputs:", tf_rep.outputs)

# All nodes in the model
print("tensor_dict:")
print(tf_rep.tensor_dict)

# Generate output filenames based on model type
tf_model_name = f"tf_model_trained{'_improved' if model_type == 'improved' else ''}"
tflite_model_name = f"tf_model_trained{'_improved' if model_type == 'improved' else ''}.tflite"

# Export the TensorFlow SavedModel
try:
    tf_rep.export_graph(path_to_dataset_w + tf_model_name)
    print(f"Successfully exported TensorFlow SavedModel: {tf_model_name}")
except Exception as e:
    print(f"Error exporting TensorFlow model: {e}")
    raise

# Convert to TensorFlow Lite
try:
    # Load the saved model
    model = tf.saved_model.load(path_to_dataset_w + tf_model_name)
    print("Successfully loaded TensorFlow SavedModel for TFLite conversion")
    
    # Define input shape
    input_shape = [1, 1, 40, 61]
    
    # Get concrete function
    func = tf.function(model).get_concrete_function(
        input=tf.TensorSpec(shape=input_shape, dtype=np.float32, name="input")
    )
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
    
    # Configure converter for better compatibility with mobile devices
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # For improved model, we might want to use more conservative quantization
    if model_type == "improved":
        # Set representative dataset for quantization (optional)
        # converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops
            tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops (fallback)
        ]
        print("Configured converter for improved model with additional optimizations")
    
    # Convert to TFLite
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(path_to_dataset_w + tflite_model_name, "wb") as f:
        f.write(tflite_model)
    
    print(f"Successfully created TensorFlow Lite model: {tflite_model_name}")
    print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")
    
except Exception as e:
    print(f"Error converting to TensorFlow Lite: {e}")
    print("This might be due to unsupported operations in the improved model.")
    print("Trying with experimental features enabled...")
    
    try:
        # Retry with more permissive settings
        converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        
        tflite_model = converter.convert()
        
        with open(path_to_dataset_w + tflite_model_name, "wb") as f:
            f.write(tflite_model)
        
        print(f"Successfully created TensorFlow Lite model with experimental features: {tflite_model_name}")
        print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")
        
    except Exception as e2:
        print(f"Failed to convert to TensorFlow Lite even with experimental features: {e2}")
        print("The model conversion to TFLite failed, but SavedModel was created successfully.")

# Optional: Test the TensorFlow SavedModel
try:
    print("\nTesting TensorFlow SavedModel...")
    
    # Create dummy input
    dummy_input = np.random.rand(1, 1, 40, 61).astype(np.float32)
    
    # Run inference
    result = model(input=dummy_input)
    print(f"Test inference successful! Output shape: {result.shape}")
    print(f"Output sample: {result.numpy()[0][:5]}...")  # Show first 5 values
    
except Exception as e:
    print(f"Error testing SavedModel: {e}")

# Optional: Test the TFLite model if conversion was successful
if os.path.exists(path_to_dataset_w + tflite_model_name):
    try:
        print(f"\nTesting TensorFlow Lite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=path_to_dataset_w + tflite_model_name)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"TFLite input details: {input_details}")
        print(f"TFLite output details: {output_details}")
        
        # Create dummy input
        dummy_input = np.random.rand(1, 1, 40, 61).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"TFLite test inference successful! Output shape: {output_data.shape}")
        print(f"TFLite output sample: {output_data[0][:5]}...")  # Show first 5 values
        
    except Exception as e:
        print(f"Error testing TFLite model: {e}")

print(f"\nConversion completed!")
print(f"Generated files:")
print(f"- TensorFlow SavedModel: {tf_model_name}")
if os.path.exists(path_to_dataset_w + tflite_model_name):
    print(f"- TensorFlow Lite model: {tflite_model_name}")
else:
    print("- TensorFlow Lite model: FAILED TO GENERATE")

# Save conversion metadata
conversion_metadata = {
    "source_onnx_model": onnx_path if 'onnx_path' in locals() else "unknown",
    "model_type": model_type,
    "tf_savedmodel": tf_model_name,
    "tflite_model": tflite_model_name if os.path.exists(path_to_dataset_w + tflite_model_name) else None,
    "input_shape": input_shape,
    "wake_words": wake_words,
    "sample_rate": sr
}

with open(path_to_dataset_w + f'conversion_metadata{"_improved" if model_type == "improved" else ""}.json', 'w') as f:
    json.dump(conversion_metadata, f, indent=2)

print(f"Conversion metadata saved to: conversion_metadata{'_improved' if model_type == 'improved' else ''}.json")