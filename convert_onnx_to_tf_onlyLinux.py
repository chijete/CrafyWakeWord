# Alternative ONNX to TensorFlow conversion without onnx-tf dependency
# Uses onnx2tf which is more modern and compatible with recent TensorFlow versions

import os
import json
import numpy as np
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_dependencies():
    """Check and install required packages"""
    required_packages = [
        "onnx2tf",
        "onnxruntime", 
        "tensorflow"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            install_package(package)
            print(f"✓ {package} installed successfully")

# Check and install dependencies
print("Checking dependencies...")
# check_and_install_dependencies()

# Now import the packages
# try:
#     import onnx2tf
#     import onnxruntime as ort
#     import tensorflow as tf
#     print("✓ All dependencies loaded successfully")
# except ImportError as e:
#     print(f"Error importing dependencies: {e}")
#     print("Please install manually with: pip install onnx2tf onnxruntime tensorflow")
#     sys.exit(1)

import onnx2tf
import onnxruntime as ort
import tensorflow as tf

# Load configuration
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

onnx_model_path = None
model_type = None

for path, model_variant in onnx_model_paths:
    if os.path.exists(path):
        onnx_model_path = path
        model_type = model_variant
        print(f"Found {model_variant} ONNX model: {path}")
        break

if onnx_model_path is None:
    raise FileNotFoundError("Could not find any ONNX model. Please ensure either 'onnx_model_trained_improved.onnx' or 'onnx_model_trained.onnx' exists in the dataset folder.")

# Generate output filenames based on model type
tf_model_name = f"tf_model_trained{'_improved' if model_type == 'improved' else ''}"
tflite_model_name = f"tf_model_trained{'_improved' if model_type == 'improved' else ''}.tflite"

print(f"\nConverting {model_type} ONNX model to TensorFlow...")
print(f"Input: {onnx_model_path}")
print(f"Output: {path_to_dataset_w + tf_model_name}")

# Method 1: Using onnx2tf (recommended)
try:
    print("\n=== Using onnx2tf for conversion ===")
    
    # Convert ONNX to TensorFlow SavedModel
    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=path_to_dataset_w + tf_model_name,
        copy_onnx_input_output_names_to_tflite=True,
        output_integer_quantized_tflite=True,
        output_dynamic_range_quantized_tflite=True,
        output_float16_quantized_tflite=True,
    )
    
    print(f"✓ Successfully converted to TensorFlow SavedModel: {tf_model_name}")
    
    # Test the SavedModel
    try:
        print("\nTesting TensorFlow SavedModel...")
        model = tf.saved_model.load(path_to_dataset_w + tf_model_name)
        
        # Create dummy input
        dummy_input = np.random.rand(1, 1, 40, 61).astype(np.float32)
        
        # Get the serving signature
        infer = model.signatures["serving_default"]
        
        # Run inference
        result = infer(input=tf.constant(dummy_input))
        output_key = list(result.keys())[0]
        output = result[output_key]
        
        print(f"✓ Test inference successful!")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output.numpy()[0][:5]}...")
        
    except Exception as e:
        print(f"⚠ Error testing SavedModel: {e}")
    
    # Check for TFLite models created by onnx2tf
    tflite_files = []
    for file in os.listdir(path_to_dataset_w + tf_model_name):
        if file.endswith('.tflite'):
            tflite_files.append(file)
    
    if tflite_files:
        print(f"\n✓ onnx2tf created {len(tflite_files)} TFLite model(s):")
        for tflite_file in tflite_files:
            print(f"  - {tflite_file}")
            
            # Test the first TFLite model
            if tflite_file == tflite_files[0]:
                try:
                    print(f"\nTesting TFLite model: {tflite_file}")
                    
                    tflite_path = os.path.join(path_to_dataset_w + tf_model_name, tflite_file)
                    interpreter = tf.lite.Interpreter(model_path=tflite_path)
                    interpreter.allocate_tensors()
                    
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    print(f"  Input details: {input_details[0]['shape']}")
                    print(f"  Output details: {output_details[0]['shape']}")
                    
                    # Create dummy input
                    dummy_input = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])
                    
                    # Run inference
                    interpreter.set_tensor(input_details[0]['index'], dummy_input)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    print(f"✓ TFLite test inference successful!")
                    print(f"  Output sample: {output_data[0][:5]}...")
                    
                    # Copy the main TFLite file to the dataset root for compatibility
                    import shutil
                    shutil.copy(tflite_path, path_to_dataset_w + tflite_model_name)
                    print(f"✓ Copied main TFLite model to: {tflite_model_name}")
                    
                except Exception as e:
                    print(f"⚠ Error testing TFLite model: {e}")
    
except Exception as e:
    print(f"❌ onnx2tf conversion failed: {e}")
    print("\nTrying alternative method...")
    
    # Method 2: Manual conversion using ONNX Runtime and TensorFlow
    try:
        print("\n=== Using ONNX Runtime + TensorFlow conversion ===")
        
        # Load ONNX model with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_model_path)
        
        # Get input/output info
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        output_name = ort_session.get_outputs()[0].name
        output_shape = ort_session.get_outputs()[0].shape
        
        print(f"ONNX model info:")
        print(f"  Input: {input_name} {input_shape}")
        print(f"  Output: {output_name} {output_shape}")
        
        # Create a TensorFlow function that mimics the ONNX model
        @tf.function
        def onnx_to_tf_model(input_tensor):
            # Convert TF tensor to numpy for ONNX Runtime
            input_np = input_tensor.numpy()
            
            # Run ONNX inference
            ort_inputs = {input_name: input_np}
            ort_outputs = ort_session.run([output_name], ort_inputs)
            
            # Convert back to TF tensor
            return tf.constant(ort_outputs[0])
        
        # Create concrete function
        dummy_input = tf.constant(np.random.rand(1, 1, 40, 61).astype(np.float32))
        concrete_function = onnx_to_tf_model.get_concrete_function(dummy_input)
        
        # This approach doesn't work well because TF functions can't call external libraries
        print("❌ Manual conversion method not suitable for this use case")
        
    except Exception as e2:
        print(f"❌ Alternative conversion also failed: {e2}")

# Save conversion metadata
conversion_metadata = {
    "source_onnx_model": onnx_model_path,
    "model_type": model_type,
    "tf_savedmodel": tf_model_name,
    "tflite_model": tflite_model_name if os.path.exists(path_to_dataset_w + tflite_model_name) else None,
    "input_shape": [1, 1, 40, 61],
    "wake_words": wake_words,
    "sample_rate": sr,
    "conversion_method": "onnx2tf"
}

metadata_filename = f'conversion_metadata{"_improved" if model_type == "improved" else ""}.json'
with open(path_to_dataset_w + metadata_filename, 'w') as f:
    json.dump(conversion_metadata, f, indent=2)

print(f"\n✓ Conversion metadata saved to: {metadata_filename}")

print(f"\n{'='*50}")
print("CONVERSION SUMMARY")
print(f"{'='*50}")
print(f"Source ONNX model: {os.path.basename(onnx_model_path)} ({model_type})")
print(f"TensorFlow SavedModel: {tf_model_name}")
if os.path.exists(path_to_dataset_w + tflite_model_name):
    print(f"TensorFlow Lite model: {tflite_model_name}")
    print(f"✓ Conversion completed successfully!")
else:
    print("❌ TensorFlow Lite conversion failed")
print(f"{'='*50}")

print("\nRecommendation:")
print("- Use the SavedModel for server/desktop deployment")
print("- Use the TFLite model for mobile/edge deployment")
print("- The onnx2tf tool creates optimized versions (quantized) automatically")