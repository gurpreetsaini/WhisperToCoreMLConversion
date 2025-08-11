#!/usr/bin/env python3
"""
Example usage script for the converted CoreML Whisper models
"""

import coremltools as ct
import numpy as np
import whisper

def load_and_test_coreml_encoder(model_path):
    """Load and test the CoreML encoder model."""
    print(f"Loading CoreML model from: {model_path}")
    
    # Load the CoreML model
    model = ct.models.MLModel(model_path)
    
    # Print model information
    print("Model input description:")
    for input_desc in model.get_spec().description.input:
        print(f"  {input_desc.name}: {input_desc}")
    
    print("Model output description:")
    for output_desc in model.get_spec().description.output:
        print(f"  {output_desc.name}: {output_desc}")
    
    # Create sample input data (mel spectrogram)
    # Shape should match the model's expected input: (1, 80, 3000)
    sample_input = np.random.randn(1, 80, 3000).astype(np.float32)
    
    # Run prediction
    print("Running prediction...")
    try:
        result = model.predict({"logmel_data": sample_input})
        print(f"Prediction successful! Output shape: {result['output'].shape}")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

def compare_original_vs_coreml():
    """Compare the original Whisper encoder with the CoreML version."""
    print("Comparing original Whisper encoder with CoreML version...")
    
    # Load original Whisper model
    original_model = whisper.load_model("base")
    
    # Create test input
    mel_input = np.random.randn(1, 80, 3000).astype(np.float32)
    
    # Run with original model
    import torch
    with torch.no_grad():
        torch_input = torch.from_numpy(mel_input)
        original_output = original_model.encoder(torch_input)
        print(f"Original output shape: {original_output.shape}")
    
    # Note: Full comparison would require the CoreML model to be loaded and run
    print("For full comparison, load the CoreML model and compare outputs.")

if __name__ == "__main__":
    import sys
    import os
    
    # Check if CoreML model exists
    base_model_path = "./converted_models/coreml-encoder-base.mlpackage"
    small_model_path = "./converted_models/coreml-encoder-small.mlpackage"
    
    if os.path.exists(base_model_path):
        print("=== Testing Base Model ===")
        load_and_test_coreml_encoder(base_model_path)
    else:
        print(f"Base model not found at: {base_model_path}")
    
    if os.path.exists(small_model_path):
        print("\n=== Testing Small Model ===")
        load_and_test_coreml_encoder(small_model_path)
    else:
        print(f"Small model not found at: {small_model_path}")
    
    print("\n=== Usage Notes ===")
    print("To use these models in your iOS/macOS app:")
    print("1. Add the .mlpackage files to your Xcode project")
    print("2. Use MLModel to load them in your Swift/Objective-C code")
    print("3. Feed mel spectrogram data as input")
    print("4. The output will be encoded audio features")
