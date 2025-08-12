#!/usr/bin/env python3
"""
Example usage script for the converted CoreML Whisper models
"""

import coremltools as ct
import numpy as np
import whisper

def load_and_test_coreml_encoder(model_path):
    """Load and test the CoreML encoder model."""
    print(f"Loading CoreML encoder from: {model_path}")
    
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
        return result['output']
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None


def load_and_test_coreml_decoder(model_path, audio_features=None):
    """Load and test the CoreML decoder model."""
    print(f"Loading CoreML decoder from: {model_path}")
    
    # Load the CoreML model
    model = ct.models.MLModel(model_path)
    
    # Print model information
    print("Model input description:")
    for input_desc in model.get_spec().description.input:
        print(f"  {input_desc.name}: {input_desc}")
    
    print("Model output description:")
    for output_desc in model.get_spec().description.output:
        print(f"  {output_desc.name}: {output_desc}")
    
    # Create sample input data
    if audio_features is None:
        # Default audio features shape for small model: (1, 1500, 768)
        audio_features = np.random.randn(1, 1500, 768).astype(np.float32)
    
    # Token data - start with special tokens
    token_data = np.array([[50258]], dtype=np.int32)  # Start token for Whisper
    
    # Run prediction
    print("Running prediction...")
    try:
        result = model.predict({
            "token_data": token_data,
            "audio_data": audio_features
        })
        print(f"Prediction successful! Output shape: {result['output'].shape}")
        return result['output']
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

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
    
    # Check if CoreML models exist
    base_encoder_path = "./converted_models/coreml-encoder-base.mlpackage"
    small_encoder_path = "./converted_models/coreml-encoder-small.mlpackage"
    small_decoder_path = "./converted_models/coreml-decoder-small.mlpackage"
    
    audio_features = None
    
    if os.path.exists(base_encoder_path):
        print("=== Testing Base Encoder Model ===")
        load_and_test_coreml_encoder(base_encoder_path)
    else:
        print(f"Base encoder model not found at: {base_encoder_path}")
    
    if os.path.exists(small_encoder_path):
        print("\n=== Testing Small Encoder Model ===")
        audio_features = load_and_test_coreml_encoder(small_encoder_path)
    else:
        print(f"Small encoder model not found at: {small_encoder_path}")
    
    if os.path.exists(small_decoder_path):
        print("\n=== Testing Small Decoder Model ===")
        load_and_test_coreml_decoder(small_decoder_path, audio_features)
    else:
        print(f"Small decoder model not found at: {small_decoder_path}")
    
    if os.path.exists(small_encoder_path) and os.path.exists(small_decoder_path):
        print("\n=== End-to-End Pipeline Test ===")
        print("Testing complete encoder → decoder pipeline...")
        
        # Test with fresh random mel spectrogram
        mel_input = np.random.randn(1, 80, 3000).astype(np.float32)
        
        # Load models
        encoder_model = ct.models.MLModel(small_encoder_path)
        decoder_model = ct.models.MLModel(small_decoder_path)
        
        try:
            # Encode audio
            encoder_result = encoder_model.predict({"logmel_data": mel_input})
            audio_features = encoder_result['output']
            print(f"✓ Encoder output shape: {audio_features.shape}")
            
            # Decode to text tokens
            token_data = np.array([[50258]], dtype=np.int32)  # Start token
            decoder_result = decoder_model.predict({
                "token_data": token_data,
                "audio_data": audio_features
            })
            logits = decoder_result['output']
            print(f"✓ Decoder output shape: {logits.shape}")
            
            # Get predicted token
            predicted_token = np.argmax(logits[0, 0, :])
            print(f"✓ Predicted token ID: {predicted_token}")
            print("✓ End-to-end pipeline test successful!")
            
        except Exception as e:
            print(f"✗ End-to-end pipeline test failed: {e}")
    
    print("\n=== Usage Notes ===")
    print("To use these models in your iOS/macOS app:")
    print("1. Add the .mlpackage files to your Xcode project")
    print("2. Use MLModel to load them in your Swift/Objective-C code")
    print("3. For encoder: Feed mel spectrogram data as input")
    print("4. For decoder: Feed token data and audio features as input")
    print("5. Chain encoder → decoder for complete speech-to-text pipeline")
