#!/usr/bin/env python3
"""
Complete Whisper CoreML Pipeline Example

This script demonstrates how to use the converted CoreML models
with the extracted tokenizer for complete speech-to-text functionality.
"""

import numpy as np
import coremltools as ct
from pathlib import Path
import sys
import os

# Add tokenizer to path
sys.path.append('./tokenizer')
from whisper_tokenizer import WhisperTokenizer


class WhisperCoreMLPipeline:
    """Complete Whisper pipeline using CoreML models and extracted tokenizer."""
    
    def __init__(self, encoder_path, decoder_path, tokenizer_dir="./tokenizer"):
        """
        Initialize the pipeline with CoreML models and tokenizer.
        
        Args:
            encoder_path: Path to the CoreML encoder model
            decoder_path: Path to the CoreML decoder model
            tokenizer_dir: Directory containing the tokenizer files
        """
        print("Loading CoreML models...")
        self.encoder = ct.models.MLModel(encoder_path)
        self.decoder = ct.models.MLModel(decoder_path)
        
        print("Loading tokenizer...")
        self.tokenizer = WhisperTokenizer(tokenizer_dir)
        
        print("Pipeline initialized successfully!")
    
    def encode_audio(self, mel_spectrogram):
        """
        Encode mel spectrogram to audio features using the encoder.
        
        Args:
            mel_spectrogram: numpy array of shape (1, 80, 3000)
            
        Returns:
            Audio features of shape (1, 1500, 768) for small model
        """
        result = self.encoder.predict({"logmel_data": mel_spectrogram})
        return result['output']
    
    def decode_step(self, audio_features, token_sequence):
        """
        Single decoding step: predict next token given audio features and current tokens.
        
        Args:
            audio_features: Encoded audio features from encoder
            token_sequence: List of token IDs generated so far
            
        Returns:
            Logits for next token (shape: 1, 1, vocab_size)
        """
        # Use the last token for prediction
        current_token = np.array([[token_sequence[-1]]], dtype=np.int32)
        
        result = self.decoder.predict({
            "token_data": current_token,
            "audio_data": audio_features
        })
        
        return result['output']
    
    def transcribe(self, mel_spectrogram, language="en", task="transcribe", max_tokens=100):
        """
        Complete transcription pipeline.
        
        Args:
            mel_spectrogram: Input mel spectrogram (1, 80, 3000)
            language: Target language code
            task: "transcribe" or "translate"
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Transcribed text
        """
        print(f"Transcribing audio (language={language}, task={task})...")
        
        # 1. Encode audio to features
        print("Encoding audio...")
        audio_features = self.encode_audio(mel_spectrogram)
        print(f"Audio features shape: {audio_features.shape}")
        
        # 2. Initialize token sequence with SOT (Start of Transcript)
        token_sequence = self.tokenizer.get_sot_sequence(language, task)
        print(f"Initial tokens: {token_sequence}")
        
        # 3. Generate tokens iteratively
        print("Generating tokens...")
        generated_tokens = []
        
        for step in range(max_tokens):
            # Get next token probabilities
            logits = self.decode_step(audio_features, token_sequence)
            
            # Simple greedy decoding (take most probable token)
            next_token_id = int(np.argmax(logits[0, 0, :]))
            
            # Check for end of transcript
            if next_token_id == self.tokenizer.get_special_token("eot"):
                print(f"End of transcript token reached at step {step}")
                break
            
            # Add token to sequence
            token_sequence.append(next_token_id)
            generated_tokens.append(next_token_id)
            
            if step % 10 == 0:
                print(f"Step {step}: Generated token {next_token_id}")
        
        # 4. Decode tokens to text
        print("Decoding tokens to text...")
        # Only decode the generated tokens (skip SOT sequence)
        text = self.tokenizer.decode(generated_tokens)
        
        print(f"Generated {len(generated_tokens)} tokens")
        print(f"Final text: {text}")
        
        return text, generated_tokens
    
    def transcribe_with_beam_search(self, mel_spectrogram, beam_size=3, language="en", task="transcribe"):
        """
        Advanced transcription with beam search (simplified version).
        
        Args:
            mel_spectrogram: Input mel spectrogram
            beam_size: Number of beams for beam search
            language: Target language
            task: "transcribe" or "translate"
            
        Returns:
            Best transcription result
        """
        print(f"Transcribing with beam search (beam_size={beam_size})...")
        
        # Encode audio
        audio_features = self.encode_audio(mel_spectrogram)
        
        # Initialize beams
        initial_sequence = self.tokenizer.get_sot_sequence(language, task)
        beams = [(initial_sequence, 0.0)]  # (token_sequence, log_probability)
        
        max_length = 100
        
        for step in range(max_length):
            new_beams = []
            
            for sequence, log_prob in beams:
                # Get next token probabilities
                logits = self.decode_step(audio_features, sequence)
                
                # Apply softmax manually
                exp_logits = np.exp(logits[0, 0, :] - np.max(logits[0, 0, :]))
                probs = exp_logits / np.sum(exp_logits)
                
                # Get top k candidates
                top_k = 5
                top_indices = np.argsort(probs)[-top_k:]
                
                for idx in top_indices:
                    token_id = int(idx)
                    token_prob = float(probs[idx])
                    
                    new_sequence = sequence + [token_id]
                    new_log_prob = log_prob + np.log(token_prob)
                    
                    new_beams.append((new_sequence, new_log_prob))
            
            # Keep only top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Check if all beams ended
            eot_token = self.tokenizer.get_special_token("eot")
            if all(seq[-1] == eot_token for seq, _ in beams):
                break
        
        # Return best beam
        best_sequence, best_score = beams[0]
        
        # Extract generated tokens (skip SOT sequence)
        sot_length = len(initial_sequence)
        generated_tokens = best_sequence[sot_length:]
        
        # Remove EOT if present
        eot_token = self.tokenizer.get_special_token("eot")
        if generated_tokens and generated_tokens[-1] == eot_token:
            generated_tokens = generated_tokens[:-1]
        
        text = self.tokenizer.decode(generated_tokens)
        
        print(f"Best sequence score: {best_score}")
        print(f"Generated text: {text}")
        
        return text, generated_tokens


def demo_pipeline():
    """Demonstrate the complete pipeline with dummy audio."""
    
    # Paths to models
    encoder_path = "./converted_models/coreml-encoder-small.mlpackage"
    decoder_path = "./converted_models/coreml-decoder-small.mlpackage"
    
    # Check if models exist
    if not os.path.exists(encoder_path):
        print(f"Encoder model not found: {encoder_path}")
        return
    
    if not os.path.exists(decoder_path):
        print(f"Decoder model not found: {decoder_path}")
        return
    
    # Initialize pipeline
    pipeline = WhisperCoreMLPipeline(encoder_path, decoder_path)
    
    # Create dummy mel spectrogram (in real use, this would come from audio preprocessing)
    print("\nGenerating dummy mel spectrogram...")
    mel_spectrogram = np.random.randn(1, 80, 3000).astype(np.float32)
    
    # Test basic transcription
    print("\n" + "="*50)
    print("BASIC TRANSCRIPTION TEST")
    print("="*50)
    
    text, tokens = pipeline.transcribe(
        mel_spectrogram, 
        language="en", 
        task="transcribe", 
        max_tokens=20  # Keep it short for demo
    )
    
    print(f"\nResult: '{text}'")
    print(f"Tokens: {tokens}")
    
    # Test beam search (if time permits)
    print("\n" + "="*50)
    print("BEAM SEARCH TEST")
    print("="*50)
    
    try:
        text_beam, tokens_beam = pipeline.transcribe_with_beam_search(
            mel_spectrogram,
            beam_size=2,
            language="en",
            task="transcribe"
        )
        
        print(f"\nBeam search result: '{text_beam}'")
        print(f"Tokens: {tokens_beam}")
        
    except Exception as e:
        print(f"Beam search failed: {e}")
    
    print("\n🎉 Pipeline demo completed!")


if __name__ == "__main__":
    demo_pipeline()
