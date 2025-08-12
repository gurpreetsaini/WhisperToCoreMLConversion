#!/usr/bin/env python3
"""
Simple Whisper Tokenizer Wrapper

This module provides easy access to the Whisper tokenizer for use with CoreML models.
"""

import json
import pickle
from pathlib import Path

class WhisperTokenizer:
    def __init__(self, tokenizer_dir="./tokenizer"):
        self.tokenizer_dir = Path(tokenizer_dir)
        
        # Load vocabulary
        with open(self.tokenizer_dir / "vocabulary.json", 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.id_to_token = {int(k): v for k, v in vocab_data.items()}
        
        # Load reverse mapping
        with open(self.tokenizer_dir / "token_to_id.json", 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        
        # Load special tokens
        with open(self.tokenizer_dir / "special_tokens.json", 'r', encoding='utf-8') as f:
            self.special_tokens = json.load(f)
        
        # Load tiktoken encoding if available
        try:
            with open(self.tokenizer_dir / "tiktoken_encoding.pkl", 'rb') as f:
                self.encoding = pickle.load(f)
        except:
            self.encoding = None
    
    def encode(self, text):
        """Encode text to token IDs."""
        if self.encoding:
            return self.encoding.encode(text)
        else:
            # Fallback: simple word-based tokenization (not recommended for production)
            tokens = []
            for char in text:
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                else:
                    tokens.append(self.special_tokens.get("unk", 0))
            return tokens
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        if self.encoding:
            return self.encoding.decode(token_ids)
        else:
            # Fallback: simple concatenation
            return ''.join([self.id_to_token.get(token_id, '<unk>') for token_id in token_ids])
    
    def get_special_token(self, name):
        """Get special token ID by name."""
        return self.special_tokens.get(name)
    
    def get_sot_sequence(self, language="en", task="transcribe"):
        """Get start-of-transcript sequence for a given language and task."""
        tokens = [self.special_tokens["sot"]]
        
        if language and language != "en":
            lang_token = self.special_tokens.get(f"lang_{language}")
            if lang_token:
                tokens.append(lang_token)
        
        if task == "translate":
            tokens.append(self.special_tokens["translate"])
        else:
            tokens.append(self.special_tokens["transcribe"])
        
        tokens.append(self.special_tokens["notimestamps"])
        return tokens

# Example usage:
if __name__ == "__main__":
    tokenizer = WhisperTokenizer()
    
    # Test encoding/decoding
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    # Get special tokens
    print(f"Start token: {tokenizer.get_special_token('sot')}")
    print(f"End token: {tokenizer.get_special_token('eot')}")
    
    # Get SOT sequence for English transcription
    sot_sequence = tokenizer.get_sot_sequence("en", "transcribe")
    print(f"SOT sequence: {sot_sequence}")
