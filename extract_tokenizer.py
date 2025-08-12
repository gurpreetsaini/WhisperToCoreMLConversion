#!/usr/bin/env python3
"""
Whisper Tokenizer Extractor

Extracts and saves the Whisper tokenizer for use with CoreML models.
The tokenizer is needed to convert between text and token IDs.
"""

import json
import pickle
import os
from pathlib import Path
import ssl
import urllib.request

# SSL fix for macOS
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

original_urlopen = urllib.request.urlopen
def patched_urlopen(*args, **kwargs):
    if 'context' not in kwargs:
        kwargs['context'] = ssl_context
    return original_urlopen(*args, **kwargs)

urllib.request.urlopen = patched_urlopen

try:
    import whisper
    import tiktoken
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Install with: pip install openai-whisper tiktoken")
    exit(1)


def extract_whisper_tokenizer(model_name="small", output_dir="./tokenizer"):
    """
    Extract the Whisper tokenizer and save it in multiple formats for easy use.
    """
    print(f"Extracting tokenizer for Whisper {model_name} model...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load the Whisper model to get the tokenizer
    model = whisper.load_model(model_name)
    
    # Get the tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=model.is_multilingual,
        num_languages=model.num_languages if hasattr(model, 'num_languages') else 99
    )
    
    print(f"Tokenizer vocabulary size: {tokenizer.encoding.n_vocab}")
    print(f"Model is multilingual: {model.is_multilingual}")
    
    # Extract vocabulary and special tokens
    vocab = {}
    special_tokens = {}
    
    # Get all tokens
    for i in range(tokenizer.encoding.n_vocab):
        try:
            token = tokenizer.encoding.decode([i])
            vocab[i] = token
        except:
            vocab[i] = f"<unk_{i}>"
    
    # Extract special tokens
    special_tokens = {
        "sot": 50258,  # Start of transcript
        "eot": 50257,  # End of transcript
        "sot_prev": 50361,  # Start of previous context
        "nospeech": 50362,  # No speech
        "notimestamps": 50363,  # No timestamps
    }
    
    # Add language tokens
    if model.is_multilingual:
        languages = whisper.tokenizer.LANGUAGES
        for lang_code, lang_name in languages.items():
            token_id = 50259 + list(languages.keys()).index(lang_code)
            special_tokens[f"lang_{lang_code}"] = token_id
            special_tokens[f"lang_{lang_name}"] = token_id
    
    # Add task tokens
    special_tokens["transcribe"] = 50359
    special_tokens["translate"] = 50358
    
    # Save vocabulary as JSON
    vocab_file = output_path / "vocabulary.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"✓ Vocabulary saved to: {vocab_file}")
    
    # Save special tokens as JSON
    special_tokens_file = output_path / "special_tokens.json"
    with open(special_tokens_file, 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f, indent=2)
    print(f"✓ Special tokens saved to: {special_tokens_file}")
    
    # Save reverse vocabulary (token -> id) as JSON
    reverse_vocab = {token: token_id for token_id, token in vocab.items()}
    reverse_vocab_file = output_path / "token_to_id.json"
    with open(reverse_vocab_file, 'w', encoding='utf-8') as f:
        json.dump(reverse_vocab, f, indent=2, ensure_ascii=False)
    print(f"✓ Token-to-ID mapping saved to: {reverse_vocab_file}")
    
    # Save the tiktoken encoding for direct use
    encoding_file = output_path / "tiktoken_encoding.pkl"
    with open(encoding_file, 'wb') as f:
        pickle.dump(tokenizer.encoding, f)
    print(f"✓ Tiktoken encoding saved to: {encoding_file}")
    
    # Create a simple Python tokenizer wrapper
    tokenizer_wrapper = f'''#!/usr/bin/env python3
"""
Simple Whisper Tokenizer Wrapper

This module provides easy access to the Whisper tokenizer for use with CoreML models.
"""

import json
import pickle
from pathlib import Path

class WhisperTokenizer:
    def __init__(self, tokenizer_dir="{output_dir}"):
        self.tokenizer_dir = Path(tokenizer_dir)
        
        # Load vocabulary
        with open(self.tokenizer_dir / "vocabulary.json", 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.id_to_token = {{int(k): v for k, v in vocab_data.items()}}
        
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
            lang_token = self.special_tokens.get(f"lang_{{language}}")
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
    
    print(f"Original: {{text}}")
    print(f"Tokens: {{tokens}}")
    print(f"Decoded: {{decoded}}")
    
    # Get special tokens
    print(f"Start token: {{tokenizer.get_special_token('sot')}}")
    print(f"End token: {{tokenizer.get_special_token('eot')}}")
    
    # Get SOT sequence for English transcription
    sot_sequence = tokenizer.get_sot_sequence("en", "transcribe")
    print(f"SOT sequence: {{sot_sequence}}")
'''
    
    wrapper_file = output_path / "whisper_tokenizer.py"
    with open(wrapper_file, 'w', encoding='utf-8') as f:
        f.write(tokenizer_wrapper)
    print(f"✓ Tokenizer wrapper saved to: {wrapper_file}")
    
    # Create a README for the tokenizer
    readme_content = f"""# Whisper Tokenizer

This directory contains the extracted tokenizer for the Whisper {model_name} model.

## Files

- `vocabulary.json`: Complete vocabulary mapping (ID → token)
- `token_to_id.json`: Reverse mapping (token → ID)
- `special_tokens.json`: Special tokens and their IDs
- `tiktoken_encoding.pkl`: Original tiktoken encoding (binary)
- `whisper_tokenizer.py`: Python wrapper for easy use

## Model Information

- Model: {model_name}
- Vocabulary size: {tokenizer.encoding.n_vocab}
- Multilingual: {model.is_multilingual}

## Usage

### Python
```python
from whisper_tokenizer import WhisperTokenizer

tokenizer = WhisperTokenizer()

# Encode text to tokens
tokens = tokenizer.encode("Hello, world!")

# Decode tokens to text
text = tokenizer.decode(tokens)

# Get special tokens
sot_token = tokenizer.get_special_token("sot")
eot_token = tokenizer.get_special_token("eot")
```

### Swift/iOS
```swift
// Load vocabulary
guard let vocabPath = Bundle.main.path(forResource: "vocabulary", ofType: "json"),
      let vocabData = NSData(contentsOfFile: vocabPath),
      let vocab = try? JSONSerialization.jsonObject(with: vocabData as Data) as? [String: String] else {{
    fatalError("Failed to load vocabulary")
}}

// Convert token IDs to text
func decodeTokens(_ tokenIds: [Int]) -> String {{
    return tokenIds.compactMap {{ vocab[String($0)] }}.joined()
}}
```

## Special Tokens

Key special tokens for Whisper:
- Start of transcript: {special_tokens["sot"]}
- End of transcript: {special_tokens["eot"]}
- No speech: {special_tokens["nospeech"]}
- Transcribe task: {special_tokens["transcribe"]}
- Translate task: {special_tokens["translate"]}
"""
    
    readme_file = output_path / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ Documentation saved to: {readme_file}")
    
    print(f"\n🎉 Tokenizer extraction completed!")
    print(f"Files saved in: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Whisper tokenizer")
    parser.add_argument(
        "--model", 
        type=str, 
        default="small",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                "medium", "medium.en", "large-v1", "large-v2", "large-v3"],
        help="Whisper model to extract tokenizer from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tokenizer",
        help="Output directory for tokenizer files"
    )
    
    args = parser.parse_args()
    extract_whisper_tokenizer(args.model, args.output_dir)
