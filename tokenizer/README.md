# Whisper Tokenizer

This directory contains the extracted tokenizer for the Whisper small model.

## Files

- `vocabulary.json`: Complete vocabulary mapping (ID → token)
- `token_to_id.json`: Reverse mapping (token → ID)
- `special_tokens.json`: Special tokens and their IDs
- `tiktoken_encoding.pkl`: Original tiktoken encoding (binary)
- `whisper_tokenizer.py`: Python wrapper for easy use

## Model Information

- Model: small
- Vocabulary size: 51865
- Multilingual: True

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
      let vocab = try? JSONSerialization.jsonObject(with: vocabData as Data) as? [String: String] else {
    fatalError("Failed to load vocabulary")
}

// Convert token IDs to text
func decodeTokens(_ tokenIds: [Int]) -> String {
    return tokenIds.compactMap { vocab[String($0)] }.joined()
}
```

## Special Tokens

Key special tokens for Whisper:
- Start of transcript: 50258
- End of transcript: 50257
- No speech: 50362
- Transcribe task: 50359
- Translate task: 50358
