# Whisper to CoreML Converter

Convert OpenAI Whisper models to Apple CoreML format with optional Apple Neural Engine (ANE) optimizations for better performance on Apple devices.

## Features

- Convert Whisper models (tiny, base, small, medium, large) to CoreML
- Support for encoder-only conversion for efficient audio feature extraction
- **Complete pipeline**: Both encoder and decoder conversion for full speech-to-text
- **Tokenizer extraction**: Extract and save Whisper tokenizer for text processing
- Apple Neural Engine (ANE) optimizations available
- 16-bit quantization support for smaller model sizes
- SSL certificate issue handling for macOS
- Full compatibility with iOS and macOS applications
- **End-to-end pipeline demo** with beam search support

## Requirements

- Python 3.10+ (tested with Python 3.12)
- macOS (for CoreML tools)
- At least 4GB free disk space (models can be large)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd WhisperConversion
```

2. Create a virtual environment:
```bash
python3 -m venv whisper_env
source whisper_env/bin/activate  # On Windows: whisper_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline (Encoder + Decoder + Tokenizer)

Convert both encoder and decoder for full speech-to-text capability:

```bash
# Convert complete pipeline for small model
python whisper_coreml_fixed.py --model small --output-dir ./converted_models

# Extract tokenizer
python extract_tokenizer.py --model small --output-dir ./tokenizer

# Test complete pipeline
python complete_pipeline_demo.py
```

### Basic Usage

Convert individual components:

```bash
# Convert base model encoder only
python whisper_coreml_fixed.py --model base --encoder-only --output-dir ./converted_models

# Convert small model with both encoder and decoder
python whisper_coreml_fixed.py --model small --output-dir ./converted_models

# Convert with quantization for smaller size
python whisper_coreml_fixed.py --model base --encoder-only --quantize --output-dir ./converted_models

# Convert with ANE optimization
python whisper_coreml_fixed.py --model base --encoder-only --optimize-ane --output-dir ./converted_models
```

### Available Models

- `tiny` (39 MB) - Fastest, lowest accuracy
- `base` (74 MB) - Good balance of speed and accuracy
- `small` (244 MB) - Better accuracy
- `medium` (769 MB) - High accuracy
- `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo` (1550 MB) - Highest accuracy

Add `.en` suffix for English-only models (e.g., `base.en`, `small.en`).

### Command Line Options

- `--model`: Whisper model to convert (required)
- `--encoder-only`: Only convert the encoder (recommended for most use cases)
- `--quantize`: Quantize weights to 16-bit for smaller model size
- `--optimize-ane`: Optimize for Apple Neural Engine
- `--output-dir`: Output directory for converted models (default: `models`)

## Testing Converted Models

Test your converted CoreML models:

```bash
python test_coreml_models.py
```

This script will:
- Load and validate the converted CoreML models
- Test prediction with sample data
- Display model input/output specifications

## Model Specifications

### Input Format
- **Shape**: `(1, 80, 3000)`
- **Type**: Float32
- **Description**: Mel spectrogram with 80 mel bins and up to 3000 time steps

### Output Format
- **Base Model**: `(1, 1500, 512)` - 512-dimensional features
- **Small Model**: `(1, 1500, 768)` - 768-dimensional features
- **Type**: Float32
- **Description**: Encoded audio features

## Integration with iOS/macOS

### Complete Speech-to-Text Pipeline

1. Add the `.mlpackage` files and tokenizer files to your Xcode project
2. Use Apple's Core ML framework for inference
3. Use the tokenizer for text processing

```swift
import CoreML

class WhisperSpeechRecognizer {
    let encoder: MLModel
    let decoder: MLModel
    let vocabulary: [Int: String]
    let specialTokens: [String: Int]
    
    init() throws {
        encoder = try MLModel(contentsOf: Bundle.main.url(forResource: "coreml-encoder-small", withExtension: "mlpackage")!)
        decoder = try MLModel(contentsOf: Bundle.main.url(forResource: "coreml-decoder-small", withExtension: "mlpackage")!)
        
        // Load tokenizer files
        let vocabData = try Data(contentsOf: Bundle.main.url(forResource: "vocabulary", withExtension: "json")!)
        vocabulary = try JSONDecoder().decode([String: String].self, from: vocabData)
            .reduce(into: [:]) { result, pair in
                result[Int(pair.key)!] = pair.value
            }
        
        let tokensData = try Data(contentsOf: Bundle.main.url(forResource: "special_tokens", withExtension: "json")!)
        specialTokens = try JSONDecoder().decode([String: Int].self, from: tokensData)
    }
    
    func transcribe(melSpectrogram: MLMultiArray) throws -> String {
        // 1. Encode audio features
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: ["logmel_data": melSpectrogram])
        let encoderOutput = try encoder.prediction(from: encoderInput)
        let audioFeatures = encoderOutput.featureValue(for: "output")!.multiArrayValue!
        
        // 2. Initialize with start-of-transcript tokens
        var tokens: [Int] = [
            specialTokens["sot"]!,           // Start of transcript
            specialTokens["transcribe"]!,    // Transcribe task
            specialTokens["notimestamps"]!   // No timestamps
        ]
        
        // 3. Generate tokens iteratively
        for _ in 0..<100 { // Max 100 tokens
            let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            tokenArray[0] = NSNumber(value: tokens.last!)
            
            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "token_data": tokenArray,
                "audio_data": audioFeatures
            ])
            
            let decoderOutput = try decoder.prediction(from: decoderInput)
            let logits = decoderOutput.featureValue(for: "output")!.multiArrayValue!
            
            // Get next token (simplified - should use proper sampling)
            let nextToken = argmax(logits)
            if nextToken == specialTokens["eot"]! { break } // End token
            tokens.append(nextToken)
        }
        
        // 4. Convert tokens to text
        let textTokens = tokens.dropFirst(3) // Skip SOT sequence
        return textTokens.compactMap { vocabulary[$0] }.joined()
    }
}
```

### Tokenizer Usage

```swift
// Load vocabulary for token decoding
func loadVocabulary() -> [Int: String] {
    guard let path = Bundle.main.path(forResource: "vocabulary", ofType: "json"),
          let data = NSData(contentsOfFile: path),
          let json = try? JSONSerialization.jsonObject(with: data as Data) as? [String: String] else {
        fatalError("Failed to load vocabulary")
    }
    
    return json.reduce(into: [:]) { result, pair in
        result[Int(pair.key)!] = pair.value
    }
}

// Convert token IDs to text
func decodeTokens(_ tokenIds: [Int], vocabulary: [Int: String]) -> String {
    return tokenIds.compactMap { vocabulary[$0] }.joined()
}
```

## File Structure

```
WhisperConversion/
├── whisper_coreml_fixed.py         # Main conversion script (SSL fixed)
├── whisper_coreml.py               # Original conversion script
├── extract_tokenizer.py            # Tokenizer extraction script
├── complete_pipeline_demo.py       # End-to-end pipeline demonstration
├── test_coreml_models.py           # Testing script for models
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── converted_models/               # Output directory for CoreML models
│   ├── coreml-encoder-base.mlpackage
│   ├── coreml-encoder-small.mlpackage
│   └── coreml-decoder-small.mlpackage
└── tokenizer/                      # Extracted tokenizer files
    ├── vocabulary.json             # Token ID to text mapping
    ├── token_to_id.json           # Text to token ID mapping
    ├── special_tokens.json        # Special tokens (SOT, EOT, etc.)
    ├── tiktoken_encoding.pkl      # Original tiktoken encoding
    ├── whisper_tokenizer.py       # Python tokenizer wrapper
    └── README.md                  # Tokenizer documentation
```

## Known Issues

### SSL Certificate Issues on macOS
The original script may encounter SSL certificate verification errors when downloading Whisper models. Use `whisper_coreml_fixed.py` which includes a workaround for this issue.

### PyTorch Version Compatibility
You may see warnings about PyTorch version compatibility with CoreMLtools. The current setup uses PyTorch 2.8.0, while CoreMLtools was tested with PyTorch 2.5.0. This typically doesn't cause issues but may generate warnings.

## Performance Notes

- **Encoder-only models** are recommended for most use cases as they provide audio feature extraction without the overhead of text generation
- **ANE optimization** can significantly improve inference speed on Apple devices with Neural Engine
- **Quantization** reduces model size by ~50% with minimal accuracy loss

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Apple for CoreML and ANE technologies
- The open source community for various optimization techniques

## Support

If you encounter issues:
1. Check the [Known Issues](#known-issues) section
2. Ensure all requirements are met
3. Try with different model sizes
4. Open an issue with detailed error logs
