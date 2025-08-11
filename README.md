# Whisper to CoreML Converter

Convert OpenAI Whisper models to Apple CoreML format with optional Apple Neural Engine (ANE) optimizations for better performance on Apple devices.

## Features

- Convert Whisper models (tiny, base, small, medium, large) to CoreML
- Support for encoder-only conversion for efficient audio feature extraction
- Apple Neural Engine (ANE) optimizations available
- 16-bit quantization support for smaller model sizes
- SSL certificate issue handling for macOS
- Full compatibility with iOS and macOS applications

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

### Basic Usage

Convert a Whisper model to CoreML format:

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

1. Add the `.mlpackage` files to your Xcode project
2. Use Apple's Core ML framework:

```swift
import CoreML

// Load the model
guard let model = try? MLModel(contentsOf: modelURL) else {
    fatalError("Failed to load model")
}

// Prepare input data (mel spectrogram)
let input = try MLDictionaryFeatureProvider(dictionary: [
    "logmel_data": MLMultiArray(/* your mel spectrogram data */)
])

// Run prediction
let output = try model.prediction(from: input)
```

## File Structure

```
WhisperConversion/
├── whisper_coreml_fixed.py     # Main conversion script (SSL fixed)
├── whisper_coreml.py           # Original conversion script
├── test_coreml_models.py       # Testing script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── converted_models/           # Output directory
    ├── coreml-encoder-base.mlpackage
    └── coreml-encoder-small.mlpackage
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
