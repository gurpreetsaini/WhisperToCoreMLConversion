import CoreML
import Foundation

// MARK: - Whisper CoreML Implementation for iOS

class WhisperCoreML {
    
    // MARK: - Properties
    private let encoder: MLModel
    private let decoder: MLModel
    private let vocabulary: [Int: String]
    private let specialTokens: [String: Int]
    
    // MARK: - Initialization
    init() throws {
        // Load CoreML models from app bundle
        guard let encoderURL = Bundle.main.url(forResource: "coreml-encoder-small", withExtension: "mlpackage"),
              let decoderURL = Bundle.main.url(forResource: "coreml-decoder-small", withExtension: "mlpackage") else {
            throw WhisperError.modelNotFound
        }
        
        self.encoder = try MLModel(contentsOf: encoderURL)
        self.decoder = try MLModel(contentsOf: decoderURL)
        
        // Load tokenizer files
        self.vocabulary = try Self.loadVocabulary()
        self.specialTokens = try Self.loadSpecialTokens()
    }
    
    // MARK: - Public Methods
    
    /// Transcribe audio from mel spectrogram
    /// - Parameter melSpectrogram: MLMultiArray of shape [1, 80, 3000]
    /// - Returns: Transcribed text
    func transcribe(melSpectrogram: MLMultiArray) throws -> String {
        // 1. Encode audio features
        let audioFeatures = try encodeAudio(melSpectrogram)
        
        // 2. Generate tokens
        let tokens = try generateTokens(from: audioFeatures)
        
        // 3. Decode to text
        let text = decodeTokens(tokens)
        
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    // MARK: - Private Methods
    
    private func encodeAudio(_ melSpectrogram: MLMultiArray) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(dictionary: ["logmel_data": melSpectrogram])
        let output = try encoder.prediction(from: input)
        
        guard let features = output.featureValue(for: "output")?.multiArrayValue else {
            throw WhisperError.encodingFailed
        }
        
        return features
    }
    
    private func generateTokens(from audioFeatures: MLMultiArray, maxTokens: Int = 100) throws -> [Int] {
        // Initialize with start-of-transcript sequence
        var tokens = getStartSequence()
        var generatedTokens: [Int] = []
        
        for _ in 0..<maxTokens {
            // Prepare current token input
            let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            tokenArray[0] = NSNumber(value: tokens.last!)
            
            // Run decoder
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "token_data": tokenArray,
                "audio_data": audioFeatures
            ])
            
            let output = try decoder.prediction(from: input)
            
            guard let logits = output.featureValue(for: "output")?.multiArrayValue else {
                throw WhisperError.decodingFailed
            }
            
            // Get next token (greedy decoding)
            let nextToken = argmax(logits)
            
            // Check for end token
            if nextToken == specialTokens["eot"] {
                break
            }
            
            tokens.append(nextToken)
            generatedTokens.append(nextToken)
        }
        
        return generatedTokens
    }
    
    private func decodeTokens(_ tokens: [Int]) -> String {
        return tokens.compactMap { vocabulary[$0] }
                    .joined()
                    .replacingOccurrences(of: "<|", with: " <|") // Add spaces around special tokens
                    .replacingOccurrences(of: "|>", with: "|> ")
    }
    
    private func getStartSequence() -> [Int] {
        return [
            specialTokens["sot"] ?? 50258,           // Start of transcript
            specialTokens["transcribe"] ?? 50359,    // Transcribe task
            specialTokens["notimestamps"] ?? 50363   // No timestamps
        ]
    }
    
    private func argmax(_ array: MLMultiArray) -> Int {
        let count = array.shape[2].intValue
        var maxIndex = 0
        var maxValue = array[0].floatValue
        
        for i in 1..<count {
            let value = array[i].floatValue
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }
        
        return maxIndex
    }
    
    // MARK: - Static Helper Methods
    
    private static func loadVocabulary() throws -> [Int: String] {
        guard let path = Bundle.main.path(forResource: "vocabulary", ofType: "json"),
              let data = Data(contentsOf: URL(fileURLWithPath: path)) else {
            throw WhisperError.vocabularyNotFound
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]
        
        return json.reduce(into: [:]) { result, pair in
            if let key = Int(pair.key) {
                result[key] = pair.value
            }
        }
    }
    
    private static func loadSpecialTokens() throws -> [String: Int] {
        guard let path = Bundle.main.path(forResource: "special_tokens", ofType: "json"),
              let data = Data(contentsOf: URL(fileURLWithPath: path)) else {
            throw WhisperError.specialTokensNotFound
        }
        
        return try JSONSerialization.jsonObject(with: data) as? [String: Int] ?? [:]
    }
}

// MARK: - Error Types

enum WhisperError: Error, LocalizedError {
    case modelNotFound
    case vocabularyNotFound
    case specialTokensNotFound
    case encodingFailed
    case decodingFailed
    case invalidInput
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "CoreML models not found in app bundle"
        case .vocabularyNotFound:
            return "Vocabulary file not found"
        case .specialTokensNotFound:
            return "Special tokens file not found"
        case .encodingFailed:
            return "Audio encoding failed"
        case .decodingFailed:
            return "Token decoding failed"
        case .invalidInput:
            return "Invalid input data"
        }
    }
}

// MARK: - Usage Example

class ViewController: UIViewController {
    
    private var whisper: WhisperCoreML?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        do {
            whisper = try WhisperCoreML()
            print("Whisper initialized successfully")
        } catch {
            print("Failed to initialize Whisper: \(error)")
        }
    }
    
    func transcribeAudio(melSpectrogram: MLMultiArray) {
        guard let whisper = whisper else { return }
        
        do {
            let transcription = try whisper.transcribe(melSpectrogram: melSpectrogram)
            print("Transcription: \(transcription)")
        } catch {
            print("Transcription failed: \(error)")
        }
    }
}
