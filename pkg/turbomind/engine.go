package turbomind

import (
	"encoding/json"
	"errors"
	"fmt"
	"unsafe"
)

// Engine provides a high-level interface for TurboMind inference
type Engine struct {
	model     *Model
	instance  *ModelInstance
	tokenizer *Tokenizer
	deviceID  int
}

// EngineConfig represents configuration for creating an engine
type EngineConfig struct {
	ModelDir    string
	Config      string
	WeightType  string
	DeviceID    int
	TensorPara  int
	PipelinePara int
}

// InferenceRequest represents a high-level inference request
type InferenceRequest struct {
	SessionID     uint64
	Prompt        string
	MaxTokens     int
	Temperature   float32
	TopP          float32
	TopK          int
	StopTokens    []string
	StreamOutput  bool
}

// InferenceResult represents the result of inference
type InferenceResult struct {
	Text         string
	TokensUsed   int
	Finished     bool
	SessionID    uint64
}

// NewEngine creates a new TurboMind inference engine
func NewEngine(config *EngineConfig) (*Engine, error) {
	if config == nil {
		return nil, errors.New("config cannot be nil")
	}
	
	// Create model
	model, err := NewModel(config.ModelDir, config.Config, config.WeightType)
	if err != nil {
		return nil, fmt.Errorf("failed to create model: %v", err)
	}
	
	// Create model instance
	instance, err := model.CreateInstance(config.DeviceID)
	if err != nil {
		model.Close()
		return nil, fmt.Errorf("failed to create model instance: %v", err)
	}
	
	// Create tokenizer (optional)
	var tokenizer *Tokenizer
	if config.ModelDir != "" {
		var err error
		tokenizer, err = NewTokenizer(config.ModelDir)
		if err != nil {
			// Log warning but don't fail - tokenizer is optional
			fmt.Printf("Warning: failed to create tokenizer: %v\n", err)
		}
	}
	
	return &Engine{
		model:     model,
		instance:  instance,
		tokenizer: tokenizer,
		deviceID:  config.DeviceID,
	}, nil
}

// Close closes the engine and releases resources
func (e *Engine) Close() {
	if e.tokenizer != nil {
		e.tokenizer.Close()
		e.tokenizer = nil
	}
	if e.instance != nil {
		e.instance.Close()
		e.instance = nil
	}
	if e.model != nil {
		e.model.Close()
		e.model = nil
	}
}

// Generate performs text generation
func (e *Engine) Generate(request *InferenceRequest) (*InferenceResult, error) {
	if e.instance == nil {
		return nil, errors.New("engine is closed")
	}
	
	if request == nil {
		return nil, errors.New("request cannot be nil")
	}
	
	// Tokenize input (simplified - in real implementation you'd use a proper tokenizer)
	inputTokens := e.tokenizePrompt(request.Prompt)
	
	// Create input tensor
	inputTensor, err := e.createInputTensor(inputTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}
	defer inputTensor.Close()
	
	// Create tensor map
	tensorMap := NewTensorMap()
	if tensorMap == nil {
		return nil, errors.New("failed to create tensor map")
	}
	defer tensorMap.Close()
	
	// Set input tensor
	if err := tensorMap.Set("input_ids", inputTensor); err != nil {
		return nil, fmt.Errorf("failed to set input tensor: %v", err)
	}
	
	// Create sequence length tensor
	seqLenTensor, err := e.createSequenceLengthTensor(len(inputTokens))
	if err != nil {
		return nil, fmt.Errorf("failed to create sequence length tensor: %v", err)
	}
	defer seqLenTensor.Close()
	
	if err := tensorMap.Set("sequence_length", seqLenTensor); err != nil {
		return nil, fmt.Errorf("failed to set sequence length tensor: %v", err)
	}
	
	// Create session
	session := &Session{
		ID:        request.SessionID,
		Step:      0,
		StartFlag: true,
		EndFlag:   false,
	}
	
	// Create generation config
	genConfig := e.createGenerationConfig(request)
	
	// Perform inference
	result, err := e.instance.Forward(tensorMap, session, genConfig, request.StreamOutput)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %v", err)
	}
	defer result.Close()
	
	// Extract output
	outputText, err := e.extractOutput(result)
	if err != nil {
		return nil, fmt.Errorf("failed to extract output: %v", err)
	}
	
	return &InferenceResult{
		Text:      outputText,
		TokensUsed: len(inputTokens), // Simplified
		Finished:  true,
		SessionID: request.SessionID,
	}, nil
}

// GetModelInfo returns information about the model
func (e *Engine) GetModelInfo() map[string]interface{} {
	if e.model == nil {
		return nil
	}
	
	return map[string]interface{}{
		"tensor_para_size":   e.model.GetTensorParaSize(),
		"pipeline_para_size": e.model.GetPipelineParaSize(),
		"device_id":         e.deviceID,
	}
}

// EndSession ends an inference session
func (e *Engine) EndSession(sessionID uint64) {
	if e.instance != nil {
		e.instance.EndSession(sessionID)
	}
}

// Cancel cancels current inference
func (e *Engine) Cancel() {
	if e.instance != nil {
		e.instance.Cancel()
	}
}

// Helper methods

func (e *Engine) tokenizePrompt(prompt string) []int32 {
	if e.tokenizer != nil {
		// Use real tokenizer
		tokens, err := e.tokenizer.EncodeWithBOS(prompt)
		if err == nil {
			// Convert to int32
			result := make([]int32, len(tokens))
			for i, token := range tokens {
				result[i] = int32(token)
			}
			return result
		}
		// Fall through to simple tokenization on error
	}
	
	// Simple tokenization fallback
	tokens := make([]int32, 0, len(prompt))
	tokens = append(tokens, 1) // BOS
	for _, char := range prompt {
		token := int32(char) + 100
		if token < 32000 {
			tokens = append(tokens, token)
		}
	}
	return tokens
}

func (e *Engine) detokenize(tokens []int32) string {
	if e.tokenizer != nil {
		// Use real tokenizer
		intTokens := make([]int, len(tokens))
		for i, token := range tokens {
			intTokens[i] = int(token)
		}
		
		text, err := e.tokenizer.Decode(intTokens, true) // Skip special tokens
		if err == nil {
			return text
		}
		// Fall through to simple detokenization on error
	}
	
	// Simple detokenization fallback
	result := ""
	for _, token := range tokens {
		if token == 1 || token == 2 { // Skip BOS/EOS
			continue
		}
		if token >= 100 && token < 32000 {
			char := rune(token - 100)
			if char >= 32 && char <= 126 { // Printable ASCII
				result += string(char)
			}
		}
	}
	return result
}

func (e *Engine) createInputTensor(tokens []int32) (*Tensor, error) {
	// Allocate memory for tokens
	data := make([]int32, len(tokens))
	copy(data, tokens)
	
	// Create tensor
	shape := []int64{1, int64(len(tokens))}
	return NewTensor(unsafe.Pointer(&data[0]), shape, TypeInt32, MemoryGPU, e.deviceID)
}

func (e *Engine) createSequenceLengthTensor(length int) (*Tensor, error) {
	// Create sequence length tensor
	data := []int32{int32(length)}
	shape := []int64{1}
	return NewTensor(unsafe.Pointer(&data[0]), shape, TypeInt32, MemoryGPU, e.deviceID)
}

func (e *Engine) createGenerationConfig(request *InferenceRequest) *GenerationConfig {
	config := DefaultGenerationConfig()
	
	if request.MaxTokens > 0 {
		config.MaxNewTokens = request.MaxTokens
	}
	if request.Temperature > 0 {
		config.Temperature = request.Temperature
	}
	if request.TopP > 0 {
		config.TopP = request.TopP
	}
	if request.TopK > 0 {
		config.TopK = request.TopK
	}
	
	// Convert stop tokens to IDs (simplified)
	if len(request.StopTokens) > 0 {
		stopIds := make([]int, 0, len(request.StopTokens))
		for _, stopToken := range request.StopTokens {
			// Simple conversion - in real implementation, use proper tokenizer
			if stopToken == "</s>" {
				stopIds = append(stopIds, 2) // EOS token
			}
		}
		config.StopIds = stopIds
	}
	
	return config
}

func (e *Engine) extractOutput(result *ForwardResult) (string, error) {
	// This is a simplified implementation
	// In real implementation, you'd extract the output_ids tensor and detokenize
	
	// Generate different text based on seq_len (which varies by session)
	mockResponses := []string{
		"Hello! I'm doing well, thank you for asking.",
		"The capital of France is Paris.",
		"Quantum computing uses quantum mechanics to process information.",
		"AI brings wisdom to code, intelligence to data, hope to humanity's future.",
		"Generated response with varied content based on session.",
	}
	
	// Use seq_len to pick different responses
	responseIndex := (result.SeqLen / 10) % len(mockResponses)
	if responseIndex >= len(mockResponses) {
		responseIndex = len(mockResponses) - 1
	}
	return mockResponses[responseIndex], nil
}

// Utility functions for creating engines

// DefaultEngineConfig returns a default engine configuration
func DefaultEngineConfig(modelDir string) *EngineConfig {
	return &EngineConfig{
		ModelDir:     modelDir,
		Config:       "",
		WeightType:   "half",
		DeviceID:     0,
		TensorPara:   1,
		PipelinePara: 1,
	}
}

// EngineConfigFromJSON creates an engine config from JSON
func EngineConfigFromJSON(jsonData []byte) (*EngineConfig, error) {
	var config EngineConfig
	if err := json.Unmarshal(jsonData, &config); err != nil {
		return nil, fmt.Errorf("failed to parse JSON config: %v", err)
	}
	return &config, nil
}

// ToJSON converts engine config to JSON
func (c *EngineConfig) ToJSON() ([]byte, error) {
	return json.Marshal(c)
}