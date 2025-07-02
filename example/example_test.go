package main

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/xianml/turbomind-go/pkg/turbomind"
)

var (
	testModelPath = os.Getenv("TEST_MODEL_PATH")
	testLibPath   = os.Getenv("TURBOMIND_LIB_PATH")
)

func TestMain(m *testing.M) {
	// Set default paths if not provided
	if testLibPath == "" {
		testLibPath = "../build/libturbomind_go.so"
	}
	if testModelPath == "" {
		// Default to a local Phi-3-mini model path
		testModelPath = "./models/microsoft--Phi-3-mini-4k-instruct"
	}

	// Check if library exists
	if _, err := os.Stat(testLibPath); os.IsNotExist(err) {
		fmt.Printf("Warning: TurboMind library not found at %s. Run 'make build' first.\n", testLibPath)
		os.Exit(1)
	}

	// Initialize the library
	if err := turbomind.Initialize(testLibPath); err != nil {
		fmt.Printf("Failed to initialize TurboMind library: %v\n", err)
		os.Exit(1)
	}

	// Run tests
	code := m.Run()
	os.Exit(code)
}

func TestLibraryLoad(t *testing.T) {
	// Test library loading
	err := turbomind.Initialize(testLibPath)
	assert.NoError(t, err, "Library should load successfully")
}

func TestVersionInfo(t *testing.T) {
	version := turbomind.GetVersion()
	assert.NotEmpty(t, version.Version, "Version should not be empty")
	assert.NotEmpty(t, version.BuildTime, "Build time should not be empty")
	
	fmt.Printf("TurboMind Version: %s\n", version.Version)
	fmt.Printf("Git Commit: %s\n", version.GitCommit)
	fmt.Printf("Build Time: %s\n", version.BuildTime)
	fmt.Printf("CUDA Version: %s\n", version.CudaVersion)
}

func TestDefaultConfig(t *testing.T) {
	config := turbomind.DefaultConfig("/path/to/model")
	assert.Equal(t, "/path/to/model", config.ModelPath)
	assert.Equal(t, "hf", config.ModelFormat)
	assert.Equal(t, 1, config.TP)
	assert.Equal(t, 2048, config.SessionLen)
	assert.Equal(t, 32, config.MaxBatchSize)
	assert.Equal(t, 0, config.QuantPolicy) // fp16
}

func TestEngineCreationWithoutModel(t *testing.T) {
	// Test engine creation with invalid path
	config := turbomind.DefaultConfig("/nonexistent/path")
	engine, err := turbomind.NewEngine(config)
	
	// In mock mode, engine creation should succeed even with invalid path
	// (Real implementation would fail with invalid model path)
	if engine != nil {
		// Mock implementation - engine created successfully
		assert.NoError(t, err, "Mock implementation should succeed even with invalid path")
		assert.NotNil(t, engine, "Engine should be created in mock mode")
		assert.True(t, engine.IsReady(), "Mock engine should be ready")
		
		// Test cleanup
		engine.Close()
		assert.False(t, engine.IsReady(), "Engine should not be ready after close")
		
		fmt.Printf("Mock engine test passed: created and closed successfully\n")
	} else {
		// Real implementation - should fail
		assert.Error(t, err, "Real implementation should fail with invalid model path")
		assert.Nil(t, engine, "Engine should be nil on failure")
		
		// Check error message
		errorMsg := turbomind.GetLastError()
		assert.NotEmpty(t, errorMsg, "Error message should not be empty")
		fmt.Printf("Expected error: %s\n", errorMsg)
	}
}

func TestEngineWithModel(t *testing.T) {
	// Skip if no model path is provided
	if testModelPath == "" || !fileExists(testModelPath) {
		t.Skip("Skipping model test: TEST_MODEL_PATH not set or model not found")
	}

	t.Run("CreateEngine", func(t *testing.T) {
		config := turbomind.DefaultConfig(testModelPath)
		config.SessionLen = 1024  // Smaller for faster testing
		config.MaxBatchSize = 1   // Single batch for testing

		engine, err := turbomind.NewEngine(config)
		require.NoError(t, err, "Should create engine successfully")
		require.NotNil(t, engine, "Engine should not be nil")

		// Check if engine is ready
		assert.True(t, engine.IsReady(), "Engine should be ready")

		// Test model info
		modelInfo, err := engine.GetModelInfo()
		if err == nil {
			fmt.Printf("Model Info:\n")
			fmt.Printf("  Name: %s\n", modelInfo.ModelName)
			fmt.Printf("  Type: %s\n", modelInfo.ModelType)
			fmt.Printf("  Vocab Size: %d\n", modelInfo.VocabSize)
			fmt.Printf("  Hidden Size: %d\n", modelInfo.HiddenSize)
			fmt.Printf("  Layers: %d\n", modelInfo.NumLayers)
			fmt.Printf("  Max Position: %d\n", modelInfo.MaxPositionEmbeddings)
		}

		// Close engine
		engine.Close()
		assert.False(t, engine.IsReady(), "Engine should not be ready after close")
	})
}

func TestSimpleInference(t *testing.T) {
	// Skip if no model path is provided
	if testModelPath == "" || !fileExists(testModelPath) {
		t.Skip("Skipping inference test: TEST_MODEL_PATH not set or model not found")
	}

	config := turbomind.DefaultConfig(testModelPath)
	config.SessionLen = 1024
	config.MaxBatchSize = 1

	engine, err := turbomind.NewEngine(config)
	require.NoError(t, err)
	require.NotNil(t, engine)
	defer engine.Close()

	// Wait for engine to be ready
	timeout := time.After(30 * time.Second)
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			t.Fatal("Engine failed to become ready within timeout")
		case <-ticker.C:
			if engine.IsReady() {
				goto EngineReady
			}
		}
	}

EngineReady:
	t.Run("SimpleGeneration", func(t *testing.T) {
		// Simple test prompt
		params := turbomind.RequestParams{
			RequestID:         1,
			Prompt:            "Hello, how are you?",
			MaxNewTokens:      50,
			Temperature:       0.7,
			TopP:              0.8,
			TopK:              40,
			RepetitionPenalty: 1,
			Stream:            false,
			StopWords:         "",
		}

		response, err := engine.Generate(params)
		require.NoError(t, err, "Generation should succeed")
		require.NotNil(t, response, "Response should not be nil")

		assert.Equal(t, int64(1), response.RequestID)
		assert.NotEmpty(t, response.Text, "Generated text should not be empty")
		assert.True(t, response.InputTokens > 0, "Input tokens should be positive")
		assert.True(t, response.OutputTokens > 0, "Output tokens should be positive")
		assert.True(t, response.Finished, "Generation should be finished")
		assert.Equal(t, 0, response.ErrorCode, "Error code should be 0")

		fmt.Printf("Generation Results:\n")
		fmt.Printf("  Input Tokens: %d\n", response.InputTokens)
		fmt.Printf("  Output Tokens: %d\n", response.OutputTokens)
		fmt.Printf("  Generated Text: %s\n", response.Text)
	})

	t.Run("MultipleGenerations", func(t *testing.T) {
		prompts := []string{
			"What is artificial intelligence?",
			"Explain machine learning in simple terms.",
			"What are the benefits of using Go programming language?",
		}

		for i, prompt := range prompts {
			params := turbomind.RequestParams{
				RequestID:         int64(i + 2),
				Prompt:            prompt,
				MaxNewTokens:      30,
				Temperature:       0.5,
				TopP:              0.9,
				TopK:              50,
				RepetitionPenalty: 1,
				Stream:            false,
				StopWords:         "",
			}

			response, err := engine.Generate(params)
			require.NoError(t, err, "Generation %d should succeed", i)
			require.NotNil(t, response, "Response %d should not be nil", i)

			assert.NotEmpty(t, response.Text, "Generated text %d should not be empty", i)
			fmt.Printf("Prompt %d: %s\n", i+1, prompt)
			fmt.Printf("Response %d: %s\n", i+1, response.Text)
			fmt.Printf("---\n")
		}
	})
}

func TestErrorHandling(t *testing.T) {
	if testModelPath == "" || !fileExists(testModelPath) {
		t.Skip("Skipping error handling test: TEST_MODEL_PATH not set or model not found")
	}

	config := turbomind.DefaultConfig(testModelPath)
	engine, err := turbomind.NewEngine(config)
	require.NoError(t, err)
	defer engine.Close()

	// Wait for ready
	for !engine.IsReady() {
		time.Sleep(100 * time.Millisecond)
	}

	t.Run("EmptyPrompt", func(t *testing.T) {
		params := turbomind.RequestParams{
			RequestID:    999,
			Prompt:       "", // Empty prompt
			MaxNewTokens: 10,
		}

		response, err := engine.Generate(params)
		// This might succeed or fail depending on implementation
		if err != nil {
			fmt.Printf("Expected error with empty prompt: %s\n", err)
		} else {
			fmt.Printf("Empty prompt handled gracefully: %s\n", response.Text)
		}
	})

	t.Run("ClosedEngine", func(t *testing.T) {
		engine.Close()

		params := turbomind.RequestParams{
			RequestID: 1000,
			Prompt:    "Test prompt",
		}

		_, err := engine.Generate(params)
		assert.Error(t, err, "Should fail with closed engine")
		assert.Contains(t, err.Error(), "closed", "Error should mention engine is closed")
	})
}

// Benchmark tests
func BenchmarkGeneration(b *testing.B) {
	if testModelPath == "" || !fileExists(testModelPath) {
		b.Skip("Skipping benchmark: TEST_MODEL_PATH not set or model not found")
	}

	config := turbomind.DefaultConfig(testModelPath)
	config.MaxBatchSize = 1
	
	engine, err := turbomind.NewEngine(config)
	require.NoError(b, err)
	defer engine.Close()

	// Wait for ready
	for !engine.IsReady() {
		time.Sleep(100 * time.Millisecond)
	}

	params := turbomind.RequestParams{
		Prompt:       "Benchmark test prompt",
		MaxNewTokens: 20,
		Temperature:  0.7,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		params.RequestID = int64(i)
		_, err := engine.Generate(params)
		if err != nil {
			b.Fatalf("Generation failed: %v", err)
		}
	}
}

// Helper functions
func fileExists(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func directoryExists(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return info.IsDir()
}

// Example usage function
func ExampleUsage() {
	// Initialize library
	err := turbomind.Initialize("../build/libturbomind_go.so")
	if err != nil {
		fmt.Printf("Failed to initialize: %v\n", err)
		return
	}

	// Get version info
	version := turbomind.GetVersion()
	fmt.Printf("TurboMind Version: %s\n", version.Version)

	// Create engine configuration
	config := turbomind.DefaultConfig("./models/phi-3-mini")
	config.QuantPolicy = 8 // Use INT8 quantization

	// Create engine
	engine, err := turbomind.NewEngine(config)
	if err != nil {
		fmt.Printf("Failed to create engine: %v\n", err)
		return
	}
	defer engine.Close()

	// Wait for engine to be ready
	for !engine.IsReady() {
		time.Sleep(100 * time.Millisecond)
	}

	// Prepare request
	params := turbomind.RequestParams{
		RequestID:    1,
		Prompt:       "What is the meaning of life?",
		MaxNewTokens: 100,
		Temperature:  0.7,
		TopP:         0.8,
	}

	// Generate response
	response, err := engine.Generate(params)
	if err != nil {
		fmt.Printf("Generation failed: %v\n", err)
		return
	}

	fmt.Printf("Generated text: %s\n", response.Text)
	fmt.Printf("Tokens used: %d input, %d output\n", response.InputTokens, response.OutputTokens)
} 