package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/xianml/turbomind-go/pkg/turbomind"
)

func main() {
	// Command line flags
	var (
		libPath     = flag.String("lib", "libturbomind_go.so.0.9.0", "Path to TurboMind shared library")
		modelPath   = flag.String("model", "", "Path to the model directory (required)")
		prompt      = flag.String("prompt", "Hello! How are you today?", "Input prompt for generation")
		maxTokens   = flag.Int("max-tokens", 100, "Maximum number of tokens to generate")
		temperature = flag.Float64("temperature", 0.7, "Temperature for sampling")
		topP        = flag.Float64("top-p", 0.8, "Top-p for sampling")
		quantPolicy = flag.Int("quant", 0, "Quantization policy (0=fp16, 4=int4, 8=int8)")
		sessionLen  = flag.Int("session-len", 2048, "Maximum session length")
		verbose     = flag.Bool("verbose", false, "Enable verbose output")
	)
	flag.Parse()

	// Validate required parameters
	if *modelPath == "" {
		fmt.Println("Error: --model parameter is required")
		fmt.Println("\nUsage example:")
		fmt.Println("  go run example_main.go --model ./models/microsoft--Phi-3-mini-4k-instruct --prompt \"What is AI?\"")
		fmt.Println("\nAvailable options:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Initialize library
	fmt.Printf("Initializing TurboMind library from: %s\n", *libPath)
	if err := turbomind.Initialize(*libPath); err != nil {
		log.Fatalf("Failed to initialize TurboMind library: %v", err)
	}

	// Print version information
	if *verbose {
		version := turbomind.GetVersion()
		fmt.Printf("\nTurboMind Version Information:\n")
		fmt.Printf("  Version: %s\n", version.Version)
		fmt.Printf("  Git Commit: %s\n", version.GitCommit)
		fmt.Printf("  Build Time: %s\n", version.BuildTime)
		fmt.Printf("  CUDA Version: %s\n", version.CudaVersion)
		fmt.Println()
	}

	// Create engine configuration
	config := turbomind.DefaultConfig(*modelPath)
	config.QuantPolicy = *quantPolicy
	config.SessionLen = *sessionLen
	config.MaxBatchSize = 1

	fmt.Printf("Creating TurboMind engine...\n")

	// Create engine
	startTime := time.Now()
	engine, err := turbomind.NewEngine(config)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	// Wait for engine to be ready
	fmt.Printf("Waiting for engine to be ready...")
	for !engine.IsReady() {
		fmt.Print(".")
		time.Sleep(500 * time.Millisecond)
	}
	initTime := time.Since(startTime)
	fmt.Printf(" Ready! (%.2fs)\n", initTime.Seconds())

	// Prepare generation parameters
	params := turbomind.RequestParams{
		RequestID:         1,
		Prompt:            *prompt,
		MaxNewTokens:      *maxTokens,
		Temperature:       float32(*temperature),
		TopP:              float32(*topP),
		TopK:              40,
		RepetitionPenalty: 1,
		Stream:            false,
		StopWords:         "",
	}

	// Perform inference
	fmt.Printf("\nGenerating response...\n")
	genStartTime := time.Now()
	response, err := engine.Generate(params)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}
	genTime := time.Since(genStartTime)

	// Display results
	fmt.Printf("\nGeneration Results:\n")
	fmt.Printf("Input: %s\n", params.Prompt)
	fmt.Printf("Output: %s\n", response.Text)
	fmt.Printf("Tokens: %d input, %d output\n", response.InputTokens, response.OutputTokens)
	fmt.Printf("Time: %.2fs (%.2f tokens/sec)\n", genTime.Seconds(), float64(response.OutputTokens)/genTime.Seconds())

	fmt.Printf("\nTurboMind-Go example completed successfully!\n")
} 