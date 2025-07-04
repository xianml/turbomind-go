# TurboMind Go Bindings

High-performance Go bindings for LMDeploy's TurboMind inference engine with real tokenizer support.

## Overview

This project provides Go language bindings for TurboMind, a high-performance inference engine for large language models. TurboMind is part of the [LMDeploy](https://github.com/InternLM/lmdeploy) project by InternLM.

## ✨ Features

- **🚀 High Performance**: Direct integration with TurboMind's C++ engine
- **🔧 Easy to Use**: Simple Go API for model loading and inference
- **📦 Production Ready**: Memory management and error handling
- **⚡ GPU Accelerated**: CUDA support for faster inference
- **🔤 Real Tokenizer**: Integrated HuggingFace-compatible tokenizer
- **🎯 Multi-Model Support**: Supports various LLM architectures
- **🛠️ Complete Build System**: Makefile-based build automation

## 🚀 Quick Start

### Prerequisites

- Go 1.22+
- CUDA 12.0+ (for GPU acceleration)
- CMake 3.18+
- GCC/Clang with C++17 support

### Installation

1. Clone the repository:
```bash
git clone https://github.com/xianml/turbomind-go.git
cd turbomind-go
```

2. Build the project:
```bash
make build
```

3. Run tests:
```bash
make test
```

4. Test tokenizer functionality:
```bash
make test-tokenizer
```

### Basic Usage

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/xianml/turbomind-go/pkg/turbomind"
)

func main() {
    // Create engine configuration
    config := turbomind.DefaultEngineConfig("/path/to/model")
    
    // Create inference engine (automatically loads tokenizer)
    engine, err := turbomind.NewEngine(config)
    if err != nil {
        log.Fatal(err)
    }
    defer engine.Close()

    // Generate text
    request := &turbomind.InferenceRequest{
        SessionID:   1,
        Prompt:      "Hello, how are you?",
        MaxTokens:   100,
        Temperature: 0.7,
    }

    response, err := engine.Generate(request)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("Response:", response.Text)
}
```

### Tokenizer Usage

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/xianml/turbomind-go/pkg/turbomind"
)

func main() {
    // Load tokenizer
    tokenizer, err := turbomind.NewTokenizer("/path/to/model")
    if err != nil {
        log.Fatal(err)
    }
    defer tokenizer.Close()

    // Encode text
    tokens, err := tokenizer.Encode("Hello, world!", false)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Tokens: %v\n", tokens)

    // Decode tokens
    text, err := tokenizer.Decode(tokens, false)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Decoded: %s\n", text)

    // Get vocabulary information
    fmt.Printf("Vocabulary size: %d\n", tokenizer.GetVocabSize())
}
```

## 🏗️ Architecture

### Core Components

- **C++ Wrapper**: Bridge between Go and TurboMind C++ API (`src/turbomind_wrapper.hpp`)
- **Go Bindings**: High-level Go interface using CGO (`pkg/turbomind/`)
- **Real Tokenizer**: HuggingFace-compatible tokenizer using `sugarme/tokenizer`
- **Memory Management**: Automatic resource cleanup with finalizers
- **Build System**: Comprehensive Makefile for automation

### Project Structure

```
├── src/                         # C++ wrapper implementation
│   ├── turbomind_wrapper.hpp    # C API header
│   ├── turbomind_wrapper_proper.cpp    # Full implementation
│   └── turbomind_wrapper_minimal_test.cpp  # Test implementation
├── pkg/turbomind/              # Go package
│   ├── turbomind_proper.go     # Main Go bindings
│   ├── engine.go               # High-level engine interface
│   ├── tokenizer.go            # Tokenizer wrapper
│   └── tokenizer_test.go       # Tokenizer tests
├── example/                    # Examples and tests
│   ├── models/                 # Example model files (phi4-mini)
│   ├── example_proper.go       # Example usage
│   └── turbomind_proper_test.go # Integration tests
├── third_party/lmdeploy/       # LMDeploy submodule
├── build/                      # Build artifacts
├── Makefile                    # Build automation
└── CLAUDE.md                   # Project documentation
```

## 🔧 Build System

### Available Commands

```bash
# Build targets
make build-cpp       # Build C++ shared library
make build-go        # Build Go bindings
make build-example   # Build example executable
make build           # Build everything

# Test targets
make test-basic      # Run basic tests
make test-tokenizer  # Run tokenizer tests with real models
make test-example    # Run example tests
make test            # Run basic and example tests
make test-all        # Run all tests including tokenizer

# Utility targets
make clean           # Clean build artifacts
make fmt             # Format Go code
make tidy            # Run go mod tidy
make deps            # Download dependencies
make dev-setup       # Set up development environment
make help            # Show all available commands
```

## 📋 Configuration

### Engine Configuration

```go
config := &turbomind.EngineConfig{
    ModelDir:     "/path/to/model",
    Config:       "",           // Optional config file
    WeightType:   "half",       // Weight precision
    DeviceID:     0,           // GPU device ID
    TensorPara:   1,           // Tensor parallelism
    PipelinePara: 1,           // Pipeline parallelism
}
```

### Tokenizer Configuration

```go
config := &turbomind.TokenizerConfig{
    TokenizerPath: "/path/to/tokenizer.json",
    BosToken:      1,      // Beginning of sequence token
    EosToken:      32000,  // End of sequence token
    PadToken:      32000,  // Padding token
}

tokenizer, err := turbomind.NewTokenizerWithConfig(config)
```

### Request Parameters

```go
request := &turbomind.InferenceRequest{
    SessionID:     1,
    Prompt:        "Your prompt here",
    MaxTokens:     100,
    Temperature:   0.7,
    TopP:          0.9,
    TopK:          50,
    StopTokens:    []string{"</s>"},
    StreamOutput:  false,
}
```

## 🧪 Testing

### Run All Tests
```bash
make test-all
```

### Test Individual Components
```bash
make test-basic      # Basic functionality tests
make test-tokenizer  # Real tokenizer tests with phi4-mini model
make test-example    # Integration tests
```

### Test with Real Model
```bash
# Place your model files in example/models/ or set TEST_MODEL_PATH
TEST_MODEL_PATH=/path/to/your/model make test-tokenizer
```

## 🎯 Tokenizer Features

- **HuggingFace Compatibility**: Load `tokenizer.json` files directly
- **Complete API**: Encode, decode, special tokens, padding, offsets
- **High Performance**: Based on `sugarme/tokenizer` (Rust tokenizers port)
- **Rich Functionality**:
  - Text encoding/decoding
  - BOS/EOS token handling
  - Padding and truncation
  - Token-to-text conversion
  - Offset tracking for tokens

## 🚀 Performance

TurboMind Go bindings provide near-native performance by:

- **Zero-copy data transfer** between Go and C++
- **Minimal overhead** CGO interface
- **Efficient memory management** with object pooling
- **CUDA acceleration** for GPU inference
- **Optimized tokenizer** with Rust-based backend

## 📦 Supported Models

The project includes a complete phi4-mini model for testing:

- **Included**: phi4-mini with tokenizer, config, and model weights
- **Supported**: LLaMA/LLaMA-2, InternLM, Qwen, ChatGLM, Vicuna, and more
- **Format**: HuggingFace-compatible model files

## 🔧 Development

### Development Setup
```bash
make dev-setup       # Install development tools
make deps           # Download dependencies
```

### Code Quality
```bash
make fmt            # Format Go code
make lint           # Run linters (requires golangci-lint)
```

### Building and Testing
```bash
make clean && make build    # Clean rebuild
make test-all              # Comprehensive testing
```

## 📖 Examples

See the `example/` directory for:
- `example_proper.go`: Complete usage example
- `turbomind_proper_test.go`: Integration tests
- `models/`: phi4-mini model for testing
- `tk.go`: Tokenizer usage example

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [LMDeploy](https://github.com/InternLM/lmdeploy) team for the excellent TurboMind engine
- [sugarme/tokenizer](https://github.com/sugarme/tokenizer) for the Go tokenizer implementation
- [InternLM](https://github.com/InternLM) for the foundational work
- Go community for CGO best practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run `make test-all` to ensure everything works
6. Submit a pull request

## 📊 Project Status

- ✅ **Core Engine**: TurboMind integration complete
- ✅ **Tokenizer**: Real HuggingFace tokenizer support
- ✅ **Testing**: Comprehensive test suite with real models
- ✅ **Build System**: Complete Makefile automation
- ✅ **Documentation**: Complete API documentation
- ✅ **Examples**: Working examples with phi4-mini model

**Ready for production use!** 🎉