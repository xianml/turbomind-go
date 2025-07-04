# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is turbomind-go, a Go binding for the TurboMind inference engine. It provides a CGO-based interface to run large language models through TurboMind's C++ implementation.

## Architecture

### Core Components

- **Go Package**: `pkg/turbomind/turbomind.go` - Main Go interface using purego for dynamic library loading
- **C++ Wrapper**: `src/turbomind_wrapper_hybrid.cpp` - C++ wrapper that interfaces with LMDeploy/TurboMind
- **Header Interface**: `src/turbomind_wrapper.hpp` - C API definition for Go-C++ interop
- **Examples**: `example/example.go` - Usage examples and CLI tool

### Key Design Patterns

- Uses `github.com/ebitengine/purego` for dynamic library loading instead of traditional CGO
- Implements proper memory management with finalizers for Engine cleanup
- C structs mirror Go structs for seamless data passing
- Hybrid implementation (`turbomind_wrapper_hybrid.cpp`) for selective LMDeploy linking

## Build System

### CMake Build
```bash
# Build the C++ shared library
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Go Build
```bash
# Build and run Go example
go mod tidy
go build -o example ./example
./example --model /path/to/model
```

### Testing
```bash
# Run Go tests
go test ./example -v

# Run tests with model (requires TEST_MODEL_PATH environment variable)
TEST_MODEL_PATH=/path/to/model go test ./example -v
```

## Development Setup

### Prerequisites
- Go 1.22+
- CUDA 12.1+ (for GPU support)
- CMake 3.18+
- LMDeploy source code in `third_party/lmdeploy/`

### Environment Variables
- `TEST_MODEL_PATH`: Path to test model for running tests
- `TURBOMIND_LIB_PATH`: Path to TurboMind shared library

### Docker Development
```bash
# Build development image
docker build --target development -t turbomind-go-dev .

# Run development container
docker run -it --gpus all turbomind-go-dev
```

## Important Implementation Details

### Memory Management
- Engine objects use finalizers for automatic cleanup
- C strings are properly freed using `C.free()`
- Response objects have dedicated cleanup functions (`turbomindFreeResponse`)

### Error Handling
- All C API calls return error codes
- `GetLastError()` provides detailed error messages
- Engine state is tracked (ready/closed) to prevent invalid operations

### Library Dependencies
- Primary dependency: `github.com/ebitengine/purego` for dynamic linking
- Test framework: `github.com/stretchr/testify`
- No direct LMDeploy Go dependencies (uses C++ library)

## Common Operations

### Initialize and Create Engine
```go
// Initialize library
err := turbomind.Initialize("./libturbomind_go.so")

// Create engine with default config
config := turbomind.DefaultConfig("/path/to/model")
engine, err := turbomind.NewEngine(config)
defer engine.Close()
```

### Generate Text
```go
params := turbomind.RequestParams{
    RequestID:    1,
    Prompt:       "Hello, world!",
    MaxNewTokens: 100,
    Temperature:  0.7,
}

response, err := engine.Generate(params)
```

### Model Information
```go
modelInfo, err := engine.GetModelInfo()
version := turbomind.GetVersion()
```