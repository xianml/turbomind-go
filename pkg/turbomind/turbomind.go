package turbomind

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// #include <stdint.h>
// #include <stdbool.h>
// #include <stdlib.h>
import "C"

// Engine handle
type Engine struct {
	handle uintptr
}

// Configuration for TurboMind engine
type Config struct {
	ModelPath            string
	ModelFormat          string  // "hf", "awq", "gptq", etc.
	TP                   int     // tensor parallelism
	SessionLen           int     // max sequence length
	MaxBatchSize         int     // max batch size
	QuantPolicy          int     // 0=fp16, 4=int4, 8=int8
	CacheMaxEntryCount   int     // cache entry count
	EnablePrefixCaching  bool
	RopeScalingFactor    float32
	RopeScalingType      int
}

// Request parameters
type RequestParams struct {
	RequestID         int64
	Prompt            string
	MaxNewTokens      int
	Temperature       float32
	TopP              float32
	TopK              float32
	RepetitionPenalty int
	Stream            bool
	StopWords         string // JSON array string
}

// Response data
type ResponseData struct {
	RequestID    int64
	Text         string
	InputTokens  int
	OutputTokens int
	Finished     bool
	ErrorCode    int
	ErrorMessage string
}

// Version information
type VersionInfo struct {
	Version     string
	GitCommit   string
	BuildTime   string
	CudaVersion string
}

// Model information
type ModelInfo struct {
	ModelName             string
	ModelType             string // "llm", "vlm"
	VocabSize             int
	HiddenSize            int
	NumLayers             int
	MaxPositionEmbeddings int
}

// C struct mappings
type cConfig struct {
	modelPath            *C.char
	modelFormat          *C.char
	tp                   C.int
	sessionLen           C.int
	maxBatchSize         C.int
	quantPolicy          C.int
	cacheMaxEntryCount   C.int
	enablePrefixCaching  C.bool
	ropeScalingFactor    C.float
	ropeScalingType      C.int
}

type cRequestParams struct {
	requestId         C.int64_t
	prompt            *C.char
	maxNewTokens      C.int
	temperature       C.float
	topP              C.float
	topK              C.float
	repetitionPenalty C.int
	stream            C.bool
	stopWords         *C.char
}

type cResponseData struct {
	requestId    C.int64_t
	text         *C.char
	inputTokens  C.int
	outputTokens C.int
	finished     C.bool
	errorCode    C.int
	errorMessage *C.char
}

type cVersionInfo struct {
	version     *C.char
	gitCommit   *C.char
	buildTime   *C.char
	cudaVersion *C.char
}

type cModelInfo struct {
	modelName             *C.char
	modelType             *C.char
	vocabSize             C.int
	hiddenSize            C.int
	numLayers             C.int
	maxPositionEmbeddings C.int
}

// Global function pointers
var (
	libPath                   string
	turbomindCreateEngine     func(config uintptr) uintptr
	turbomindDestroyEngine    func(engine uintptr)
	turbomindIsEngineReady    func(engine uintptr) bool
	turbomindGenerate         func(engine uintptr, request uintptr, response uintptr) int
	turbomindGenerateAsync    func(engine uintptr, request uintptr) int
	turbomindGetResponse      func(engine uintptr, requestId int64, response uintptr) int
	turbomindGenerateBatch    func(engine uintptr, requests uintptr, batchSize int, responses uintptr) int
	turbomindGetVersion       func(info uintptr) int
	turbomindFreeResponse     func(response uintptr)
	turbomindGetLastError     func() uintptr
	turbomindGetModelInfo     func(engine uintptr, info uintptr) int
	turbomindFreeModelInfo    func(info uintptr)
)

// Initialize loads the TurboMind shared library
func Initialize(libraryPath string) error {
	if libraryPath == "" {
		libraryPath = "./libturbomind_go.so"
	}
	libPath = libraryPath

	// Open the shared library
	lib, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("failed to load library %s: %v", libPath, err)
	}

	// Register functions
	purego.RegisterLibFunc(&turbomindCreateEngine, lib, "turbomind_create_engine")
	purego.RegisterLibFunc(&turbomindDestroyEngine, lib, "turbomind_destroy_engine")
	purego.RegisterLibFunc(&turbomindIsEngineReady, lib, "turbomind_is_engine_ready")
	purego.RegisterLibFunc(&turbomindGenerate, lib, "turbomind_generate")
	purego.RegisterLibFunc(&turbomindGenerateAsync, lib, "turbomind_generate_async")
	purego.RegisterLibFunc(&turbomindGetResponse, lib, "turbomind_get_response")
	purego.RegisterLibFunc(&turbomindGenerateBatch, lib, "turbomind_generate_batch")
	purego.RegisterLibFunc(&turbomindGetVersion, lib, "turbomind_get_version")
	purego.RegisterLibFunc(&turbomindFreeResponse, lib, "turbomind_free_response")
	purego.RegisterLibFunc(&turbomindGetLastError, lib, "turbomind_get_last_error")
	purego.RegisterLibFunc(&turbomindGetModelInfo, lib, "turbomind_get_model_info")
	purego.RegisterLibFunc(&turbomindFreeModelInfo, lib, "turbomind_free_model_info")

	return nil
}

// NewEngine creates a new TurboMind engine
func NewEngine(config Config) (*Engine, error) {
	// Convert Go config to C config
	cConfig := cConfig{
		modelPath:            C.CString(config.ModelPath),
		modelFormat:          C.CString(config.ModelFormat),
		tp:                   C.int(config.TP),
		sessionLen:           C.int(config.SessionLen),
		maxBatchSize:         C.int(config.MaxBatchSize),
		quantPolicy:          C.int(config.QuantPolicy),
		cacheMaxEntryCount:   C.int(config.CacheMaxEntryCount),
		enablePrefixCaching:  C.bool(config.EnablePrefixCaching),
		ropeScalingFactor:    C.float(config.RopeScalingFactor),
		ropeScalingType:      C.int(config.RopeScalingType),
	}

	// Ensure cleanup
	defer func() {
		C.free(unsafe.Pointer(cConfig.modelPath))
		C.free(unsafe.Pointer(cConfig.modelFormat))
	}()

	handle := turbomindCreateEngine(uintptr(unsafe.Pointer(&cConfig)))
	if handle == 0 {
		return nil, fmt.Errorf("failed to create engine: %s", GetLastError())
	}

	engine := &Engine{handle: handle}
	runtime.SetFinalizer(engine, (*Engine).Close)
	return engine, nil
}

// IsReady checks if the engine is ready
func (e *Engine) IsReady() bool {
	if e.handle == 0 {
		return false
	}
	return turbomindIsEngineReady(e.handle)
}

// Generate performs inference
func (e *Engine) Generate(params RequestParams) (*ResponseData, error) {
	if e.handle == 0 {
		return nil, errors.New("engine is closed")
	}

	// Convert params to C struct
	cRequest := cRequestParams{
		requestId:         C.int64_t(params.RequestID),
		prompt:            C.CString(params.Prompt),
		maxNewTokens:      C.int(params.MaxNewTokens),
		temperature:       C.float(params.Temperature),
		topP:              C.float(params.TopP),
		topK:              C.float(params.TopK),
		repetitionPenalty: C.int(params.RepetitionPenalty),
		stream:            C.bool(params.Stream),
		stopWords:         C.CString(params.StopWords),
	}

	defer func() {
		C.free(unsafe.Pointer(cRequest.prompt))
		C.free(unsafe.Pointer(cRequest.stopWords))
	}()

	var cResponse cResponseData
	result := turbomindGenerate(e.handle, uintptr(unsafe.Pointer(&cRequest)), uintptr(unsafe.Pointer(&cResponse)))
	if result != 0 {
		return nil, fmt.Errorf("generation failed: %s", GetLastError())
	}

	// Convert response
	response := &ResponseData{
		RequestID:    int64(cResponse.requestId),
		Text:         C.GoString(cResponse.text),
		InputTokens:  int(cResponse.inputTokens),
		OutputTokens: int(cResponse.outputTokens),
		Finished:     bool(cResponse.finished),
		ErrorCode:    int(cResponse.errorCode),
	}

	if cResponse.errorMessage != nil {
		response.ErrorMessage = C.GoString(cResponse.errorMessage)
	}

	// Free C response
	turbomindFreeResponse(uintptr(unsafe.Pointer(&cResponse)))

	return response, nil
}

// GetModelInfo returns model information
func (e *Engine) GetModelInfo() (*ModelInfo, error) {
	if e.handle == 0 {
		return nil, errors.New("engine is closed")
	}

	var cInfo cModelInfo
	result := turbomindGetModelInfo(e.handle, uintptr(unsafe.Pointer(&cInfo)))
	if result != 0 {
		return nil, fmt.Errorf("failed to get model info: %s", GetLastError())
	}

	info := &ModelInfo{
		ModelName:             C.GoString(cInfo.modelName),
		ModelType:             C.GoString(cInfo.modelType),
		VocabSize:             int(cInfo.vocabSize),
		HiddenSize:            int(cInfo.hiddenSize),
		NumLayers:             int(cInfo.numLayers),
		MaxPositionEmbeddings: int(cInfo.maxPositionEmbeddings),
	}

	// Free C info
	turbomindFreeModelInfo(uintptr(unsafe.Pointer(&cInfo)))

	return info, nil
}

// Close destroys the engine
func (e *Engine) Close() {
	if e.handle != 0 {
		turbomindDestroyEngine(e.handle)
		e.handle = 0
		runtime.SetFinalizer(e, nil)
	}
}

// GetVersion returns version information
func GetVersion() VersionInfo {
	var cInfo cVersionInfo
	result := turbomindGetVersion(uintptr(unsafe.Pointer(&cInfo)))
	if result != 0 {
		return VersionInfo{
			Version:     "unknown",
			GitCommit:   "unknown",
			BuildTime:   "unknown",
			CudaVersion: "unknown",
		}
	}
	
	return VersionInfo{
		Version:     C.GoString(cInfo.version),
		GitCommit:   C.GoString(cInfo.gitCommit),
		BuildTime:   C.GoString(cInfo.buildTime),
		CudaVersion: C.GoString(cInfo.cudaVersion),
	}
}

// GetLastError returns the last error message
func GetLastError() string {
	ptr := turbomindGetLastError()
	if ptr == 0 {
		return "unknown error"
	}
	return C.GoString((*C.char)(unsafe.Pointer(ptr)))
}

// DefaultConfig returns a default configuration
func DefaultConfig(modelPath string) Config {
	return Config{
		ModelPath:            modelPath,
		ModelFormat:          "hf",
		TP:                   1,
		SessionLen:           2048,
		MaxBatchSize:         32,
		QuantPolicy:          0, // fp16
		CacheMaxEntryCount:   0,
		EnablePrefixCaching:  false,
		RopeScalingFactor:    1.0,
		RopeScalingType:      0,
	}
} 