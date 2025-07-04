package turbomind

/*
#cgo CFLAGS: -I../../src
#cgo CXXFLAGS: -I../../src -I../../third_party/lmdeploy/src -std=c++17
#cgo LDFLAGS: -L../../build -L/usr/local/cuda/lib64 -lturbomind_go -lcudart -lstdc++

#include "turbomind_wrapper.hpp"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// DataType represents TurboMind data types
type DataType int

const (
	TypeInvalid DataType = iota
	TypeBool
	TypeUint8
	TypeUint16
	TypeUint32
	TypeUint64
	TypeInt8
	TypeInt16
	TypeInt32
	TypeInt64
	TypeFP16
	TypeFP32
	TypeFP64
	TypeBF16
)

// MemoryType represents memory location types
type MemoryType int

const (
	MemoryCPU MemoryType = iota
	MemoryCPUPinned
	MemoryGPU
)

// RequestStatus represents request execution status
type RequestStatus int

const (
	RequestPending RequestStatus = iota
	RequestRunning
	RequestCompleted
	RequestCancelled
	RequestFailed
)

// Model represents a TurboMind model
type Model struct {
	handle *C.TurboMindModel
}

// ModelInstance represents a model instance for inference
type ModelInstance struct {
	handle *C.TurboMindModelInstance
}

// Tensor represents a TurboMind tensor
type Tensor struct {
	handle *C.TurboMindTensor
	shape  []int64
	dtype  DataType
	memory MemoryType
}

// TensorMap represents a collection of tensors
type TensorMap struct {
	handle *C.TurboMindTensorMap
}

// Session represents inference session parameters
type Session struct {
	ID        uint64
	Step      int
	StartFlag bool
	EndFlag   bool
}

// GenerationConfig represents generation configuration
type GenerationConfig struct {
	MaxNewTokens             int
	MinNewTokens             int
	EosIds                   []int
	StopIds                  []int
	BadIds                   []int
	TopP                     float32
	TopK                     int
	MinP                     float32
	Temperature              float32
	RepetitionPenalty        float32
	RandomSeed               uint64
	OutputLogprobs           bool
	OutputLastHiddenState    bool
	OutputLogits             bool
}

// ForwardResult represents the result of forward inference
type ForwardResult struct {
	handle  *C.TurboMindForwardResult
	Tensors *TensorMap
	Status  RequestStatus
	SeqLen  int
}

// NewModel creates a new TurboMind model
func NewModel(modelDir, config, weightType string) (*Model, error) {
	cModelDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cModelDir))
	
	var cConfig *C.char
	if config != "" {
		cConfig = C.CString(config)
		defer C.free(unsafe.Pointer(cConfig))
	}
	
	var cWeightType *C.char
	if weightType != "" {
		cWeightType = C.CString(weightType)
		defer C.free(unsafe.Pointer(cWeightType))
	}
	
	handle := C.turbomind_create_model(cModelDir, cConfig, cWeightType)
	if handle == nil {
		return nil, fmt.Errorf("failed to create model: %s", GetLastError())
	}
	
	model := &Model{handle: handle}
	runtime.SetFinalizer(model, (*Model).Close)
	return model, nil
}

// Close destroys the model
func (m *Model) Close() {
	if m.handle != nil {
		C.turbomind_destroy_model(m.handle)
		m.handle = nil
		runtime.SetFinalizer(m, nil)
	}
}

// CreateInstance creates a model instance for inference
func (m *Model) CreateInstance(deviceID int) (*ModelInstance, error) {
	if m.handle == nil {
		return nil, errors.New("model is closed")
	}
	
	// Step 1: Create shared weights (uses rank index - must be 0 for single GPU)
	C.turbomind_create_shared_weights(m.handle, C.int(0), C.int(0)) // device_id=0, rank=0 for single GPU
	
	// Step 2: Process weights (uses device_id index - must be 0 for single GPU)
	C.turbomind_process_weights(m.handle, C.int(0), C.int(0)) // device_id=0, rank=0 for single GPU
	
	// Step 3: Create engine (uses device_id index - must be 0 for single GPU) 
	C.turbomind_create_engine(m.handle, C.int(0), C.int(0)) // device_id=0, rank=0 for single GPU
	
	// Step 3: Create model instance
	handle := C.turbomind_create_model_instance(m.handle, C.int(deviceID))
	if handle == nil {
		return nil, fmt.Errorf("failed to create model instance: %s", GetLastError())
	}
	
	instance := &ModelInstance{handle: handle}
	runtime.SetFinalizer(instance, (*ModelInstance).Close)
	return instance, nil
}

// GetTensorParaSize returns tensor parallelism size
func (m *Model) GetTensorParaSize() int {
	if m.handle == nil {
		return -1
	}
	return int(C.turbomind_get_tensor_para_size(m.handle))
}

// GetPipelineParaSize returns pipeline parallelism size
func (m *Model) GetPipelineParaSize() int {
	if m.handle == nil {
		return -1
	}
	return int(C.turbomind_get_pipeline_para_size(m.handle))
}

// Close destroys the model instance
func (mi *ModelInstance) Close() {
	if mi.handle != nil {
		C.turbomind_destroy_model_instance(mi.handle)
		mi.handle = nil
		runtime.SetFinalizer(mi, nil)
	}
}

// Forward performs forward inference
func (mi *ModelInstance) Forward(inputTensors *TensorMap, session *Session, genConfig *GenerationConfig, streamOutput bool) (*ForwardResult, error) {
	if mi.handle == nil {
		return nil, errors.New("model instance is closed")
	}
	
	// Convert session
	cSession := C.TurboMindSession{
		id:         C.uint64_t(session.ID),
		step:       C.int(session.Step),
		start_flag: C.bool(session.StartFlag),
		end_flag:   C.bool(session.EndFlag),
	}
	
	// Convert generation config
	cGenConfig := C.TurboMindGenerationConfig{
		max_new_tokens:              C.int(genConfig.MaxNewTokens),
		min_new_tokens:              C.int(genConfig.MinNewTokens),
		top_p:                       C.float(genConfig.TopP),
		top_k:                       C.int(genConfig.TopK),
		min_p:                       C.float(genConfig.MinP),
		temperature:                 C.float(genConfig.Temperature),
		repetition_penalty:          C.float(genConfig.RepetitionPenalty),
		random_seed:                 C.uint64_t(genConfig.RandomSeed),
		output_logprobs:             C.bool(genConfig.OutputLogprobs),
		output_last_hidden_state:    C.bool(genConfig.OutputLastHiddenState),
		output_logits:               C.bool(genConfig.OutputLogits),
	}
	
	// Convert arrays
	var cEosIds *C.int
	var cStopIds *C.int
	var cBadIds *C.int
	
	if len(genConfig.EosIds) > 0 {
		cEosIds = (*C.int)(C.malloc(C.size_t(len(genConfig.EosIds)) * C.sizeof_int))
		defer C.free(unsafe.Pointer(cEosIds))
		cEosSlice := (*[1 << 30]C.int)(unsafe.Pointer(cEosIds))
		for i, id := range genConfig.EosIds {
			cEosSlice[i] = C.int(id)
		}
		cGenConfig.eos_ids = cEosIds
		cGenConfig.eos_ids_count = C.int(len(genConfig.EosIds))
	}
	
	if len(genConfig.StopIds) > 0 {
		cStopIds = (*C.int)(C.malloc(C.size_t(len(genConfig.StopIds)) * C.sizeof_int))
		defer C.free(unsafe.Pointer(cStopIds))
		cStopSlice := (*[1 << 30]C.int)(unsafe.Pointer(cStopIds))
		for i, id := range genConfig.StopIds {
			cStopSlice[i] = C.int(id)
		}
		cGenConfig.stop_ids = cStopIds
		cGenConfig.stop_ids_count = C.int(len(genConfig.StopIds))
	}
	
	if len(genConfig.BadIds) > 0 {
		cBadIds = (*C.int)(C.malloc(C.size_t(len(genConfig.BadIds)) * C.sizeof_int))
		defer C.free(unsafe.Pointer(cBadIds))
		cBadSlice := (*[1 << 30]C.int)(unsafe.Pointer(cBadIds))
		for i, id := range genConfig.BadIds {
			cBadSlice[i] = C.int(id)
		}
		cGenConfig.bad_ids = cBadIds
		cGenConfig.bad_ids_count = C.int(len(genConfig.BadIds))
	}
	
	// Call forward
	handle := C.turbomind_forward(mi.handle, inputTensors.handle, &cSession, &cGenConfig, C.bool(streamOutput))
	if handle == nil {
		return nil, fmt.Errorf("forward inference failed: %s", GetLastError())
	}
	
	result := &ForwardResult{handle: handle}
	runtime.SetFinalizer(result, (*ForwardResult).Close)
	return result, nil
}

// EndSession ends an inference session
func (mi *ModelInstance) EndSession(sessionID uint64) {
	if mi.handle != nil {
		C.turbomind_end_session(mi.handle, C.uint64_t(sessionID))
	}
}

// Cancel cancels current request
func (mi *ModelInstance) Cancel() {
	if mi.handle != nil {
		C.turbomind_cancel_request(mi.handle)
	}
}

// NewTensor creates a new tensor
func NewTensor(data unsafe.Pointer, shape []int64, dtype DataType, memory MemoryType, deviceID int) (*Tensor, error) {
	if data == nil || len(shape) == 0 {
		return nil, errors.New("invalid tensor parameters")
	}
	
	// Convert shape to C array
	cShape := (*C.int64_t)(C.malloc(C.size_t(len(shape)) * C.sizeof_int64_t))
	defer C.free(unsafe.Pointer(cShape))
	cShapeSlice := (*[1 << 30]C.int64_t)(unsafe.Pointer(cShape))
	for i, s := range shape {
		cShapeSlice[i] = C.int64_t(s)
	}
	
	handle := C.turbomind_create_tensor(data, cShape, C.int(len(shape)), 
		C.TurboMindDataType(dtype), C.TurboMindMemoryType(memory), C.int(deviceID))
	if handle == nil {
		return nil, fmt.Errorf("failed to create tensor: %s", GetLastError())
	}
	
	tensor := &Tensor{
		handle: handle,
		shape:  make([]int64, len(shape)),
		dtype:  dtype,
		memory: memory,
	}
	copy(tensor.shape, shape)
	
	runtime.SetFinalizer(tensor, (*Tensor).Close)
	return tensor, nil
}

// Close destroys the tensor
func (t *Tensor) Close() {
	if t.handle != nil {
		C.turbomind_destroy_tensor(t.handle)
		t.handle = nil
		runtime.SetFinalizer(t, nil)
	}
}

// Shape returns tensor shape
func (t *Tensor) Shape() []int64 {
	return t.shape
}

// DataType returns tensor data type
func (t *Tensor) DataType() DataType {
	return t.dtype
}

// MemoryType returns tensor memory type
func (t *Tensor) MemoryType() MemoryType {
	return t.memory
}

// Size returns tensor size in bytes
func (t *Tensor) Size() int {
	if t.handle == nil {
		return 0
	}
	return int(C.turbomind_get_tensor_size(t.handle))
}

// CopyFrom copies data from another tensor
func (t *Tensor) CopyFrom(src *Tensor) error {
	if t.handle == nil || src.handle == nil {
		return errors.New("tensor is closed")
	}
	
	C.turbomind_copy_tensor(t.handle, src.handle)
	if err := GetLastError(); err != "" {
		return fmt.Errorf("copy failed: %s", err)
	}
	
	return nil
}

// NewTensorMap creates a new tensor map
func NewTensorMap() *TensorMap {
	handle := C.turbomind_create_tensor_map()
	if handle == nil {
		return nil
	}
	
	tensorMap := &TensorMap{handle: handle}
	runtime.SetFinalizer(tensorMap, (*TensorMap).Close)
	return tensorMap
}

// Close destroys the tensor map
func (tm *TensorMap) Close() {
	if tm.handle != nil {
		C.turbomind_destroy_tensor_map(tm.handle)
		tm.handle = nil
		runtime.SetFinalizer(tm, nil)
	}
}

// Set sets a tensor in the map
func (tm *TensorMap) Set(key string, tensor *Tensor) error {
	if tm.handle == nil {
		return errors.New("tensor map is closed")
	}
	
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	
	result := C.turbomind_tensor_map_set(tm.handle, cKey, tensor.handle)
	if result != 0 {
		return fmt.Errorf("failed to set tensor: %s", GetLastError())
	}
	
	return nil
}

// Get gets a tensor from the map
func (tm *TensorMap) Get(key string) (*Tensor, error) {
	if tm.handle == nil {
		return nil, errors.New("tensor map is closed")
	}
	
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	
	handle := C.turbomind_tensor_map_get(tm.handle, cKey)
	if handle == nil {
		return nil, fmt.Errorf("tensor not found: %s", key)
	}
	
	// Note: This is a simplified implementation - proper lifetime management needed
	return &Tensor{handle: handle}, nil
}

// Close destroys the forward result
func (fr *ForwardResult) Close() {
	if fr.handle != nil {
		C.turbomind_destroy_forward_result(fr.handle)
		fr.handle = nil
		runtime.SetFinalizer(fr, nil)
	}
}

// Utility functions

// SetDevice sets the current CUDA device
func SetDevice(deviceID int) {
	C.turbomind_set_device(C.int(deviceID))
}

// GetLastError returns the last error message
func GetLastError() string {
	return C.GoString(C.turbomind_get_last_error())
}

// DefaultGenerationConfig returns a default generation configuration
func DefaultGenerationConfig() *GenerationConfig {
	return &GenerationConfig{
		MaxNewTokens:      100,
		MinNewTokens:      1,
		TopP:              0.8,
		TopK:              40,
		MinP:              0.0,
		Temperature:       1.0,
		RepetitionPenalty: 1.0,
		RandomSeed:        0,
		OutputLogprobs:    false,
		OutputLastHiddenState: false,
		OutputLogits:      false,
	}
}