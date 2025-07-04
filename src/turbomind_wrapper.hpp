#ifndef TURBOMIND_WRAPPER_HPP
#define TURBOMIND_WRAPPER_HPP

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct TurboMindModel TurboMindModel;
typedef struct TurboMindModelInstance TurboMindModelInstance;

// Data types (matching Python bindings)
typedef enum {
    TM_TYPE_INVALID = 0,
    TM_TYPE_BOOL,
    TM_TYPE_UINT8,
    TM_TYPE_UINT16, 
    TM_TYPE_UINT32,
    TM_TYPE_UINT64,
    TM_TYPE_INT8,
    TM_TYPE_INT16,
    TM_TYPE_INT32,
    TM_TYPE_INT64,
    TM_TYPE_FP16,
    TM_TYPE_FP32,
    TM_TYPE_FP64,
    TM_TYPE_BF16
} TurboMindDataType;

// Memory types
typedef enum {
    TM_MEMORY_CPU = 0,
    TM_MEMORY_CPU_PINNED,
    TM_MEMORY_GPU
} TurboMindMemoryType;

// Forward declarations for opaque types
typedef struct TurboMindTensor TurboMindTensor;

// Session parameters
typedef struct {
    uint64_t id;
    int step;
    bool start_flag;
    bool end_flag;
} TurboMindSession;

// Generation configuration
typedef struct {
    int max_new_tokens;
    int min_new_tokens;
    int* eos_ids;
    int eos_ids_count;
    int* stop_ids;
    int stop_ids_count;
    int* bad_ids;
    int bad_ids_count;
    float top_p;
    int top_k;
    float min_p;
    float temperature;
    float repetition_penalty;
    uint64_t random_seed;
    bool output_logprobs;
    bool output_last_hidden_state;
    bool output_logits;
} TurboMindGenerationConfig;

// Request state
typedef enum {
    TM_REQUEST_PENDING = 0,
    TM_REQUEST_RUNNING,
    TM_REQUEST_COMPLETED,
    TM_REQUEST_CANCELLED,
    TM_REQUEST_FAILED
} TurboMindRequestStatus;

// TensorMap - collection of tensors
typedef struct TurboMindTensorMap TurboMindTensorMap;

// Forward declarations for opaque types
typedef struct TurboMindForwardResult TurboMindForwardResult;

// API Functions

// Model creation and management
TurboMindModel* turbomind_create_model(const char* model_dir, const char* config, const char* weight_type);
void turbomind_destroy_model(TurboMindModel* model);

// Model setup (must be called before creating model instance)
void turbomind_create_shared_weights(TurboMindModel* model, int device_id, int rank);
void turbomind_process_weights(TurboMindModel* model, int device_id, int rank);
void turbomind_create_engine(TurboMindModel* model, int device_id, int rank);

// Model instance management
TurboMindModelInstance* turbomind_create_model_instance(TurboMindModel* model, int device_id);
void turbomind_destroy_model_instance(TurboMindModelInstance* instance);

// Tensor management
TurboMindTensor* turbomind_create_tensor(void* data, int64_t* shape, int ndim, TurboMindDataType dtype, TurboMindMemoryType memory_type, int device_id);
void turbomind_destroy_tensor(TurboMindTensor* tensor);

// TensorMap management
TurboMindTensorMap* turbomind_create_tensor_map();
void turbomind_destroy_tensor_map(TurboMindTensorMap* tensor_map);
int turbomind_tensor_map_set(TurboMindTensorMap* tensor_map, const char* key, TurboMindTensor* tensor);
TurboMindTensor* turbomind_tensor_map_get(TurboMindTensorMap* tensor_map, const char* key);

// Forward inference
TurboMindForwardResult* turbomind_forward(TurboMindModelInstance* instance, 
                                         TurboMindTensorMap* input_tensors,
                                         TurboMindSession* session,
                                         TurboMindGenerationConfig* gen_config,
                                         bool stream_output);
void turbomind_destroy_forward_result(TurboMindForwardResult* result);

// Session management
void turbomind_end_session(TurboMindModelInstance* instance, uint64_t session_id);
void turbomind_cancel_request(TurboMindModelInstance* instance);

// Model information
int turbomind_get_tensor_para_size(TurboMindModel* model);
int turbomind_get_pipeline_para_size(TurboMindModel* model);

// Utility functions
const char* turbomind_get_last_error();
void turbomind_set_device(int device_id);

// Helper functions for tensor operations
size_t turbomind_get_tensor_size(TurboMindTensor* tensor);
void turbomind_copy_tensor(TurboMindTensor* dst, TurboMindTensor* src);

#ifdef __cplusplus
}
#endif

#endif // TURBOMIND_WRAPPER_HPP 