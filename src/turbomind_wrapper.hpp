#ifndef TURBOMIND_WRAPPER_HPP
#define TURBOMIND_WRAPPER_HPP

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct TurboMindEngine TurboMindEngine;
typedef struct TurboMindRequest TurboMindRequest;
typedef struct TurboMindResponse TurboMindResponse;

// Engine configuration
typedef struct {
    const char* model_path;
    const char* model_format;  // "hf", "awq", "gptq", etc.
    int tp;                    // tensor parallelism
    int session_len;           // max sequence length
    int max_batch_size;        // max batch size
    int quant_policy;          // 0=fp16, 4=int4, 8=int8
    int cache_max_entry_count; // cache entry count
    bool enable_prefix_caching;
    float rope_scaling_factor;
    int rope_scaling_type;
} TurboMindConfig;

// Request structure
typedef struct {
    int64_t request_id;
    const char* prompt;
    int max_new_tokens;
    float temperature;
    float top_p;
    float top_k;
    int repetition_penalty;
    bool stream;
    const char* stop_words;  // JSON array string
} RequestParams;

// Response structure  
typedef struct {
    int64_t request_id;
    const char* text;
    int input_tokens;
    int output_tokens;
    bool finished;
    int error_code;
    const char* error_message;
} ResponseData;

// Version info
typedef struct {
    const char* version;
    const char* git_commit;
    const char* build_time;
    const char* cuda_version;
} VersionInfo;

// API Functions
// Engine management
TurboMindEngine* turbomind_create_engine(const TurboMindConfig* config);
void turbomind_destroy_engine(TurboMindEngine* engine);
bool turbomind_is_engine_ready(TurboMindEngine* engine);

// Inference
int turbomind_generate(TurboMindEngine* engine, const RequestParams* request, ResponseData* response);
int turbomind_generate_async(TurboMindEngine* engine, const RequestParams* request);
int turbomind_get_response(TurboMindEngine* engine, int64_t request_id, ResponseData* response);

// Batch inference
int turbomind_generate_batch(TurboMindEngine* engine, const RequestParams* requests, int batch_size, ResponseData* responses);

// Utility functions
VersionInfo turbomind_get_version(void);
void turbomind_free_response(ResponseData* response);
const char* turbomind_get_last_error(void);

// Model info
typedef struct {
    const char* model_name;
    const char* model_type;  // "llm", "vlm"
    int vocab_size;
    int hidden_size;
    int num_layers;
    int max_position_embeddings;
} ModelInfo;

int turbomind_get_model_info(TurboMindEngine* engine, ModelInfo* info);
void turbomind_free_model_info(ModelInfo* info);

#ifdef __cplusplus
}
#endif

#endif // TURBOMIND_WRAPPER_HPP 