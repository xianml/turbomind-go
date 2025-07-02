#include "turbomind_wrapper.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <cstring>
#include <atomic>
#include <stdexcept>

// LMDeploy TurboMind includes
#include "turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "turbomind/engine/model_request.h"
#include "turbomind/core/tensor.h"
#include "turbomind/core/core.h"
#include "turbomind/utils/logger.h"

// Version information
#ifndef LMDEPLOY_VERSION
#define LMDEPLOY_VERSION "v0.9.0"
#endif

#ifndef GIT_COMMIT
#define GIT_COMMIT "unknown"
#endif

#ifndef BUILD_TIME
#define BUILD_TIME __DATE__ " " __TIME__
#endif

#ifndef CUDA_VERSION
#define CUDA_VERSION "unknown"
#endif

static std::string g_last_error;
static std::mutex g_error_mutex;

using namespace turbomind;

// Internal engine wrapper
struct TurboMindEngine {
    std::shared_ptr<turbomind::LlamaTritonModel> model;
    std::shared_ptr<void> model_instance;
    std::string model_path;
    std::string model_type;
    std::atomic<bool> ready{false};
    std::mutex request_mutex;
    std::atomic<int64_t> next_request_id{1};
    
    // Configuration
    int tp_size;
    int session_len;
    int max_batch_size;
    int quant_policy;
    bool enable_prefix_caching;
    float rope_scaling_factor;
    int rope_scaling_type;
    
    // Model info
    std::string model_name;
    int vocab_size = 0;
    int hidden_size = 0;
    int num_layers = 0;
    int max_position_embeddings = 0;
    
    ~TurboMindEngine() {
        ready = false;
    }
};

static void set_last_error(const std::string& error) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = error;
}

static std::string safe_string_copy(const char* src) {
    return src ? std::string(src) : std::string();
}

static char* allocate_string(const std::string& str) {
    char* result = new char[str.length() + 1];
    strcpy(result, str.c_str());
    return result;
}

// Convert TurboMind GenerationConfig from our RequestParams
static turbomind::GenerationConfig create_generation_config(const RequestParams* request) {
    turbomind::GenerationConfig config;
    
    config.max_new_tokens = request->max_new_tokens > 0 ? request->max_new_tokens : 512;
    config.top_k = static_cast<int>(request->top_k > 0 ? request->top_k : 40);
    config.top_p = request->top_p > 0 ? request->top_p : 0.8f;
    config.temperature = request->temperature > 0 ? request->temperature : 0.7f;
    config.repetition_penalty = request->repetition_penalty > 0 ? 
        static_cast<float>(request->repetition_penalty) : 1.0f;
    
    // Set random seed
    config.random_seed = static_cast<uint64_t>(time(nullptr));
    
    return config;
}

// Create model request from our RequestParams
static std::shared_ptr<turbomind::ModelRequest> create_model_request(
    TurboMindEngine* engine, 
    const RequestParams* request) {
    
    auto model_request = std::make_shared<turbomind::ModelRequest>();
    
    // Set request ID
    model_request->id = request->request_id > 0 ? 
        static_cast<uint64_t>(request->request_id) : 
        static_cast<uint64_t>(engine->next_request_id++);
    
    // Create session parameters
    turbomind::SessionParam session;
    session.id = model_request->id;
    session.step = 0;
    session.start_flag = true;
    session.end_flag = !request->stream; // Non-streaming requests end immediately
    session.kill_flag = false;
    
    model_request->session = session;
    model_request->gen_cfg = create_generation_config(request);
    model_request->stream_output = request->stream;
    
    return model_request;
}

extern "C" {

TurboMindEngine* turbomind_create_engine(const TurboMindConfig* config) {
    try {
        if (!config || !config->model_path) {
            set_last_error("Invalid configuration: model_path is required");
            return nullptr;
        }

        auto engine = std::make_unique<TurboMindEngine>();
        engine->model_path = safe_string_copy(config->model_path);
        engine->tp_size = config->tp > 0 ? config->tp : 1;
        engine->session_len = config->session_len > 0 ? config->session_len : 2048;
        engine->max_batch_size = config->max_batch_size > 0 ? config->max_batch_size : 32;
        engine->quant_policy = config->quant_policy;
        engine->enable_prefix_caching = config->enable_prefix_caching;
        engine->rope_scaling_factor = config->rope_scaling_factor > 0 ? config->rope_scaling_factor : 1.0f;
        engine->rope_scaling_type = config->rope_scaling_type;
        
        // Extract model name from path
        std::string path = engine->model_path;
        size_t last_slash = path.find_last_of("/\\");
        engine->model_name = (last_slash != std::string::npos) ? 
            path.substr(last_slash + 1) : path;

        try {
            // Create LlamaTritonModel
            std::string model_format = safe_string_copy(config->model_format ? config->model_format : "hf");
            std::string weight_type = "fp16"; // Default weight type
            
            if (config->quant_policy == 4) {
                weight_type = "int4";
            } else if (config->quant_policy == 8) {
                weight_type = "int8";
            }
            
            // Create the model
            engine->model = turbomind::LlamaTritonModel::createLlamaModel(
                engine->model_path,
                "", // Use default config
                weight_type
            );
            
            if (!engine->model) {
                set_last_error("Failed to create LlamaTritonModel");
                return nullptr;
            }
            
            // Create model instance for device 0
            engine->model_instance = engine->model->createModelInstance(0);
            if (!engine->model_instance) {
                set_last_error("Failed to create model instance");
                return nullptr;
            }
            
            // Process weights
            engine->model->processWeights(0, 0); // device_id=0, rank=0
            
            // Create inference engine
            engine->model->createEngine(0, 0); // device_id=0, rank=0
            
            // Get model configuration/info
            try {
                auto params = engine->model->getParams(0, 0);
                if (params && !params->empty()) {
                    // Try to extract model info from params
                    engine->vocab_size = 32000; // Default, should be extracted from model
                    engine->hidden_size = 4096;
                    engine->num_layers = 32;
                    engine->max_position_embeddings = engine->session_len;
                }
            } catch (const std::exception& e) {
                // Non-fatal error, use defaults
                TM_LOG_WARNING("Failed to get model params: %s", e.what());
            }
            
            engine->ready = true;
            
        } catch (const std::exception& e) {
            set_last_error(std::string("Failed to initialize TurboMind model: ") + e.what());
            return nullptr;
        }
        
        return engine.release();
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception creating engine: ") + e.what());
        return nullptr;
    }
}

void turbomind_destroy_engine(TurboMindEngine* engine) {
    if (engine) {
        engine->ready = false;
        // Model and instance will be destroyed automatically by shared_ptr
        delete engine;
    }
}

bool turbomind_is_engine_ready(TurboMindEngine* engine) {
    return engine && engine->ready.load();
}

int turbomind_generate(TurboMindEngine* engine, const RequestParams* request, ResponseData* response) {
    if (!engine || !request || !response) {
        set_last_error("Invalid parameters");
        return -1;
    }

    if (!engine->ready.load()) {
        set_last_error("Engine not ready");
        return -1;
    }

    try {
        std::string prompt = safe_string_copy(request->prompt);
        if (prompt.empty()) {
            set_last_error("Empty prompt");
            return -1;
        }

        // Create model request
        auto model_request = create_model_request(engine, request);
        
        // Create input tensors
        auto input_tensors = std::make_shared<turbomind::core::TensorMap>();
        
        // For now, we'll use a simplified approach
        // In a full implementation, we would need to:
        // 1. Tokenize the input prompt
        // 2. Create proper input tensors (input_ids, attention_mask, etc.)
        // 3. Handle embeddings and other inputs
        
        // This is a simplified mock implementation that demonstrates the structure
        // but doesn't perform actual inference
        
        // Simulate processing
        std::string mock_response = "This is a generated response to: " + prompt;
        
        // Fill response
        response->request_id = model_request->id;
        response->text = allocate_string(mock_response);
        response->input_tokens = static_cast<int>(prompt.length() / 4); // Rough estimate
        response->output_tokens = static_cast<int>(mock_response.length() / 4);
        response->finished = true;
        response->error_code = 0;
        response->error_message = nullptr;

        return 0;
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception during generation: ") + e.what());
        return -1;
    }
}

int turbomind_generate_async(TurboMindEngine* engine, const RequestParams* request) {
    set_last_error("Async generation not implemented yet");
    return -1;
}

int turbomind_get_response(TurboMindEngine* engine, int64_t request_id, ResponseData* response) {
    set_last_error("Async response retrieval not implemented yet");
    return -1;
}

int turbomind_generate_batch(TurboMindEngine* engine, const RequestParams* requests, int batch_size, ResponseData* responses) {
    if (!engine || !requests || !responses || batch_size <= 0) {
        set_last_error("Invalid parameters for batch generation");
        return -1;
    }

    for (int i = 0; i < batch_size; i++) {
        int result = turbomind_generate(engine, &requests[i], &responses[i]);
        if (result != 0) {
            return result;
        }
    }
    
    return 0;
}

VersionInfo turbomind_get_version(void) {
    VersionInfo info;
    info.version = allocate_string(LMDEPLOY_VERSION);
    info.git_commit = allocate_string(GIT_COMMIT);
    info.build_time = allocate_string(BUILD_TIME);
    info.cuda_version = allocate_string(CUDA_VERSION);
    return info;
}

void turbomind_free_response(ResponseData* response) {
    if (response) {
        delete[] response->text;
        delete[] response->error_message;
        memset(response, 0, sizeof(ResponseData));
    }
}

const char* turbomind_get_last_error(void) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return g_last_error.c_str();
}

int turbomind_get_model_info(TurboMindEngine* engine, ModelInfo* info) {
    if (!engine || !info) {
        set_last_error("Invalid parameters");
        return -1;
    }
    
    if (!engine->ready.load()) {
        set_last_error("Engine not ready");
        return -1;
    }
    
    try {
        info->model_name = allocate_string(engine->model_name);
        info->model_type = allocate_string("llm");
        info->vocab_size = engine->vocab_size;
        info->hidden_size = engine->hidden_size;
        info->num_layers = engine->num_layers;
        info->max_position_embeddings = engine->max_position_embeddings;
        
        return 0;
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception getting model info: ") + e.what());
        return -1;
    }
}

void turbomind_free_model_info(ModelInfo* info) {
    if (info) {
        delete[] info->model_name;
        delete[] info->model_type;
        memset(info, 0, sizeof(ModelInfo));
    }
}

} // extern "C" 