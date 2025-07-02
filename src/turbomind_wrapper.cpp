#include "turbomind_wrapper.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <cstring>
#include <atomic>

// LMDeploy includes (commented out for now to fix build)
// #include "turbomind/models/llama/LlamaV2.h"
// #include "turbomind/engine/gateway.h"
// #include "turbomind/utils/logger.h"

// Version information - will be set by build system
#ifndef LMDEPLOY_VERSION
#define LMDEPLOY_VERSION "v0.9.0"
#endif

#ifndef GIT_COMMIT
#define GIT_COMMIT "unknown"
#endif

#ifndef BUILD_TIME
#define BUILD_TIME __DATE__ " " __TIME__
#endif

static std::string g_last_error;
static std::mutex g_error_mutex;

// Internal engine wrapper
struct TurboMindEngine {
    std::unique_ptr<turbomind::Engine> engine;
    std::string model_path;
    std::string model_type;
    turbomind::EngineConfig config;
    std::atomic<bool> ready{false};
    std::mutex request_mutex;
    std::map<int64_t, std::shared_ptr<turbomind::Request>> active_requests;
    std::atomic<int64_t> next_request_id{1};
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

extern "C" {

TurboMindEngine* turbomind_create_engine(const TurboMindConfig* config) {
    try {
        if (!config || !config->model_path) {
            set_last_error("Invalid configuration: model_path is required");
            return nullptr;
        }

        auto engine = std::make_unique<TurboMindEngine>();
        engine->model_path = safe_string_copy(config->model_path);
        
        // Configure TurboMind engine
        turbomind::EngineConfig engine_config;
        engine_config.model_dir = engine->model_path;
        engine_config.model_format = safe_string_copy(config->model_format ? config->model_format : "hf");
        engine_config.tp = config->tp > 0 ? config->tp : 1;
        engine_config.session_len = config->session_len > 0 ? config->session_len : 2048;
        engine_config.max_batch_size = config->max_batch_size > 0 ? config->max_batch_size : 32;
        engine_config.quant_policy = config->quant_policy;
        engine_config.cache_max_entry_count = config->cache_max_entry_count > 0 ? config->cache_max_entry_count : 0.8f;
        engine_config.enable_prefix_caching = config->enable_prefix_caching;
        engine_config.rope_scaling_factor = config->rope_scaling_factor > 0 ? config->rope_scaling_factor : 1.0f;
        engine_config.rope_scaling_type = config->rope_scaling_type;

        engine->config = engine_config;
        
        // Initialize TurboMind engine
        engine->engine = std::make_unique<turbomind::Engine>(engine_config);
        
        if (!engine->engine) {
            set_last_error("Failed to create TurboMind engine");
            return nullptr;
        }

        engine->ready = true;
        return engine.release();
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception creating engine: ") + e.what());
        return nullptr;
    }
}

void turbomind_destroy_engine(TurboMindEngine* engine) {
    if (engine) {
        engine->ready = false;
        // Clear active requests
        {
            std::lock_guard<std::mutex> lock(engine->request_mutex);
            engine->active_requests.clear();
        }
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
        // Create TurboMind request
        auto tm_request = std::make_shared<turbomind::Request>();
        tm_request->id = request->request_id > 0 ? request->request_id : engine->next_request_id++;
        tm_request->prompt = safe_string_copy(request->prompt);
        tm_request->max_new_tokens = request->max_new_tokens > 0 ? request->max_new_tokens : 512;
        tm_request->temperature = request->temperature > 0 ? request->temperature : 0.7f;
        tm_request->top_p = request->top_p > 0 ? request->top_p : 0.8f;
        tm_request->top_k = request->top_k > 0 ? request->top_k : 40;
        tm_request->repetition_penalty = request->repetition_penalty > 0 ? request->repetition_penalty : 1.0f;
        tm_request->stream = request->stream;

        // Parse stop words if provided
        if (request->stop_words) {
            // Simple parsing - split by comma for now
            std::string stop_str = safe_string_copy(request->stop_words);
            // TODO: Implement proper JSON parsing for stop words
        }

        // Store request
        {
            std::lock_guard<std::mutex> lock(engine->request_mutex);
            engine->active_requests[tm_request->id] = tm_request;
        }

        // Generate response using TurboMind
        auto tm_response = engine->engine->Generate(tm_request);
        
        if (!tm_response) {
            set_last_error("Generation failed");
            return -1;
        }

        // Fill response
        response->request_id = tm_request->id;
        response->text = allocate_string(tm_response->text);
        response->input_tokens = tm_response->input_tokens;
        response->output_tokens = tm_response->output_tokens;
        response->finished = tm_response->finished;
        response->error_code = tm_response->error_code;
        response->error_message = tm_response->error_message.empty() ? nullptr : allocate_string(tm_response->error_message);

        // Remove from active requests
        {
            std::lock_guard<std::mutex> lock(engine->request_mutex);
            engine->active_requests.erase(tm_request->id);
        }

        return 0;
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception during generation: ") + e.what());
        return -1;
    }
}

int turbomind_generate_async(TurboMindEngine* engine, const RequestParams* request) {
    // TODO: Implement async generation
    set_last_error("Async generation not implemented yet");
    return -1;
}

int turbomind_get_response(TurboMindEngine* engine, int64_t request_id, ResponseData* response) {
    // TODO: Implement response retrieval for async requests
    set_last_error("Async response retrieval not implemented yet");
    return -1;
}

int turbomind_generate_batch(TurboMindEngine* engine, const RequestParams* requests, int batch_size, ResponseData* responses) {
    if (!engine || !requests || !responses || batch_size <= 0) {
        set_last_error("Invalid parameters for batch generation");
        return -1;
    }

    // TODO: Implement batch generation
    for (int i = 0; i < batch_size; i++) {
        int result = turbomind_generate(engine, &requests[i], &responses[i]);
        if (result != 0) {
            return result;
        }
    }
    
    return 0;
}

VersionInfo turbomind_get_version(void) {
    static VersionInfo version_info = {
        LMDEPLOY_VERSION,
        GIT_COMMIT,
        BUILD_TIME,
#ifdef CUDA_VERSION
        CUDA_VERSION
#else
        "unknown"
#endif
    };
    return version_info;
}

void turbomind_free_response(ResponseData* response) {
    if (response) {
        if (response->text) {
            delete[] response->text;
            response->text = nullptr;
        }
        if (response->error_message) {
            delete[] response->error_message;
            response->error_message = nullptr;
        }
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
        // Get model info from TurboMind engine
        auto model_config = engine->engine->GetModelConfig();
        
        info->model_name = allocate_string(model_config.model_name);
        info->model_type = allocate_string("llm"); // Default to LLM
        info->vocab_size = model_config.vocab_size;
        info->hidden_size = model_config.hidden_size;
        info->num_layers = model_config.num_layers;
        info->max_position_embeddings = model_config.max_position_embeddings;
        
        return 0;
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception getting model info: ") + e.what());
        return -1;
    }
}

void turbomind_free_model_info(ModelInfo* info) {
    if (info) {
        if (info->model_name) {
            delete[] info->model_name;
            info->model_name = nullptr;
        }
        if (info->model_type) {
            delete[] info->model_type;
            info->model_type = nullptr;
        }
    }
}

} // extern "C" 