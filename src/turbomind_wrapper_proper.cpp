#include "turbomind_wrapper.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <iostream>
#include <cstring>

// TurboMind headers (matching Python bindings)
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/model_request.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace ft = turbomind;

// Global error state
static std::string g_last_error;
static std::mutex g_error_mutex;

static void set_last_error(const std::string& error) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = error;
    std::cerr << "TurboMind Error: " << error << std::endl;
}

// Convert C data type to TurboMind data type
ft::DataType convert_data_type(TurboMindDataType type) {
    switch (type) {
        case TM_TYPE_BOOL: return ft::kBool;
        case TM_TYPE_UINT8: return ft::kUint8;
        case TM_TYPE_UINT16: return ft::kUint16;
        case TM_TYPE_UINT32: return ft::kUint32;
        case TM_TYPE_UINT64: return ft::kUint64;
        case TM_TYPE_INT8: return ft::kInt8;
        case TM_TYPE_INT16: return ft::kInt16;
        case TM_TYPE_INT32: return ft::kInt32;
        case TM_TYPE_INT64: return ft::kInt64;
        case TM_TYPE_FP16: return ft::kFloat16;
        case TM_TYPE_FP32: return ft::kFloat32;
        case TM_TYPE_FP64: return ft::kFloat64;
        case TM_TYPE_BF16: return ft::kBfloat16;
        default: return ft::kNull;
    }
}

// Convert C memory type to TurboMind device type
ft::DeviceType convert_memory_type(TurboMindMemoryType type) {
    switch (type) {
        case TM_MEMORY_CPU: return ft::kCPU;
        case TM_MEMORY_CPU_PINNED: return ft::kCPUpinned;
        case TM_MEMORY_GPU: return ft::kDEVICE;
        default: return ft::kCPU;
    }
}

// TurboMind Model wrapper
struct TurboMindModel {
    std::shared_ptr<ft::LlamaTritonModel> model;
    std::string model_dir;
    std::string config;
    std::string weight_type;
    
    TurboMindModel(const std::string& dir, const std::string& cfg, const std::string& wt) 
        : model_dir(dir), config(cfg), weight_type(wt) {
        
        // Convert weight type to data type
        ft::DataType data_type;
        if (weight_type == "half" || weight_type == "fp16" || weight_type == "float16" || weight_type == "int4") {
            data_type = ft::kFloat16;
        } else if (weight_type == "bf16" || weight_type == "bfloat16") {
#ifdef ENABLE_BF16
            data_type = ft::kBfloat16;
#else
            throw std::runtime_error("turbomind not built with bf16 support");
#endif
        } else if (weight_type == "fp8") {
            data_type = ft::kBfloat16;
        } else {
#ifdef ENABLE_FP32
            data_type = ft::kFloat32;
#else
            throw std::runtime_error("turbomind not built with fp32 support");
#endif
        }
        
        // Create GIL factory for Python compatibility (empty for C++)
        auto gil_factory = []() -> std::shared_ptr<void> {
            return nullptr;
        };
        
        // Create the model
        model = std::make_shared<ft::LlamaTritonModel>(data_type, model_dir, config, gil_factory);
    }
};

// TurboMind Model Instance wrapper
struct TurboMindModelInstance {
    std::unique_ptr<ft::ModelRequest> request;
    int device_id;
    
    TurboMindModelInstance(TurboMindModel* model, int dev_id) : device_id(dev_id) {
        request = model->model->createModelInstance(device_id);
        if (!request) {
            throw std::runtime_error("Failed to create model instance");
        }
    }
};

// TurboMind Tensor wrapper
struct TurboMindTensor {
    std::shared_ptr<ft::core::Tensor> tensor;
    std::vector<int64_t> shape_storage; // Store shape data
    
    TurboMindTensor(void* data, int64_t* shape, int ndim, TurboMindDataType dtype, 
                    TurboMindMemoryType memory_type, int device_id) {
        // Copy shape data
        shape_storage.assign(shape, shape + ndim);
        
        // Create tensor
        auto ft_dtype = convert_data_type(dtype);
        auto ft_memory = convert_memory_type(memory_type);
        
        std::vector<ft::core::ssize_t> ft_shape(shape, shape + ndim);
        
        // Create device
        ft::core::Device device{ft_memory, device_id};
        
        // Create tensor with shared pointer to data
        std::shared_ptr<void> data_ptr(data, [](void*) {}); // Non-owning
        tensor = std::make_shared<ft::core::Tensor>(data_ptr, std::move(ft_shape), ft_dtype, device);
    }
};

// TurboMind TensorMap wrapper
struct TurboMindTensorMap {
    std::shared_ptr<ft::core::TensorMap> tensor_map;
    
    TurboMindTensorMap() {
        tensor_map = std::make_shared<ft::core::TensorMap>();
    }
};

// TurboMind Forward Result wrapper
struct TurboMindForwardResult {
    std::shared_ptr<ft::core::TensorMap> tensors;
    TurboMindRequestStatus status;
    int seq_len;
    
    TurboMindForwardResult(std::shared_ptr<ft::core::TensorMap> t, TurboMindRequestStatus s, int len)
        : tensors(t), status(s), seq_len(len) {}
};

// C API Implementation

extern "C" {

// Model creation and management
TurboMindModel* turbomind_create_model(const char* model_dir, const char* config, const char* weight_type) {
    if (!model_dir) {
        set_last_error("model_dir cannot be null");
        return nullptr;
    }
    
    try {
        std::string cfg = config ? config : "";
        std::string wt = weight_type ? weight_type : "half";
        
        return new TurboMindModel(model_dir, cfg, wt);
    } catch (const std::exception& e) {
        set_last_error("Failed to create model: " + std::string(e.what()));
        return nullptr;
    }
}

void turbomind_destroy_model(TurboMindModel* model) {
    delete model;
}

// Model setup functions
void turbomind_create_shared_weights(TurboMindModel* model, int device_id, int rank) {
    if (!model || !model->model) {
        set_last_error("model cannot be null");
        return;
    }
    
    try {
        model->model->createSharedWeights(device_id, rank);
    } catch (const std::exception& e) {
        set_last_error("Failed to create shared weights: " + std::string(e.what()));
    }
}

void turbomind_process_weights(TurboMindModel* model, int device_id, int rank) {
    if (!model || !model->model) {
        set_last_error("model cannot be null");
        return;
    }
    
    try {
        model->model->processWeights(device_id, rank);
    } catch (const std::exception& e) {
        set_last_error("Failed to process weights: " + std::string(e.what()));
    }
}

void turbomind_create_engine(TurboMindModel* model, int device_id, int rank) {
    if (!model || !model->model) {
        set_last_error("model cannot be null");
        return;
    }
    
    try {
        model->model->createEngine(device_id, rank);
    } catch (const std::exception& e) {
        set_last_error("Failed to create engine: " + std::string(e.what()));
    }
}

// Model instance management
TurboMindModelInstance* turbomind_create_model_instance(TurboMindModel* model, int device_id) {
    if (!model) {
        set_last_error("model cannot be null");
        return nullptr;
    }
    
    try {
        return new TurboMindModelInstance(model, device_id);
    } catch (const std::exception& e) {
        set_last_error("Failed to create model instance: " + std::string(e.what()));
        return nullptr;
    }
}

void turbomind_destroy_model_instance(TurboMindModelInstance* instance) {
    delete instance;
}

// Tensor management
TurboMindTensor* turbomind_create_tensor(void* data, int64_t* shape, int ndim, 
                                        TurboMindDataType dtype, TurboMindMemoryType memory_type, int device_id) {
    if (!data || !shape || ndim <= 0) {
        set_last_error("Invalid tensor parameters");
        return nullptr;
    }
    
    try {
        return new TurboMindTensor(data, shape, ndim, dtype, memory_type, device_id);
    } catch (const std::exception& e) {
        set_last_error("Failed to create tensor: " + std::string(e.what()));
        return nullptr;
    }
}

void turbomind_destroy_tensor(TurboMindTensor* tensor) {
    delete tensor;
}

// TensorMap management
TurboMindTensorMap* turbomind_create_tensor_map() {
    try {
        return new TurboMindTensorMap();
    } catch (const std::exception& e) {
        set_last_error("Failed to create tensor map: " + std::string(e.what()));
        return nullptr;
    }
}

void turbomind_destroy_tensor_map(TurboMindTensorMap* tensor_map) {
    delete tensor_map;
}

int turbomind_tensor_map_set(TurboMindTensorMap* tensor_map, const char* key, TurboMindTensor* tensor) {
    if (!tensor_map || !key || !tensor) {
        set_last_error("Invalid parameters for tensor map set");
        return -1;
    }
    
    try {
        (*tensor_map->tensor_map)[key] = *tensor->tensor;
        return 0;
    } catch (const std::exception& e) {
        set_last_error("Failed to set tensor in map: " + std::string(e.what()));
        return -1;
    }
}

TurboMindTensor* turbomind_tensor_map_get(TurboMindTensorMap* tensor_map, const char* key) {
    if (!tensor_map || !key) {
        set_last_error("Invalid parameters for tensor map get");
        return nullptr;
    }
    
    try {
        auto it = tensor_map->tensor_map->find(key);
        if (it == tensor_map->tensor_map->end()) {
            set_last_error("Tensor not found in map: " + std::string(key));
            return nullptr;
        }
        
        // Create wrapper for existing tensor
        // Note: This is a simplified implementation - proper implementation would need
        // to handle tensor lifetime properly
        return nullptr; // TODO: Implement proper tensor extraction
    } catch (const std::exception& e) {
        set_last_error("Failed to get tensor from map: " + std::string(e.what()));
        return nullptr;
    }
}

// Forward inference
TurboMindForwardResult* turbomind_forward(TurboMindModelInstance* instance, 
                                         TurboMindTensorMap* input_tensors,
                                         TurboMindSession* session,
                                         TurboMindGenerationConfig* gen_config,
                                         bool stream_output) {
    if (!instance || !input_tensors || !session || !gen_config) {
        set_last_error("Invalid parameters for forward");
        return nullptr;
    }
    
    try {
        // Create session param
        ft::SessionParam session_param;
        session_param.id = session->id;
        session_param.step = session->step;
        session_param.start_flag = session->start_flag;
        session_param.end_flag = session->end_flag;
        
        // Create generation config
        ft::GenerationConfig generation_config;
        generation_config.max_new_tokens = gen_config->max_new_tokens;
        generation_config.min_new_tokens = gen_config->min_new_tokens;
        generation_config.top_p = gen_config->top_p;
        generation_config.top_k = gen_config->top_k;
        generation_config.min_p = gen_config->min_p;
        generation_config.temperature = gen_config->temperature;
        generation_config.repetition_penalty = gen_config->repetition_penalty;
        generation_config.random_seed = gen_config->random_seed;
        generation_config.output_logprobs = gen_config->output_logprobs;
        generation_config.output_last_hidden_state = gen_config->output_last_hidden_state;
        generation_config.output_logits = gen_config->output_logits;
        
        // Convert arrays
        if (gen_config->eos_ids && gen_config->eos_ids_count > 0) {
            generation_config.eos_ids.assign(gen_config->eos_ids, gen_config->eos_ids + gen_config->eos_ids_count);
        }
        if (gen_config->stop_ids && gen_config->stop_ids_count > 0) {
            generation_config.stop_ids[0].assign(gen_config->stop_ids, gen_config->stop_ids + gen_config->stop_ids_count);
        }
        if (gen_config->bad_ids && gen_config->bad_ids_count > 0) {
            generation_config.bad_ids[0].assign(gen_config->bad_ids, gen_config->bad_ids + gen_config->bad_ids_count);
        }
        
        // Prepare input param
        ft::ModelRequest::InputParam input_param;
        input_param.tensors = input_tensors->tensor_map;
        input_param.session = session_param;
        input_param.gen_cfg = generation_config;
        input_param.stream_output = stream_output;
        
        // Call forward
        auto output_param = instance->request->Forward(std::move(input_param), nullptr);
        
        // Create result
        TurboMindRequestStatus status = TM_REQUEST_COMPLETED; // Simplified
        return new TurboMindForwardResult(output_param.tensors, status, 0);
        
    } catch (const std::exception& e) {
        set_last_error("Forward inference failed: " + std::string(e.what()));
        return nullptr;
    }
}

void turbomind_destroy_forward_result(TurboMindForwardResult* result) {
    delete result;
}

// Session management
void turbomind_end_session(TurboMindModelInstance* instance, uint64_t session_id) {
    if (!instance) {
        set_last_error("Invalid instance for end session");
        return;
    }
    
    try {
        instance->request->End([](int){}, session_id);
    } catch (const std::exception& e) {
        set_last_error("Failed to end session: " + std::string(e.what()));
    }
}

void turbomind_cancel_request(TurboMindModelInstance* instance) {
    if (!instance) {
        set_last_error("Invalid instance for cancel request");
        return;
    }
    
    try {
        instance->request->Cancel();
    } catch (const std::exception& e) {
        set_last_error("Failed to cancel request: " + std::string(e.what()));
    }
}

// Model information
int turbomind_get_tensor_para_size(TurboMindModel* model) {
    if (!model) {
        set_last_error("Invalid model for tensor para size");
        return -1;
    }
    
    try {
        return model->model->getTensorParaSize();
    } catch (const std::exception& e) {
        set_last_error("Failed to get tensor para size: " + std::string(e.what()));
        return -1;
    }
}

int turbomind_get_pipeline_para_size(TurboMindModel* model) {
    if (!model) {
        set_last_error("Invalid model for pipeline para size");
        return -1;
    }
    
    try {
        return model->model->getPipelineParaSize();
    } catch (const std::exception& e) {
        set_last_error("Failed to get pipeline para size: " + std::string(e.what()));
        return -1;
    }
}

// Utility functions
const char* turbomind_get_last_error() {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return g_last_error.c_str();
}

void turbomind_set_device(int device_id) {
    try {
        ft::check_cuda_error(cudaSetDevice(device_id));
    } catch (const std::exception& e) {
        set_last_error("Failed to set device: " + std::string(e.what()));
    }
}

// Helper functions for tensor operations
size_t turbomind_get_tensor_size(TurboMindTensor* tensor) {
    if (!tensor) {
        set_last_error("Invalid tensor for size calculation");
        return 0;
    }
    
    try {
        return tensor->tensor->byte_size();
    } catch (const std::exception& e) {
        set_last_error("Failed to get tensor size: " + std::string(e.what()));
        return 0;
    }
}

void turbomind_copy_tensor(TurboMindTensor* dst, TurboMindTensor* src) {
    if (!dst || !src) {
        set_last_error("Invalid tensors for copy");
        return;
    }
    
    try {
        if (dst->tensor->byte_size() != src->tensor->byte_size()) {
            set_last_error("Tensor size mismatch for copy");
            return;
        }
        
        // Use CUDA memcpy for tensor copying
        ft::check_cuda_error(cudaMemcpy(dst->tensor->raw_data(), src->tensor->raw_data(), 
                                       dst->tensor->byte_size(), cudaMemcpyDefault));
    } catch (const std::exception& e) {
        set_last_error("Failed to copy tensor: " + std::string(e.what()));
    }
}

} // extern "C"