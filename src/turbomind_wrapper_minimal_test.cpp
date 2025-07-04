#include "turbomind_wrapper.hpp"
#include <iostream>
#include <string>
#include <memory>
#include <cstring>
#include <map>
#include <vector>
#include <sys/stat.h>
#include <sstream>

// Simple test implementation without complex dependencies
static std::string g_last_error;

static void set_last_error(const std::string& error) {
    g_last_error = error;
    std::cerr << "Error: " << error << std::endl;
}

// Minimal struct implementations for testing
struct TurboMindModel {
    std::string model_dir;
    bool initialized = false;
    
    TurboMindModel(const std::string& dir, const std::string& config, const std::string& weight_type) 
        : model_dir(dir) {
        std::cout << "Created model with dir: " << dir << std::endl;
        
        // Check if model directory exists (for better error handling)
        struct stat info;
        if (stat(dir.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            throw std::runtime_error("Model directory does not exist: " + dir);
        }
        
        initialized = true;
    }
};

struct TurboMindModelInstance {
    TurboMindModel* model;
    int device_id;
    
    TurboMindModelInstance(TurboMindModel* m, int dev_id) : model(m), device_id(dev_id) {
        std::cout << "Created model instance on device: " << device_id << std::endl;
    }
};

struct TurboMindTensor {
    std::vector<int64_t> shape;
    TurboMindDataType dtype;
    TurboMindMemoryType memory_type;
    size_t size_bytes;
    
    TurboMindTensor(void* data, int64_t* shape_ptr, int ndim, TurboMindDataType dt, 
                    TurboMindMemoryType mt, int device_id) 
        : dtype(dt), memory_type(mt) {
        shape.assign(shape_ptr, shape_ptr + ndim);
        
        // Calculate size
        size_bytes = 1;
        for (int i = 0; i < ndim; i++) {
            size_bytes *= shape_ptr[i];
        }
        
        // Size based on data type
        switch (dt) {
            case TM_TYPE_INT32: size_bytes *= 4; break;
            case TM_TYPE_FP16: size_bytes *= 2; break;
            case TM_TYPE_FP32: size_bytes *= 4; break;
            default: size_bytes *= 4; break;
        }
        
        std::cout << "Created tensor with " << ndim << " dimensions, size: " << size_bytes << " bytes" << std::endl;
    }
};

struct TurboMindTensorMap {
    std::map<std::string, std::shared_ptr<TurboMindTensor>> tensors;
    
    TurboMindTensorMap() {
        std::cout << "Created tensor map" << std::endl;
    }
};

struct TurboMindForwardResult {
    std::shared_ptr<TurboMindTensorMap> tensors;
    TurboMindRequestStatus status;
    int seq_len;
    
    TurboMindForwardResult() : status(TM_REQUEST_COMPLETED), seq_len(0) {
        tensors = std::make_shared<TurboMindTensorMap>();
        std::cout << "Created forward result" << std::endl;
    }
};

// C API Implementation
extern "C" {

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
        tensor_map->tensors[key] = std::shared_ptr<TurboMindTensor>(tensor, [](TurboMindTensor*) {
            // Don't delete, let Go handle it
        });
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
        auto it = tensor_map->tensors.find(key);
        if (it == tensor_map->tensors.end()) {
            set_last_error("Tensor not found in map: " + std::string(key));
            return nullptr;
        }
        return it->second.get();
    } catch (const std::exception& e) {
        set_last_error("Failed to get tensor from map: " + std::string(e.what()));
        return nullptr;
    }
}

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
        std::cout << "Running forward inference..." << std::endl;
        std::cout << "Session ID: " << session->id << std::endl;
        std::cout << "Max new tokens: " << gen_config->max_new_tokens << std::endl;
        std::cout << "Temperature: " << gen_config->temperature << std::endl;
        
        // Create mock result with session-specific output
        auto result = new TurboMindForwardResult();
        result->seq_len = static_cast<int>(session->id) * 10; // Vary by session
        return result;
    } catch (const std::exception& e) {
        set_last_error("Forward inference failed: " + std::string(e.what()));
        return nullptr;
    }
}

void turbomind_destroy_forward_result(TurboMindForwardResult* result) {
    delete result;
}

void turbomind_end_session(TurboMindModelInstance* instance, uint64_t session_id) {
    if (!instance) {
        set_last_error("Invalid instance for end session");
        return;
    }
    std::cout << "Ended session: " << session_id << std::endl;
}

void turbomind_cancel_request(TurboMindModelInstance* instance) {
    if (!instance) {
        set_last_error("Invalid instance for cancel request");
        return;
    }
    std::cout << "Cancelled request" << std::endl;
}

int turbomind_get_tensor_para_size(TurboMindModel* model) {
    if (!model) {
        set_last_error("Invalid model for tensor para size");
        return -1;
    }
    return 1; // Mock value
}

int turbomind_get_pipeline_para_size(TurboMindModel* model) {
    if (!model) {
        set_last_error("Invalid model for pipeline para size");
        return -1;
    }
    return 1; // Mock value
}

const char* turbomind_get_last_error() {
    return g_last_error.c_str();
}

void turbomind_set_device(int device_id) {
    std::cout << "Set device to: " << device_id << std::endl;
}

size_t turbomind_get_tensor_size(TurboMindTensor* tensor) {
    if (!tensor) {
        set_last_error("Invalid tensor for size calculation");
        return 0;
    }
    return tensor->size_bytes;
}

void turbomind_copy_tensor(TurboMindTensor* dst, TurboMindTensor* src) {
    if (!dst || !src) {
        set_last_error("Invalid tensors for copy");
        return;
    }
    
    if (dst->size_bytes != src->size_bytes) {
        set_last_error("Tensor size mismatch for copy");
        return;
    }
    
    std::cout << "Copied tensor (" << src->size_bytes << " bytes)" << std::endl;
}

} // extern "C"