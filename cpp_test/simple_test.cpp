#include "turbomind_wrapper.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing TurboMind Wrapper C++ Interface..." << std::endl;

    // Test version info
    auto version = turbomind_get_version();
    std::cout << "Version: " << version.version << std::endl;
    std::cout << "Git Commit: " << version.git_commit << std::endl;
    std::cout << "Build Time: " << version.build_time << std::endl;

    // Test engine creation with valid config
    TurboMindConfig config = {};
    config.model_path = "/fake/model/path";
    config.model_format = "hf";
    config.tp = 1;
    config.session_len = 1024;
    config.max_batch_size = 8;
    config.quant_policy = 0;

    std::cout << "\nCreating engine..." << std::endl;
    auto engine = turbomind_create_engine(&config);
    assert(engine != nullptr);

    // Test engine ready
    bool ready = turbomind_is_engine_ready(engine);
    assert(ready);
    std::cout << "Engine is ready: " << ready << std::endl;

    // Test model info
    ModelInfo model_info = {};
    int result = turbomind_get_model_info(engine, &model_info);
    assert(result == 0);
    std::cout << "Model Name: " << model_info.model_name << std::endl;
    std::cout << "Model Type: " << model_info.model_type << std::endl;
    std::cout << "Vocab Size: " << model_info.vocab_size << std::endl;

    // Test generation
    RequestParams request = {};
    request.request_id = 1;
    request.prompt = "Hello, world!";
    request.max_new_tokens = 50;
    request.temperature = 0.7f;
    request.top_p = 0.8f;
    request.top_k = 40.0f;
    request.repetition_penalty = 1;
    request.stream = false;
    request.stop_words = "";

    ResponseData response = {};
    std::cout << "\nGenerating response..." << std::endl;
    result = turbomind_generate(engine, &request, &response);
    assert(result == 0);
    
    std::cout << "Request ID: " << response.request_id << std::endl;
    std::cout << "Input Tokens: " << response.input_tokens << std::endl;
    std::cout << "Output Tokens: " << response.output_tokens << std::endl;
    std::cout << "Generated Text: " << response.text << std::endl;
    std::cout << "Finished: " << response.finished << std::endl;

    // Cleanup
    turbomind_free_response(&response);
    turbomind_free_model_info(&model_info);
    turbomind_destroy_engine(engine);

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
} 