cmake_minimum_required(VERSION 3.18)

project(turbomind-go 
    VERSION 0.9.0
    DESCRIPTION "Golang bindings for TurboMind inference engine"
    LANGUAGES CXX CUDA
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Version and build info
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

string(TIMESTAMP BUILD_TIME "%Y-%m-%d %H:%M:%S")

# Find required packages
find_package(CUDA 12.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# LMDeploy path configuration
set(LMDEPLOY_ROOT "${CMAKE_SOURCE_DIR}/third_party/lmdeploy" CACHE PATH "Path to LMDeploy source")
set(LMDEPLOY_BUILD_DIR "${LMDEPLOY_ROOT}/build" CACHE PATH "Path to LMDeploy build directory")

# Check if LMDeploy exists
if(NOT EXISTS ${LMDEPLOY_ROOT})
    message(FATAL_ERROR "LMDeploy not found at ${LMDEPLOY_ROOT}. Please clone it first.")
endif()

# Include directories
include_directories(
    ${LMDEPLOY_ROOT}/src
    ${LMDEPLOY_ROOT}/src/turbomind
    ${LMDEPLOY_ROOT}/3rdparty
    ${CUDA_INCLUDE_DIRS}
)

# Add LMDeploy include directories to handle all header reference styles
include_directories(${LMDEPLOY_ROOT})
include_directories(${LMDEPLOY_ROOT}/src)

# Link directories
link_directories(
    ${LMDEPLOY_BUILD_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --expt-extended-lambda")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
endif()

# Define build macros
add_definitions(
    -DLMDEPLOY_VERSION="${PROJECT_VERSION}"
    -DGIT_COMMIT="${GIT_COMMIT}"
    -DBUILD_TIME="${BUILD_TIME}"
    -DCUDA_VERSION="${CUDA_VERSION_STRING}"
)

# Source files - real implementation
set(SOURCES
    src/turbomind_wrapper_proper.cpp
)

# LMDeploy static libraries to link (actual built libraries)
# Order matters for static linking - parser must come before libraries that use it
set(LMDEPLOY_LIBRARIES
    # Core TurboMind libraries
    Llama
    LlamaTritonBackend
    engine
    core
    logger
    
    # Layer components first
    DynamicDecodeLayer
    SamplingLayer
    LogitsProcessorLayer
    StopCriteriaLayer
    
    # Essential kernels
    attention
    unfused_attention_kernels
    decoding_kernels
    gpt_kernels
    activation_kernels
    quantization_kernels
    rms_norm
    logprob_kernels
    
    # Sampling and stopping
    sampling_kernels
    sampling_penalty_kernels
    sampling_topk_kernels
    sampling_topp_kernels
    ban_bad_words
    stop_criteria
    
    # Communication
    device_comm
    host_comm
    cuda_ipc_comm
    
    # CUDA utilities
    cuda_utils
    memory_utils
    nvtx_utils
    anomaly_handler
    
    # GEMM after parser (gemm2 uses parser functions)
    gemm2
    
    # Parser library last (provides ParseArgsList, ParseListOrTuple)
    parser
)

# Create shared library
add_library(turbomind_go SHARED ${SOURCES})

# Target properties
set_target_properties(turbomind_go PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

# Link libraries  
target_link_libraries(turbomind_go PRIVATE
    ${LMDEPLOY_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    cuda
    cudart
    cublas
    cublasLt
)

# Link LMDeploy libraries for real implementation
foreach(lib ${LMDEPLOY_LIBRARIES})
    find_library(${lib}_PATH 
        NAMES lib${lib}.a ${lib} 
        PATHS ${LMDEPLOY_BUILD_DIR}/lib 
        NO_DEFAULT_PATH
    )
    if(${lib}_PATH)
        target_link_libraries(turbomind_go PRIVATE ${${lib}_PATH})
        message(STATUS "Linking library: ${${lib}_PATH}")
    else()
        message(WARNING "Required library ${lib} not found in ${LMDEPLOY_BUILD_DIR}/lib")
    endif()
endforeach()

# Link YAML library for real implementation
find_library(YAML_CPP_PATH 
    NAMES libyaml-cpp.a yaml-cpp
    PATHS ${LMDEPLOY_BUILD_DIR}/_deps/yaml-cpp-build ${LMDEPLOY_BUILD_DIR}/lib
    NO_DEFAULT_PATH
)
if(YAML_CPP_PATH)
    target_link_libraries(turbomind_go PRIVATE ${YAML_CPP_PATH})
    message(STATUS "Linking YAML library: ${YAML_CPP_PATH}")
else()
    message(WARNING "YAML library not found")
endif()

# Version information
set_target_properties(turbomind_go PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation
install(TARGETS turbomind_go
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES src/turbomind_wrapper.hpp
    DESTINATION include
)

# C++ test executable (optional)
if(EXISTS "${CMAKE_SOURCE_DIR}/cpp_test/simple_test.cpp")
    add_executable(test_wrapper
        cpp_test/simple_test.cpp
    )
    target_link_libraries(test_wrapper turbomind_go)
endif()

# Print configuration summary
message(STATUS "TurboMind-Go Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  LMDeploy Root: ${LMDEPLOY_ROOT}")
message(STATUS "  LMDeploy Build Dir: ${LMDEPLOY_BUILD_DIR}")
message(STATUS "  CUDA Version: ${CUDA_VERSION_STRING}")
message(STATUS "  Git Commit: ${GIT_COMMIT}")
message(STATUS "  Build Time: ${BUILD_TIME}") 