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
find_package(CUDA REQUIRED)

# LMDeploy path configuration
set(LMDEPLOY_ROOT "${CMAKE_SOURCE_DIR}/third_party/lmdeploy" CACHE PATH "Path to LMDeploy source")
set(LMDEPLOY_BUILD_DIR "${LMDEPLOY_ROOT}/build" CACHE PATH "Path to LMDeploy build directory")

# Check if LMDeploy exists
if(NOT EXISTS ${LMDEPLOY_ROOT})
    message(FATAL_ERROR "LMDeploy not found at ${LMDEPLOY_ROOT}. Please clone it first.")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${LMDEPLOY_ROOT}/src
    ${LMDEPLOY_ROOT}/src/turbomind
    ${LMDEPLOY_ROOT}/3rdparty
    ${CUDA_INCLUDE_DIRS}
)

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

# Source files - Use hybrid implementation
set(SOURCES
    src/turbomind_wrapper_hybrid.cpp
)

# LMDeploy static libraries to link (selective linking for hybrid approach)
set(LMDEPLOY_LIBRARIES
    # Core libraries - essential ones only
    core
    logger
    cuda_utils
    memory_utils
    
    # Optional libraries - will be linked if available
    # Llama
    # engine
    # LlamaTritonBackend
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
    cudart
)

# Handle missing libraries gracefully
foreach(lib ${LMDEPLOY_LIBRARIES})
    if(NOT TARGET ${lib})
        find_library(${lib}_FOUND NAMES ${lib} lib${lib}.a PATHS ${LMDEPLOY_BUILD_DIR}/lib NO_DEFAULT_PATH)
        if(${lib}_FOUND)
            target_link_libraries(turbomind_go PRIVATE ${${lib}_FOUND})
        else()
            message(WARNING "Library ${lib} not found, skipping...")
        endif()
    endif()
endforeach()

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