# Multi-stage Dockerfile for TurboMind-Go
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    pkg-config \
    ninja-build \
    pybind11-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Go
ENV GO_VERSION=1.21.0
RUN wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz
ENV PATH=/usr/local/go/bin:${PATH}

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Build LMDeploy first
RUN make setup-lmdeploy
RUN make build-lmdeploy

# Build TurboMind-Go
RUN make build

# Runtime stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Go runtime
ENV GO_VERSION=1.21.0
RUN apt-get update && apt-get install -y wget && \
    wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/go/bin:${PATH}

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create working directory
WORKDIR /app

# Copy built artifacts
COPY --from=builder /workspace/build/libturbomind_go.so /app/lib/
COPY --from=builder /workspace/src/turbomind_wrapper.hpp /app/include/
COPY --from=builder /workspace/test /app/test

# Set library path
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

# Default command
CMD ["/bin/bash"]

# Development stage with all tools
FROM builder AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Go development tools
RUN go install golang.org/x/tools/gopls@latest && \
    go install github.com/go-delve/delve/cmd/dlv@latest

WORKDIR /workspace

# Keep container running for development
CMD ["/bin/bash"] 