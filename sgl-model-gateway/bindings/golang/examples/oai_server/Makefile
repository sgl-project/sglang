# Makefile for OAI Server
# Builds binary, runs tests, and provides basic targets

# Configuration
APP_NAME = oai_server
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Paths
ROOT_DIR := $(shell pwd)
BINDINGS_DIR := $(shell cd $(ROOT_DIR)/../.. && pwd)
BUILD_DIR := $(ROOT_DIR)/build
BINARY := $(BUILD_DIR)/$(APP_NAME)

# Rust FFI library paths
LIB_DIR := $(BINDINGS_DIR)/lib
LIB_NAME = libsgl_model_gateway_go

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB_EXT = .so
    LD_LIBRARY_PATH_VAR = LD_LIBRARY_PATH
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),x86_64)
        GOARCH = amd64
    else ifeq ($(ARCH),aarch64)
        GOARCH = arm64
    endif
endif
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = .dylib
    LD_LIBRARY_PATH_VAR = DYLD_LIBRARY_PATH
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),x86_64)
        GOARCH = amd64
    else ifeq ($(ARCH),arm64)
        GOARCH = arm64
    endif
endif

# Build flags
LDFLAGS = -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME) -X main.GitCommit=$(GIT_COMMIT)
GO_BUILD_FLAGS = -ldflags "$(LDFLAGS)"

# Python LDFLAGS (needed for Rust FFI that depends on Python)
PYTHON_LDFLAGS := $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || python-config --ldflags --embed 2>/dev/null || python-config --ldflags 2>/dev/null || echo "")

# CGO flags
CGO_LDFLAGS = -L$(LIB_DIR) $(PYTHON_LDFLAGS)

.PHONY: all build build-dev test e2e clean help lib run stream check-rust-lib check-server

# E2E test configuration
E2E_HOST ?= localhost
E2E_PORT ?= 8080
E2E_MODEL ?= default
E2E_TOKENIZER ?= $(shell echo $$SGL_TOKENIZER_PATH || echo "./examples/tokenizer")
E2E_NUM_PROMPTS ?= 100
E2E_INPUT_LEN ?= 1024
E2E_OUTPUT_LEN ?= 512
E2E_REQUEST_RATE ?= 20
E2E_MAX_CONCURRENCY ?= 20
E2E_BASE_URL ?= http://$(E2E_HOST):$(E2E_PORT)

help:
	@echo "OAI Server Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  lib             - Build Rust FFI library"
	@echo "  build           - Build binary (release mode)"
	@echo "  build-dev       - Build binary (debug mode)"
	@echo "  test            - Run tests"
	@echo "  e2e             - Run end-to-end test with bench_serving.py"
	@echo "  run             - Run the server (development)"
	@echo "  stream          - Run streaming example"
	@echo "  clean           - Clean build artifacts"
	@echo ""
	@echo "E2E test variables:"
	@echo "  E2E_HOST          - OAI Server host (default: localhost)"
	@echo "  E2E_PORT          - OAI Server port (default: 8080)"
	@echo "  E2E_MODEL         - Model name (default: default)"
	@echo "  E2E_TOKENIZER     - Tokenizer path"
	@echo "  E2E_NUM_PROMPTS   - Number of prompts (default: 100)"
	@echo "  E2E_INPUT_LEN     - Input token length (default: 1024)"
	@echo "  E2E_OUTPUT_LEN    - Output token length (default: 512)"
	@echo "  E2E_REQUEST_RATE  - Request rate per second (default: 20)"
	@echo "  E2E_MAX_CONCURRENCY - Max concurrent requests (default: 20)"

all: build

# Build Rust FFI library
lib:
	@echo "Building Rust FFI library..."
	@cd $(BINDINGS_DIR) && $(MAKE) lib
	@echo "✓ Rust FFI library built"

# Check if Rust FFI library exists
check-rust-lib:
	@if [ ! -f "$(LIB_DIR)/$(LIB_NAME)$(LIB_EXT)" ]; then \
		echo "Error: Rust FFI library not found at $(LIB_DIR)/$(LIB_NAME)$(LIB_EXT)"; \
		echo "Building Rust library..."; \
		cd $(BINDINGS_DIR) && $(MAKE) lib; \
	fi
	@echo "✓ Rust FFI library found"

# Build binary (release)
build: check-rust-lib
	@echo "Building $(APP_NAME) (release mode)..."
	@mkdir -p $(BUILD_DIR)
	@CGO_ENABLED=1 \
	 CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	 GOOS=$(shell go env GOOS) \
	 GOARCH=$(GOARCH) \
	 go build $(GO_BUILD_FLAGS) -o $(BINARY) .
	@echo "✓ Binary built: $(BINARY)"

# Build binary (debug)
build-dev: check-rust-lib
	@echo "Building $(APP_NAME) (debug mode)..."
	@mkdir -p $(BUILD_DIR)
	@CGO_ENABLED=1 \
	 CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	 go build -o $(BINARY) .
	@echo "✓ Binary built (debug): $(BINARY)"

# Run tests
test: check-rust-lib
	@echo "Running tests..."
	@CGO_ENABLED=1 \
	 CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	 export $(LD_LIBRARY_PATH_VAR)="$(LIB_DIR):$$$(LD_LIBRARY_PATH_VAR)" && \
	 go test -v ./...
	@echo "✓ Tests completed"

# Check if OAI Server is running
check-server:
	@echo "Checking if OAI Server is running at $(E2E_BASE_URL)..."
	@if curl -s -f $(E2E_BASE_URL)/health > /dev/null 2>&1; then \
		echo "✓ OAI Server is running"; \
		exit 0; \
	else \
		echo "✗ OAI Server is not running at $(E2E_BASE_URL)"; \
		echo "  Start it with: make run"; \
		exit 1; \
	fi

# Find sglang project root (4 levels up from oai_server)
SGLANG_ROOT := $(shell cd $(ROOT_DIR)/../../../../.. && pwd)

# Run end-to-end test with bench_serving.py
e2e: check-server
	@echo "Checking if bench_serving.py is available..."
	@if python -m sglang.bench_serving --help > /dev/null 2>&1; then \
		echo "✓ Using installed bench_serving.py module"; \
		USE_SGLANG_ROOT=false; \
	elif [ -f "$(SGLANG_ROOT)/python/sglang/bench_serving.py" ]; then \
		echo "✓ Using bench_serving.py from $(SGLANG_ROOT)"; \
		USE_SGLANG_ROOT=true; \
	else \
		echo "✗ bench_serving.py is not available"; \
		echo "  Install dependencies: pip install aiohttp numpy datasets transformers tqdm pillow pybase64"; \
		exit 1; \
	fi
	@echo "Running end-to-end test with bench_serving.py..."
	@echo "Configuration:"
	@echo "  Server: $(E2E_BASE_URL)"
	@if [ "$(E2E_MODEL)" != "default" ]; then \
		echo "  Model: $(E2E_MODEL)"; \
	fi
	@if [ -n "$(E2E_TOKENIZER)" ]; then \
		echo "  Tokenizer: $(E2E_TOKENIZER)"; \
	fi
	@echo "  Prompts: $(E2E_NUM_PROMPTS)"
	@echo "  Input/Output: $(E2E_INPUT_LEN)/$(E2E_OUTPUT_LEN) tokens"
	@echo "  Request rate: $(E2E_REQUEST_RATE) req/s"
	@echo "  Max concurrency: $(E2E_MAX_CONCURRENCY)"
	@echo ""
	@TOKENIZER_ABS=$$(cd $(ROOT_DIR) && python3 -c "import os; path='$(E2E_TOKENIZER)'; print(os.path.abspath(path) if not os.path.isabs(path) else path)" 2>/dev/null || echo "$(E2E_TOKENIZER)"); \
	if [ -n "$(E2E_TOKENIZER)" ]; then \
		if [ -n "$$TOKENIZER_ABS" ] && ([ -d "$$TOKENIZER_ABS" ] || [ -f "$$TOKENIZER_ABS" ]); then \
			TOKENIZER_ARG="--tokenizer $$TOKENIZER_ABS"; \
		else \
			TOKENIZER_ARG="--tokenizer $(E2E_TOKENIZER)"; \
		fi; \
	else \
		TOKENIZER_ARG=""; \
	fi; \
	if [ "$$USE_SGLANG_ROOT" = "true" ]; then \
		cd $(SGLANG_ROOT) && PYTHONPATH=$(SGLANG_ROOT)/python:$$PYTHONPATH python python/sglang/bench_serving.py \
			--backend sglang-oai-chat \
			--base-url $(E2E_BASE_URL) \
			$$([ "$(E2E_MODEL)" != "default" ] && echo "--model $(E2E_MODEL)") \
			$$TOKENIZER_ARG \
			--dataset-name random \
			--num-prompts $(E2E_NUM_PROMPTS) \
			--random-input-len $(E2E_INPUT_LEN) \
			--random-output-len $(E2E_OUTPUT_LEN) \
			--request-rate $(E2E_REQUEST_RATE) \
			--max-concurrency $(E2E_MAX_CONCURRENCY) \
			--warmup-requests 5 \
			--disable-tqdm || (echo "✗ E2E test failed"; exit 1); \
	else \
		python -m sglang.bench_serving \
			--backend sglang-oai-chat \
			--base-url $(E2E_BASE_URL) \
			$$([ "$(E2E_MODEL)" != "default" ] && echo "--model $(E2E_MODEL)") \
			$$TOKENIZER_ARG \
			--dataset-name random \
			--num-prompts $(E2E_NUM_PROMPTS) \
			--random-input-len $(E2E_INPUT_LEN) \
			--random-output-len $(E2E_OUTPUT_LEN) \
			--request-rate $(E2E_REQUEST_RATE) \
			--max-concurrency $(E2E_MAX_CONCURRENCY) \
			--warmup-requests 5 \
			--disable-tqdm || (echo "✗ E2E test failed"; exit 1); \
	fi
	@echo ""
	@echo "✓ E2E test completed"

# Run the server (development)
run: build-dev
	@echo "Running server..."
	@export $(LD_LIBRARY_PATH_VAR)="$(LIB_DIR):$$$(LD_LIBRARY_PATH_VAR)" && \
	 $(BINARY)

# Run streaming example
stream: check-rust-lib
	@echo "Running streaming example..."
	@cd $(BINDINGS_DIR)/examples/streaming && \
	 export $(LD_LIBRARY_PATH_VAR)="$(LIB_DIR):$$$(LD_LIBRARY_PATH_VAR)" && \
	 bash run.sh

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "✓ Clean complete"
