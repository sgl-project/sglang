# Makefile for sglang-router golang bindings
# This builds the Rust FFI library and provides convenience targets for Go development

# Configuration
CARGO_BUILD_DIR ?= $(HOME)/cargobuild
BUILD_MODE ?= release
LIB_NAME = libsglang_router_rs

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB_EXT = .so
    LD_LIBRARY_PATH_VAR = LD_LIBRARY_PATH
endif
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = .dylib
    LD_LIBRARY_PATH_VAR = DYLD_LIBRARY_PATH
endif

# Paths
ROOT_DIR := $(shell pwd)
RUST_SRC_DIR := $(ROOT_DIR)/src
LIB_DIR := $(CARGO_BUILD_DIR)/$(BUILD_MODE)
LIB_PATH := $(LIB_DIR)/$(LIB_NAME)$(LIB_EXT)

# Python LDFLAGS (needed for Rust FFI that depends on Python)
PYTHON_LDFLAGS := $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

# CGO flags
export CGO_LDFLAGS = -L$(LIB_DIR) -lsglang_router_rs $(PYTHON_LDFLAGS) -ldl
export $(LD_LIBRARY_PATH_VAR) := $(LIB_DIR):$($(LD_LIBRARY_PATH_VAR))

.PHONY: all build build-dev clean test examples help

help:
	@echo "Available targets:"
	@echo "  build       - Build release version of Rust FFI library"
	@echo "  build-dev   - Build debug version of Rust FFI library"
	@echo "  clean       - Clean build artifacts"
	@echo "  test        - Run Go tests"
	@echo "  examples    - Build example programs"
	@echo "  run-simple  - Run simple example"
	@echo "  run-streaming - Run streaming example"

all: build

build:
	@echo "Building Rust FFI library (release mode)..."
	@CARGO_TARGET_DIR=$(CARGO_BUILD_DIR) cargo build --release --manifest-path Cargo.toml
	@echo "Library built at: $(LIB_PATH)"

build-dev:
	@echo "Building Rust FFI library (debug mode)..."
	@CARGO_TARGET_DIR=$(CARGO_BUILD_DIR) cargo build --manifest-path Cargo.toml
	@echo "Library built at: $(LIB_DIR)/debug/$(LIB_NAME)$(LIB_EXT)"

clean:
	@echo "Cleaning build artifacts..."
	@CARGO_TARGET_DIR=$(CARGO_BUILD_DIR) cargo clean --manifest-path Cargo.toml
	@echo "Clean complete"

test:
	@echo "Running Go tests..."
	@go test ./...

examples: build
	@echo "Building example programs..."
	@cd examples/simple && go build -o simple main.go
	@cd examples/streaming && go build -o streaming main.go
	@echo "Examples built"

run-simple: build
	@echo "Running simple example..."
	@cd examples/simple && bash run.sh

run-streaming: build
	@echo "Running streaming example..."
	@cd examples/streaming && bash run.sh

# Check if library exists
check-lib:
	@if [ ! -f "$(LIB_PATH)" ]; then \
		echo "Error: Library not found at $(LIB_PATH)"; \
		echo "Run 'make build' first"; \
		exit 1; \
	fi
