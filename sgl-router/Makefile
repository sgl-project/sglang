# Model Gateway Makefile
# Provides convenient shortcuts for common development tasks

# Python bindings directory
PYTHON_DIR := bindings/python

# Auto-detect CPU cores and cap at reasonable limit to avoid thread exhaustion
# Can be overridden: make python-dev JOBS=4
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
JOBS ?= $(shell echo $$(($(NPROC) > 16 ? 16 : $(NPROC))))

# Check if sccache is available and set RUSTC_WRAPPER accordingly
SCCACHE := $(shell which sccache 2>/dev/null)
ifdef SCCACHE
    export RUSTC_WRAPPER := $(SCCACHE)
    $(info Using sccache for compilation caching)
else
    $(info sccache not found. Install it for faster builds: cargo install sccache)
endif

.PHONY: help build test clean docs check fmt dev-setup pre-commit setup-sccache sccache-stats sccache-clean sccache-stop \
        python-dev python-build python-build-release python-install python-clean python-test python-check \
        release-notes

help: ## Show this help message
	@echo "Model Gateway Development Commands"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

build: ## Build the project in release mode
	@echo "Building SGLang Model Gateway..."
	@cargo build --release

test: ## Run all tests
	@echo "Running tests..."
	@cargo test

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@cargo clean

docs: ## Generate and open documentation
	@echo "Generating documentation..."
	@cargo doc --open

check: ## Run cargo check and clippy
	@echo "Running cargo check..."
	@cargo check
	@echo "Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

fmt: ## Format code with rustfmt
	@echo "Formatting code..."
	@rustup run nightly cargo fmt

# Development workflow shortcuts
dev-setup: build test ## Set up development environment
	@echo "Development environment ready!"

pre-commit: fmt check test ## Run pre-commit checks
	@echo "Pre-commit checks passed!"

# sccache management targets
setup-sccache: ## Install and configure sccache
	@echo "Setting up sccache..."
	@./scripts/setup-sccache.sh

sccache-stats: ## Show sccache statistics
	@if [ -n "$(SCCACHE)" ]; then \
		echo "sccache statistics:"; \
		sccache -s; \
	else \
		echo "sccache not installed. Run 'make setup-sccache' to install it."; \
	fi

sccache-clean: ## Clear sccache cache
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Clearing sccache cache..."; \
		sccache -C; \
		echo "sccache cache cleared"; \
	else \
		echo "sccache not installed"; \
	fi

sccache-stop: ## Stop the sccache server
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Stopping sccache server..."; \
		sccache --stop-server || true; \
	else \
		echo "sccache not installed"; \
	fi

# Python bindings (maturin) targets
python-dev: ## Build Python bindings in development mode (fast, debug build)
	@echo "Building Python bindings in development mode (using $(JOBS) parallel jobs with sccache)..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin develop

python-build: ## Build Python wheel (release mode with vendored OpenSSL)
	@echo "Building Python wheel (release, vendored OpenSSL, using $(JOBS) parallel jobs with sccache)..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin build --release --out dist --features vendored-openssl

python-build-release: python-build ## Alias for python-build

python-install: python-build ## Build and install Python wheel
	@echo "Installing Python wheel..."
	@pip install --force-reinstall $(PYTHON_DIR)/dist/*.whl
	@echo "Python package installed!"

python-clean: ## Clean Python build artifacts
	@echo "Cleaning Python build artifacts..."
	@rm -rf $(PYTHON_DIR)/dist/
	@rm -rf $(PYTHON_DIR)/target/
	@rm -rf $(PYTHON_DIR)/sglang_router.egg-info/
	@rm -rf $(PYTHON_DIR)/sglang_router/__pycache__/
	@find $(PYTHON_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find $(PYTHON_DIR) -name "*.pyc" -delete 2>/dev/null || true
	@echo "Python build artifacts cleaned!"

python-test: ## Run Python tests
	@echo "Running Python tests..."
	@pytest py_test/ -v

python-check: ## Check Python package with twine
	@echo "Checking Python package..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin build --release --out dist --features vendored-openssl
	@pip install twine 2>/dev/null || true
	@twine check $(PYTHON_DIR)/dist/*
	@echo "Python package check passed!"

# Combined shortcuts
dev: python-dev ## Quick development setup (build Python bindings in dev mode)

install: python-install ## Build and install everything

# Release management
release-notes: ## Generate release notes for gateway (usage: make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0)
	@if [ -z "$(PREV)" ] || [ -z "$(CURR)" ]; then \
		echo "Usage: make release-notes PREV=<previous-tag> CURR=<current-tag>"; \
		echo "Example: make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0"; \
		echo ""; \
		echo "Options:"; \
		echo "  OUTPUT=<file>     Save to file (default: stdout)"; \
		echo "  CREATE_RELEASE=1  Create GitHub draft release via gh CLI (default: draft)"; \
		echo "  DRAFT=0           Publish release immediately (skip draft)"; \
		exit 1; \
	fi
	@ARGS="$(PREV) $(CURR)"; \
	if [ -n "$(OUTPUT)" ]; then \
		ARGS="$$ARGS --output $(OUTPUT)"; \
	fi; \
	if [ "$(CREATE_RELEASE)" = "1" ]; then \
		ARGS="$$ARGS --create-release"; \
		if [ "$(DRAFT)" = "0" ]; then \
			ARGS="$$ARGS --no-draft"; \
		fi; \
	fi; \
	./scripts/generate_gateway_release_notes.sh $$ARGS
