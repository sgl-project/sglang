# SGLang Router Makefile
# Provides convenient shortcuts for common development tasks

.PHONY: help bench bench-quick bench-baseline bench-compare test build clean

help: ## Show this help message
	@echo "SGLang Router Development Commands"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

build: ## Build the project in release mode
	@echo "Building SGLang Router..."
	@cargo build --release

test: ## Run all tests
	@echo "Running tests..."
	@cargo test

bench: ## Run full benchmark suite
	@echo "Running full benchmarks..."
	@python3 scripts/run_benchmarks.py

bench-quick: ## Run quick benchmarks only
	@echo "Running quick benchmarks..."
	@python3 scripts/run_benchmarks.py --quick

bench-baseline: ## Save current performance as baseline
	@echo "Saving performance baseline..."
	@python3 scripts/run_benchmarks.py --save-baseline main

bench-compare: ## Compare with saved baseline
	@echo "Comparing with baseline..."
	@python3 scripts/run_benchmarks.py --compare-baseline main

bench-ci: ## Run benchmarks suitable for CI (quick mode)
	@echo "Running CI benchmarks..."
	@python3 scripts/run_benchmarks.py --quick

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
	@cargo clippy

fmt: ## Format code with rustfmt
	@echo "Formatting code..."
	@cargo fmt

# Development workflow shortcuts
dev-setup: build test ## Set up development environment
	@echo "Development environment ready!"

pre-commit: fmt check test bench-quick ## Run pre-commit checks
	@echo "Pre-commit checks passed!"

# Benchmark analysis shortcuts
bench-report: ## Open benchmark HTML report
	@if [ -f "target/criterion/request_processing/report/index.html" ]; then \
		echo "Opening benchmark report..."; \
		if command -v xdg-open >/dev/null 2>&1; then \
			xdg-open target/criterion/request_processing/report/index.html; \
		elif command -v open >/dev/null 2>&1; then \
			open target/criterion/request_processing/report/index.html; \
		else \
			echo "Please open target/criterion/request_processing/report/index.html in your browser"; \
		fi \
	else \
		echo "No benchmark report found. Run 'make bench' first."; \
	fi

bench-clean: ## Clean benchmark results
	@echo "Cleaning benchmark results..."
	@rm -rf target/criterion

# Performance monitoring
perf-monitor: ## Run continuous performance monitoring
	@echo "Starting performance monitoring..."
	@if command -v watch >/dev/null 2>&1; then \
		watch -n 300 'make bench-quick'; \
	else \
		echo "Warning: 'watch' command not found. Install it or run 'make bench-quick' manually."; \
	fi
