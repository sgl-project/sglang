# SGLang Router Makefile
# Provides convenient shortcuts for common development tasks

.PHONY: help bench bench-quick bench-baseline bench-compare bench-load bench-policies test build clean

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
	@cargo bench

bench-quick: ## Run quick benchmarks only
	@echo "Running quick benchmarks..."
	@cargo bench -- --sample-size 10 --measurement-time 2

bench-baseline: ## Save current performance as baseline
	@echo "Saving performance baseline..."
	@python3 scripts/run_benchmarks.py --save-baseline main

bench-compare: ## Compare with saved baseline
	@echo "Comparing with baseline..."
	@python3 scripts/run_benchmarks.py --compare-baseline main

bench-ci: ## Run benchmarks suitable for CI (quick mode)
	@echo "Running CI benchmarks..."
	@cargo bench -- --sample-size 10 --measurement-time 2

bench-load: ## Run load test benchmarks
	@echo "Running load test benchmarks..."
	@cargo run --release --bin load_test -- 1000 4 50 --no-stream

bench-policies: ## Test different load balancing policies
	@echo "Testing load balancing policies..."
	@cargo run --release --bin load_test -- 1000 4 50 --policy random --no-stream
	@cargo run --release --bin load_test -- 1000 4 50 --policy round_robin --no-stream
	@cargo run --release --bin load_test -- 1000 4 50 --policy power_of_two --no-stream
	@cargo run --release --bin load_test -- 1000 4 50 --policy cache_aware --no-stream

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
	@if [ -d "target/criterion" ]; then \
		echo "Opening benchmark report..."; \
		if command -v xdg-open >/dev/null 2>&1; then \
			xdg-open target/criterion/report/index.html 2>/dev/null || xdg-open target/criterion/; \
		elif command -v open >/dev/null 2>&1; then \
			open target/criterion/report/index.html 2>/dev/null || open target/criterion/; \
		else \
			echo "Please open target/criterion/ in your browser"; \
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
