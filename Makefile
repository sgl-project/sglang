.PHONY: check-deps install-deps format update help

# Show help for each target
help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check-deps: ## Check and install required Python formatting dependencies
	@command -v isort >/dev/null 2>&1 || (echo "Installing isort..." && pip install isort)
	@command -v black >/dev/null 2>&1 || (echo "Installing black..." && pip install black)

install-deps: ## Install Python formatting tools (isort and black)
	pip install isort black

format: check-deps ## Format modified Python files using isort and black
	@echo "Formatting modified Python files..."
	git diff --name-only --diff-filter=M | grep '\.py$$' | xargs -I {} sh -c 'isort {} && black {}'

FILES_TO_UPDATE = docker/Dockerfile.rocm \
                 python/pyproject.toml \
                 python/pyproject_other.toml \
                 python/sglang/version.py \
                 docs/developer_guide/setup_github_runner.md \
                 docs/get_started/install.md \
                 docs/platforms/amd_gpu.md \
                 docs/platforms/ascend_npu.md \
				 docs/platforms/cpu_server.md \
				 docs/platforms/xpu.md \
				 benchmark/deepseek_v3/README.md

update: ## Update version numbers across project files. Usage: make update <new_version>
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Version required. Usage: make update <new_version>"; \
		exit 1; \
	fi
	@OLD_VERSION=$$(grep "version" python/sglang/version.py | cut -d '"' -f2); \
	NEW_VERSION=$(filter-out $@,$(MAKECMDGOALS)); \
	echo "Updating version from $$OLD_VERSION to $$NEW_VERSION"; \
	for file in $(FILES_TO_UPDATE); do \
		if [ "$(shell uname)" = "Darwin" ]; then \
			sed -i '' -e "s/$$OLD_VERSION/$$NEW_VERSION/g" $$file; \
		else \
			sed -i -e "s/$$OLD_VERSION/$$NEW_VERSION/g" $$file; \
		fi \
	done; \
	echo "Version update complete"

%:
	@:
