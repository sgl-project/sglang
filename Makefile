.PHONY: check-deps install-deps format

check-deps:
	@command -v isort >/dev/null 2>&1 || (echo "Installing isort..." && pip install isort)
	@command -v black >/dev/null 2>&1 || (echo "Installing black..." && pip install black)

install-deps:
	pip install isort black

format: check-deps
	@echo "Formatting modified Python files..."
	git diff --name-only --diff-filter=M | grep '\.py$$' | xargs -I {} sh -c 'isort {} && black {}'
