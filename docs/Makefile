# Minimal Makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build
PORT          ?= 8003

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Additional targets:"
	@echo "  serve       to build and serve documentation with auto-build and live reload"
	@echo "  markdown    to build Markdown documentation for downstream consumption"
	@echo "  export      to export source documentation files (md/rst/ipynb)"
	@echo "  export-all  to run compile, export, and markdown in sequence"

# Compile Notebook files and record execution time
compile:
	@set -e; \
	echo "Starting Notebook compilation..."; \
	mkdir -p logs; \
	echo "Notebook execution timings:" > logs/timing.log; \
	START_TOTAL=$$(date +%s); \
	find $(SOURCEDIR) -path "*/_build/*" -prune -o -name "*.ipynb" -print0 | \
		parallel -0 -j3 --halt soon,fail=1 ' \
		NB_NAME=$$(basename {}); \
		START_TIME=$$(date +%s); \
		retry --delay=0 --times=2 -- \
			jupyter nbconvert --to notebook --execute --inplace "{}" \
			--ExecutePreprocessor.timeout=600 \
			--ExecutePreprocessor.kernel_name=python3; \
		RET_CODE=$$?; \
		END_TIME=$$(date +%s); \
		ELAPSED_TIME=$$((END_TIME - START_TIME)); \
		echo "$${NB_NAME}: $${ELAPSED_TIME}s" >> logs/timing.log; \
		exit $$RET_CODE' || exit 1; \
	END_TOTAL=$$(date +%s); \
	TOTAL_ELAPSED=$$((END_TOTAL - START_TOTAL)); \
	echo "---------------------------------" >> logs/timing.log; \
	echo "Total execution time: $${TOTAL_ELAPSED}s" >> logs/timing.log; \
	echo "All Notebook execution timings:" && cat logs/timing.log

# Serve documentation with auto-build and live reload
serve:
	@echo "Starting auto-build server at http://0.0.0.0:$(PORT)"
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--host 0.0.0.0 \
		--port $(PORT) \
		--watch $(SOURCEDIR) \
		--re-ignore ".*\.(ipynb_checkpoints|pyc|pyo|pyd|git)"

# Build Markdown documentation for downstream consumption (RAG, LLM, etc.)
markdown:
	@echo "Building Markdown documentation..."
	@$(SPHINXBUILD) -M markdown "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Markdown documentation built in $(BUILDDIR)/markdown/"

# Export source documentation files to a structured directory
export:
	@echo "Exporting documentation source files..."
	@mkdir -p $(BUILDDIR)/export
	@find $(SOURCEDIR) -path "$(BUILDDIR)" -prune -o \( -name "*.md" -o -name "*.rst" -o -name "*.ipynb" \) -type f -print0 | \
		while IFS= read -r -d '' f; do \
			rel=$${f#./}; \
			mkdir -p "$(BUILDDIR)/export/$$(dirname "$$rel")"; \
			cp "$$f" "$(BUILDDIR)/export/$$rel"; \
		done
	@echo "Documentation exported to $(BUILDDIR)/export/"
	@echo "Files exported: $$(find $(BUILDDIR)/export -type f | wc -l | tr -d ' ')"

# Full export pipeline: compile notebooks, export sources, build markdown
export-all: compile export markdown
	@echo ""
	@echo "==============================================="
	@echo "Full documentation export complete!"
	@echo "==============================================="
	@echo "  - Executed notebooks: source files (in-place)"
	@echo "  - Source export:      $(BUILDDIR)/export/"
	@echo "  - Markdown output:    $(BUILDDIR)/markdown/"
	@echo ""

.PHONY: help Makefile compile clean serve markdown export export-all

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	find . -name "*.ipynb" -exec nbstripout {} \;
	rm -rf $(BUILDDIR)
	rm -rf logs
