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

# Convert Notebook files to Markdown artifacts (no execution)
markdown:
	@set -e; \
	echo "Exporting docs to Markdown..."; \
	mkdir -p "$(BUILDDIR)/html/markdown"; \
	\
	# 1) Copy .md and .rst files as-is; additionally convert .rst -> .md \
	find $(SOURCEDIR) -path "*/_build/*" -prune -o \( -name "*.md" -o -name "*.rst" \) -print0 | \
		parallel -0 -j3 --halt soon,fail=1 ' \
		SRC="{}"; \
		REL_DIR=$$(dirname "$$SRC"); \
		OUT_DIR="$(BUILDDIR)/html/markdown/$$REL_DIR"; \
		mkdir -p "$$OUT_DIR"; \
		cp -f "$$SRC" "$$OUT_DIR/"; \
		case "$$SRC" in \
		  *.rst) \
			BASE=$$(basename "$$SRC" .rst); \
			pandoc -f rst -t gfm "$$SRC" -o "$$OUT_DIR/$$BASE.md" ;; \
		esac \
		' || exit 1; \
	\
	# 2) Convert .ipynb -> .md \
	find $(SOURCEDIR) -path "*/_build/*" -prune -o -name "*.ipynb" -print0 | \
		parallel -0 -j3 --halt soon,fail=1 ' \
		NB_SRC="{}"; \
		REL_DIR=$$(dirname "$$NB_SRC"); \
		NB_NAME=$$(basename "$$NB_SRC"); \
		NB_BASE=$${NB_NAME%.ipynb}; \
		OUT_DIR="$(BUILDDIR)/html/markdown/$$REL_DIR"; \
		mkdir -p "$$OUT_DIR"; \
		jupyter nbconvert --to markdown "$$NB_SRC" \
			--output "$$NB_BASE.md" \
			--output-dir "$$OUT_DIR" \
			>/dev/null; \
		' || exit 1; \
	\
	echo "Markdown artifacts written to: $(BUILDDIR)/html/markdown"



# Serve documentation with auto-build and live reload
serve:
	@echo "Starting auto-build server at http://0.0.0.0:$(PORT)"
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--host 0.0.0.0 \
		--port $(PORT) \
		--watch $(SOURCEDIR) \
		--re-ignore ".*\.(ipynb_checkpoints|pyc|pyo|pyd|git)"

.PHONY: help Makefile compile clean serve

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	find . -name "*.ipynb" -exec nbstripout {} \;
	rm -rf $(BUILDDIR)
	rm -rf logs
