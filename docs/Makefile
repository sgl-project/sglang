# Minimal makefile for Sphinx documentation
#

# You can set these variables from the terminal, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# New target to compile Markdown and Jupyter Notebook files
compile:
	find $(SOURCEDIR) -path "*/_build/*" -prune -o -name "*.ipynb" -print | while read nb; do \
		if [ -f "$$nb" ]; then \
			echo "Executing $$nb"; \
			jupyter nbconvert --to notebook --execute --inplace "$$nb" \
				--ExecutePreprocessor.timeout=600 \
				--ExecutePreprocessor.kernel_name=python3; \
		fi; \
	done

.PHONY: help Makefile compile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*
