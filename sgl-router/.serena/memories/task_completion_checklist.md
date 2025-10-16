# Task Completion Checklist
- Run relevant cargo commands (`cargo fmt`, `cargo clippy`, `cargo test`) to ensure formatting, linting, and tests pass.
- For tokenizer or routing changes, consider running targeted benchmarks (`python3 scripts/run_benchmarks.py --quick`) if performance may be affected.
- Update documentation or README snippets when altering CLI flags, tokenization features, or deployment instructions.
- Verify Python bindings (`python3 -m build && pip install dist/*.whl`) if changes impact the package interface.
- Summarize changes and provide follow-up steps (e.g., benchmark reruns, deployment considerations) when handing off work.