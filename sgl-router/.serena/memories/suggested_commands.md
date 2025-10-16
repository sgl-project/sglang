# Suggested Commands
- `cargo build --release` – Build the Rust router binary.
- `cargo test` – Run Rust unit and integration tests.
- `cargo check` / `cargo clippy` – Static analysis and linting (Make target `make check`).
- `cargo fmt` – Format Rust sources (available via `make fmt`).
- `make test` / `make build` – Convenience wrappers for cargo tasks.
- `python3 scripts/run_benchmarks.py [--quick]` – Execute performance benchmarks (`make bench`, `make bench-quick`).
- `python3 -m build && pip install dist/*.whl` – Build/install Python package when iterating on bindings.
- `python3 -m sglang_router.launch_router ...` – Launch router via Python utilities (see README for flags).
- `./target/release/sglang-router ...` – Run compiled router binary with desired options.