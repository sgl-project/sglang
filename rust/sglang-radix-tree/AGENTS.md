# Radix tree validation

- Run the native suite from this crate as documented in `README.md`; Python
  extension tests do not compile the native `src/tests/` modules.
- For torch 2.13, use the documented `torch_2_13_compat.h` CXX include and put
  the installed torch library directory on `LD_LIBRARY_PATH` for native tests.
- Install the configured Rust toolchain once before running checks. Concurrent
  first-time rustup installs from pre-commit and cargo can race and leave an
  incomplete toolchain. Run formatting and clippy after installation finishes.
- For external linker component changes, test both Python and Rust tree backends
  and keep offload descriptors consistent across implementations. A checkpoint
  belongs to the cached prefix endpoint, not to every token page.
