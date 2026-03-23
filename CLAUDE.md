# Claude Project Guide - SGLang Optimization

## Project Context
- **Project**: SGLang (High-performance LLM serving framework).
- **Focus**: Inference optimization, CUDA kernels, and distributed serving.
- **Hardware Target**: NVIDIA H20 GPU.

## Environment & Constraints (Critical)
- **Local Environment**: This is where Claude Code runs. It has **NO GPU** access and is restricted to file editing and static analysis.
- **Remote Environment**: A separate, offline GPU cluster where actual execution and benchmarking happen.
- **File Sync**: A distributed file system is in place. Files saved locally are automatically synced to the remote GPU environment.
- **Execution Rule**:
    - **DO NOT** attempt to run any scripts that require GPU (e.g., `python -m sglang.launch_server`, `pytest` with GPU tests, or CUDA profiling).
    - **DO** use shell commands for file system operations (`ls`, `grep`, `find`, `cat`).
    - **DO** perform static code analysis, type checking, and logic refactoring.
