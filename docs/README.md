# SGLang Documentation
We recommend new contributors start from writing documentation, which helps you quickly understand SGLang codebase. Most documentation files are located under the `docs/` folder. We prefer **Jupyter Notebooks** over Markdown so that all examples can be executed and validated by our docs CI pipeline.

## Docs Workflow

### Install Dependency

```bash
pip install -r requirements.txt
```

### Update Documentation

Update your Jupyter notebooks in the appropriate subdirectories under `docs/`. If you add new files, remember to update `index.rst` (or relevant `.rst` files) accordingly.

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.

```bash
# 1) Compile all Jupyter notebooks
make compile

# 2) Compile and Preview documentation locally
# Open your browser at the displayed port to view the docs
bash serve.sh

# 3) Clean notebook outputs
# nbstripout removes notebook outputs so your PR stays clean
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;

# 4) Pre-commit checks and create a PR
# After these checks pass, push your changes and open a PR on your branch
pre-commit run --all-files
```
---

### **Port Allocation and CI Efficiency**

**To launch and kill the server:**

```python
from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

server_process, port = launch_server_cmd(
    """
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
 --host 0.0.0.0
"""
)

wait_for_server(f"http://localhost:{port}")

# Terminate Server
terminate_process(server_process)
```

**To launch and kill the engine:**

```python
# Launch Engine
import sglang as sgl
import asyncio
from sglang.test.test_utils import is_in_ci

if is_in_ci():
    import patch

llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

# Terminalte Engine
llm.shutdown()
```

### **Why this approach?**

- **Dynamic Port Allocation**: Avoids port conflicts by selecting an available port at runtime, enabling multiple server instances to run in parallel.
- **Optimized for CI**: The `patch` version of `launch_server_cmd` and `sgl.Engine()` in CI environments helps manage GPU memory dynamically, preventing conflicts and improving test parallelism.
- **Better Parallel Execution**: Ensures smooth concurrent tests by avoiding fixed port collisions and optimizing memory usage.

### **Model Selection**

For demonstrations in the docs, **prefer smaller models** to reduce memory consumption and speed up inference. Running larger models in CI can lead to instability due to memory constraints.

### **Prompt Alignment Example**

When designing prompts, ensure they align with SGLang’s structured formatting. For example:

```python
prompt = """You are an AI assistant. Answer concisely and accurately.

User: What is the capital of France?
Assistant: The capital of France is Paris."""
```

This keeps responses aligned with expected behavior and improves reliability across different files.
