# Contribution Guide

Welcome to **SGLang**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR). Whether you’re fixing a small bug or developing a major feature, we encourage following these steps for a smooth contribution process.

## Setting Up & Building from Source

### Fork and Clone the Repository

**Note**: SGLang does **not** accept PRs on the main repo. Please fork the repository under your GitHub account, then clone your fork locally.

```bash
git clone https://github.com/<your_user_name>/sglang.git
cd sglang
```

### Install Dependencies & Build

Refer to [Install SGLang](https://sgl-project.github.io/start/install.html) documentation for more details on setting up the necessary dependencies.

Install correct version of flashinfer according to your PyTorch and CUDA versions.
```bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Version:', torch.version.cuda)"
```

Below is an example on PyTorch 2.4 with CUDA 12.4:
```bash
pip install --upgrade pip
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# Install the lateset SGLang on your own fork repo
cd sglang/python
pip install .
```

## Code Formatting with Pre-Commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code style checks. Before pushing your changes, please run:

```bash
pip3 install pre-commit
cd sglang
pre-commit run --all-files
```

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.

## Writing Documentation & Running Docs CI

Most documentation files are located under the `docs/` folder. We prefer **Jupyter Notebooks** over Markdown so that all examples can be executed and validated by our docs CI pipeline.

### Docs Workflow

Add or update your Jupyter notebooks in the appropriate subdirectories under `docs/`. If you add new files, remember to update `index.rst` (or relevant `.rst` files) accordingly.

```bash
# 1) Compile all Jupyter notebooks
make compile

# 2) Generate static HTML
make html

# 3) Preview documentation locally
bash serve.sh
# Open your browser at the displayed port to view the docs

# 4) Clean notebook outputs
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;
# nbstripout removes notebook outputs so your PR stays clean

# 5) Pre-commit checks and create a PR
pre-commit run --all-files
# After these checks pass, push your changes and open a PR on your branch
```


If you need to run and shut up a SGLang server or engine, following these examples:

1. Launch and close Sever:

```python
#Launch Sever

from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)

server_process = execute_shell_command(
    "python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000 --host 0.0.0.0"
)

wait_for_server("http://localhost:30000")

# Terminate Sever

terminate_process(server_process)
```
2. Launch Engine and close Engine

```python
# Launch Engine

import sglang as sgl
import asyncio

llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

# Terminalte Engine
llm.shutdown()
```


## Running Unit Tests & Adding to CI

SGLang uses Python’s built-in [unittest](https://docs.python.org/3/library/unittest.html) framework. You can run tests either individually or in suites.

### Test Backend Runtime

```bash
cd sglang/test/srt

# Run a single file
python3 test_srt_endpoint.py

# Run a single test
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_simple_decode

# Run a suite with multiple files
python3 run_suite.py --suite minimal
```

### Test Frontend Language

```bash
cd sglang/test/lang
export OPENAI_API_KEY=sk-*****

# Run a single file
python3 test_openai_backend.py

# Run a single test
python3 -m unittest test_openai_backend.TestOpenAIBackend.test_few_shot_qa

# Run a suite with multiple files
python3 run_suite.py --suite minimal
```

### Adding or Updating Tests in CI

- Create new test files under `test/srt` or `test/lang` depending on the type of test.
- Ensure they are referenced in the respective `run_suite.py` (e.g., `test/srt/run_suite.py` or `test/lang/run_suite.py`) so they’re picked up in CI.
- In CI, all tests run automatically. You may modify the workflows in [`.github/workflows/`](https://github.com/sgl-project/sglang/tree/main/.github/workflows) to add custom test groups or extra checks.

### Writing Elegant Test Cases

- Examine existing tests in [sglang/test](https://github.com/sgl-project/sglang/tree/main/test) for practical examples.
- Keep each test function focused on a single scenario or piece of functionality.
- Give tests descriptive names reflecting their purpose.
- Use robust assertions (e.g., assert, unittest methods) to validate outcomes.
- Clean up resources to avoid side effects and preserve test independence.


## Tips for Newcomers

If you want to contribute but don’t have a specific idea in mind, pick issues labeled [“good first issue” or “help wanted”](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22). These tasks typically have lower complexity and provide an excellent introduction to the codebase. Also check out this [code walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through) for a deeper look into SGLang’s workflow.

If you have any questions or want to start a discussion, please feel free to ask in our [Slack channel](https://join.slack.com/t/sgl-fru7574/shared_invite/zt-2um0ad92q-LkU19KQTxCGzlCgRiOiQEw).

Thank you for your interest in SGLang—**happy coding**!
