# SGLang Development Commands

## Installation
```bash
# Install with pip/uv
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.5.0rc2"

# Install from source for development
git clone -b v0.5.0rc2 https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[dev]"
```

## Code Formatting and Linting
```bash
# Format modified Python files
make format

# Run pre-commit hooks manually
pre-commit run --all-files

# Individual formatting tools
black <file.py>
isort <file.py>
```

## Testing
```bash
# Run pytest (from test directory)
cd test
pytest <test_file.py>

# Common test patterns
pytest test/srt/test_*.py
pytest test/lang/test_*.py
```

## Running the Server
```bash
# Launch SGLang server
python -m sglang.launch_server --model-path <model> --host 0.0.0.0 --port 30000

# With Docker
docker run --gpus all --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<token>" --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path <model> --host 0.0.0.0 --port 30000
```

## Utility Commands
```bash
# Kill all SGLang processes
bash scripts/killall_sglang.sh

# Update version across project
make update <new_version>

# Check environment
python -m sglang.check_env
```

## Building Components
```bash
# Build sgl-router (Rust component)
cd sgl-router
python -m build
pip install dist/*.whl

# Development mode
pip install -e .
```