# SGLang Router (Experimental)

SGLang router is a standalone module implemented in Rust to achieve data parallelism across SGLang instances.

## Installation

WIP. Ideally just

```bash
pip install sglang-router
```

## Development

### Rust

1. Install Rust

```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

2. Build the router

```bash
# Navigate to the rust directory
cd ./rust

# Build the project
cargo build

# Verify the binary works correctly
./target/debug/router --help
```

The help command will show available options:
```
Usage: router [OPTIONS]

Options:
      --host <HOST>                [default: 127.0.0.1]
      --port <PORT>                [default: 3001]
      --worker-urls <WORKER_URLS>
      --policy <POLICY>            [default: round_robin] [possible values: round_robin, random]
  -h, --help                       Print help
  -V, --version                    Print version
```

### Python Binding

1. Create a virtual environment

```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

2. Install python dependencies

```bash
$ pip install maturin
$ pip install patchelf
```

3. Install rust python binding

```bash
$ maturin develop
🔗 Found pyo3 bindings
🐍 Found CPython 3.10 at /home/jobuser/resources/sglang/rust/.venv/bin/python
📡 Using build options bindings from pyproject.toml
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.11s
📦 Built wheel for CPython 3.10 to /tmp/.tmpJb65sc/sglang_router-0.0.0-cp310-cp310-linux_x86_64.whl
✏️  Setting installed package as editable
🛠 Installed sglang_router-0.0.0
```

## Usage

1. Launch worker instances
```bash
# Launch first worker on GPU 0
export CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 30000

# Launch second worker on GPU 1
export CUDA_VISIBLE_DEVICES=1
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 30002
```

2. Launch router and connect to workers
```bash
./target/debug/router --worker-urls http://127.0.0.1:30000,http://127.0.0.1:30002
```
