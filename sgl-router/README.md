# SGLang Router

SGLang router is a standalone module implemented in Rust to achieve data parallelism across SGLang instances.

## User docs

Please check https://docs.sglang.ai/router/router.html

## Developer docs

### Prerequisites

- Rust and Cargo installed

```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

- Python with pip installed


### Build Process

#### 1. Build Rust Project

```bash
$ cargo build
```

#### 2. Build Python Binding

##### Option A: Build and Install Wheel
1. Build the wheel package:
```bash
$ pip install setuptools-rust wheel build
$ python -m build
```

2. Install the generated wheel:
```bash
$ pip install <path-to-wheel>
```

If you want one handy command to do build + install for every change you make:

```bash
$ python -m build && pip install --force-reinstall dist/*.whl
```

##### Option B: Development Mode

For development purposes, you can install the package in editable mode:

Warning: Using editable python binding can suffer from performance degradation!! Please build a fresh wheel for every update if you want to test performance.

```bash
$ pip install -e .
```

**Note:** When modifying Rust code, you must rebuild the wheel for changes to take effect.

### Logging

The SGL Router includes structured logging with console output by default. To enable log files:

```python
# Enable file logging when creating a router
router = Router(
    worker_urls=["http://worker1:8000", "http://worker2:8000"],
    log_dir="./logs"  # Daily log files will be created here
)
```

Use the `--verbose` flag with the CLI for more detailed logs.

### Metrics

SGL Router exposes a Prometheus HTTP scrape endpoint for monitoring, which by default listens at 127.0.0.1:29000.

To change the endpoint to listen on all network interfaces and set the port to 9000, configure the following options when launching the router:
```
python -m sglang_router.launch_router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --prometheus-host 0.0.0.0 \
  --prometheus-port 9000
```

### Kubernetes Service Discovery

SGL Router supports automatic service discovery for worker nodes in Kubernetes environments. When enabled, the router will automatically:

- Discover and add worker pods with matching labels
- Remove unhealthy or deleted worker pods
- Dynamically adjust the worker pool based on pod health and availability

#### Command Line Usage

```bash
python -m sglang_router.launch_router \
    --service-discovery \
    --selector app=sglang-worker role=inference \
    --service-discovery-port 8000 \
    --service-discovery-namespace default
```

#### Service Discovery Arguments

- `--service-discovery`: Enable Kubernetes service discovery feature
- `--selector`: One or more label key-value pairs for pod selection (format: key1=value1 key2=value2)
- `--service-discovery-port`: Port to use when generating worker URLs (default: 80)
- `--service-discovery-namespace`: Optional. Kubernetes namespace to watch for pods. If not provided, watches all namespaces (requires cluster-wide permissions)

#### RBAC Requirements

When using service discovery, you must configure proper Kubernetes RBAC permissions:

- **If using namespace-scoped discovery** (with `--service-discovery-namespace`):
  Set up a ServiceAccount, Role, and RoleBinding

- **If watching all namespaces** (without specifying namespace):
  Set up a ServiceAccount, ClusterRole, and ClusterRoleBinding with permissions to list/watch pods at the cluster level

### Troubleshooting

1. If rust analyzer is not working in VSCode, set `rust-analyzer.linkedProjects` to the absolute path of `Cargo.toml` in your repo. For example:

```json
{
  "rust-analyzer.linkedProjects":  ["/workspaces/sglang/sgl-router/Cargo.toml"]
}
```

### CI/CD Setup

The continuous integration pipeline consists of three main steps:

#### 1. Build Wheels
- Uses `cibuildwheel` to create manylinux x86_64 packages
- Compatible with major Linux distributions (Ubuntu, CentOS, etc.)
- Additional configurations can be added to support other OS/architectures
- Reference: [cibuildwheel documentation](https://cibuildwheel.pypa.io/en/stable/)

#### 2. Build Source Distribution
- Creates a source distribution containing the raw, unbuilt code
- Enables `pip` to build the package from source when prebuilt wheels are unavailable

#### 3. Publish to PyPI
- Uploads both wheels and source distribution to PyPI

The CI configuration is based on the [tiktoken workflow](https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/.github/workflows/build_wheels.yml#L1).
