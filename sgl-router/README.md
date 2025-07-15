# SGLang Router

SGLang router is a standalone Rust module that enables data parallelism across SGLang instances, providing high-performance request routing and advanced load balancing. The router supports multiple load balancing algorithms including cache-aware, power of two, random, and round robin, and acts as a specialized load balancer for prefill-decode disaggregated serving architectures.

## Documentation

- **User Guide**: [docs.sglang.ai/router/router.html](https://docs.sglang.ai/router/router.html)

## Quick Start

### Prerequisites

**Rust and Cargo:**
```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

**Python with pip installed**

### Installation

#### Option A: Build and Install Wheel (Recommended)
```bash
# Install build dependencies
pip install setuptools-rust wheel build

# Build the wheel package
python -m build

# Install the generated wheel
pip install dist/*.whl

# One-liner for development (rebuild + install)
python -m build && pip install --force-reinstall dist/*.whl
```

#### Option B: Development Mode
```bash
pip install -e .
```

⚠️ **Warning**: Editable installs may suffer performance degradation. Use wheel builds for performance testing.

### Basic Usage

```bash
# Build Rust components
cargo build

# Launch router with worker URLs
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000
```

## Configuration

### Logging

Enable structured logging with optional file output:

```python
from sglang_router import Router

# Console logging (default)
router = Router(worker_urls=["http://worker1:8000", "http://worker2:8000"])

# File logging enabled
router = Router(
    worker_urls=["http://worker1:8000", "http://worker2:8000"],
    log_dir="./logs"  # Daily log files created here
)
```

Set log level with `--log-level` flag ([documentation](https://docs.sglang.ai/backend/server_arguments.html#logging)).

### Metrics

Prometheus metrics endpoint available at `127.0.0.1:29000` by default.

```bash
# Custom metrics configuration
python -m sglang_router.launch_router \
    --worker-urls http://localhost:8080 http://localhost:8081 \
    --prometheus-host 0.0.0.0 \
    --prometheus-port 9000
```

## Advanced Features

### Kubernetes Service Discovery

Automatic worker discovery and management in Kubernetes environments.

#### Basic Service Discovery

```bash
python -m sglang_router.launch_router \
    --service-discovery \
    --selector app=sglang-worker role=inference \
    --service-discovery-namespace default
```

#### PD (Prefill-Decode) Mode

For disaggregated prefill/decode routing:

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --service-discovery \
    --prefill-selector app=sglang component=prefill \
    --decode-selector app=sglang component=decode \
    --service-discovery-namespace sglang-system
```

#### Kubernetes Pod Configuration

**Prefill Server Pod:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sglang-prefill-1
  labels:
    app: sglang
    component: prefill
  annotations:
    sglang.ai/bootstrap-port: "9001"  # Optional: Bootstrap port
spec:
  containers:
  - name: sglang
    image: lmsys/sglang:latest
    ports:
    - containerPort: 8000  # Main API port
    - containerPort: 9001  # Optional: Bootstrap port
```

**Decode Server Pod:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sglang-decode-1
  labels:
    app: sglang
    component: decode
spec:
  containers:
  - name: sglang
    image: lmsys/sglang:latest
    ports:
    - containerPort: 8000
```

#### RBAC Configuration

**Namespace-scoped (recommended):**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sglang-router
  namespace: sglang-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: sglang-system
  name: sglang-router
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-router
  namespace: sglang-system
subjects:
- kind: ServiceAccount
  name: sglang-router
  namespace: sglang-system
roleRef:
  kind: Role
  name: sglang-router
  apiGroup: rbac.authorization.k8s.io
```

#### Complete PD Example

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --service-discovery \
    --prefill-selector app=sglang component=prefill environment=production \
    --decode-selector app=sglang component=decode environment=production \
    --service-discovery-namespace production \
    --host 0.0.0.0 \
    --port 8080 \
    --prometheus-host 0.0.0.0 \
    --prometheus-port 9090
```

### Command Line Arguments Reference

#### Service Discovery
- `--service-discovery`: Enable Kubernetes service discovery
- `--service-discovery-port`: Port for worker URLs (default: 8000)
- `--service-discovery-namespace`: Kubernetes namespace to watch
- `--selector`: Label selectors for regular mode (format: `key1=value1 key2=value2`)

#### PD Mode
- `--pd-disaggregation`: Enable Prefill-Decode disaggregated mode
- `--prefill`: Initial prefill server (format: `URL BOOTSTRAP_PORT`)
- `--decode`: Initial decode server URL
- `--prefill-selector`: Label selector for prefill pods
- `--decode-selector`: Label selector for decode pods
- `--policy`: Routing policy (`cache_aware`, `random`, `power_of_two`)

## Development

### Build Process

```bash
# Build Rust project
cargo build

# Build Python binding (see Installation section above)
```

**Note**: When modifying Rust code, you must rebuild the wheel for changes to take effect.

### Troubleshooting

**VSCode Rust Analyzer Issues:**
Set `rust-analyzer.linkedProjects` to the absolute path of `Cargo.toml`:

```json
{
  "rust-analyzer.linkedProjects": ["/workspaces/sglang/sgl-router/Cargo.toml"]
}
```

### CI/CD Pipeline

The continuous integration pipeline includes comprehensive testing, benchmarking, and publishing:

#### Build & Test
1. **Build Wheels**: Uses `cibuildwheel` for manylinux x86_64 packages
2. **Build Source Distribution**: Creates source distribution for pip fallback
3. **Rust HTTP Server Benchmarking**: Performance testing of router overhead
4. **Basic Inference Testing**: End-to-end validation through the router
5. **PD Disaggregation Testing**: Benchmark and sanity checks for prefill-decode load balancing

#### Publishing
- **PyPI Publishing**: Wheels and source distributions are published only when the version changes in `pyproject.toml`
- **Container Images**: Docker images published using `/docker/Dockerfile.router`

## Features

- **High Performance**: Rust-based routing with connection pooling and optimized request handling
- **Advanced Load Balancing**: Multiple algorithms including:
  - **Cache-Aware**: Intelligent routing based on cache locality for optimal performance
  - **Power of Two**: Chooses the less loaded of two randomly selected workers
  - **Random**: Distributes requests randomly across available workers
  - **Round Robin**: Sequential distribution across workers in rotation
- **Prefill-Decode Disaggregation**: Specialized load balancing for separated prefill and decode servers
- **Service Discovery**: Automatic Kubernetes worker discovery and health management
- **Monitoring**: Comprehensive Prometheus metrics and structured logging
- **Scalability**: Handles thousands of concurrent connections with efficient resource utilization
