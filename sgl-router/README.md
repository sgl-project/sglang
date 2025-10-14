# SGLang Router

SGLang router is a standalone Rust module that enables data parallelism across SGLang instances, providing high-performance request routing and advanced load balancing. The router supports multiple load balancing algorithms including cache-aware, power of two, random, and round robin, and acts as a specialized load balancer for prefill-decode disaggregated serving architectures.

## Documentation

- **User Guide**: [docs.sglang.ai/advanced_features/router.html](https://docs.sglang.ai/advanced_features/router.html)

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
# Currently broken
pip install -e .
```

⚠️ **Warning**: Editable installs may suffer performance degradation. Use wheel builds for performance testing.

### Basic Usage

```bash
# Build Rust components
cargo build
```

#### Using the Rust Binary Directly (Alternative to Python)
```bash
# Build the Rust binary
cargo build --release

# Launch router with worker URLs in regular mode
./target/release/sglang-router \
    --worker-urls http://worker1:8000 http://worker2:8000

# Or use cargo run
cargo run --release -- \
    --worker-urls http://worker1:8000 http://worker2:8000
```

#### Launch Router with Python (Original Method)
```bash
# Launch router with worker URLs
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000
```

#### Launch Router with Worker URLs in prefill-decode mode
```bash
# Note that the prefill and decode URLs must be provided in the following format:
# http://<ip>:<port> for  decode nodes
# http://<ip>:<port> bootstrap-port for  prefill nodes, where bootstrap-port is optional

# Using Rust binary directly
./target/release/sglang-router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 9002 \
    --prefill http://127.0.0.3:30003 9003 \
    --prefill http://127.0.0.4:30004 9004 \
    --decode http://127.0.0.5:30005 \
    --decode http://127.0.0.6:30006 \
    --decode http://127.0.0.7:30007 \
    --host 0.0.0.0 \
    --port 8080

# Or using Python launcher
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 9002 \
    --prefill http://127.0.0.3:30003 9003 \
    --prefill http://127.0.0.4:30004 9004 \
    --decode http://127.0.0.5:30005 \
    --decode http://127.0.0.6:30006 \
    --decode http://127.0.0.7:30007 \
    --host 0.0.0.0 \
    --port 8080
````

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

### Retries and Circuit Breakers

- Retries (regular router) are enabled by default with exponential backoff and jitter. You can tune them via CLI:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --retry-max-retries 3 \
  --retry-initial-backoff-ms 100 \
  --retry-max-backoff-ms 10000 \
  --retry-backoff-multiplier 2.0 \
  --retry-jitter-factor 0.1
```

- Circuit Breaker defaults protect workers and auto-recover. Tune thresholds/timeouts:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://localhost:8080 http://localhost:8081 \
  --cb-failure-threshold 5 \
  --cb-success-threshold 2 \
  --cb-timeout-duration-secs 30 \
  --cb-window-duration-secs 60
```

Behavior summary:
- Closed → Open after N consecutive failures (failure-threshold)
- Open → HalfOpen after timeout (timeout-duration-secs)
- HalfOpen → Closed after M consecutive successes (success-threshold)
- Any failure in HalfOpen reopens immediately

Retry predicate (regular router): retry on 408/429/500/502/503/504, otherwise return immediately. Backoff/jitter observed between attempts.

### Request ID Tracking

Track requests across distributed systems with configurable headers:

```bash
# Use custom request ID headers
python -m sglang_router.launch_router \
    --worker-urls http://localhost:8080 \
    --request-id-headers x-trace-id x-request-id
```

Default headers: `x-request-id`, `x-correlation-id`, `x-trace-id`, `request-id`

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

# With separate routing policies:
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill-policy cache_aware \
    --decode-policy power_of_two \
    --service-discovery \
    --prefill-selector app=sglang component=prefill \
    --decode-selector app=sglang component=decode \
    --service-discovery-namespace sglang-system

# in lws case, such as tp16(1 leader pod, 1 worker pod)
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --service-discovery \
    --prefill-selector app=sglang component=prefill role=leader\
    --decode-selector app=sglang component=decode role=leader\
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

### API Key Authentication

The router supports multi-level API key authentication for both the router itself and individual workers:

#### Router API Key
Protect access to the router endpoints:

```bash
python -m sglang_router.launch_router \
    --api-key "your-router-api-key" \
    --worker-urls http://worker1:8000 http://worker2:8000
```

When router API key is set, clients must include the Bearer token:
```bash
curl -H "Authorization: Bearer your-router-api-key" http://localhost:8080/v1/chat/completions
```

#### Worker API Keys
Workers can have their own API keys for authentication:

```bash
# Workers specified in --worker-urls automatically inherit the router's API key
python -m sglang_router.launch_router \
    --api-key "shared-api-key" \
    --worker-urls http://worker1:8000 http://worker2:8000
# Both workers will use "shared-api-key" for authentication

# Adding workers dynamically WITHOUT inheriting router's key
curl -X POST http://localhost:8080/add_worker?url=http://worker3:8000
# WARNING: This worker has NO API key even though router has one!

# Adding workers with specific API keys dynamically
curl -X POST http://localhost:8080/add_worker?url=http://worker3:8000&api_key=worker3-specific-key
```

#### Security Configurations

1. **No Authentication** (default):
   - Router and workers accessible without keys
   - Suitable for trusted environments

2. **Router-only Authentication**:
   - Clients need key to access router
   - Router can access workers freely

3. **Worker-only Authentication**:
   - Router accessible without key
   - Each worker requires authentication
   ```bash
   # Add workers with their API keys
   curl -X POST http://localhost:8080/add_worker?url=http://worker:8000&api_key=worker-key
   ```

4. **Full Authentication**:
   - Router requires key from clients
   - Each worker requires its own key
   ```bash
   # Start router with its key
   python -m sglang_router.launch_router --api-key "router-key"

   # Add workers with their keys
   curl -H "Authorization: Bearer router-key" \
        -X POST http://localhost:8080/add_worker?url=http://worker:8000&api_key=worker-key
   ```

#### Important Notes

- **Initial Workers**: Workers specified in `--worker-urls` automatically inherit the router's API key
- **Dynamic Workers**: When adding workers via API, you must explicitly specify their API keys - they do NOT inherit the router's key
- **Security Warning**: When adding workers without API keys while the router has one configured, a warning will be logged
- **Common Pitfall**: If router and workers use the same API key, you must still specify the key when adding workers dynamically

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
- `--policy`: Routing policy (`cache_aware`, `random`, `power_of_two`, `round_robin`)
- `--prefill-policy`: Separate routing policy for prefill nodes (optional, overrides `--policy` for prefill)
- `--decode-policy`: Separate routing policy for decode nodes (optional, overrides `--policy` for decode)

#### Authentication
- `--api-key`: API key for router authentication (clients must provide this as Bearer token)

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
