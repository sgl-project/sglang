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

Use the `--log-level` flag with the CLI to set [log level](https://docs.sglang.ai/backend/server_arguments.html#logging).

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

SGL Router supports automatic service discovery for worker nodes in Kubernetes environments. This feature works with both regular (single-server) routing and PD (Prefill-Decode) routing modes. When enabled, the router will automatically:

- Discover and add worker pods with matching labels
- Remove unhealthy or deleted worker pods
- Dynamically adjust the worker pool based on pod health and availability
- For PD mode: distinguish between prefill and decode servers based on labels

#### Regular Mode Service Discovery

For traditional single-server routing:

```bash
python -m sglang_router.launch_router \
    --service-discovery \
    --selector app=sglang-worker role=inference \
    --service-discovery-namespace default
```

#### PD Mode Service Discovery

For PD (Prefill-Decode) disaggregated routing, service discovery can automatically discover and classify pods as either prefill or decode servers based on their labels:

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --service-discovery \
    --prefill-selector app=sglang component=prefill \
    --decode-selector app=sglang component=decode \
    --service-discovery-namespace sglang-system
```

You can also specify initial prefill and decode servers and let service discovery add more:

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://prefill-1:8000 8001 \
    --decode http://decode-1:8000 \
    --service-discovery \
    --prefill-selector app=sglang component=prefill \
    --decode-selector app=sglang component=decode \
    --service-discovery-namespace sglang-system
```

#### Kubernetes Pod Configuration for PD Mode

When using PD service discovery, your Kubernetes pods need specific labels to be classified as prefill or decode servers:

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
    sglang.ai/bootstrap-port: "9001"  # Optional: Bootstrap port for Mooncake prefill coordination
spec:
  containers:
  - name: sglang
    image: lmsys/sglang:latest
    ports:
    - containerPort: 8000  # Main API port
    - containerPort: 9001  # Optional: Bootstrap coordination port
    # ... rest of configuration
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
    - containerPort: 8000  # Main API port
    # ... rest of configuration
```

**Key Requirements:**
- Prefill pods must have labels matching your `--prefill-selector`
- Decode pods must have labels matching your `--decode-selector`
- Prefill pods can optionally include bootstrap port in annotations using `sglang.ai/bootstrap-port` (defaults to None if not specified)

#### Service Discovery Arguments

**General Arguments:**
- `--service-discovery`: Enable Kubernetes service discovery feature
- `--service-discovery-port`: Port to use when generating worker URLs (default: 8000)
- `--service-discovery-namespace`: Optional. Kubernetes namespace to watch for pods. If not provided, watches all namespaces (requires cluster-wide permissions)
- `--selector`: One or more label key-value pairs for pod selection in regular mode (format: key1=value1 key2=value2)

**PD Mode Arguments:**
- `--pd-disaggregation`: Enable PD (Prefill-Decode) disaggregated mode
- `--prefill`: Specify initial prefill server URL and bootstrap port (format: URL BOOTSTRAP_PORT, can be used multiple times)
- `--decode`: Specify initial decode server URL (can be used multiple times)
- `--prefill-selector`: Label selector for prefill server pods in PD mode (format: key1=value1 key2=value2)
- `--decode-selector`: Label selector for decode server pods in PD mode (format: key1=value1 key2=value2)
- `--policy`: Routing policy (cache_aware, random, power_of_two - note: power_of_two only works in PD mode)

**Notes:**
- Bootstrap port annotation is automatically set to `sglang.ai/bootstrap-port` for Mooncake deployments
- Advanced cache tuning parameters use sensible defaults and are not exposed via CLI

#### RBAC Requirements

When using service discovery, you must configure proper Kubernetes RBAC permissions:

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

**Cluster-wide (if watching all namespaces):**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sglang-router
  namespace: sglang-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: sglang-router
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: sglang-router
subjects:
- kind: ServiceAccount
  name: sglang-router
  namespace: sglang-system
roleRef:
  kind: ClusterRole
  name: sglang-router
  apiGroup: rbac.authorization.k8s.io
```

#### Complete Example: PD Mode with Service Discovery

Here's a complete example of running SGLang Router with PD mode and service discovery:

```bash
# Start the router with PD mode and automatic prefill/decode discovery
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

This setup will:
1. Enable PD (Prefill-Decode) disaggregated routing mode with automatic pod classification
2. Watch for pods in the `production` namespace
3. Automatically add prefill servers with labels `app=sglang`, `component=prefill`, `environment=production`
4. Automatically add decode servers with labels `app=sglang`, `component=decode`, `environment=production`
5. Extract bootstrap ports from the `sglang.ai/bootstrap-port` annotation on prefill pods
6. Use cache-aware load balancing for optimal performance
7. Expose the router API on port 8080 and metrics on port 9090

**Note:** In PD mode with service discovery, pods MUST match either the prefill or decode selector to be added. Pods that don't match either selector are ignored.

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
