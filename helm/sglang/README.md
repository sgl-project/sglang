# SGLang Helm Chart

A production-ready Helm chart for deploying [SGLang](https://github.com/sgl-project/sglang) - a high-performance serving framework for large language models and multimodal models.

## Features

- 🚀 **Single and Multi-Node Support**: Deploy on single node or distributed across multiple nodes
- 📊 **Built-in Monitoring**: Prometheus metrics and ServiceMonitor support
- 🔐 **Secure Secret Management**: Support for inline secrets, existing secrets, or External Secrets Operator
- 🎯 **Production Ready**: Includes PodDisruptionBudget, health probes, and resource management
- 🔧 **Highly Configurable**: Comprehensive values.yaml with sensible defaults

## Prerequisites

- Kubernetes 1.20+
- Helm 3.0+
- GPU nodes with NVIDIA runtime (for GPU workloads)
- (Optional) External Secrets Operator for secret management
- (Optional) Prometheus Operator for monitoring

## Installation

### Quick Start

```bash
# Install from OCI registry (GitHub Container Registry)
helm install my-sglang oci://ghcr.io/eladmotola/sglang --version 0.1.0

# Or install from local chart
helm install my-sglang ./sglang-helm-chart
```

### Single-Node Deployment

```bash
helm install my-sglang ./sglang-helm-chart \
  --set replicaCount=1 \
  --set model.tensorParallelSize=1 \
  --set model.path=meta-llama/Llama-3.1-8B-Instruct \
  --set hfSecret.token=your-hf-token
```

### Multi-Node Distributed Deployment

```bash
helm install my-sglang ./sglang-helm-chart \
  --set replicaCount=2 \
  --set model.tensorParallelSize=8 \
  --set model.path=meta-llama/Llama-3.1-70B-Instruct \
  --set hfSecret.existingSecret=my-hf-secret
```

## Configuration

The following table lists the key configurable parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas (nodes) | `1` |
| `image.repository` | SGLang image repository | `lmsysorg/sglang` |
| `image.tag` | Image tag | `latest` |
| `hostNetwork` | Use host network for RDMA/InfiniBand (requires privileged mode) | `false` |
| `model.path` | HuggingFace model path | `meta-llama/Llama-3.1-8B-Instruct` |
| `model.tensorParallelSize` | Tensor parallel size (GPUs per node) | `1` |
| `model.distInitPort` | Port for distributed initialization | `5000` |
| `hfSecret.token` | HuggingFace token (not recommended for production) | `""` |
| `hfSecret.existingSecret` | Name of existing secret containing HF token | `""` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `monitoring.serviceMonitor.enabled` | Enable Prometheus ServiceMonitor | `false` |
| `persistence.size` | Size of persistent volume per replica | `30Gi` |

See [values.yaml](values.yaml) for the complete list of configurable parameters.

## Architecture

This chart uses a **StatefulSet** for both single-node and multi-node deployments to provide:

- Stable network identities for distributed serving
- Ordered deployment and scaling
- Persistent storage per replica via volumeClaimTemplates

### Single-Node Mode

- `replicaCount: 1`
- Uses tensor parallelism only within a single node
- Suitable for smaller models or development

### Multi-Node Mode

- `replicaCount: N` (where N > 1)
- Automatically configures distributed serving with stable pod indices
- Each pod gets a unique `node-rank` for coordination
- Requires proper network connectivity between nodes

## Examples

### Basic Single GPU

```yaml
replicaCount: 1
distributed:
  tensorParallelSize: 1
resources:
  limits:
    nvidia.com/gpu: 1
```

### Multi-GPU Single Node

```yaml
replicaCount: 1
model:
  tensorParallelSize: 4
resources:
  limits:
    nvidia.com/gpu: 4
```

## Monitoring

Enable Prometheus monitoring:

```yaml
model:
  enableMetrics: true
service:
  metrics:
    enabled: true
monitoring:
  serviceMonitor:
    enabled: true
```

## Security

### HuggingFace Token Management

Three options (in order of preference):

1. **External Secrets Operator** (Recommended for production):
```yaml
hfSecret:
  externalSecret:
    secretStoreRef:
      name: my-secret-store
      kind: SecretStore
    remoteRef:
      key: sglang-hf-token
```

2. **Existing Kubernetes Secret**:
```yaml
hfSecret:
  existingSecret: my-hf-secret
  existingSecretKey: HF_TOKEN
```

3. **Inline Token** (Not recommended for production):
```yaml
hfSecret:
  token: hf_xxxxxxxxxxxx
```

## Troubleshooting

### Pods not starting

Check logs:
```bash
kubectl logs -n <namespace> <pod-name>
```

### Storage issues

Check PVCs:
```bash
kubectl get pvc -n <namespace>
```

## Uninstalling

```bash
helm uninstall my-sglang
```

## Contributing

Contributions are welcome! Please submit issues and pull requests to the [repository](https://github.com/sgl-project/sglang).

## License

This Helm chart is provided as-is under the Apache 2.0 License.

## Links

- [SGLang Documentation](https://docs.sglang.io/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
