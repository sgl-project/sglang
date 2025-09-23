# SGLang LeaderWorkerSet Helm Chart Deployment Guide

This Helm Chart is used to deploy SGLang's distributed inference architecture in Kubernetes clusters, implementing multi-node distributed inference based on LeaderWorkerSet (LWS).

## Project Structure

```
lws_pd_helm/
├── lws_pd_helm.md                     # Documentation
├── deploy.sh                          # Automated deployment script
└── sglang-leaderworkerset/            # Helm Chart directory
    ├── Chart.yaml                     # Chart metadata
    ├── values.yaml                    # Default configuration values
    ├── values-example.yaml            # Production environment configuration example
    ├── charts/                        # Dependency Chart directory
    └── templates/                     # Kubernetes template directory
        ├── _helpers.tpl               # Helm helper templates
        ├── NOTES.txt                  # Post-deployment instructions
        ├── prefill-leaderworkerset.yaml  # Prefill deployment template
        ├── decode-leaderworkerset.yaml   # Decode deployment template
        ├── services.yaml              # Service template
        └── loadbalancer.yaml          # LoadBalancer template
```

## Prerequisites

1. Kubernetes >= 1.26
2. LeaderWorkerSet (LWS) Operator installed
3. Nodes with NVIDIA GPU support
4. RDMA network support (optional but recommended)

## Quick Start

### 1. Clone Repository and Navigate to Directory

```bash
cd docs/references/multi_node_deployment/lws_pd_helm
```

### 2. Modify Configuration

Edit the `sglang-leaderworkerset/values.yaml` file, mainly need to modify the following configuration items:

```yaml
global:
  # Modify to actual model path
  model:
    path: "/your/model/path"
  
  # Modify node selector
  nodeSelector:
    your-label: "value"
  
  # Modify tolerations
  tolerations:
    - key: your-taint-key
      operator: Exists
  
  # Modify RDMA configuration (according to actual network environment)
  rdma:
    ibDevice: "your_ib_devices"
    
  # Modify storage volume paths
  volumes:
    model:
      hostPath: "/your/model/path"
    config:
      hostPath: "/your/config/path"
    cache:
      hostPath: "/your/cache/path"
```

### 3. Deploy Application

#### Using Automated Script Deployment (Recommended)

```bash
# Deploy with default configuration
./deploy.sh install

# Specify release name and namespace
./deploy.sh install my-sglang sglang-system

# Use custom configuration file
./deploy.sh install my-sglang sglang-system ./sglang-leaderworkerset/values-example.yaml
```

#### Using Helm Commands Directly

```bash
# Install Helm Chart
helm install sglang-leaderworkerset ./sglang-leaderworkerset

# Or specify namespace
helm install sglang-leaderworkerset ./sglang-leaderworkerset -n sglang-system --create-namespace
```

### 4. Check Deployment Status

```bash
# Check LeaderWorkerSet status
kubectl get leaderworkerset

# Check Pod status
kubectl get pods

# Check service status
kubectl get svc
```

### 5. Test API

Get LoadBalancer service address:

```bash
# If using NodePort
# Replace <NAME_PREFIX> with your global.namePrefix value from values.yaml
export NODE_PORT=$(kubectl get -o jsonpath="{.spec.ports[0].nodePort}" services <NAME_PREFIX>-lb-service)
export NODE_IP=$(kubectl get nodes -o jsonpath="{.items[0].status.addresses[0].address}")
echo http://$NODE_IP:$NODE_PORT

# Test API call
# Replace <MODEL_NAME> with your actual model identifier
curl -X POST "http://$NODE_IP:$NODE_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer None" \
  -d '{
    "rid": "test123",
    "model": "<MODEL_NAME>", 
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant"},
      {"role": "user", "content": "Hello, please introduce yourself"}
    ],
    "max_tokens": 200
  }'
```

## Configuration Description

### Global Configuration

- `global.namePrefix`: Resource name prefix
- `global.image`: Docker image configuration
- `global.model`: Model-related configuration
- `global.nodeSelector`: Node selector
- `global.tolerations`: Tolerations configuration
- `global.rdma`: RDMA network configuration
- `global.volumes`: Storage volume configuration

### Prefill Configuration

- `prefill.enabled`: Whether to enable Prefill nodes
- `prefill.leaderWorkerSet.size`: Number of Worker nodes
- `prefill.config.*`: Prefill-related parameter configuration
- `prefill.leader.env`: Leader node environment variables
- `prefill.worker.env`: Worker node environment variables
- `prefill.worker.port`: Worker node container port (default 30001)

### Decode Configuration

- `decode.enabled`: Whether to enable Decode nodes
- `decode.leaderWorkerSet.size`: Number of Worker nodes
- `decode.config.*`: Decode-related parameter configuration
- `decode.leader.env`: Leader node environment variables
- `decode.worker.env`: Worker node environment variables
- `decode.worker.port`: Worker node container port (default 30001)

### LoadBalancer Configuration

- `loadBalancer.enabled`: Whether to enable load balancer
- `loadBalancer.service.type`: Service type (NodePort/LoadBalancer/ClusterIP)
- `loadBalancer.service.nodePort`: NodePort port number (when type is NodePort)

## Upgrade and Uninstall

### Upgrade

```bash
# Upgrade using script
./deploy.sh upgrade my-sglang sglang-system ./sglang-leaderworkerset/values-example.yaml

# Or use Helm commands
helm upgrade sglang-leaderworkerset ./sglang-leaderworkerset --reset-values
```

### Rollback

```bash
# View version history
helm history sglang-leaderworkerset

# Rollback to specified version
helm rollback sglang-leaderworkerset 1
```

### Uninstall

```bash
# Uninstall using script
./deploy.sh uninstall my-sglang sglang-system

# Or use Helm commands
helm uninstall sglang-leaderworkerset
```

## Troubleshooting

### 1. Check Pod Status

```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name> -c <container-name>
```

### 2. Common Issues and Solutions

#### Insufficient GPU Resources
```yaml
# Adjust GPU requests in values.yaml
prefill:
  leader:
    resources:
      limits:
        nvidia.com/gpu: "4"  # Reduce GPU count
```

#### Network Connection Issues
Check RDMA configuration and firewall settings:
```bash
# Check RDMA devices
kubectl exec -it <pod-name> -- ls /dev/infiniband/
```

#### Storage Path Issues
Ensure model paths are accessible on all nodes:
```bash
# Check model path
kubectl exec -it <pod-name> -- ls -la /work/models
```

#### Pod Cannot Be Scheduled
- Check nodeSelector and tolerations configuration
- Confirm sufficient node resources

#### Model Loading Failed
- Check if model path is correct
- Confirm all nodes can access model path

#### Insufficient Resources
- Adjust GPU and memory requests
- Reduce concurrent request count

### 3. Debug Commands

```bash
# Check deployment status
kubectl get leaderworkerset,pods,svc

# View Pod details
kubectl describe pod <pod-name>

# View logs
kubectl logs <pod-name> -c <container-name>

# View Helm status
helm status <release-name>

# View Helm history
helm history <release-name>
```

### 4. Performance Tuning

Adjust parameters according to actual hardware environment:

```yaml
# Adjust according to GPU memory
prefill:
  config:
    memFractionStatic: "0.8"  # Adjust memory usage ratio
    maxRunningRequests: "512"  # Adjust maximum concurrent requests

decode:
  config:
    memFractionStatic: "0.9"
    maxRunningRequests: "1024"
```

## Advanced Configuration

### Custom Environment Variables

```yaml
prefill:
  leader:
    env:
      CUSTOM_ENV_VAR: "value"
      NCCL_DEBUG: "INFO"
```

### Multi-Instance Deployment

```yaml
decode:
  leaderWorkerSet:
    replicas: 2  # Deploy multiple decode instances
```

### Custom Port Configuration

```yaml
prefill:
  worker:
    port: 30002  # Custom worker port

decode:
  worker:
    port: 30003  # Custom worker port
```

### Resource Limits

```yaml
prefill:
  leader:
    resources:
      limits:
        nvidia.com/gpu: "8"
        memory: "64Gi"
        cpu: "16"
      requests:
        memory: "32Gi"
        cpu: "8"
```

## Monitoring and Logs

### View Real-time Logs

```bash
# Replace <name-prefix> with your global.namePrefix value from values.yaml

# Prefill logs
kubectl logs -f -l leaderworkerset.sigs.k8s.io/name=<name-prefix>-prefill,role=leader

# Decode logs  
kubectl logs -f -l leaderworkerset.sigs.k8s.io/name=<name-prefix>-decode,role=leader

# LoadBalancer logs
kubectl logs -f -l app=<name-prefix>-lb
```

### Performance Monitoring

You can monitor performance through the following methods:

1. Use kubectl top to view resource usage
2. Deploy Prometheus and Grafana for detailed monitoring
3. View SGLang built-in performance metrics

## Quick Usage Guide Summary

### 1. Basic Deployment
```bash
# Deploy with default configuration
./deploy.sh install

# Specify release name and namespace
./deploy.sh install my-sglang sglang-system
```

### 2. Custom Configuration Deployment
```bash
# Copy example configuration
cp sglang-leaderworkerset/values-example.yaml my-values.yaml

# Modify configuration file
vim my-values.yaml

# Deploy with custom configuration
./deploy.sh install my-sglang sglang-system my-values.yaml
```

### 3. Upgrade and Management
```bash
# Upgrade deployment
./deploy.sh upgrade my-sglang sglang-system my-values.yaml

# Uninstall deployment
./deploy.sh uninstall my-sglang sglang-system
```

## Core Configuration Items

### Required Configuration Changes
```yaml
global:
  model:
    path: "/your/model/path"          # Model path
  nodeSelector:
    your-label: "value"               # Node selector
  tolerations:                        # Tolerations configuration
    - key: your-taint
      operator: Exists
  volumes:
    model:
      hostPath: "/your/model/path"    # Storage volume path
```

### Optional Optimization Configuration
```yaml
prefill:
  config:
    maxRunningRequests: "1024"        # Maximum concurrent requests
    memFractionStatic: "0.7"          # Memory usage ratio
  worker:
    port: 30001                       # Worker port

decode:
  config:
    maxRunningRequests: "2048"        # Maximum concurrent requests
    memFractionStatic: "0.849"        # Memory usage ratio
  worker:
    port: 30001                       # Worker port
```

## Summary

By deploying SGLang in Helm Chart format, we have achieved:

1. **Standardized Deployment**: Complies with Kubernetes and Helm best practices
2. **Simplified Management**: Simplified from 5 yaml files to 1 Helm command
3. **Improved Maintainability**: Centralized configuration management, templated reuse
4. **Enhanced Reliability**: Version control, rollback support
5. **Improved User Experience**: Automated scripts, detailed documentation
6. **Complete Parameterization**: All configuration items can be customized through values.yaml

This Helm Chart can serve as the official deployment solution for the SGLang project, providing users with a better Kubernetes deployment experience.

## Technical Support

For issues, please refer to:

1. [SGLang Official Documentation](https://github.com/sgl-project/sglang)
2. [LeaderWorkerSet Documentation](https://github.com/kubernetes-sigs/lws)
3. [Helm Documentation](https://helm.sh/docs/)

Or submit an Issue to the SGLang project repository.