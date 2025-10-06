# Checkpoint Engine Design Documentation

## Overview

ckpt-engine is a lightweight library specifically designed to accelerate weight synchronization in large-scale distributed training. It operates on a parameter server architecture (ps.py, worker.py). It support two deployment methods: co-locate and disaggregation. Its core mechanism is to establish an asynchronous, pipelined data transfer process based mooncake transfer engine. This allows sglang inference engine to offload the weight update task to background workers, effectively hiding the I/O and communication latency.

Two key scenarios can benefit from this ckpt-engine:

- Reinforcement Learning (RL) Workloads – including RLHF, DPO, and continual pre-training – where model weights are updated frequently. Current methods for synchronizing these updates into the inference engine introduce significant latency, creating a bottleneck. This underutilizes GPUs during weight updates and slows the overall training-inference loop.
- Bulk Deployment – The boot time is a performance bottleneck when launching multiple SGLang instances. 

## Use Cases

Prerequisites: installing checkpoint-engine
```bash
pip install 'checkpoint-engine[p2p]'  # install checkpoint engine
```

Running Methods:

- sglang
```bash
python3 -m sglang.launch_server --model /opt/models/Qwen/Qwen3-8b --tp 8 --load-format ckpt_engine --port 30001
```

- checkpoint engine
```bash
torchrun --nproc-per-node 8 ckptengine_update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-8b/
```

## Architecture

### Core Components

The checkpoint engine consists of several key components:

1. **CkptEngineConnector** - The main connector that handles checkpoint engine communication
2. **CkptEngineModelLoader** - Specialized model loader for checkpoint engine format
3. **CkptEngineUpdate** - Standalone script for updating weights via checkpoint engine
4. **IPC-based Weight Transfer** - Efficient inter-process communication for weight updates

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SGLang Server                               │
├─────────────────────────────────────────────────────────────────┤
│  HTTP API                                                       │
│  ├── /update_weights_from_ckpt_engine                          │
│  └── /update_weights_from_distributed                          │
├─────────────────────────────────────────────────────────────────┤
│  Scheduler                                                      │
│  ├── SchedulerUpdateWeightsMixin                               │
│  └── Request Dispatcher                                        │
├─────────────────────────────────────────────────────────────────┤
│  Model Runner                                                  │
│  ├── ModelRunner.update_weights_from_ckpt_engine()             │
│  └── ModelRunner.update_weights_from_distributed()             │
├─────────────────────────────────────────────────────────────────┤
│  Connector                                                     │
│  ├── CkptEngineConnector                                       │
│  └── BaseConnector Interface                                   │
├─────────────────────────────────────────────────────────────────┤
│  Model Loader.                                                 │
│  ├── CkptEngineModelLoader                                     │
│  └── get_model_loader()                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Checkpoint Engine Format Support

The system supports a new load format called `ckpt_engine` that enables:

- **Efficient Weight Loading**: Load models from checkpoint engine format
- **Distributed Loading**: Support for tensor parallel weight distribution
- **Memory Optimization**: Optimized memory usage during weight loading

### 2. In-Place Weight Updates

The checkpoint engine enables updating model weights without restarting the server:

- **Hot Swapping**: Update weights while the server is running
- **Rollback Support**: Automatic rollback on update failures
- **Memory Safety**: Safe memory management during updates

### 3. Inter-Process Communication (IPC)

Efficient IPC-based weight transfer:

- **Shared Memory**: Utilizes shared memory for efficient tensor transfer
- **Metadata Management**: Handles tensor metadata for proper reconstruction
- **Error Handling**: Robust error handling and cleanup

### 4. Distributed Weight Synchronization

Supports distributed weight updates across tensor parallel workers:

- **Broadcast Updates**: Broadcast weight updates to all workers
- **P2P Updates**: Point-to-point weight updates for specific workers
- **Synchronization**: Proper synchronization barriers for consistency

## Implementation Details

### CkptEngineConnector

The `CkptEngineConnector` class implements the core checkpoint engine functionality:

```python
class CkptEngineConnector(BaseConnector):
    def __init__(self, url: str, device: torch.device = "cpu"):
        super().__init__(url)
        self.url = url
        self.device = device
        self.zmq_handle = None
        self.zmq_ctx = None
        self.device_uuid = None
        self.socket = None
        self.buffer: Optional[torch.Tensor] = None
        self.local_rank = None
        self.final_state_dict = OrderedDict()
        self.pending_weights: Dict[str, torch.Tensor] = {}
```

Key methods:
- `get_zmq_handle()`: Establishes ZMQ connection for weight transfer
- `update_weights_from_ipc()`: Handles IPC-based weight updates
- `_extract_weights()`: Extracts individual tensors from shared buffer

### CkptEngineModelLoader

The `CkptEngineModelLoader` handles loading models from checkpoint engine format:

```python
class CkptEngineModelLoader(BaseModelLoader):
    def load_model(self, *, model_config: ModelConfig, device_config: DeviceConfig) -> nn.Module:
        """Load model using checkpoint engine format."""
        logger.info("Loading weights from checkpoint engine format ...")
        
        model_weights = f"ckptengine://"
        
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config)
                
            with create_remote_connector(model_weights, device_config.device) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.CKPTENGINE:
                    self.load_model_from_ckpt_engine(
                        model, client, model_config, device_config
                    )
                else:
                    raise ValueError(f"Unsupported connector type {connector_type}")
        
        return model.eval()
```

### Weight Update Process

The weight update process involves several steps:

1. **Initialization**: Set up ZMQ connections and shared memory
2. **Metadata Transfer**: Send tensor metadata (shapes, dtypes, offsets)
3. **Buffer Transfer**: Transfer shared memory buffer containing weights
4. **Weight Loading**: Load weights into model using standard load_weights method
5. **Cleanup**: Clean up resources and synchronize

### IPC Protocol

The IPC protocol uses ZMQ for communication:

- **Port Assignment**: Dynamic port assignment (base port 33001 + rank)
- **Message Types**: Support for tensor metadata, buffer handles, and termination signals
- **Error Handling**: Robust error handling with proper cleanup

## API Integration

### HTTP Endpoints

The system exposes HTTP endpoints for weight updates:

```python
@app.post("/update_weights_from_ckpt_engine")
async def update_weights_from_ckpt_engine(
    obj: UpdateWeightsFromCkptEngineReqInput, request: Request
):
    """Update the weights from disk inplace without re-launching the server."""
```

### Request Structure

Weight update requests include:
- `model_path`: Path to the new model weights
- `load_format`: Format of the weights (e.g., "ckpt_engine")

## Configuration

### Load Format Configuration

The checkpoint engine format is registered in the load configuration:

```python
class LoadFormat(str, enum.Enum):
    # ... existing formats ...
    CKPT_ENGINE = "ckpt_engine"
```

### Server Arguments

The system supports configuration through server arguments:
- `--load-format ckpt_engine`: Use checkpoint engine format for initial loading
- Custom weight loader support for extensibility



## Use Cases

### 1. Online Model Updates

Update model weights without server downtime:
```bash
curl -X POST http://localhost:30000/update_weights_from_ckpt_engine \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/new/checkpoint", "load_format": "ckpt_engine"}'
```

### 2. Distributed Training Integration

Integrate with distributed training systems for seamless model updates.


### 3. Model Serving at Scale

Efficient weight management for large-scale model serving deployments.

## Future Enhancements

### Planned Features

1. **Incremental Updates**: Support for incremental weight updates
2. **Compression**: Advanced compression algorithms for weight transfer
3. **Caching**: Intelligent caching for frequently used weights
4. **Monitoring**: Enhanced monitoring and metrics for weight updates

### Performance Optimizations

1. **Parallel Transfer**: Parallel weight transfer for large models
2. **Streaming**: Streaming weight updates for very large models
3. **GPU Direct**: GPU-direct memory transfer for improved performance

