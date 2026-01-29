# Loading Models from Object Storage

SGLang supports direct loading of models from object storage (S3 and Google Cloud Storage) without requiring a full local download. This feature uses the `runai_streamer` load format powered by [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) to stream model weights directly from cloud storage, significantly reducing startup time and local storage requirements.

Run:ai Model Streamer is a library designed to read tensors concurrently while streaming them to GPU memory. For more details, see the [Run:ai Model Streamer Documentation](https://github.com/run-ai/runai-model-streamer).

## Installation

Before using object storage features, install the required dependencies:

```bash
# For S3 (Amazon S3, MinIO, Ceph, etc.)
pip install runai_model_streamer[s3]

# For Google Cloud Storage
pip install runai_model_streamer[gs]

# For both S3 and GCS
pip install runai_model_streamer[s3,gs]
```

**Note**: These are optional dependencies. Install only what you need based on your storage backend.

## Overview

When loading models from object storage, SGLang uses a two-phase approach:

1. **Metadata Download** (once, before process launch): Configuration files and tokenizer files are downloaded to a local cache
2. **Weight Streaming** (lazy, during model loading): Model weights are streamed directly from object storage as needed

This design avoids file locks, race conditions, and duplicate downloads across multiple worker processes.

## Supported Storage Backends

- **Amazon S3**: `s3://bucket-name/path/to/model/`
- **Google Cloud Storage**: `gs://bucket-name/path/to/model/`
- **Local Storage**: `/path/to/model/` (SSD, NVMe, shared file systems)

**Note**: While this guide focuses on object storage, the `runai_streamer` load format can also be used with local file paths to accelerate weight loading from SSDs, NVMe drives, and shared file systems (NFS, Lustre, etc.). The concurrent streaming approach can significantly speed up model loading even from local storage.

## Quick Start

### Basic Usage

Simply provide an object storage URI as the model path:

```bash
# S3
python -m sglang.launch_server \
  --model-path s3://my-bucket/models/llama-3-8b/ \
  --load-format runai_streamer

# Google Cloud Storage
python -m sglang.launch_server \
  --model-path gs://my-bucket/models/llama-3-8b/ \
  --load-format runai_streamer

# Local storage (for faster loading from SSD/NVMe)
python -m sglang.launch_server \
  --model-path /path/to/model/ \
  --load-format runai_streamer
```

**Note**: The `--load-format runai_streamer` is automatically detected when using object storage URIs (s3://, gs://), so you can omit it for cloud storage:

```bash
python -m sglang.launch_server \
  --model-path s3://my-bucket/models/llama-3-8b/
```

### With Tensor Parallelism

```bash
python -m sglang.launch_server \
  --model-path gs://my-bucket/models/llama-70b/ \
  --tp 4 \
  --model-loader-extra-config '{"distributed": true}'
```

## Configuration

### Load Format

The `runai_streamer` load format is specifically designed for object storage:

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --load-format runai_streamer
```

### Extended Configuration Parameters

Use `--model-loader-extra-config` to pass additional configuration as a JSON string:

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --model-loader-extra-config '{
    "distributed": true,
    "concurrency": 8,
    "memory_limit": 2147483648
  }'
```

#### Available Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `distributed` | bool | Enable distributed streaming for multi-GPU setups. Automatically set to `true` for object storage paths. | Auto-detected |
| `concurrency` | int | Controls the level of concurrency and number of OS threads reading tensors from the file to the CPU buffer. For reading from S3, this will be the number of client instances the host opens to the S3 server. Higher values can improve throughput for large models. | 4 |
| `memory_limit` | int | Size of the CPU memory buffer (in bytes) to which tensors are read from the file. You can limit this size to control memory usage. See [CPU buffer memory limiting](https://github.com/run-ai/runai-model-streamer#memory-limit) for more details. | System-dependent |

**Example with all parameters:**

```bash
python -m sglang.launch_server \
  --model-path gs://my-bucket/models/deepseek-v3/ \
  --tp 8 \
  --model-loader-extra-config '{
    "distributed": true,
    "concurrency": 16,
    "memory_limit": 4294967296
  }'
```

## Environment Variables

### Required Credentials

#### S3 Configuration

Use standard AWS credential methods:

```bash
# Option 1: AWS CLI configuration
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# Option 3: Custom S3 endpoint (for S3-compatible storage)
export AWS_ENDPOINT_URL=https://s3.custom-endpoint.com
export RUNAI_STREAMER_S3_ENDPOINT=https://s3.custom-endpoint.com
```

**Note**: `RUNAI_STREAMER_S3_ENDPOINT` will automatically fall back to `AWS_ENDPOINT_URL` if not set.

#### Google Cloud Storage Configuration

Use Google Cloud SDK authentication:

```bash
# Option 1: Application Default Credentials
gcloud auth application-default login

# Option 2: Service account key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Option 3: For public/anonymous access (testing only)
export RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS=true
```

### Optional Configuration

```bash
# Cache directory for metadata files
export SGLANG_CACHE_DIR=~/.cache/sglang/

# Control streaming behavior
export RUNAI_STREAMER_CONCURRENCY=8
export RUNAI_STREAMER_MEMORY_LIMIT=2147483648

# S3 virtual addressing (for S3-compatible storage)
export RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0

# Disable EC2 metadata (when using S3-compatible storage outside AWS)
export AWS_EC2_METADATA_DISABLED=true
```

For a complete list of environment variables and additional tunable parameters, see:
- [Environment Variables Reference](../references/environment_variables.md#object-storage-and-runai-streamer)
- [Run:ai Model Streamer Environment Variables Documentation](https://github.com/run-ai/runai-model-streamer#environment-variables)

## Architecture

### Design Pattern: Single Metadata Download

SGLang uses a carefully designed pattern to avoid complexity:

1. **Engine Entrypoint** (main process):
   - Downloads config files and tokenizer once before launching worker processes
   - Stores metadata in `~/.cache/sglang/model_streamer/<model_hash>/`

2. **Worker Processes**:
   - Use cached metadata from the local directory
   - Stream weights directly from object storage during model loading
   - No file locks or inter-process coordination needed

3. **Weight Streaming**:
   - Calls `list_safetensors()` to enumerate weight files from object storage
   - Streams weights lazily using `SafetensorsStreamer` from `runai_model_streamer` library

This avoids file locks, race conditions, and duplicate downloads.

## Performance Considerations

### Use Cases for RunAI Streamer

The `runai_streamer` load format provides performance benefits for various storage scenarios:

1. **Object Storage (S3/GCS)**: Eliminates need for full local download, reduces startup time
2. **High-Speed Local Storage (NVMe/SSD)**: Concurrent streaming can speed up loading by 2-3x compared to sequential loading
3. **Shared File Systems (NFS, Lustre, GPFS)**: Parallel reads improve throughput on network file systems
4. **Multi-Node Deployments**: Each node streams weights independently, avoiding contention

### Network Bandwidth (Object Storage)

- **Critical Factor**: Network bandwidth between the compute instance and object storage
- **Recommendation**: Use instances in the same region as your storage bucket
- **Example**: On AWS, use EC2 instances in the same region as your S3 bucket

### Concurrency Tuning

Adjust concurrency based on your storage type and bandwidth:

```bash
# For large models (70B+) with high bandwidth or fast NVMe
--model-loader-extra-config '{"concurrency": 16}'

# For smaller models or limited bandwidth/slower storage
--model-loader-extra-config '{"concurrency": 4}'

# For shared file systems, tune based on network capacity
--model-loader-extra-config '{"concurrency": 8}'
```

### Distributed Streaming

For multi-GPU setups, enable distributed streaming to parallelize weight loading:

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --tp 8 \
  --model-loader-extra-config '{"distributed": true}'
```

## Comparison with Local Loading

| Aspect | Object Storage | Local Storage |
|--------|----------------|---------------|
| **Startup Time** | Faster (only metadata) | Slower (full download) |
| **Disk Space** | Minimal (~100MB) | Full model size (100GB+) |
| **First Request** | Slightly slower (streaming) | Faster |
| **Subsequent Requests** | Same as local | Same |
| **Network Dependency** | Yes | No |
| **Multi-Node** | Easier (shared storage) | Requires file sync |

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

**Error**: `Access Denied` or `Permission Denied`

**Solution**: Verify credentials are properly configured:

```bash
# For S3
aws s3 ls s3://your-bucket/models/

# For GCS
gsutil ls gs://your-bucket/models/
```

#### 2. Metadata Download Already Called

**Error**: `Metadata download already called for <model_path>`

**Cause**: `ObjectStorageModel.download_and_get_path()` was called multiple times

**Solution**: This should only happen once in the Engine entrypoint. Check for duplicate server initialization.

#### 3. Network Timeout

**Error**: Connection timeout during weight loading

**Solution**:
- Check network connectivity to object storage
- Increase timeout or reduce concurrency
- Ensure compute instance and storage are in the same region

#### 4. Missing Metadata Files

**Error**: `Essential metadata files missing from cache`

**Solution**:
- Verify the object storage path contains all necessary files (config.json, tokenizer files, weight files)
- Check that the model was uploaded completely
- Clear cache and retry: `rm -rf ~/.cache/sglang/model_streamer/`

### Debug Mode

Enable detailed logging:

```bash
export SGLANG_LOG_LEVEL=DEBUG
python -m sglang.launch_server --model-path s3://bucket/model/ --log-level debug
```

## Examples

### Example 1: Single GPU with S3

```bash
python -m sglang.launch_server \
  --model-path s3://my-models/llama-3-8b-instruct/ \
  --host 0.0.0.0 \
  --port 30000
```

### Example 2: Multi-GPU with GCS

```bash
python -m sglang.launch_server \
  --model-path gs://my-models/llama-3-70b-instruct/ \
  --tp 4 \
  --model-loader-extra-config '{"distributed": true, "concurrency": 12}' \
  --host 0.0.0.0 \
  --port 30000
```

### Example 3: Custom S3 Endpoint (MinIO, Ceph, etc.)

For S3-compatible storage like MinIO or Google Cloud Storage with S3 compatibility:

```bash
export AWS_ENDPOINT_URL=https://minio.example.com
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

python -m sglang.launch_server \
  --model-path s3://models/llama-3-8b/ \
  --host 0.0.0.0 \
  --port 30000
```

For Google Cloud Storage using S3 compatibility mode:

```bash
export RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0
export AWS_EC2_METADATA_DISABLED=true
export AWS_ENDPOINT_URL=https://storage.googleapis.com

python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --host 0.0.0.0 \
  --port 30000
```

### Example 4: Multi-Node Tensor Parallelism

```bash
# Node 0
python -m sglang.launch_server \
  --model-path s3://models/llama-3-405b/ \
  --tp 8 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr node0:50000 \
  --model-loader-extra-config '{"distributed": true, "concurrency": 16}'

# Node 1
python -m sglang.launch_server \
  --model-path s3://models/llama-3-405b/ \
  --tp 8 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr node0:50000 \
  --model-loader-extra-config '{"distributed": true, "concurrency": 16}'
```

## Limitations

- **AMD/ROCm**: Object storage with runai_streamer is currently disabled on AMD ROCm due to compatibility issues
- **Network Dependency**: Requires stable network connection to object storage during model loading
- **First Load Latency**: First model load may be slower than local loading due to streaming
- **Supported Formats**: Currently only supports `.safetensors` weight format (recommended format)

## See Also

- [Server Arguments Reference](server_arguments.md#model-and-tokenizer) - Complete list of server arguments
- [Environment Variables](../references/environment_variables.md#object-storage-and-runai-streamer) - Environment variable configuration
- [Hyperparameter Tuning](hyperparameter_tuning.md) - Performance optimization guide
- [Multi-Node Deployment](../references/multi_node_deployment/multi_node_index.rst) - Distributed setup guide
