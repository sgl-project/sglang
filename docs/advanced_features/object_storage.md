# Loading Models from Object Storage

SGLang supports direct loading of models from object storage (S3 and Google Cloud Storage) without requiring a full local download. This feature uses the `runai_streamer` load format to stream model weights directly from cloud storage, significantly reducing startup time and local storage requirements.

## Overview

When loading models from object storage, SGLang uses a two-phase approach:

1. **Metadata Download** (once, before process launch): Configuration files and tokenizer files are downloaded to a local cache
2. **Weight Streaming** (lazy, during model loading): Model weights are streamed directly from object storage as needed

## Supported Storage Backends

- **Amazon S3**: `s3://bucket-name/path/to/model/`
- **Amazon S3**: `s3://bucket-name/path/to/model/`
- **S3 compatible**: `s3://bucket-name/path/to/model/`

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
```

**Note**: The `--load-format runai_streamer` is automatically detected when using object storage URIs, so you can omit it:

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

The `runai_streamer` load format is specifically designed for object storage, ssd and shared file systems

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
| `distributed` | bool | Enable distributed streaming for multi-GPU setups. Automatically set to `true` for object storage paths and cuda alike devices. | Auto-detected |
| `concurrency` | int | Number of concurrent download streams. Higher values can improve throughput for large models. | 4 |
| `memory_limit` | int | Memory limit (in bytes) for the streaming buffer. | System-dependent |


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
```

## Performance Considerations

### Distributed Streaming

For multi-GPU setups, enable distributed streaming to parallelize weight loading between the processes:

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --tp 8 \
  --model-loader-extra-config '{"distributed": true}'
```

## Limitations

- **Supported Formats**: Currently only supports `.safetensors` weight format (recommended format)
- **Supported Device**: Distributed streaming is supported on cuda alike devices. Otherwise fallback to non distributed streaming

## See Also

- [Runai model streamer documentation](https://github.com/run-ai/runai-model-streamer)
