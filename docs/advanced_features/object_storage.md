# Loading Models from Object Storage

SGLang supports direct loading of models from object storage (S3 and Google Cloud Storage) without requiring a full local download. This feature uses the `runai_streamer` load format to stream model weights directly from cloud storage, significantly reducing startup time and local storage requirements.

## Overview

When loading models from object storage, SGLang uses a two-phase approach:

1. **Metadata Download** (once, before process launch): Configuration files and tokenizer files are downloaded to a local cache
2. **Weight Streaming** (lazy, during model loading): Model weights are streamed directly from object storage as needed

## Supported Storage Backends

1. **Amazon S3**: `s3://bucket-name/path/to/model/`
2. **Google Cloud Storage**: `gs://bucket-name/path/to/model/`
3. **Azure Blob**: `az://some-azure-container/path/`
4. **S3-compatible** (e.g. Backblaze B2): `s3://bucket-name/path/to/model/`

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


## Loading from Backblaze B2 (S3-compatible)

The `s3://` scheme also works against [Backblaze B2](https://www.backblaze.com/cloud-storage)'s S3-compatible API by pointing the underlying boto3 client at the B2 endpoint. SGLang accepts the endpoint, region, and credentials either as query-string parameters on the model URI or as standard AWS environment variables.

Create a bucket-scoped [Application Key](https://www.backblaze.com/docs/cloud-storage-application-keys) in the B2 console first. The `keyID` maps to `AWS_ACCESS_KEY_ID` and the `applicationKey` maps to `AWS_SECRET_ACCESS_KEY`. The endpoint is shown on the bucket detail page (e.g. `https://s3.us-west-004.backblazeb2.com`).

### URI form (query-string parameters)

```bash
python -m sglang.launch_server \
  --model-path "s3://my-bucket/llama-3-70b?endpoint_url=https://s3.us-west-004.backblazeb2.com&region=us-west-004"
```

Recognized query parameters: `endpoint_url`, `region` (alias `region_name`),
`aws_access_key_id`, `aws_secret_access_key`. They are stripped from the URI
before listing or downloading objects.

### Environment-variable alternative

```bash
export AWS_ENDPOINT_URL=https://s3.us-west-004.backblazeb2.com
export AWS_DEFAULT_REGION=us-west-004
export AWS_ACCESS_KEY_ID=<your-B2-application-key-id>
export AWS_SECRET_ACCESS_KEY=<your-B2-application-key>

python -m sglang.launch_server --model-path s3://my-bucket/llama-3-70b/
```

Explicit query-string values take precedence over environment variables, which in turn take precedence over the default boto3 credential chain.

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
