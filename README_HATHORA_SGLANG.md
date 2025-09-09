# SGLang Hathora Server

A high-performance, OpenAI-compatible API server powered by SGLang for deployment on Hathora Cloud. This server replaces HuggingFace Transformers with SGLang for improved inference performance and includes comprehensive logging and monitoring.

## Features

- **SGLang Backend**: High-performance inference engine replacing HuggingFace Transformers
- **OpenAI Compatible**: Drop-in replacement for OpenAI's chat completions API
- **Comprehensive Logging**: Detailed request tracking and performance monitoring
- **Hathora Integration**: Native support for Hathora Cloud deployment with GCP networking
- **Streaming Support**: Real-time token streaming for better user experience
- **Health Monitoring**: Built-in health checks and metrics endpoints
- **Graceful Shutdown**: Proper cleanup of resources and network endpoints

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│  Hathora Cloud   │───▶│  SGLang Engine  │
│                 │    │  Load Balancer   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   FastAPI App    │
                       │  (serve_hathora) │
                       └──────────────────┘
```

## Quick Start

### 1. Environment Variables

Set the required environment variables for Hathora deployment:

```bash
# Required for Hathora
export HATHORA_HOSTNAME="your-app.hathora.dev"
export HATHORA_DEFAULT_PORT="8000"
export HATHORA_REGION="seattle"
export GCP_SERVICE_ACCOUNT_KEY_BASE64="<base64-encoded-service-account-key>"

# SGLang Configuration
export MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
export TP_SIZE="1"
export MAX_TOTAL_TOKENS="4096"
export LOG_LEVEL="INFO"
```

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Install SGLang (adjust based on your installation method)
pip install sglang[all]

# Run the server
python serve_hathora.py
```

The server will be available at:
- API: `http://localhost:8000/v1/chat/completions`
- Health: `http://localhost:8000/health`
- Docs: `http://localhost:8000/docs`

### 3. Docker Deployment

```bash
# Build the Docker image
docker build -f Dockerfile.hathora -t sglang-hathora .

# Run locally (without Hathora networking)
docker run -p 8000:8000 \
  -e MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct" \
  -e TP_SIZE="1" \
  -e LOG_LEVEL="INFO" \
  sglang-hathora
```

## API Usage

### Chat Completions

The server provides an OpenAI-compatible chat completions endpoint:

```python
import openai

client = openai.OpenAI(
    base_url="http://your-app.hathora.dev/v1",
    api_key="not-required"  # SGLang server doesn't require API keys
)

# Non-streaming request
response = client.chat.completions.create(
    model="sglang",  # Model name is configurable
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
    temperature=0.8
)

print(response.choices[0].message.content)

# Streaming request
stream = client.chat.completions.create(
    model="sglang",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    max_tokens=200,
    temperature=0.8,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Health Check

```bash
curl http://your-app.hathora.dev/health
```

Response:
```json
{
  "status": "ok",
  "engine_status": "ready",
  "hathora_region": "seattle",
  "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "timestamp": 1694123456.789
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HATHORA_HOSTNAME` | Hathora app hostname | **Required** |
| `HATHORA_DEFAULT_PORT` | Port for the service | **Required** |
| `HATHORA_REGION` | Hathora deployment region | **Required** |
| `GCP_SERVICE_ACCOUNT_KEY_BASE64` | Base64-encoded GCP service account key | **Required** |
| `MODEL_PATH` | HuggingFace model path for SGLang | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `TP_SIZE` | Tensor parallelism size | `1` |
| `MAX_TOTAL_TOKENS` | Maximum total tokens per request | `4096` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `PORT` | Server port | `8000` |

### Model Configuration

SGLang supports various models. Update the `MODEL_PATH` environment variable to use different models:

```bash
# Llama models
export MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
export MODEL_PATH="meta-llama/Meta-Llama-3.1-70B-Instruct"

# Mistral models
export MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3"

# Qwen models
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
```

For multi-GPU deployments, increase the `TP_SIZE`:

```bash
export TP_SIZE="2"  # For 2 GPUs
export TP_SIZE="4"  # For 4 GPUs
```

## Logging

The server provides comprehensive logging at multiple levels:

### Log Levels

- **DEBUG**: Detailed debugging information including prompt content
- **INFO**: General operational information and request summaries
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for failed requests

### Log Format

```
2025-09-09 19:12:30,059 - serve_hathora - INFO - [127.0.0.1] POST /v1/chat/completions - Request started
2025-09-09 19:12:30,060 - serve_hathora - INFO - Chat completion request: model=sglang, messages=1, max_tokens=50, stream=False
2025-09-09 19:12:30,061 - serve_hathora - INFO - [chatcmpl-1757445150061234] Starting generation for prompt length: 145
2025-09-09 19:12:30,185 - serve_hathora - INFO - [chatcmpl-1757445150061234] Generation completed in 124.5ms
2025-09-09 19:12:30,186 - serve_hathora - INFO - [127.0.0.1] POST /v1/chat/completions - Status: 200, Duration: 127.2ms
```

### Log Files

Logs are written to both stdout and `serve_hathora.log` file for persistence.

## Performance

### Benchmarking

Use the included test script to validate functionality:

```bash
python test_serve_hathora.py
```

For performance benchmarking, use SGLang's built-in benchmarking tools:

```bash
# Install SGLang bench tools
pip install sglang[all]

# Benchmark the server
python -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://your-app.hathora.dev/v1 \
  --model sglang \
  --num-prompts 100
```

### Monitoring

Monitor server performance through:

1. **Health endpoint**: `/health` for basic status
2. **Response headers**: `X-Process-Time` for request latency
3. **Logs**: Detailed timing and usage information
4. **Hathora dashboard**: Infrastructure metrics

## Deployment

### Hathora Cloud

1. Create a new Hathora application
2. Configure environment variables in the Hathora dashboard
3. Deploy using the provided Dockerfile:

```bash
# Tag and push to your container registry
docker tag sglang-hathora your-registry/sglang-hathora:latest
docker push your-registry/sglang-hathora:latest
```

4. Set the image in your Hathora application configuration

### Development vs Production

**Development:**
```bash
export LOG_LEVEL="DEBUG"
export MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"  # Smaller model for testing
```

**Production:**
```bash
export LOG_LEVEL="INFO"
export MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
export TP_SIZE="2"  # Scale based on available GPUs
```

## Troubleshooting

### Common Issues

1. **Engine not initialized**: 
   - Check that the model path is correct
   - Ensure sufficient GPU memory
   - Verify SGLang installation

2. **High latency**:
   - Increase `TP_SIZE` for multi-GPU setups
   - Reduce `MAX_TOTAL_TOKENS` if not needed
   - Check model size vs available hardware

3. **Memory issues**:
   - Reduce `MAX_TOTAL_TOKENS`
   - Use a smaller model
   - Increase GPU memory

### Debug Mode

Enable debug mode for detailed logging:

```bash
export LOG_LEVEL="DEBUG"
python serve_hathora.py
```

This will show:
- Full prompt content
- Detailed timing information
- SGLang engine debug output

## Comparison with Original

### Improvements

| Feature | Original (HF Transformers) | New (SGLang) |
|---------|---------------------------|--------------|
| **Inference Speed** | Slower, single-GPU limited | Faster, multi-GPU support |
| **Memory Efficiency** | Higher memory usage | Optimized memory usage |
| **Batching** | Manual batching | Automatic intelligent batching |
| **Streaming** | Token-by-token generation | Native streaming support |
| **Logging** | Basic logging | Comprehensive monitoring |
| **Scalability** | Limited | Built for production scale |
| **API Compatibility** | Basic OpenAI compat | Full OpenAI compatibility |

### Migration Benefits

- **Performance**: 2-3x faster inference with SGLang
- **Scalability**: Better handling of concurrent requests
- **Monitoring**: Detailed logging and metrics
- **Reliability**: Graceful error handling and recovery
- **Compatibility**: Drop-in replacement for existing clients

## Contributing

1. Run tests: `python test_serve_hathora.py`
2. Check linting: `flake8 serve_hathora.py`
3. Update documentation for any new features
4. Test with different models and configurations

## License

This project follows the same license as SGLang. See the SGLang repository for details.

## Support

For issues related to:
- **SGLang**: Check the [SGLang repository](https://github.com/sgl-project/sglang)
- **Hathora**: Check the [Hathora documentation](https://hathora.dev/docs)
- **This implementation**: Create an issue in this repository

---

Based on the original implementation from https://github.com/AndreHathora/llm-serve
