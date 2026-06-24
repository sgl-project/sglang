# SGLang Engine

SGLang provides a direct inference engine without the need for an HTTP server. There are generally these use cases:

- [Offline Batch Inference](#offline-batch-inference)
- [Embedding Generation](#embedding-generation)
- [Custom Server](#custom-server)
- [Token-In-Token-Out for RLHF](#token-in-token-out-for-rlhf)
- [Inference Using FastAPI](#inference-using-fastapi)

## Examples

### [Offline Batch Inference](./offline_batch_inference.py)

In this example, we launch an SGLang engine and feed a batch of inputs for inference. If you provide a very large batch, the engine will intelligently schedule the requests to process efficiently and prevent OOM (Out of Memory) errors.

### [Embedding Generation](./embedding.py)

In this example, we launch an SGLang engine and feed a batch of inputs for embedding generation.

### [Custom Server](./custom_server.py)

This example demonstrates how to create a custom server on top of the SGLang Engine. We use [Sanic](https://sanic.dev/en/) as an example. The server supports both non-streaming and streaming endpoints.

#### Steps

1. Install Sanic:

   ```bash
   pip install sanic
   ```

2. Run the server:

   ```bash
   python custom_server
   ```

3. Send requests:

   ```bash
   curl -X POST http://localhost:8000/generate  -H "Content-Type: application/json"  -d '{"prompt": "The Transformer architecture is..."}'
   curl -X POST http://localhost:8000/generate_stream  -H "Content-Type: application/json"  -d '{"prompt": "The Transformer architecture is..."}' --no-buffer
   ```

   This will send both non-streaming and streaming requests to the server.

### [Token-In-Token-Out for RLHF](../token_in_token_out)

In this example, we launch an SGLang engine, feed tokens as input and generate tokens as output.

### [Inference Using FastAPI](fastapi_engine_inference.py)

This example demonstrates how to create a FastAPI server that uses the SGLang engine for text generation.
