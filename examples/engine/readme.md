# SGLang Engine

## Introduction
SGLang provides a direct inference engine without the need for an HTTP server. There are generallt two use cases

1. Offline Batch Inference
2. Custom Server on top of Engine

## Example

1. [Offline batch inference](./offline_batch_inference.py)

In the example, we launch a sglang engine and feed a batch of inputs to do inference. Note that if you feed a very large batch, the engine will intelligently schedule the requests to process efficiently and avoid OOM.

2. [Custom Server](./custom_server.py)

The example shows how to create a custom server on top of SGLang Engine. We use [Sanic](https://sanic.dev/en/) as an example. The server supports non-streaming and streaming endpoints.

1. Install sanic

```bsah
pip install sanic
```

2. Run the server

```bash
python custom_server
```

3. Send requests!

```bash
curl -X POST http://localhost:8000/generate  -H "Content-Type: application/json"  -d '{"prompt": "Transformer architecture is..."}'
curl -X POST http://localhost:8000/generate_stream  -H "Content-Type: application/json"  -d '{"prompt": "Transformer architecture is.."}' --no-buffer
```
