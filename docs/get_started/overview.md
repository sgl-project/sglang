SGLang performs inference on models for you.

It takes < 5 min to set up an SGLang inference-ready deployment with your model.

Some common use cases are: TODO

To use SGLang you need two ingredients:

1). Model (locally or on huggingface)
2). Hardware (SGLang supports NVIDIA GPUs, AMD GPUs, ARM CPUs, ...)

SGLang provides two deployment methods:
1). Server
2). Serverless

## Server

All communication between users and the model deployment is managed through HTTP or gRPC API calls.

For detailed instructions on how to use SGLang as a server, read the deployment instructions here: [Router guide](../advanced_features/router.md).

An SGLang server deployment consists of only two components: router(s) and worker(s). Both components must be present. The router(s) and worker(s) can all be deployed on the same device (i.e., NVIDIA GPU) or they can all be deployed on different devices (i.e., the worker is on an NVIDIA GPU and the router is on an ARM CPU).

### Routers

The router(s) receive API calls and act(s) according to the received API calls.

### Workers


## Serverless

For detailed instructions on how to use SGLang serverless, read here: [Offline Engine](../basic_usage/offline_engine_api.ipynb)
