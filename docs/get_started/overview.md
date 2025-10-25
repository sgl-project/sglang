SGLang performs inference on models for you.

All communication between users and the model deployment is managed through HTTP or gRPC API calls.

It takes < 5 min to set up an SGLang inference-ready deployment with your model.

To use SGLang you need two ingrediants:

1). Model (locally or on huggingface)
2). Hardware (SGLang supports NVIDIA GPUs, AMD GPUs, ARM CPUs, ...)

For detailed instructions on how to use SGLang, read the deployment instructions here: [Router guide](../advanced_features/router.md).

SGLang


An SGLang deployment consists of only two components: router(s) and worker(s). Both components must be present. The router(s) and worker(s) can all be deployed on the same device (i.e., NVIDIA GPU) or they can all be deployed on different devices (i.e., the worker is on an NVIDIA GPU and the router is on an ARM CPU).

## Routers

The router(s) receive API calls and act(s) according to the received API calls.

## Workers
