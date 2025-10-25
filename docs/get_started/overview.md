SGLang performs inference on models for you.

All communication between users and the model deployment is managed through HTTP or gRPC API calls.

It takes < 5 min to set up an SGLang inference-ready deployment with your model.

To deploy an SGLang instance you need two ingrediants:

1). Model (locally or on huggingface)
2). Hardware (SGLang supports NVIDIA GPUs, AMD GPUs, ARM CPUs, ...)


An SGLang deployment consists of only two components: router(s) and worker(s).

## Routers

The router(s) receive API calls and act according to the received API calls.

## Workers
