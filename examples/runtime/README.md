# Runtime examples

The below examples will mostly need you to start a server in a separate terminal before you can execute them. Please see in the code for detailed instruction.

## Native API

* `lora.py`: An example how to use LoRA adapters.
* `multimodal_embedding.py`: An example how perform [multi modal embedding](Alibaba-NLP/gme-Qwen2-VL-2B-Instruct).
* `openai_batch_chat.py`: An example how to process batch requests for chat completions.
* `openai_batch_complete.py`: An example how to process batch requests for text completions.
* `openai_chat_with_response_prefill.py`: An example how to [prefill](https://eugeneyan.com/writing/prompting/#prefill-claudes-responses) a response using OpenAI API.
* `reward_model.py`: An example how to extract scores from a reward model.
* `vertex_predict.py`: An example how to deploy a model to [Vertex AI](https://cloud.google.com/vertex-ai?hl=en).

## Engine

The `engine` folder contains that examples that show how to use [Offline Engine API](https://docs.sglang.ai/backend/offline_engine_api.html#Offline-Engine-API) for common workflows.

* `custom_server.py`: An example how to deploy a custom server.
* `embedding.py`: An example how to extract embeddings.
* `launch_engine.py`: An example how to launch the Engine.
* `offline_batch_inference_eagle.py`: An example how to perform speculative decoding using [EAGLE](https://docs.sglang.ai/backend/speculative_decoding.html).
* `offline_batch_inference_torchrun.py`: An example how to perform inference using [torchrun](https://pytorch.org/docs/stable/elastic/run.html).
* `offline_batch_inference_vlm.py`: An example how to use VLMs with the engine.
* `offline_batch_inference.py`: An example how to use the engine to perform inference on a batch of examples.

## Hidden States

The `hidden_states` folder contains examples on how to extract hidden states using SGLang. Please note that this might degrade throughput due to cuda graph rebuilding.

* `hidden_states_engine.py`: An example how to extract hidden states using the Engine API.
* `hidden_states_server.py`: An example how to extract hidden states using the Server API.

## LLaVA-NeXT

SGLang support LLaVA-OneVision with single-image, multi-image and video are supported. The folder `llava_onevision` shows how to do this.

## Token In, Token Out

The folder `token_in_token_out` shows how to perform inference, where we provide tokens and get tokens as response.

* `token_in_token_out_{llm|vlm}_{engine|server}.py`: Shows how to perform token in, token out workflow for llm/vlm using either the engine or native API.
