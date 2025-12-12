# Runtime examples

The below examples will mostly need you to start a server in a separate terminal before you can execute them. Please see in the code for detailed instruction.

## Native API

* `lora.py`: An example how to use LoRA adapters.
* `multimodal_embedding.py`: An example how perform [multi modal embedding](Alibaba-NLP/gme-Qwen2-VL-2B-Instruct).
* `openai_batch_chat.py`: An example how to process batch requests for chat completions.
* `openai_batch_complete.py`: An example how to process batch requests for text completions.
* **`openai_chat_with_response_prefill.py`**:
  An example that demonstrates how to [prefill a response](https://eugeneyan.com/writing/prompting/#prefill-claudes-responses) using the OpenAI API by enabling the `continue_final_message` parameter.
  When enabled, the final (partial) assistant message is removed and its content is used as a prefill so that the model continues that message rather  than starting a new turn. See [Anthropic's prefill example](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response#example-structured-data-extraction-with-prefilling) for more context.
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

## Multimodal

SGLang supports multimodal inputs for various model architectures. The `multimodal` folder contains examples showing how to use urls, files or encoded data to make requests to multimodal models. Examples include querying the [Llava-OneVision](multimodal/llava_onevision_server.py) model (image, multi-image, video), Llava-backed [Qwen-Llava](multimodal/qwen_llava_server.py) and [Llama3-Llava](multimodal/llama3_llava_server.py) models (image, multi-image), and Mistral AI's [Pixtral](multimodal/pixtral_server.py) (image, multi-image).


## Token In, Token Out

The folder `token_in_token_out` shows how to perform inference, where we provide tokens and get tokens as response.

* `token_in_token_out_{llm|vlm}_{engine|server}.py`: Shows how to perform token in, token out workflow for llm/vlm using either the engine or native API.
