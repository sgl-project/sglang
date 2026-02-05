SGLang Documentation
====================

.. raw:: html

  <a class="github-button" href="https://github.com/sgl-project/sglang" data-size="large" data-show-count="true" aria-label="Star sgl-project/sglang on GitHub">Star</a>
  <a class="github-button" href="https://github.com/sgl-project/sglang/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork sgl-project/sglang on GitHub">Fork</a>
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <br></br>

SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.
- **RL & Post-Training Backbone**: SGLang is a proven rollout backend across the world, with native RL integrations and adoption by well-known post-training frameworks such as AReaL, Miles, slime, Tunix, verl and more.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install.md

.. toctree::
   :maxdepth: 1
   :caption: Basic Usage

   basic_usage/send_request.ipynb
   basic_usage/openai_api.rst
   basic_usage/ollama_api.md
   basic_usage/offline_engine_api.ipynb
   basic_usage/native_api.ipynb
   basic_usage/sampling_params.md
   basic_usage/popular_model_usage.rst

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced_features/server_arguments.md
   advanced_features/hyperparameter_tuning.md
   advanced_features/attention_backend.md
   advanced_features/speculative_decoding.ipynb
   advanced_features/structured_outputs.ipynb
   advanced_features/structured_outputs_for_reasoning_models.ipynb
   advanced_features/tool_parser.ipynb
   advanced_features/separate_reasoning.ipynb
   advanced_features/quantization.md
   advanced_features/quantized_kv_cache.md
   advanced_features/expert_parallelism.md
   advanced_features/lora.ipynb
   advanced_features/pd_disaggregation.md
   advanced_features/epd_disaggregation.md
   advanced_features/pipeline_parallelism.md
   advanced_features/hicache.rst
   advanced_features/pd_multiplexing.md
   advanced_features/vlm_query.ipynb
   advanced_features/dp_for_multi_modal_encoder.md
   advanced_features/cuda_graph_for_multi_modal_encoder.md
   advanced_features/sgl_model_gateway.md
   advanced_features/deterministic_inference.md
   advanced_features/observability.md
   advanced_features/checkpoint_engine.md
   advanced_features/sglang_for_rl.md

.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   supported_models/generative_models.md
   supported_models/multimodal_language_models.md
   supported_models/diffusion_language_models.md
   supported_models/diffusion_models.md
   supported_models/embedding_models.md
   supported_models/reward_models.md
   supported_models/rerank_models.md
   supported_models/classify_models.md
   supported_models/support_new_models.md
   supported_models/transformers_fallback.md
   supported_models/modelscope.md
   supported_models/mindspore_models.md

.. toctree::
   :maxdepth: 1
   :caption: Hardware Platforms

   platforms/amd_gpu.md
   platforms/cpu_server.md
   platforms/tpu.md
   platforms/nvidia_jetson.md
   platforms/ascend_npu_support.rst
   platforms/xpu.md

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/contribution_guide.md
   developer_guide/development_guide_using_docker.md
   developer_guide/development_jit_kernel_guide.md
   developer_guide/benchmark_and_profiling.md
   developer_guide/bench_serving.md
   developer_guide/evaluating_new_models.md

.. toctree::
   :maxdepth: 1
   :caption: References

   references/faq.md
   references/environment_variables.md
   references/production_metrics.md
   references/production_request_trace.md
   references/multi_node_deployment/multi_node_index.rst
   references/custom_chat_template.md
   references/frontend/frontend_index.rst
   references/post_training_integration.md
   references/learn_more.md

.. toctree::
   :maxdepth: 1
   :caption: Security Acknowledgement

   security/acknowledgements.md
