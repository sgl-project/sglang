SGLang Documentation
====================

SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

<<<<<<< HEAD
- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor parallelism, pipeline parallelism, expert parallelism, structured outputs, chunked prefill, quantization (FP8/INT4/AWQ/GPTQ), and multi-lora batching.
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   start/install.md

.. toctree::
   :maxdepth: 1
   :caption: Backend Tutorial

   references/deepseek
   references/llama4
   backend/send_request.ipynb
   backend/openai_api_completions.ipynb
   backend/openai_api_vision.ipynb
   backend/openai_api_embeddings.ipynb
   backend/native_api.ipynb
   backend/offline_engine_api.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Advanced Backend Configurations

   backend/server_arguments.md
   backend/sampling_params.md
   backend/hyperparameter_tuning.md
   backend/attention_backend.md
=======
- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-lora batching.
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with wide industry adoption.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install.md

.. toctree::
   :maxdepth: 1
   :caption: Basic Usage

   basic_usage/send_request.ipynb
   basic_usage/openai_api.rst
   basic_usage/offline_engine_api.ipynb
   basic_usage/native_api.ipynb
   basic_usage/sampling_params.md
   basic_usage/deepseek.md
   basic_usage/gpt_oss.md
   basic_usage/llama4.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced_features/server_arguments.md
   advanced_features/hyperparameter_tuning.md
   advanced_features/speculative_decoding.ipynb
   advanced_features/structured_outputs.ipynb
   advanced_features/structured_outputs_for_reasoning_models.ipynb
   advanced_features/function_calling.ipynb
   advanced_features/separate_reasoning.ipynb
   advanced_features/quantization.md
   advanced_features/lora.ipynb
   advanced_features/pd_disaggregation.md
   advanced_features/vlm_query.ipynb
   advanced_features/router.md
   advanced_features/observability.md
   advanced_features/attention_backend.md
>>>>>>> origin/main

.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   supported_models/generative_models.md
   supported_models/multimodal_language_models.md
   supported_models/embedding_models.md
   supported_models/reward_models.md
<<<<<<< HEAD
   supported_models/support_new_models.md
   supported_models/transformers_fallback.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   backend/speculative_decoding.ipynb
   backend/structured_outputs.ipynb
   backend/function_calling.ipynb
   backend/separate_reasoning.ipynb
   backend/structured_outputs_for_reasoning_models.ipynb
   backend/custom_chat_template.md
   backend/quantization.md
   backend/lora.ipynb
   backend/pd_disaggregation.md

.. toctree::
   :maxdepth: 1
   :caption: Frontend Tutorial

   frontend/frontend.ipynb
   frontend/choices_methods.md

.. toctree::
   :maxdepth: 1
   :caption: SGLang Router

   router/router.md

.. toctree::
      :maxdepth: 1
      :caption: References

      references/general
      references/hardware
      references/advanced_deploy
      references/performance_analysis_and_optimization
      references/developer
=======
   supported_models/rerank_models.md
   supported_models/support_new_models.md
   supported_models/transformers_fallback.md
   supported_models/modelscope.md

.. toctree::
   :maxdepth: 1
   :caption: Hardware Platforms

   platforms/amd_gpu.md
   platforms/blackwell_gpu.md
   platforms/cpu_server.md
   platforms/tpu.md
   platforms/nvidia_jetson.md
   platforms/ascend_npu.md

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/contribution_guide.md
   developer_guide/development_guide_using_docker.md
   developer_guide/benchmark_and_profiling.md
   developer_guide/bench_serving.md

.. toctree::
   :maxdepth: 1
   :caption: References

   references/faq.md
   references/environment_variables.md
   references/production_metrics.md
   references/multi_node_deployment/multi_node_index.rst
   references/custom_chat_template.md
   references/frontend/frontend_index.rst
   references/learn_more.md
>>>>>>> origin/main
