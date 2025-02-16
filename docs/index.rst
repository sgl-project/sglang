SGLang Documentation
====================================

SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, jump-forward constrained decoding, overhead-free CPU scheduler, continuous batching, token attention (paged attention), tensor parallelism, FlashInfer kernels, chunked prefill, and quantization (FP8/INT4/AWQ/GPTQ).
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, QWen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   start/install.md

.. toctree::
   :maxdepth: 1
   :caption: Backend Tutorial

   backend/send_request.ipynb
   backend/openai_api_completions.ipynb
   backend/openai_api_vision.ipynb
   backend/openai_api_embeddings.ipynb
   backend/native_api.ipynb
   backend/offline_engine_api.ipynb
   backend/structured_outputs.ipynb
   backend/speculative_decoding.ipynb
   backend/function_calling.ipynb
   backend/server_arguments.md

.. toctree::
   :maxdepth: 1
   :caption: Frontend Tutorial

   frontend/frontend.md
   frontend/choices_methods.md


.. toctree::
   :maxdepth: 1
   :caption: SGLang Router

   router/router.md


.. toctree::
   :maxdepth: 1
   :caption: References

   references/supported_models.md
   references/sampling_params.md
   references/hyperparameter_tuning.md
   references/benchmark_and_profiling.md
   references/accuracy_evaluation.md
   references/custom_chat_template.md
   references/amd_configure.md
   references/deepseek.md
   references/multi_node.md
   references/modelscope.md
   references/quantization.md
   references/contribution_guide.md
   references/troubleshooting.md
   references/nvidia_jetson.md
   references/faq.md
   references/learn_more.md
