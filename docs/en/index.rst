SGLang Documentation
====================================

SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, jump-forward constrained decoding, continuous batching, token attention (paged attention), tensor parallelism, FlashInfer kernels, chunked prefill, and quantization (INT4/FP8/AWQ/GPTQ).
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama 3, Gemma 2, Mistral, QWen, DeepSeek, LLaVA, etc.) and embedding models (e5-mistral), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install.md
   backend.md
   frontend.md

.. toctree::
   :maxdepth: 1
   :caption: References

   sampling_params.md
   hyperparameter_tuning.md
   model_support.md
   contributor_guide.md
   choices_methods.md
   benchmark_and_profiling.md
   troubleshooting.md
