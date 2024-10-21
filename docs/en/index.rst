SGLang Documentation
====================================

Welcome to SGLang!

SGLang is a fast, open-source framework for serving large and vision-language models. It enhances speed and control by co-designing the backend runtime and frontend language. Join our vibrant community and contribute to advancing LLM and VLM serving.

The core features of SGLang include:

* **Fast Backend Runtime**: Efficient serving with RadixAttention for prefix caching, jump-forward constrained decoding, continuous batching, token attention (paged attention), tensor parallelism, flashinfer kernels, and quantization (AWQ/FP8/GPTQ/Marlin).

* **Flexible Frontend Language**: Enables easy programming of LLM applications with chained generation calls, advanced prompting, control flow, multiple modalities, parallelism, and external interactions.

* **Extensive Model Support**: SGLang supports a wide range of generative models including the Llama series (up to Llama 3.1), Mistral, Gemma, Qwen, DeepSeek, LLaVA, Yi-VL, StableLM, Command-R, DBRX, Grok, ChatGLM, InternLM 2 and Exaone 3. It also supports embedding models such as e5-mistral and gte-Qwen2. Easily extensible to support new models.

Documentation
-------------

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