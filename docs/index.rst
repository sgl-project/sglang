SGLang Documentation
====================

SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-lora batching.
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with wide industry adoption.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   start/install.md

.. toctree::
   :maxdepth: 1
   :caption: Basic Usage


.. toctree::
   :maxdepth: 1
   :caption: Advanced Features


.. toctree::
   :maxdepth: 1
   :caption: Performance Tuning


.. toctree::
   :maxdepth: 1
   :caption: Supported Models


.. toctree::
   :maxdepth: 1
   :caption: Hardware Platforms

   platform/amd.md
   platform/cpu.md
   platform/nvidia_jetson.md

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide


.. toctree::
   :maxdepth: 1
   :caption: References
