Welcome to SGLang!
====================================

.. figure:: ./_static/image/logo.png
  :width: 50%
  :align: center
  :alt: SGLang
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>SGLang is yet another fast serving framework for large language models and vision language models.
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/sgl-project/sglang" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/sgl-project/sglang/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/sgl-project/sglang/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

SGLang has the following core features:

* **Fast Backend Runtime**: Efficient serving with RadixAttention for prefix caching, jump-forward constrained decoding, continuous batching, token attention (paged attention), tensor parallelism, flashinfer kernels, and quantization (AWQ/FP8/GPTQ/Marlin).

* **Flexible Frontend Language**: Enables easy programming of LLM applications with chained generation calls, advanced prompting, control flow, multiple modalities, parallelism, and external interactions.

* **Extensive Model Support**: SGLang supports a wide range of generative models including the Llama series (up to Llama 3.1), Mistral, Gemma, Qwen, DeepSeek, LLaVA, Yi-VL, StableLM, Command-R, DBRX, Grok, ChatGLM, InternLM 2 and Exaone 3. It also supports embedding models such as e5-mistral and gte-Qwen2. Easily extensible to support new models.

* **Open Source Community**: SGLang is an open source project with a vibrant community of contributors. We welcome contributions from anyone interested in advancing the state of the art in LLM and VLM serving.

Documentation
-------------

.. In this documentation, we'll dive into these following areas to help you get the most out of SGLang.

.. _installation:
.. toctree::
   :maxdepth: 1
   :caption: Installation

   install.md

.. _hyperparameter_tuning:
.. toctree::
   :maxdepth: 1
   :caption: Hyperparameter Tuning

   hyperparameter_tuning.md

.. _custom_chat_template:
.. toctree::
   :maxdepth: 1
   :caption: Custom Chat Template

   custom_chat_template.md

.. _model_support:
.. toctree::
   :maxdepth: 1
   :caption: Model Support

   model_support.md

.. _sampling_params:
.. toctree::
   :maxdepth: 1
   :caption: Sampling Params

   sampling_params.md


.. _benchmark_and_profilling:
.. toctree::
   :maxdepth: 1
   :caption: Benchmark and Profilling

   benchmark_and_profiling.md