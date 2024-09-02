Welcome to SGLang's tutorials!
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

Documentation
-------------

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

Search Bar
==================

* :ref:`search`
