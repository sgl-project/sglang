# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Monkey-patches on transformers internals.

Mix of backward-compat shims (re-add symbols removed in v5), workarounds
for transformers v5 bugs, fixes for remote-model-code (trust_remote_code)
that hasn't been updated for v5 yet, and CI-only patches (e.g. neutralize
HF API calls to avoid rate limits).

Import this module early (before any ``from_pretrained`` call) to activate
all patches.  It is safe to import multiple times -- patches are idempotent.
"""

import inspect

from sglang.srt.utils import logger

_applied = False


# ---------------------------------------------------------------------------
# Public API: apply_all() -- import-time patches (idempotent)
# ---------------------------------------------------------------------------


def apply_all():
    """Apply all transformers compatibility patches (idempotent).

    Call this once at import time.  It is safe to call multiple times.

    No-op when the ``transformers`` package is not installed -- frontend-only
    sglang users should not be forced to install transformers just to import
    the top-level ``sglang`` package.
    """
    global _applied
    if _applied:
        return
    try:
        import transformers  # noqa: F401
    except ImportError:
        _applied = True
        return
    _applied = True

    # v5.4 patches
    _patch_flash_attn_availability()
    _patch_rope_parameters_validation()
    _patch_removed_symbols()
    _patch_image_processor_kwargs()
    _patch_image_process_cuda_tensor()
    _patch_nemotron_h_pattern()
    _patch_default_rope_init_function()
    _patch_mask_function_input_embeds_kwarg()

    # v5 general patches
    _ensure_clean_up_tokenization_compat()
    _ensure_is_torch_fx_available_compat()

    # CI-only: neutralize HF API calls inside tokenizer from_pretrained
    patch_is_base_mistral_in_ci()

    logger.debug("transformers compatibility patches applied")


# ---------------------------------------------------------------------------
# Public API: on-demand helpers (called explicitly by other modules)
# ---------------------------------------------------------------------------


def normalize_rope_scaling_compat(config) -> None:
    """Ensure rope_scaling dicts have ``"type"`` alongside ``"rope_type"``.

    Transformers v5 standardises rope_scaling to use ``"rope_type"`` and may
    omit the legacy ``"type"`` key.  Remote-code models (e.g. Kimi-VL) still
    read ``rope_scaling["type"]``, causing a ``KeyError``.  This helper adds
    ``"type"`` from ``"rope_type"`` whenever it is missing, recursively across
    the config and all its sub-configs.
    """

    def _patch(cfg):
        rs = getattr(cfg, "rope_scaling", None)
        if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
        # Recurse into sub-configs
        for attr in (
            "text_config",
            "llm_config",
            "language_config",
            "vision_config",
            "thinker_config",
        ):
            sub = getattr(cfg, attr, None)
            if sub is not None:
                _patch(sub)

    _patch(config)


def _ensure_gguf_version():
    """Workaround for transformers v5 bug where is_gguf_available() fails
    when the gguf package lacks __version__ and metadata lookup also fails,
    resulting in packaging.version.InvalidVersion: Invalid version: 'N/A'."""
    try:
        import gguf

        if not hasattr(gguf, "__version__"):
            import importlib.metadata

            try:
                gguf.__version__ = importlib.metadata.version("gguf")
            except importlib.metadata.PackageNotFoundError:
                gguf.__version__ = "0.0.0"
            except (ValueError, OSError, TypeError) as e:
                logger.warning(
                    "Failed to determine gguf package version: %s. "
                    "Falling back to '0.0.0'.",
                    e,
                )
                gguf.__version__ = "0.0.0"
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# v5.4 patches (merged from transformers_v54_compat.py)
# ---------------------------------------------------------------------------


def _patch_rope_parameters_validation():
    """Guard ``standardize_rope_params()`` against missing
    ``max_position_embeddings``.

    For ``PretrainedConfig``, ``standardize_rope_params()`` accesses
    ``self.max_position_embeddings`` during ``__post_init__`` before extra
    kwargs are set as attributes, causing ``AttributeError``.

    Fix: guard ``standardize_rope_params`` against missing
    ``max_position_embeddings``.
    """
    from transformers import PretrainedConfig

    # standardize_rope_params accesses self.max_position_embeddings before
    # __post_init__ sets extra kwargs — skip when the attribute is absent.
    if hasattr(PretrainedConfig, "standardize_rope_params"):
        _orig_standardize = PretrainedConfig.standardize_rope_params

        def _safe_standardize(self):
            if not hasattr(self, "max_position_embeddings"):
                return
            return _orig_standardize(self)

        PretrainedConfig.standardize_rope_params = _safe_standardize


def _compute_default_rope_parameters(
    config=None, device=None, seq_len=None, layer_type=None, **rope_kwargs
):
    """Pre-refactor "default" (non-scaling) RoPE init formula.

    Newer transformers versions moved this case into
    ``RotaryEmbeddingConfigMixin`` and dropped it from the
    ``ROPE_INIT_FUNCTIONS`` dispatch dict / off legacy ``RotaryEmbedding``
    modules entirely, since they no longer need to compute it themselves.
    Legacy remote model code (e.g. ByteDance/Ouro) predates this and still
    expects it to be reachable both ways -- see the two call sites patched
    by ``_patch_default_rope_init_function`` below.
    """
    import torch

    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually "
            "exclusive in `_compute_default_rope_parameters`, got "
            f"`rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    else:
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor


def _patch_default_rope_init_function():
    """Restore the ``"default"`` RoPE case for legacy remote model code.

    Two independent transformers-internal spots special-case
    ``rope_type == "default"`` and no longer resolve it the way legacy
    ``trust_remote_code`` models expect:

    1. ``ROPE_INIT_FUNCTIONS`` (a dispatch dict) dropped its ``"default"``
       entry, keeping only the scaling strategies (linear, dynamic, yarn,
       ...). Legacy code doing ``ROPE_INIT_FUNCTIONS[self.rope_type]`` with
       ``rope_type == "default"`` gets ``KeyError``.
    2. ``PreTrainedModel._init_weights``'s RotaryEmbedding branch resolves
       the "default" case via ``module.compute_default_rope_parameters`` --
       a method only the new ``RotaryEmbeddingConfigMixin``-based modules
       define -- instead of consulting ``ROPE_INIT_FUNCTIONS``. Legacy
       ``RotaryEmbedding`` modules (which only set ``rope_init_fn``) get
       ``AttributeError`` during ``post_init()``.

    Both are fixed with the same pre-refactor formula
    (``_compute_default_rope_parameters``), since the underlying math for
    the non-scaling case hasn't changed.
    """
    from transformers import modeling_rope_utils
    from transformers.modeling_utils import PreTrainedModel

    rope_init_functions = getattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", None)
    if rope_init_functions is not None and "default" not in rope_init_functions:
        rope_init_functions["default"] = _compute_default_rope_parameters

    _orig_init_weights = PreTrainedModel._init_weights

    def _init_weights_with_legacy_rope_fallback(self, module):
        if (
            "RotaryEmbedding" in module.__class__.__name__
            and hasattr(module, "original_inv_freq")
            and getattr(module, "rope_type", None) == "default"
            and not hasattr(module, "compute_default_rope_parameters")
        ):
            module.compute_default_rope_parameters = _compute_default_rope_parameters
        return _orig_init_weights(self, module)

    PreTrainedModel._init_weights = _init_weights_with_legacy_rope_fallback


def _patch_mask_function_input_embeds_kwarg():
    """Accept pre-refactor kwargs on mask-creation helpers.

    ``transformers.masking_utils.create_causal_mask`` (and its
    sliding-window/chunked/bidirectional siblings) dropped some legacy
    kwargs. Some remote model code (e.g. ByteDance/Ouro) still calls them
    the old way:

    - ``input_embeds`` (singular) instead of the current ``inputs_embeds``.
    - ``cache_position``, which the current docstring explicitly documents
      as "Deprecated and unused" -- safe to drop rather than rename.

    Wrap each to remap/drop these before forwarding, rather than reactively
    special-casing every legacy kwarg one at a time: anything not accepted
    by the *current* signature is filtered out, since passing it through
    unfiltered would just raise ``TypeError`` anyway.
    """
    import inspect

    from transformers import masking_utils

    for name in (
        "create_causal_mask",
        "create_sliding_window_causal_mask",
        "create_chunked_causal_mask",
        "create_bidirectional_sliding_window_mask",
    ):
        orig_fn = getattr(masking_utils, name, None)
        if orig_fn is None:
            continue
        accepted_params = set(inspect.signature(orig_fn).parameters)

        def _make_wrapper(fn, accepted_params):
            def _wrapper(*args, **kwargs):
                if "input_embeds" in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
                return fn(*args, **kwargs)

            return _wrapper

        setattr(masking_utils, name, _make_wrapper(orig_fn, accepted_params))


def _patch_flash_attn_availability():
    """Prevent flash-attn-4 from masquerading as flash-attn-2.

    flash-attn-4 registers a bare ``flash_attn`` namespace that makes
    ``is_flash_attn_2_available()`` return True, but lacks the v2 API.
    Remote model code (e.g. Kimi-VL) guarded by that check will crash.

    TODO(upstream): model authors should check for specific API symbols.
    """
    try:
        import flash_attn as _fa

        if not hasattr(_fa, "flash_attn_func"):
            import transformers.utils as _u
            import transformers.utils.import_utils as _ui

            _ui.is_flash_attn_2_available = lambda: False
            _u.is_flash_attn_2_available = lambda: False
    except ImportError:
        pass


def _patch_removed_symbols():
    """Re-export symbols removed in transformers v5.4.0.

    Remote model code (e.g. DeepSeek-OCR) still imports these.
    ``check_imports`` in ``dynamic_module_utils.py`` validates imports at
    config-load time, so these must exist before any ``from_pretrained``.

    Removed symbols:
    - ``LlamaFlashAttention2`` -- replaced by unified ``LlamaAttention``
    - ``is_flash_attn_greater_or_equal_2_10`` -- replaced by
      ``is_flash_attn_greater_or_equal("2.10.0")``

    TODO(upstream): DeepSeek-OCR / deepseek_vl_v2 remote code needs update.
    """
    # LlamaFlashAttention2
    try:
        import logging

        # Importing modeling_llama triggers a deep import chain:
        #   modeling_llama -> modeling_utils -> quantizers -> torchao
        # torchao emits a noisy warning about incompatible torch versions
        # that is irrelevant here — suppress it during this import.
        _torchao_logger = logging.getLogger("torchao")
        _prev_level = _torchao_logger.level
        _torchao_logger.setLevel(logging.ERROR)
        try:
            from transformers.models.llama import modeling_llama
        finally:
            _torchao_logger.setLevel(_prev_level)

        if not hasattr(modeling_llama, "LlamaFlashAttention2"):
            if hasattr(modeling_llama, "LlamaAttention"):
                modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
    except ImportError:
        logger.warning(
            "Could not import transformers.models.llama.modeling_llama; "
            "LlamaFlashAttention2 compat patch not applied."
        )

    # is_flash_attn_greater_or_equal_2_10
    try:
        import transformers.utils as _u

        if not hasattr(_u, "is_flash_attn_greater_or_equal_2_10"):
            if hasattr(_u, "is_flash_attn_greater_or_equal"):
                _u.is_flash_attn_greater_or_equal_2_10 = (
                    lambda: _u.is_flash_attn_greater_or_equal("2.10.0")
                )
            else:
                _u.is_flash_attn_greater_or_equal_2_10 = lambda: False
    except ImportError:
        logger.warning(
            "Could not import transformers.utils; "
            "is_flash_attn_greater_or_equal_2_10 compat patch not applied."
        )


def _patch_image_processor_kwargs():
    """Allow remote image processors that lack ``**kwargs`` in preprocess().

    Transformers v5.4 passes new kwargs (e.g. ``device``) through
    ``BaseImageProcessor.__call__`` -> ``preprocess()``.  Remote model code
    (e.g. KimiVL) that defines ``preprocess()`` without ``**kwargs`` will
    crash with ``TypeError``.

    Fix: wrap ``__call__`` and filter unsupported kwargs before invoking
    ``preprocess()``.  The accepted-kwargs set is cached per processor class:
    apart from avoiding the exception/logging slow path, this matters for VLM
    requests that preprocess many images on the request critical path.

    TODO(upstream): KimiVL image_processing_kimi_vl.py needs ``**kwargs``.
    """
    try:
        from transformers.image_processing_utils import BaseImageProcessor

        original = BaseImageProcessor.__call__
        accepted_kwargs_cache = {}
        warned_unsupported_kwargs = set()

        def safe_call(self, images, *args, **kwargs):
            processor_type = type(self)
            accepted_kwargs = accepted_kwargs_cache.get(processor_type)
            if accepted_kwargs is None and processor_type not in accepted_kwargs_cache:
                sig = inspect.signature(self.preprocess)
                params = sig.parameters
                if any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    accepted_kwargs = None
                else:
                    accepted_kwargs = frozenset(params)
                accepted_kwargs_cache[processor_type] = accepted_kwargs

            if accepted_kwargs is None:
                return original(self, images, *args, **kwargs)

            dropped = frozenset(kwargs) - accepted_kwargs
            if dropped:
                warning_key = (processor_type, dropped)
                if warning_key not in warned_unsupported_kwargs:
                    logger.warning(
                        "Image processor %s.preprocess() does not accept %s; "
                        "filtering them before preprocessing. Update the model's image "
                        "processor to accept **kwargs.",
                        processor_type.__name__,
                        sorted(dropped),
                    )
                    warned_unsupported_kwargs.add(warning_key)
                kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}
            return original(self, images, *args, **kwargs)

        BaseImageProcessor.__call__ = safe_call
    except ImportError:
        logger.debug(
            "_patch_image_processor_kwargs: BaseImageProcessor not importable, patch skipped"
        )


def _patch_image_process_cuda_tensor():
    """Fix ``process_image()`` crashing on CUDA tensors.

    Transformers v5.4's PIL image processing backend calls
    ``image.numpy()`` on torch tensors, which fails for CUDA tensors.
    Patch to call ``.cpu().numpy()`` instead.

    TODO(upstream): report to HF transformers.
    """
    try:
        import torch
        import transformers.image_processing_backends as ipb

        for cls_name in ("PilBackend", "PilImageProcessingMixin"):
            cls = getattr(ipb, cls_name, None)
            if cls is None or not hasattr(cls, "process_image"):
                continue
            original = cls.process_image

            def patched_process_image(
                self, image, *args, _orig=original, _Tensor=torch.Tensor, **kwargs
            ):
                if isinstance(image, _Tensor) and image.is_cuda:
                    image = image.cpu()
                return _orig(self, image, *args, **kwargs)

            cls.process_image = patched_process_image
    except ImportError:
        logger.debug(
            "_patch_image_process_cuda_tensor: required modules not importable, patch skipped"
        )


def _patch_nemotron_h_pattern():
    """Fix ``_pattern_to_list()`` crashing on ``-`` in hybrid_override_pattern.

    Nemotron-H models (e.g. NVIDIA-Nemotron-Nano-9B-v2) use patterns like
    ``M-M-M-MM-M-*-...`` where ``-`` denotes an MLP layer.  The upstream
    ``_pattern_to_list`` tries to map every character and crashes with
    ``KeyError: '-'``.  We skip ``-`` (and any other unmapped chars)
    since ``layers_block_type`` only tracks mamba/moe/attention layers.
    SGLang reads MLP positions from ``hybrid_override_pattern`` directly.

    TODO(upstream): report to HF transformers.
    """
    try:
        from transformers.models.nemotron_h.configuration_nemotron_h import (
            NemotronHConfig,
        )

        @staticmethod
        def _pattern_to_list(pattern: str) -> list:
            pattern_mapping = {
                "M": "mamba",
                "E": "moe",
                "*": "attention",
            }
            return [
                pattern_mapping[char] for char in pattern if char in pattern_mapping
            ]

        NemotronHConfig._pattern_to_list = _pattern_to_list
    except ImportError:
        logger.debug(
            "_patch_nemotron_h_pattern: NemotronHConfig not importable, patch skipped"
        )


# ---------------------------------------------------------------------------
# v5 general patches
# ---------------------------------------------------------------------------


def _ensure_clean_up_tokenization_compat() -> None:
    """Re-add ``clean_up_tokenization`` removed in transformers v5.

    Remote-code tokenizers (e.g. InternLM2Tokenizer) call
    ``self.clean_up_tokenization()`` which was a static method on
    ``PreTrainedTokenizerBase`` in v4 but removed in v5. Patch it back
    so existing HuggingFace Hub tokenizer code keeps working.
    """
    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "clean_up_tokenization"):
        return

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    PreTrainedTokenizerBase.clean_up_tokenization = clean_up_tokenization


def _ensure_is_torch_fx_available_compat() -> None:
    """Re-add ``is_torch_fx_available`` removed in transformers v5.

    Remote-code models (e.g. MiniCPM-V) import ``is_torch_fx_available``
    from ``transformers.utils.import_utils``.  The function was removed
    in v5.  Patch it back so existing HuggingFace Hub model code keeps
    working.  torch.fx is always available in PyTorch >= 2.0.
    """
    import transformers.utils.import_utils as _import_utils

    if hasattr(_import_utils, "is_torch_fx_available"):
        return

    _import_utils.is_torch_fx_available = lambda: True


# ---------------------------------------------------------------------------
# CI-only patches
# ---------------------------------------------------------------------------

_is_base_mistral_patched = False


def patch_is_base_mistral_in_ci():
    """Patch transformers' _patch_mistral_regex to avoid HF API calls in CI.

    transformers defines is_base_mistral as a local function inside
    _patch_mistral_regex, so it cannot be patched via module attribute.
    Instead we replace the entire _patch_mistral_regex classmethod with a
    version that simply returns the tokenizer unchanged.

    In CI this prevents exhausting the 3000 req/5min HF API rate limit.

    TODO(upstream): remove once transformers stops calling model_info()
    inside _patch_mistral_regex (or removes the method entirely).
    """
    global _is_base_mistral_patched
    if _is_base_mistral_patched:
        return

    from sglang.srt.environ import envs

    if not envs.SGLANG_IS_IN_CI.get():
        return

    from transformers import PreTrainedTokenizerFast

    if hasattr(PreTrainedTokenizerFast, "_patch_mistral_regex"):

        @classmethod
        def _noop_patch_mistral_regex(cls, tokenizer, *args, **kwargs):
            return tokenizer

        PreTrainedTokenizerFast._patch_mistral_regex = _noop_patch_mistral_regex
        logger.info("CI: patched _patch_mistral_regex to skip HF API calls")

    _is_base_mistral_patched = True
