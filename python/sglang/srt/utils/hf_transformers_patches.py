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
    """Fix rope_parameters validation for unregistered model types.

    For unregistered model types (e.g. ``deepseek_v32``), the generic
    ``PretrainedConfig`` lacks a ``rope_parameters`` field so the conversion
    that injects ``rope_theta`` from the top-level config is skipped.
    Additionally, ``standardize_rope_params()`` accesses
    ``self.max_position_embeddings`` during ``__post_init__`` before extra
    kwargs are set as attributes, causing ``AttributeError``.

    Fix: (1) patch ``from_dict`` to inject ``rope_theta`` into
    ``rope_scaling``, (2) guard ``standardize_rope_params`` against missing
    ``max_position_embeddings``.

    TODO(upstream): remove once unregistered model types handle rope
    standardization correctly in transformers.
    """
    from transformers import PretrainedConfig

    original = PretrainedConfig.from_dict.__func__

    @classmethod  # type: ignore[misc]
    def patched(cls, config_dict, **kwargs):
        rope_scaling = config_dict.get("rope_scaling")
        rope_theta = config_dict.get("rope_theta")
        if (
            isinstance(rope_scaling, dict)
            and rope_theta is not None
            and "rope_theta" not in rope_scaling
        ):
            config_dict = config_dict.copy()
            config_dict["rope_scaling"] = {**rope_scaling, "rope_theta": rope_theta}
        return original(cls, config_dict, **kwargs)

    PretrainedConfig.from_dict = patched

    # standardize_rope_params accesses self.max_position_embeddings before
    # __post_init__ sets extra kwargs — skip when the attribute is absent.
    if hasattr(PretrainedConfig, "standardize_rope_params"):
        _orig_standardize = PretrainedConfig.standardize_rope_params

        def _safe_standardize(self):
            if not hasattr(self, "max_position_embeddings"):
                return
            return _orig_standardize(self)

        PretrainedConfig.standardize_rope_params = _safe_standardize


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
        from transformers.models.llama import modeling_llama

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

    Fix: wrap ``__call__`` to catch ``TypeError`` and retry with only the
    kwargs that ``preprocess()`` actually accepts.

    TODO(upstream): KimiVL image_processing_kimi_vl.py needs ``**kwargs``.
    """
    try:
        from transformers.image_processing_utils import BaseImageProcessor

        original = BaseImageProcessor.__call__

        def safe_call(self, images, *args, **kwargs):
            try:
                return original(self, images, *args, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument" not in str(e):
                    raise
                sig = inspect.signature(self.preprocess)
                params = sig.parameters
                if any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    raise
                dropped = {k for k in kwargs if k not in params}
                if dropped:
                    logger.warning(
                        "Image processor %s.preprocess() does not accept %s; "
                        "retrying without them. Update the model's image processor "
                        "to accept **kwargs.",
                        type(self).__name__,
                        dropped,
                    )
                valid = {k: v for k, v in kwargs.items() if k in params}
                return original(self, images, *args, **valid)

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
