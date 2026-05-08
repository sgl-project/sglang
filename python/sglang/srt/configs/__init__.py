"""Lazy re-exports for sglang.srt.configs.

Previously this module eagerly imported 30+ model-specific config submodules
at package-load time. Many of those drag transformers internals with them
(e.g. ``sglang.srt.configs.bailing_hybrid`` and
``sglang.srt.configs.deepseekvl2`` together account for ~9s of transformers
eager loads). Since most callers only need one config, switch to PEP-562
``__getattr__`` so submodules are loaded on first access.

Existing code can still do ``from sglang.srt.configs import ChatGLMConfig``;
only the touched submodule is loaded.
"""

from __future__ import annotations

# Map of attribute name -> (submodule, attr) for lazy loading.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "AfmoeConfig": ("afmoe", "AfmoeConfig"),
    "BailingHybridConfig": ("bailing_hybrid", "BailingHybridConfig"),
    "ChatGLMConfig": ("chatglm", "ChatGLMConfig"),
    "DbrxConfig": ("dbrx", "DbrxConfig"),
    "DeepseekVL2Config": ("deepseekvl2", "DeepseekVL2Config"),
    "DotsOCRConfig": ("dots_ocr", "DotsOCRConfig"),
    "DotsVLMConfig": ("dots_vlm", "DotsVLMConfig"),
    "ExaoneConfig": ("exaone", "ExaoneConfig"),
    "FalconH1Config": ("falcon_h1", "FalconH1Config"),
    "GraniteMoeHybridConfig": ("granitemoehybrid", "GraniteMoeHybridConfig"),
    "MultiModalityConfig": ("janus_pro", "MultiModalityConfig"),
    "JetNemotronConfig": ("jet_nemotron", "JetNemotronConfig"),
    "JetVLMConfig": ("jet_vlm", "JetVLMConfig"),
    "KimiK25Config": ("kimi_k25", "KimiK25Config"),
    "KimiLinearConfig": ("kimi_linear", "KimiLinearConfig"),
    "KimiVLConfig": ("kimi_vl", "KimiVLConfig"),
    "MoonViTConfig": ("kimi_vl_moonvit", "MoonViTConfig"),
    "Lfm2Config": ("lfm2", "Lfm2Config"),
    "Lfm2MoeConfig": ("lfm2_moe", "Lfm2MoeConfig"),
    "Lfm2VlConfig": ("lfm2_vl", "Lfm2VlConfig"),
    "LongcatFlashConfig": ("longcat_flash", "LongcatFlashConfig"),
    "NemotronH_Nano_Omni_Reasoning_V3_Config": (
        "nano_nemotron_vl",
        "NemotronH_Nano_Omni_Reasoning_V3_Config",
    ),
    "NemotronH_Nano_VL_V2_Config": (
        "nano_nemotron_vl",
        "NemotronH_Nano_VL_V2_Config",
    ),
    "NemotronHConfig": ("nemotron_h", "NemotronHConfig"),
    "Olmo3Config": ("olmo3", "Olmo3Config"),
    "Qwen3_5Config": ("qwen3_5", "Qwen3_5Config"),
    "Qwen3_5MoeConfig": ("qwen3_5", "Qwen3_5MoeConfig"),
    "Qwen3ASRConfig": ("qwen3_asr", "Qwen3ASRConfig"),
    "Qwen3NextConfig": ("qwen3_next", "Qwen3NextConfig"),
    "Step3TextConfig": ("step3_vl", "Step3TextConfig"),
    "Step3VisionEncoderConfig": ("step3_vl", "Step3VisionEncoderConfig"),
    "Step3VLConfig": ("step3_vl", "Step3VLConfig"),
    "Step3p5Config": ("step3p5", "Step3p5Config"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'sglang.srt.configs' has no attribute {name!r}")
    submodule_name, attr_name = _LAZY_ATTRS[name]
    import importlib

    module = importlib.import_module(f"sglang.srt.configs.{submodule_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
