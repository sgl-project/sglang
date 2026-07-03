from importlib import import_module


_CONFIG_IMPORTS = [
    ("sglang.srt.configs.afmoe", ("AfmoeConfig",)),
    ("sglang.srt.configs.bailing_hybrid", ("BailingHybridConfig",)),
    ("sglang.srt.configs.chatglm", ("ChatGLMConfig",)),
    ("sglang.srt.configs.cohere2_moe", ("Cohere2MoeConfig",)),
    ("sglang.srt.configs.dbrx", ("DbrxConfig",)),
    ("sglang.srt.configs.deepseekvl2", ("DeepseekVL2Config",)),
    ("sglang.srt.configs.dots_ocr", ("DotsOCRConfig",)),
    ("sglang.srt.configs.dots_vlm", ("DotsVLMConfig",)),
    ("sglang.srt.configs.exaone", ("ExaoneConfig",)),
    ("sglang.srt.configs.falcon_h1", ("FalconH1Config",)),
    ("sglang.srt.configs.granitemoehybrid", ("GraniteMoeHybridConfig",)),
    ("sglang.srt.configs.interns2preview", ("InternS2PreviewConfig",)),
    ("sglang.srt.configs.janus_pro", ("MultiModalityConfig",)),
    ("sglang.srt.configs.jet_nemotron", ("JetNemotronConfig",)),
    ("sglang.srt.configs.jet_vlm", ("JetVLMConfig",)),
    ("sglang.srt.configs.kimi_k25", ("KimiK25Config",)),
    ("sglang.srt.configs.kimi_linear", ("KimiLinearConfig",)),
    ("sglang.srt.configs.kimi_vl", ("KimiVLConfig",)),
    ("sglang.srt.configs.kimi_vl_moonvit", ("MoonViTConfig",)),
    ("sglang.srt.configs.laguna", ("LagunaConfig",)),
    ("sglang.srt.configs.lfm2", ("Lfm2Config",)),
    ("sglang.srt.configs.lfm2_moe", ("Lfm2MoeConfig",)),
    ("sglang.srt.configs.lfm2_vl", ("Lfm2VlConfig",)),
    ("sglang.srt.configs.locate_anything", ("LocateAnythingConfig",)),
    ("sglang.srt.configs.longcat_flash", ("LongcatFlashConfig",)),
    (
        "sglang.srt.configs.minicpmv4_6",
        ("MiniCPMV4_6Config", "MiniCPMV4_6VisionConfig"),
    ),
    (
        "sglang.srt.configs.nano_nemotron_vl",
        (
            "NemotronH_Nano_Omni_Reasoning_V3_Config",
            "NemotronH_Nano_VL_V2_Config",
        ),
    ),
    (
        "sglang.srt.configs.nemotron_h",
        ("NemotronHConfig", "NemotronHPuzzleConfig"),
    ),
    ("sglang.srt.configs.olmo3", ("Olmo3Config",)),
    ("sglang.srt.configs.qwen3_5", ("Qwen3_5Config", "Qwen3_5MoeConfig")),
    ("sglang.srt.configs.qwen3_asr", ("Qwen3ASRConfig",)),
    ("sglang.srt.configs.qwen3_next", ("Qwen3NextConfig",)),
    (
        "sglang.srt.configs.step3_vl",
        ("Step3TextConfig", "Step3VisionEncoderConfig", "Step3VLConfig"),
    ),
    ("sglang.srt.configs.step3p5", ("Step3p5Config",)),
    ("sglang.srt.configs.step3p7", ("Step3p7Config",)),
    ("sglang.srt.configs.unlimited_ocr", ("UnlimitedVLConfig",)),
    ("sglang.srt.configs.zaya", ("ZayaConfig",)),
]


def _try_import_configs(module_name: str, names: tuple[str, ...]) -> None:
    try:
        module = import_module(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return
    for name in names:
        globals()[name] = getattr(module, name)


for _module_name, _names in _CONFIG_IMPORTS:
    _try_import_configs(_module_name, _names)


__all__ = [
    "AfmoeConfig",
    "BailingHybridConfig",
    "ExaoneConfig",
    "ChatGLMConfig",
    "DbrxConfig",
    "DeepseekVL2Config",
    "LongcatFlashConfig",
    "MultiModalityConfig",
    "KimiVLConfig",
    "MoonViTConfig",
    "Step3VLConfig",
    "Step3TextConfig",
    "Step3VisionEncoderConfig",
    "Olmo3Config",
    "KimiLinearConfig",
    "KimiK25Config",
    "LagunaConfig",
    "Qwen3NextConfig",
    "Qwen3_5Config",
    "Qwen3_5MoeConfig",
    "InternS2PreviewConfig",
    "DotsVLMConfig",
    "DotsOCRConfig",
    "FalconH1Config",
    "GraniteMoeHybridConfig",
    "Lfm2Config",
    "Lfm2MoeConfig",
    "Lfm2VlConfig",
    "LocateAnythingConfig",
    "MiniCPMV4_6Config",
    "MiniCPMV4_6VisionConfig",
    "NemotronHConfig",
    "NemotronHPuzzleConfig",
    "NemotronH_Nano_VL_V2_Config",
    "NemotronH_Nano_Omni_Reasoning_V3_Config",
    "JetNemotronConfig",
    "JetVLMConfig",
    "Step3p5Config",
    "Step3p7Config",
    "Qwen3ASRConfig",
    "UnlimitedVLConfig",
    "ZayaConfig",
]

__all__ = [name for name in __all__ if name in globals()]
