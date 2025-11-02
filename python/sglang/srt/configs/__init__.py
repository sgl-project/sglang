from sglang.srt.configs.chatglm import ChatGLMConfig
from sglang.srt.configs.dbrx import DbrxConfig
from sglang.srt.configs.deepseekvl2 import DeepseekVL2Config
from sglang.srt.configs.dots_ocr import DotsOCRConfig
from sglang.srt.configs.dots_vlm import DotsVLMConfig
from sglang.srt.configs.exaone import ExaoneConfig
from sglang.srt.configs.falcon_h1 import FalconH1Config
from sglang.srt.configs.janus_pro import MultiModalityConfig
from sglang.srt.configs.jet_nemotron import JetNemotronConfig
from sglang.srt.configs.jet_vlm import JetVLMConfig
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.configs.kimi_vl import KimiVLConfig
from sglang.srt.configs.kimi_vl_moonvit import MoonViTConfig
from sglang.srt.configs.longcat_flash import LongcatFlashConfig
from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.srt.configs.olmo3 import Olmo3Config
from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.configs.step3_vl import (
    Step3TextConfig,
    Step3VisionEncoderConfig,
    Step3VLConfig,
)

__all__ = [
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
    "Qwen3NextConfig",
    "DotsVLMConfig",
    "DotsOCRConfig",
    "FalconH1Config",
    "NemotronHConfig",
    "NemotronH_Nano_VL_V2_Config",
    "JetNemotronConfig",
    "JetVLMConfig"
]


def __getattr__(name):
    """Lazily import config classes on first access."""
    if name == "ChatGLMConfig":
        from sglang.srt.configs.chatglm import ChatGLMConfig

        return ChatGLMConfig
    elif name == "DbrxConfig":
        from sglang.srt.configs.dbrx import DbrxConfig

        return DbrxConfig
    elif name == "DeepseekVL2Config":
        from sglang.srt.configs.deepseekvl2 import DeepseekVL2Config

        return DeepseekVL2Config
    elif name == "DotsOCRConfig":
        from sglang.srt.configs.dots_ocr import DotsOCRConfig

        return DotsOCRConfig
    elif name == "DotsVLMConfig":
        from sglang.srt.configs.dots_vlm import DotsVLMConfig

        return DotsVLMConfig
    elif name == "ExaoneConfig":
        from sglang.srt.configs.exaone import ExaoneConfig

        return ExaoneConfig
    elif name == "FalconH1Config":
        from sglang.srt.configs.falcon_h1 import FalconH1Config

        return FalconH1Config
    elif name == "MultiModalityConfig":
        from sglang.srt.configs.janus_pro import MultiModalityConfig

        return MultiModalityConfig
    elif name == "KimiLinearConfig":
        from sglang.srt.configs.kimi_linear import KimiLinearConfig

        return KimiLinearConfig
    elif name == "KimiVLConfig":
        from sglang.srt.configs.kimi_vl import KimiVLConfig

        return KimiVLConfig
    elif name == "MoonViTConfig":
        from sglang.srt.configs.kimi_vl_moonvit import MoonViTConfig

        return MoonViTConfig
    elif name == "LongcatFlashConfig":
        from sglang.srt.configs.longcat_flash import LongcatFlashConfig

        return LongcatFlashConfig
    elif name == "NemotronHConfig":
        from sglang.srt.configs.nemotron_h import NemotronHConfig

        return NemotronHConfig
    elif name == "NemotronH_Nano_VL_V2_Config":
        from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
        
        return NemotronH_Nano_VL_V2_Config
    elif name == "JetNemotronConfig":
        from sglang.srt.configs.jet_nemotron import JetNemotronConfig

        return JetNemotronConfig
    elif name == "JetVLMConfig":
        from sglang.srt.configs.jet_vlm import JetVLMConfig

        return JetVLMConfig
    elif name == "Olmo3Config":
        from sglang.srt.configs.olmo3 import Olmo3Config

        return Olmo3Config
    elif name == "Qwen3NextConfig":
        from sglang.srt.configs.qwen3_next import Qwen3NextConfig

        return Qwen3NextConfig
    elif name == "Step3TextConfig":
        from sglang.srt.configs.step3_vl import Step3TextConfig

        return Step3TextConfig
    elif name == "Step3VisionEncoderConfig":
        from sglang.srt.configs.step3_vl import Step3VisionEncoderConfig

        return Step3VisionEncoderConfig
    elif name == "Step3VLConfig":
        from sglang.srt.configs.step3_vl import Step3VLConfig

        return Step3VLConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
