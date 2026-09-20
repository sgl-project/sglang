# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.ltx_2 import (
    LTX2SamplingParams,
    LTX23HQSamplingParams,
)
from sglang.multimodal_gen.configs.sample.pi05 import Pi05SamplingParams
from sglang.multimodal_gen.configs.sample.qwenimage import QwenImageSamplingParams
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.configs.sample.wan import WanT2V_1_3B_SamplingParams
from sglang.multimodal_gen.registry import (
    _CONFIG_REGISTRY,
    _configs_discovered,
    _discover_and_register_configs,
    _get_config_info,
    get_pipeline_config_classes,
)

NO_REGISTER_MODULES = {
    "base",
    "diffusers_generic",
    "flux_finetuned",
    "longcat_image",
    "ltx_2",
    "ltx_2_5",
    "model_deployment_config",
    "qwen_image",
    "sana",
    "sana_video",
    "sana_wm",
    "sensenova_u1",
}

if __name__ == "__main__":
    assert _configs_discovered
    assert len(_CONFIG_REGISTRY) == 62

    package = importlib.import_module("sglang.multimodal_gen.configs.pipeline_configs")
    modules = [
        importlib.import_module(info.name)
        for info in pkgutil.iter_modules(package.__path__, package.__name__ + ".")
    ]
    assert len(modules) == 33
    for module in modules:
        short_name = module.__name__.rsplit(".", 1)[-1]
        assert short_name in NO_REGISTER_MODULES or hasattr(module, "register")

    _discover_and_register_configs()
    assert len(_CONFIG_REGISTRY) == 62

    def check(path, pipeline_config_cls, sampling_param_cls):
        info = _get_config_info(path)
        assert info.pipeline_config_cls is pipeline_config_cls
        assert info.sampling_param_cls is sampling_param_cls

    check(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        WanT2V480PConfig,
        WanT2V_1_3B_SamplingParams,
    )
    check("lerobot/pi05_base", Pi05PipelineConfig, Pi05SamplingParams)
    check("nvidia/Cosmos3-Nano", Cosmos3Config, Cosmos3SamplingParams)
    check("Lightricks/LTX-2", LTX2PipelineConfig, LTX2SamplingParams)
    check("Qwen/Qwen-Image", QwenImagePipelineConfig, QwenImageSamplingParams)
    check(
        "Efficient-Large-Model/SANA-WM_bidirectional",
        SanaWMPipelineConfig,
        SanaWMSamplingParams,
    )

    assert get_pipeline_config_classes("LTX2TwoStageHQPipeline") == (
        LTX2PipelineConfig,
        LTX23HQSamplingParams,
    )

    print("OK")
