# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import LongLive2T2VConfig
from sglang.multimodal_gen.configs.sample.longlive2 import LongLive2SamplingParams
from sglang.multimodal_gen.runtime.pipelines.wan_causal_dmd_pipeline import (
    WanCausalDMDPipeline,
)

class LongLive2Pipeline(WanCausalDMDPipeline):
    pipeline_name = "LongLive2Pipeline"
    pipeline_config_cls = LongLive2T2VConfig
    sampling_params_cls = LongLive2SamplingParams

EntryClass = LongLive2Pipeline
