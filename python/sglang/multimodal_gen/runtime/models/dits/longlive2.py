# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.models.dits.longlive2 import LongLive2VideoConfig
from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
    CausalWanTransformer3DModel,
)


class LongLive2Transformer3DModel(CausalWanTransformer3DModel):
    _fsdp_shard_conditions = LongLive2VideoConfig()._fsdp_shard_conditions
    _compile_conditions = LongLive2VideoConfig()._compile_conditions
    param_names_mapping = LongLive2VideoConfig().param_names_mapping
    reverse_param_names_mapping = LongLive2VideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LongLive2VideoConfig().lora_param_names_mapping


EntryClass = LongLive2Transformer3DModel
