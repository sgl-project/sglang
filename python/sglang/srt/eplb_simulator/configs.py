from dataclasses import dataclass
from typing import Optional

from sglang.srt.managers.expert_location import ModelConfigForExpertLocation


@dataclass
class MyServerArgs:
    # When prefill, this is equivalent to `chunked_prefill_size`
    num_tokens_in_batch_overall: int
    ep_num_redundant_experts: int
    nnodes: int
    tp_size: int
    enable_expert_location_by_eplb: bool
    eplb_rebalance_num_iterations: Optional[int] = None
    # init_expert_location: Optional[str]
    deepseek_eplb_hack_shuffle: bool = False


# TODO generalize
# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
MY_MODEL_CONFIG_FOR_EXPERT_LOCATION = ModelConfigForExpertLocation(
    num_layers=61,
    num_logical_experts=256,
    num_groups=8,
)
MY_MODEL_CONFIG_NUM_EXPERTS_PER_TOK = 8
