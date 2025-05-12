from dataclasses import dataclass
from typing import Literal, Optional

from sglang.srt.managers.expert_location import ModelConfigForExpertLocation


@dataclass
class MyServerArgs:
    # When prefill, this is equivalent to `chunked_prefill_size`
    num_tokens_in_batch_overall: int
    ep_num_redundant_experts: int
    nnodes: int
    tp_size: int
    expert_location_mode: Optional[Literal["previous_chunk", "global_average"]]
    phase: Literal["prefill", "decode"]
    eplb_rebalance_num_iterations: Optional[int] = None
    # init_expert_location: Optional[str]
    deepseek_eplb_hack_shuffle: bool = False
    decode_max_left_padding: int = 500


# TODO generalize
# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
MY_MODEL_CONFIG_FOR_EXPERT_LOCATION = ModelConfigForExpertLocation(
    num_layers=61,
    num_logical_experts=256,
    num_groups=8,
)
MY_MODEL_CONFIG_NUM_EXPERTS_PER_TOK = 8
