from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dispatch_preserves_negative_routing_sentinel():
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.moe_ep_size = 2
    dispatcher.skip_local_expert_mapping = False
    dispatcher.local_expert_mapping = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
    dispatcher.use_aiter_moe_runner = False
    dispatcher.expert_mask_gpu = None

    topk_ids = torch.tensor([[-1, 3], [2, 0]], dtype=torch.int32)
    topk_output = StandardTopKOutput(
        topk_weights=torch.ones_like(topk_ids, dtype=torch.float32),
        topk_ids=topk_ids,
        router_logits=None,
    )

    with patch(
        "sglang.srt.layers.moe.token_dispatcher.standard."
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
        return_value=False,
    ):
        output = dispatcher.dispatch(torch.ones((2, 4)), topk_output)

    expected = torch.tensor([[-1, 1], [0, -1]], dtype=torch.int32)
    assert torch.equal(output.topk_output.topk_ids, expected)
