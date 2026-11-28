"""CPU-only checks for the existing token-count helper and ForwardBatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.kernels.fused_op as fused_op
import sglang.kernels.ops.elementwise.local_token_count as local_count
import sglang.srt.layers.dp_attention as dp_attention
import sglang.srt.model_executor.forward_batch_info as forward_batch_info
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLocalTokenCountDispatch(CustomTestCase):
    def test_forward_batch_passes_cpu_mirror_to_platform(self):
        op = local_count.LocalTokenCountOp()

        def from_host(
            op, global_count, rank_offset, tokens_per_rank, *, global_count_cpu
        ):
            self.assertEqual(
                (rank_offset, tokens_per_rank, global_count_cpu), (8, 4, 10)
            )
            return torch.tensor(2, dtype=global_count.dtype)

        empty = torch.empty(0, dtype=torch.int32)
        batch = forward_batch_info.ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=empty,
            req_pool_indices=empty,
            seq_lens=empty,
            out_cache_loc=empty,
            seq_lens_sum=0,
        )
        batch.global_num_token_non_padded = torch.tensor(10, dtype=torch.int32)
        batch.global_num_token_non_padded_cpu = 10
        batch.global_num_tokens_cpu = [16]
        with (
            patch.object(
                forward_batch_info,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=4, attn_tp_rank=2),
            ),
            patch.object(local_count, "_LOCAL_TOKEN_COUNT", op),
            patch.object(fused_op.BaseFusedOp, "_oot_forward_registry", {}),
            patch.object(fused_op, "_oot_dispatch_key", return_value="test_platform"),
            patch.object(fused_op, "_forced_backend", None),
            patch.object(dp_attention, "require_mlp_tp_gather", return_value=False),
            patch.object(op, "forward_native", side_effect=AssertionError),
        ):
            local_count.LocalTokenCountOp.register_oot_forward(
                op_cls=local_count.LocalTokenCountOp,
                fn=from_host,
                platform_key="test_platform",
            )
            batch.set_local_num_token_non_padded(sharded=True)
        self.assertEqual(batch.num_token_non_padded.item(), 2)


if __name__ == "__main__":
    unittest.main()
