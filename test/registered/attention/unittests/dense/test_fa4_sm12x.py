import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
    run_dense_fixture_eager,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "the SM12x FlashAttention-4 backend path",
)
class TestFA4Sm12xGatheredExtend(CustomTestCase):
    def test_gathered_prefix_matches_the_paged_call(self):
        # Per-rank heads of Qwen3-30B-A3B at TP 4; the prefix ends mid-page and the
        # extend is long enough for _dense_extend_wins.
        case = DenseAttentionCase(
            name="fa4_sm12x_extend_over_cached_prefix",
            backend="fa4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(4100,),
            extend_lens=(4096,),
        )
        fixture = build_dense_attention_fixture(
            self, case, head_dim=128, max_context_len=8208, dtype=torch.bfloat16
        )
        # With a cached prefix only the gathered branch calls the dense kernel.
        with patch.object(
            fixture.backend,
            "flash_attn_varlen_func",
            wraps=fixture.backend.flash_attn_varlen_func,
        ) as dense_call:
            gathered = run_dense_fixture_eager(fixture)
        dense_call.assert_called_once()

        fixture.backend._gathers_extend_kv = False
        paged = run_dense_fixture_eager(fixture)
        self.assertTrue(torch.equal(gathered, paged))


if __name__ == "__main__":
    unittest.main()
