import unittest

from sglang.srt.layers.attention.minimax_sparse_ops.indexer_cp import (
    draft_is_chain_layout,
    unsupported_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The run shape MiniMax-M3 CP was validated on; only the draft layout varies below.
SUPPORTED = dict(
    gfx950=True,
    tp_size=4,
    attn_tp_size=4,
    attn_cp_size=1,
    attn_dp_size=1,
    index_heads=4,
    kv_heads=4,
    head_dim=128,
    block_size=128,
    topk=16,
    score_type="max",
    max_context_len=1 << 20,
    radix_topk=True,
    draft_is_chain=True,
    tbo=False,
    hisparse=False,
    fp8_query=False,
    dense_sparse_decode=False,
)


class TestMiniMaxIndexerCPGate(CustomTestCase):
    def test_only_validated_draft_layouts_enable_cp(self):
        """The gate keyed on EAGLE's top-k, so every other algorithm read as a chain
        and enabled CP on a verify layout it cannot score (silently wrong top-k)."""
        enabled = {
            (None, None): True,
            ("EAGLE", 1): True,
            ("EAGLE3", 1): True,
            ("EAGLE3", 4): False,  # tree draft
            ("DSPARK", None): False,  # ragged verify lengths
            ("NGRAM", None): False,  # tree lives in the verify mask
            ("DFLASH", None): False,
            ("STANDALONE", 1): False,
        }
        for (algorithm, eagle_topk), expected in enabled.items():
            with self.subTest(algorithm=algorithm, eagle_topk=eagle_topk):
                reason = unsupported_reason(
                    **{
                        **SUPPORTED,
                        "draft_is_chain": draft_is_chain_layout(algorithm, eagle_topk),
                    }
                )
                self.assertEqual(reason is None, expected, reason)


if __name__ == "__main__":
    unittest.main()
