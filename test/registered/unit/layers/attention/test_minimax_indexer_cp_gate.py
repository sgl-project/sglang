import unittest

from sglang.srt.layers.attention.minimax_sparse_ops.indexer_cp import (
    draft_is_chain_layout,
    unsupported_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# The run shape MiniMax-M3 CP was validated on; only the draft layout varies below.
SUPPORTED = dict(
    gfx950=True,
    sm90=False,
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

    def test_sm90_enables_only_validated_topologies(self):
        """Each rank must serve exactly one index head of the four; any other split,
        or another attention CP/DP layout, would merge candidates for the wrong query."""
        enabled = {
            (4, 4, 1, 1): True,
            (8, 8, 1, 1): True,  # head pairs replicated across ranks
            (2, 2, 1, 1): False,  # two index heads per rank
            (8, 4, 1, 2): False,  # DP attention not validated
            (8, 4, 2, 1): False,
        }
        for (tp, attn_tp, attn_cp, attn_dp), expected in enabled.items():
            with self.subTest(tp=tp, attn_tp=attn_tp, attn_cp=attn_cp, attn_dp=attn_dp):
                reason = unsupported_reason(
                    **{
                        **SUPPORTED,
                        "gfx950": False,
                        "sm90": True,
                        "tp_size": tp,
                        "attn_tp_size": attn_tp,
                        "attn_cp_size": attn_cp,
                        "attn_dp_size": attn_dp,
                    }
                )
                self.assertEqual(reason is None, expected, reason)
        for gfx950, sm90, attn_tp, expected in (
            (True, False, 8, False),  # TP8 is validated on SM90 only
            (False, False, 4, False),  # other CUDA architectures stay off
        ):
            with self.subTest(gfx950=gfx950, sm90=sm90, attn_tp=attn_tp):
                reason = unsupported_reason(
                    **{
                        **SUPPORTED,
                        "gfx950": gfx950,
                        "sm90": sm90,
                        "tp_size": attn_tp,
                        "attn_tp_size": attn_tp,
                    }
                )
                self.assertEqual(reason is None, expected, reason)


if __name__ == "__main__":
    unittest.main()
