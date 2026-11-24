"""CPU-only test: routed MXFP4 MoE row slicing is output-identical to a single call.

Above _ROUTED_MOE_MAX_ROWS the routed path calls the FlashInfer kernel once per row
slice instead of once for the whole batch. Rows are independent, so the finalized
output must be bit-identical either way. A slice that misaligned the hidden rows
against the routing ids, the routing weights or the output rows would not be, and
the fake kernel here derives each output row from all of them so a misalignment
shows up as a value mismatch.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

HIDDEN = 64
TOPK = 2
NUM_EXPERTS = 8
INTERMEDIATE = 32


def _fake_kernel(**kw):
    """Per-row value derived from every per-row input the slicing has to keep aligned."""
    x = kw["hidden_states"]
    ids, weights = kw["topk_ids"]
    out = kw["output"]
    value = (
        x.float().sum(dim=1)
        + ids.float().sum(dim=1) * 7.0
        + weights.float().sum(dim=1) * 13.0
    )
    out.copy_(value[:, None].expand_as(out).to(out.dtype))
    # The FlashInfer op returns a list whose first entry is the output tensor; the
    # unsliced path reads result[0], so the fake has to keep that shape.
    return [out]


def _layer():
    return SimpleNamespace(
        w13_weight=torch.zeros(
            NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.uint8
        ),
        w2_weight=torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16),
        w13_weight_scale_inv=torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, 4),
        w2_weight_scale_inv=torch.zeros(NUM_EXPERTS, HIDDEN, 4),
        num_local_experts=NUM_EXPERTS,
        num_experts=NUM_EXPERTS,
        moe_ep_rank=0,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        use_cuda_mxfp4_decode_moe=False,
    )


class TestMxfp4RoutedMoeRowSlicing(CustomTestCase):
    def _apply(self, num_tokens, max_rows, seed=1234):
        torch.manual_seed(seed)
        hidden = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16)
        ids = torch.randint(0, NUM_EXPERTS, (num_tokens, TOPK), dtype=torch.int32)
        weights = torch.rand(num_tokens, TOPK, dtype=torch.float32)

        method = mod.Mxfp4FlashinferTrtllmMoEMethod.__new__(
            mod.Mxfp4FlashinferTrtllmMoEMethod
        )
        method.flashinfer_mxfp4_moe_precision = "bf16"
        method._gemm1_clamp_limit_tensor = None
        method._bypassed_routing_method = None

        topk_output = SimpleNamespace(
            topk_ids=ids, topk_weights=weights, format="standard"
        )
        dispatch_output = SimpleNamespace(hidden_states=hidden, topk_output=topk_output)
        sizes = []

        def recording(**kw):
            sizes.append(kw["hidden_states"].shape[0])
            return _fake_kernel(**kw)

        checker = SimpleNamespace(
            format_is_standard=staticmethod(lambda _: True),
            format_is_bypassed=staticmethod(lambda _: False),
        )
        runner = SimpleNamespace(
            is_deferred_finalize_enabled=lambda: False,
            trtllm_moe_enable_pdl=lambda _n: False,
            run_cuda_mxfp4_decode_moe=None,
            _make_deferred_finalize_output=None,
        )
        with (
            patch.object(
                mod, "trtllm_fp4_block_scale_routed_moe", recording, create=True
            ),
            patch.object(mod, "_ROUTED_MOE_MAX_ROWS", max_rows),
            patch.object(mod, "routed_hidden_size", lambda _layer: HIDDEN, create=True),
            patch.object(
                mod, "use_symmetric_memory", lambda *a, **k: nullcontext(), create=True
            ),
            patch.object(mod, "is_allocation_symmetric", lambda: False, create=True),
            patch.object(
                mod, "get_parallel", lambda: SimpleNamespace(tp_group=None), create=True
            ),
            patch.dict(
                "sys.modules",
                {"sglang.srt.layers.moe.moe_runner.flashinfer_trtllm": runner},
            ),
            patch("sglang.srt.layers.moe.topk.TopKOutputChecker", checker),
        ):
            result = method.apply(_layer(), dispatch_output)
        return result.hidden_states, sizes

    def test_sliced_matches_unsliced_with_a_partial_final_slice(self):
        # 4096 is the shipped bucket; 9000 rows give two full slices and a 808-row tail.
        for num_tokens in (4097, 8192, 9000):
            with self.subTest(num_tokens=num_tokens):
                whole, whole_sizes = self._apply(num_tokens, max_rows=1 << 30)
                sliced, sliced_sizes = self._apply(num_tokens, max_rows=4096)
                self.assertEqual(whole_sizes, [num_tokens])
                self.assertEqual(sum(sliced_sizes), num_tokens)
                self.assertGreater(len(sliced_sizes), 1)
                self.assertTrue(torch.equal(whole, sliced))

    def test_at_or_below_the_bucket_is_one_call(self):
        for num_tokens in (1, 4096):
            with self.subTest(num_tokens=num_tokens):
                _, sizes = self._apply(num_tokens, max_rows=4096)
                self.assertEqual(sizes, [num_tokens])


if __name__ == "__main__":
    unittest.main()
