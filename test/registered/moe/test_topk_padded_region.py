import unittest

import torch

import sglang.srt.layers.moe.topk as topk_mod
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _can_fuse_padded_region,
    _fill_padded_rows,
    _mask_topk_ids_padded_region,
    _post_process_topk_ids,
    _zero_topk_weights_padded_region,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd")

_IS_HIP = is_hip()

torch.manual_seed(1234)


def _eager_fill_padded_rows(x, num_token_non_padded, fill_value):
    out = x.clone()
    indices = torch.arange(0, x.shape[0], device=x.device)
    out[indices >= num_token_non_padded, :] = fill_value
    return out


@unittest.skipUnless(
    torch.cuda.is_available(), "fused padded-region kernel needs a GPU"
)
class TestTopkPaddedRegion(CustomTestCase):
    DEVICE = "cuda"

    def test_matches_eager_across_shapes(self):
        configs = [
            # (n_tokens, topk, dtype, fill, helper)
            (37, 9, torch.float32, 0.0, _zero_topk_weights_padded_region),
            (256, 8, torch.float32, 0.0, _zero_topk_weights_padded_region),
            (1, 9, torch.float32, 0.0, _zero_topk_weights_padded_region),
            (37, 9, torch.int32, -1, _mask_topk_ids_padded_region),
            (512, 16, torch.int32, -1, _mask_topk_ids_padded_region),
        ]
        for n, k, dtype, fill, helper in configs:
            for n_valid in (0, 1, 5, n - 1, n):
                with self.subTest(n=n, k=k, dtype=dtype, n_valid=n_valid):
                    if dtype.is_floating_point:
                        x = torch.rand((n, k), device=self.DEVICE, dtype=dtype) + 0.5
                    else:
                        x = torch.randint(
                            0, 100, (n, k), device=self.DEVICE, dtype=dtype
                        )
                    self.assertTrue(_can_fuse_padded_region(x))
                    num_token_non_padded = torch.tensor(
                        n_valid, device=self.DEVICE, dtype=torch.int32
                    )
                    expected = _eager_fill_padded_rows(x, num_token_non_padded, fill)
                    fused = x.clone()
                    helper(fused, num_token_non_padded)
                    self.assertTrue(torch.equal(fused, expected))

    def test_none_pad_count_is_noop(self):
        x = torch.rand((16, 8), device=self.DEVICE, dtype=torch.float32) + 0.5
        ref = x.clone()
        _zero_topk_weights_padded_region(x, None)
        self.assertTrue(torch.equal(x, ref))

    def test_non_contiguous_falls_back_to_eager(self):
        # A column slice is not row-major contiguous, so the fused path must be
        # skipped while still producing the correct result via the eager branch.
        base = torch.rand((32, 16), device=self.DEVICE, dtype=torch.float32) + 0.5
        view = base[:, ::2]
        self.assertFalse(_can_fuse_padded_region(view))
        num_token_non_padded = torch.tensor(5, device=self.DEVICE, dtype=torch.int32)
        expected = _eager_fill_padded_rows(view, num_token_non_padded, 0.0)
        _zero_topk_weights_padded_region(view, num_token_non_padded)
        self.assertTrue(torch.equal(view, expected))

    def test_cuda_graph_capture_and_replay(self):
        n, k = 256, 9
        weights = torch.rand((n, k), device=self.DEVICE, dtype=torch.float32) + 0.5
        num_token_non_padded = torch.tensor(n, device=self.DEVICE, dtype=torch.int32)

        # Warmup on a side stream before capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                tmp = weights.clone()
                _zero_topk_weights_padded_region(tmp, num_token_non_padded)
        torch.cuda.current_stream().wait_stream(side)

        work = weights.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _zero_topk_weights_padded_region(work, num_token_non_padded)

        for n_valid in (n, 5, 0, 100):
            work.copy_(weights)
            num_token_non_padded.fill_(n_valid)
            graph.replay()
            torch.cuda.synchronize()
            expected = _eager_fill_padded_rows(weights, num_token_non_padded, 0.0)
            self.assertTrue(torch.equal(work, expected))

    def test_invalid_pad_count_tensor_raises(self):
        x = torch.rand((8, 8), device=self.DEVICE, dtype=torch.float32)
        with self.assertRaises(TypeError):
            _fill_padded_rows(x, 4, 0.0)  # python int, not a tensor
        with self.assertRaises(ValueError):
            _fill_padded_rows(
                x,
                torch.tensor([1, 2], device=self.DEVICE, dtype=torch.int32),
                0.0,
            )
        with self.assertRaises(TypeError):
            _fill_padded_rows(
                x,
                torch.tensor(4.0, device=self.DEVICE, dtype=torch.float32),
                0.0,
            )


@unittest.skipUnless(torch.cuda.is_available(), "padded-region masking needs a GPU")
class TestZeroPaddedRegionIdempotent(CustomTestCase):
    """The HIP post-process keeps a single padded-row zeroing pass. Removing the
    earlier (redundant) pass is only safe if zeroing is idempotent."""

    DEVICE = "cuda"

    def test_zeroing_twice_equals_once(self):
        for n, k in [(37, 9), (256, 8), (512, 16)]:
            for n_valid in (0, 5, n - 1, n):
                with self.subTest(n=n, k=k, n_valid=n_valid):
                    base = torch.rand((n, k), device=self.DEVICE) + 0.5
                    pad = torch.tensor(n_valid, device=self.DEVICE, dtype=torch.int32)

                    once = base.clone()
                    _zero_topk_weights_padded_region(once, pad)

                    twice = base.clone()
                    _zero_topk_weights_padded_region(twice, pad)
                    _zero_topk_weights_padded_region(twice, pad)

                    self.assertTrue(torch.equal(once, twice))


@unittest.skipUnless(
    _IS_HIP and torch.cuda.is_available(),
    "_post_process_topk_ids padded masking is HIP-only",
)
class TestPostProcessPaddedMaskingHip(CustomTestCase):
    DEVICE = "cuda"

    def _run(self, n=256, k=8, n_valid=5):
        topk_weights = torch.rand((n, k), device=self.DEVICE, dtype=torch.float32) + 0.5
        topk_ids = torch.randint(0, 64, (n, k), device=self.DEVICE, dtype=torch.int32)
        router_logits = torch.rand((n, 64), device=self.DEVICE, dtype=torch.float32)
        pad = torch.tensor(n_valid, device=self.DEVICE, dtype=torch.int32)
        cfg = TopKConfig(top_k=k, num_fused_shared_experts=0)
        _, out_weights, _ = _post_process_topk_ids(
            topk_ids,
            topk_weights,
            cfg,
            router_logits,
            layer_id=0,
            num_token_non_padded=pad,
        )
        return out_weights, n_valid

    def test_padded_rows_zeroed_by_default(self):
        # Flag off (default): padded rows must be fully zeroed, valid rows kept.
        self.assertFalse(topk_mod._skip_hip_pad_mask)
        out, n_valid = self._run()
        self.assertTrue(torch.all(out[n_valid:] == 0.0))
        self.assertTrue(torch.all(out[:n_valid] > 0.0))

    def test_flag_skips_masking(self):
        # Flag on: padded rows are left untouched (kept non-zero here).
        orig = topk_mod._skip_hip_pad_mask
        topk_mod._skip_hip_pad_mask = True
        try:
            out, n_valid = self._run()
            self.assertTrue(torch.all(out[n_valid:] > 0.0))
        finally:
            topk_mod._skip_hip_pad_mask = orig


@unittest.skipUnless(torch.cuda.is_available(), "padded-region masking needs a GPU")
@unittest.skipIf(_IS_HIP, "HIP keeps the pre-mask routing contract (AITER/MORI)")
class TestSelectExpertsCustomRoutingPadMask(CustomTestCase):
    """``select_experts`` must accept ``num_token_non_padded`` together with a
    ``custom_routing_function`` and mask the padded region to -1.

    Bug regression (EP MoE dispatch overflow): DP-attention/SP pad rows
    carry garbage router input; a custom routing function can emit the same
    expert id in every top-k slot for them (the masked argmax degenerates on
    non-finite scores), which overflows an EP dispatch pool's
    min(top_k, experts_per_rank) distinct-ids sizing bound. The fix routes
    padded-region masking through the shared post-process; previously this
    combination was rejected with ``assert num_token_non_padded is None``,
    so no model with a custom router could mask its pad rows at all.
    """

    DEVICE = "cuda"

    def test_padded_tail_masked_after_custom_routing(self):
        from sglang.srt.layers.moe.topk import select_experts

        num_tokens, num_experts, top_k, n_valid = 12, 32, 8, 10
        hidden = torch.randn((num_tokens, 64), device=self.DEVICE, dtype=torch.bfloat16)
        router_logits = torch.randn(
            (num_tokens, num_experts), device=self.DEVICE, dtype=torch.float32
        )
        # Pad rows carry non-finite router input, like real DP-pad rows.
        router_logits[n_valid:] = float("nan")

        def _degenerate_router(hidden_states, gating_output, topk, renormalize):
            # Mimic the incident: NaN rows collapse to one expert id x top_k.
            weights = torch.softmax(gating_output.nan_to_num(0.0), dim=-1).topk(
                topk, dim=-1
            )
            ids = weights.indices.to(torch.int32)
            ids[gating_output.isnan().any(dim=-1)] = 7
            return weights.values.float(), ids

        out = select_experts(
            hidden_states=hidden,
            router_logits=router_logits,
            topk_config=TopKConfig(
                top_k=top_k,
                renormalize=True,
                custom_routing_function=_degenerate_router,
            ),
            layer_id=0,
            num_token_non_padded=torch.tensor(
                n_valid, device=self.DEVICE, dtype=torch.int32
            ),
        )
        # Padded tail fully -1 (skipped by every EP dispatch path)...
        self.assertTrue(torch.all(out.topk_ids[n_valid:] == -1))
        # ...and real rows untouched (in-range, no -1 leakage).
        self.assertTrue(torch.all(out.topk_ids[:n_valid] >= 0))
        self.assertTrue(torch.all(out.topk_ids[:n_valid] < num_experts))


if __name__ == "__main__":
    unittest.main()
