import unittest
from unittest.mock import patch

import torch

import sglang.kernels.ops.moe.fill_padded_rows as fill_mod
import sglang.srt.layers.moe.topk as topk_mod
from sglang.kernels.ops.moe.fill_padded_rows import (
    _can_fuse_padded_region,
    _fill_dp_pad_rows,
    _fill_padded_rows,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _mask_topk_ids_padded_region,
    _post_process_topk_ids,
    _zero_topk_weights_padded_region,
    select_experts,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    set_forward_context,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd")

_IS_HIP = is_hip()

torch.manual_seed(1234)


def _eager_fill_padded_rows(x, num_token_non_padded, fill_value):
    out = x.clone()
    indices = torch.arange(0, x.shape[0], device=x.device)
    out[indices >= num_token_non_padded, :] = fill_value
    return out


def _eager_fill_dp_pad_rows(x, dp_real_rows, fill_value):
    out = x.clone()
    rows_per_rank = x.shape[0] // dp_real_rows.numel()
    for rank, n_real in enumerate(dp_real_rows.tolist()):
        start = rank * rows_per_rank
        out[start + min(n_real, rows_per_rank) : start + rows_per_rank] = fill_value
    return out


def _dp_live_rows(num_rows, dp_real_rows):
    rows = torch.arange(num_rows, device=dp_real_rows.device)
    rows_per_rank = num_rows // dp_real_rows.numel()
    return rows % rows_per_rank < dp_real_rows[rows // rows_per_rank]


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


@unittest.skipUnless(torch.cuda.is_available(), "fused padded-row fill needs a GPU")
class TestFillPaddedRowsStrided(CustomTestCase):
    def test_matches_eager(self):
        # A cap of 3 takes the strided branch from 6 rows and hits every residue.
        with patch.object(fill_mod, "_MAX_PROGRAMS", 3):
            for rows in range(1, 14):
                for n_valid in range(rows + 3):
                    self._check(rows, 8, n_valid)
        self._check(4096, 1024, 4000)  # the strided branch at the real cap

    def _check(self, rows, cols, n_valid):
        x = torch.randn((rows, cols), device="cuda").to(torch.bfloat16)
        count = torch.tensor([n_valid], device="cuda", dtype=torch.int32)
        expected = _eager_fill_padded_rows(x, count, 0.0)
        _fill_padded_rows(x, count, 0.0)
        self.assertTrue(torch.equal(x, expected))


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
        cfg = TopKConfig(
            top_k=k, shared_experts=SharedExpertsConfig(num_fused_shared_experts=0)
        )
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

    def test_flag_skips_weight_masking_with_shared_append(self):
        n, k, n_valid = 16, 8, 5
        topk_weights = torch.rand((n, k), device=self.DEVICE) + 0.5
        topk_ids = torch.randint(0, 64, (n, k), device=self.DEVICE, dtype=torch.int32)
        router_logits = torch.rand((n, 64), device=self.DEVICE)
        pad = torch.tensor(n_valid, device=self.DEVICE, dtype=torch.int32)
        cfg = TopKConfig(
            top_k=k + 1, shared_experts=SharedExpertsConfig(num_fused_shared_experts=1)
        )
        orig_skip = topk_mod._skip_hip_pad_mask
        orig_aiter = topk_mod._use_aiter
        topk_mod._skip_hip_pad_mask = True
        topk_mod._use_aiter = True
        try:
            out_ids, out_weights, _ = _post_process_topk_ids(
                topk_ids,
                topk_weights,
                cfg,
                router_logits,
                layer_id=0,
                num_token_non_padded=pad,
            )
        finally:
            topk_mod._skip_hip_pad_mask = orig_skip
            topk_mod._use_aiter = orig_aiter

        self.assertTrue(torch.all(out_ids[n_valid:, :k] == 0))
        self.assertTrue(torch.all(out_weights[n_valid:] > 0))


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


@unittest.skipUnless(torch.cuda.is_available(), "padded-region masking needs a GPU")
@unittest.skipIf(_IS_HIP, "HIP marks padded rows with zero weights instead")
class TestSelectExpertsPerRankSharedSlotPadMask(CustomTestCase):
    """With per-rank fused shared slots, padded rows must leave select_experts
    fully -1, the appended shared column included. Masking only before the
    append left them routed ids of -1 - S and a live home-rank shared slot."""

    DEVICE = "cuda"

    def test_padded_tail_masked_after_shared_append(self):
        num_tokens, num_experts, top_k, n_valid = 12, 256, 8, 9
        # EP8 rank 3: 32 routed slots per rank, then the rank's shared slot.
        home_shared_id = 3 * 33 + 32
        # Grouped sigmoid is masked after routing, sqrtsoftplus by its router.
        for scoring_func, num_expert_group in (("sigmoid", 8), ("sqrtsoftplus", None)):
            with self.subTest(scoring_func=scoring_func):
                g = torch.Generator(device="cpu").manual_seed(0)
                logits = torch.randn(num_tokens, num_experts, generator=g)
                bias = 0.1 * torch.randn(num_experts, generator=g)
                topk_config = TopKConfig(
                    top_k=top_k + 1,
                    use_grouped_topk=num_expert_group is not None,
                    num_expert_group=num_expert_group,
                    topk_group=4 if num_expert_group else None,
                    shared_experts=SharedExpertsConfig(num_fused_shared_experts=1),
                    correction_bias=bias.to(self.DEVICE),
                    scoring_func=scoring_func,
                    routed_scaling_factor=2.5,
                    allow_routed_experts_capture=False,
                )
                with (
                    patch.object(
                        topk_mod,
                        "has_per_rank_fused_shared_slots",
                        side_effect=lambda num_fused: num_fused > 0,
                    ),
                    get_parallel().override(moe_ep_size=8, moe_ep_rank=3),
                ):
                    out = topk_mod.select_experts(
                        hidden_states=torch.empty(num_tokens, 16, device=self.DEVICE),
                        router_logits=logits.to(self.DEVICE),
                        topk_config=topk_config,
                        layer_id=0,
                        num_token_non_padded=torch.tensor(
                            n_valid, device=self.DEVICE, dtype=torch.int32
                        ),
                    )
                self.assertTrue(torch.all(out.topk_ids[n_valid:] == -1))
                self.assertTrue(torch.all(out.topk_ids[:n_valid] >= 0))
                self.assertTrue(torch.all(out.topk_ids[:n_valid, -1] == home_shared_id))


@unittest.skipUnless(torch.cuda.is_available(), "DP padded-row fill needs a GPU")
class TestDpPadRows(CustomTestCase):
    """A MAX_LEN DP gather: one equal slot per DP rank, real rows first."""

    DEVICE = "cuda"

    def test_matches_eager_across_layouts(self):
        # (DP ranks, rows per rank, cols): one row per rank, a padded graph
        # bucket, and rows wider than one block of columns.
        for num_ranks, rows_per_rank, cols in [(8, 1, 9), (8, 16, 9), (4, 3, 129)]:
            for idle_rank in range(3):
                counts = torch.randint(
                    0, rows_per_rank + 1, (num_ranks,), dtype=torch.int32
                )
                counts[idle_rank % num_ranks] = 0
                counts = counts.to(self.DEVICE)
                ids = torch.randint(
                    0,
                    288,
                    (num_ranks * rows_per_rank, cols),
                    device=self.DEVICE,
                    dtype=torch.int32,
                )
                with self.subTest(
                    num_ranks=num_ranks, rows_per_rank=rows_per_rank, cols=cols
                ):
                    expected = _eager_fill_dp_pad_rows(ids, counts, -1)
                    _fill_dp_pad_rows(ids, counts, -1)
                    self.assertTrue(torch.equal(ids, expected))

    def test_cuda_graph_replay_reads_counts(self):
        # The decode graph is captured with every row live and replayed with
        # each step's counts, so the kernel has to read them from device memory.
        num_ranks, rows_per_rank, k = 8, 4, 9
        ids = torch.randint(
            0,
            288,
            (num_ranks * rows_per_rank, k),
            device=self.DEVICE,
            dtype=torch.int32,
        )
        counts = torch.full(
            (num_ranks,), rows_per_rank, device=self.DEVICE, dtype=torch.int32
        )
        work = ids.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _fill_dp_pad_rows(work.clone(), counts, -1)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fill_dp_pad_rows(work, counts, -1)

        for step in ([4] * 8, [1, 0, 0, 0, 0, 0, 0, 0], [0, 2, 4, 1, 3, 0, 4, 1]):
            work.copy_(ids)
            counts.copy_(torch.tensor(step, dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            with self.subTest(counts=step):
                self.assertTrue(
                    torch.equal(work, _eager_fill_dp_pad_rows(ids, counts, -1))
                )

    def test_fused_gate_masks_only_dp_pad_rows(self):
        # The Triton router (fused shared expert) and the fast path it may take
        # without one must both return the unmasked routes on real rows.
        counts = torch.tensor([1, 0, 3, 2], device=self.DEVICE, dtype=torch.int64)
        num_rows, num_experts = 3 * counts.numel(), 64
        live = _dp_live_rows(num_rows, counts)
        scores = torch.randn(num_rows, num_experts, device=self.DEVICE)
        bias = 0.1 * torch.randn(num_experts, device=self.DEVICE)
        for num_shared in (1, 0):
            with self.subTest(num_fused_shared_experts=num_shared):
                kwargs = dict(
                    topk=8 + num_shared,
                    num_fused_shared_experts=num_shared,
                    routed_scaling_factor=2.5,
                )
                weights_all, ids_all = moe_fused_gate(scores, bias, **kwargs)
                weights, ids = moe_fused_gate(
                    scores, bias, dp_real_rows=counts, **kwargs
                )
                self.assertTrue(torch.equal(ids[live], ids_all[live]))
                self.assertTrue(torch.equal(weights[live], weights_all[live]))
                self.assertTrue(torch.all(ids[~live] == -1))
                self.assertTrue(torch.all(weights[~live] == 0))

    def test_fused_moe_skips_dp_pad_rows(self):
        # Under a DP-gathered forward, select_experts routes the padding rows to
        # -1, in the Triton router (the biased sigmoid paths, grouped as
        # GLM-5.3-Flash configures it, with a fused shared expert) or after it
        # (softmax); the TP fused MoE then returns zeros there and real rows
        # unchanged.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        num_experts, hidden, inter = 16, 256, 128
        counts = torch.tensor([1, 0, 3, 2], device=self.DEVICE, dtype=torch.int64)
        num_rows = 3 * counts.numel()
        live = _dp_live_rows(num_rows, counts)
        x = torch.randn(num_rows, hidden, device=self.DEVICE).to(torch.bfloat16)
        logits = torch.randn(num_rows, num_experts, device=self.DEVICE)
        # One more expert for the fused shared slot (id num_experts).
        w1 = torch.randn(num_experts + 1, 2 * inter, hidden, device=self.DEVICE)
        w2 = torch.randn(num_experts + 1, hidden, inter, device=self.DEVICE)
        w1 = (w1 / hidden**0.5).to(torch.bfloat16)
        w2 = (w2 / inter**0.5).to(torch.bfloat16)
        bias = 0.1 * torch.randn(num_experts, device=self.DEVICE)
        shared = SharedExpertsConfig(num_fused_shared_experts=1)
        configs = {
            "softmax": TopKConfig(top_k=4, renormalize=False),
            "sigmoid_shared": TopKConfig(
                top_k=4 + 1,
                shared_experts=shared,
                correction_bias=bias,
                scoring_func="sigmoid",
                routed_scaling_factor=2.5,
                allow_routed_experts_capture=False,
            ),
            # More than 8 routed slots takes the grouped path's Triton router.
            "grouped_sigmoid_shared": TopKConfig(
                top_k=9 + 1,
                use_grouped_topk=True,
                num_expert_group=1,
                topk_group=1,
                shared_experts=shared,
                correction_bias=bias,
                scoring_func="sigmoid",
                routed_scaling_factor=2.5,
                allow_routed_experts_capture=False,
            ),
        }
        for name, topk_config in configs.items():
            with self.subTest(router=name):
                routed_all = select_experts(
                    hidden_states=x, router_logits=logits, topk_config=topk_config
                )
                prev = set_forward_context(
                    ForwardContext(attn_backend=None, moe_dp_real_rows=lambda: counts)
                )
                try:
                    routed = select_experts(
                        hidden_states=x, router_logits=logits, topk_config=topk_config
                    )
                finally:
                    set_forward_context(prev)
                self.assertTrue(
                    torch.equal(routed.topk_ids[live], routed_all.topk_ids[live])
                )
                self.assertTrue(torch.all(routed.topk_ids[~live] == -1))
                out_all = fused_moe(x, w1, w2, routed_all)
                out = fused_moe(x, w1, w2, routed)
                self.assertTrue(torch.equal(out[live], out_all[live]))
                self.assertTrue(torch.all(out[~live] == 0))


if __name__ == "__main__":
    unittest.main()
