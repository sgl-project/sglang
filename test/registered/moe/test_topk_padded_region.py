import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import sglang.kernels.ops.moe.fill_padded_rows as fill_mod
import sglang.kernels.ops.moe.moe_fused_gate as moe_fused_gate_mod
import sglang.srt.layers.moe.topk as topk_mod
from sglang.kernels.ops.moe.fill_padded_rows import (
    _can_fuse_padded_region,
    _fill_padded_rows,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.environ import envs
from sglang.srt.layers import dp_attention as dp_attn_mod
from sglang.srt.layers.moe import ep_remap as ep_remap_mod
from sglang.srt.layers.moe import utils as moe_utils_mod
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import standard as standard_mod
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import (
    StandardTopKOutputLocalMapped,
    TopKConfig,
    _mask_topk_ids_padded_region,
    _post_process_topk_ids,
    _zero_topk_weights_padded_region,
    select_experts,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import get_context, get_exec, get_flags, get_parallel
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, stage="base-b", runner_config="1-gpu-small-amd")

_IS_HIP = is_hip()

torch.manual_seed(1234)


def _eager_fill_padded_rows(x, num_token_non_padded, fill_value):
    out = x.clone()
    indices = torch.arange(0, x.shape[0], device=x.device)
    out[indices >= num_token_non_padded, :] = fill_value
    return out


class _ComparisonDtypeRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.input_dtypes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.ge.Tensor:
            self.input_dtypes.append((args[0].dtype, args[1].dtype))
        return func(*args, **(kwargs or {}))


class TestTopkPaddedRegionEagerDtype(CustomTestCase):
    def test_comparison_operands_follow_count_dtype(self):
        """Mixed integer comparisons can fall back to CPU on accelerators that
        require matching input dtypes, so every eager mask must use the routing
        count's dtype while preserving the padded-row result.
        """
        cases = (
            (
                "ids",
                torch.full((5, 3), 7, dtype=torch.int32),
                _mask_topk_ids_padded_region,
                -1,
            ),
            (
                "weights",
                torch.full((5, 3), 1.25, dtype=torch.float32),
                _zero_topk_weights_padded_region,
                0.0,
            ),
        )
        for count_dtype in (torch.int32, torch.int64):
            for name, source, helper, fill_value in cases:
                with self.subTest(count_dtype=count_dtype, name=name):
                    actual = source.clone()
                    expected = source.clone()
                    expected[3:] = fill_value
                    count = torch.tensor(3, dtype=count_dtype)
                    recorder = _ComparisonDtypeRecorder()
                    with recorder:
                        helper(actual, count)
                    self.assertEqual(
                        recorder.input_dtypes, [(count_dtype, count_dtype)]
                    )
                    self.assertTrue(torch.equal(actual, expected))


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
        cases = (
            (
                "ids",
                torch.randint(1, 100, (32, 16), device=self.DEVICE, dtype=torch.int32),
                _mask_topk_ids_padded_region,
                0,
            ),
            (
                "weights",
                torch.rand((32, 16), device=self.DEVICE, dtype=torch.float32) + 0.5,
                _zero_topk_weights_padded_region,
                0.0,
            ),
        )
        for name, base, helper, fill_value in cases:
            with self.subTest(name=name):
                view = base[:, ::2]
                self.assertFalse(_can_fuse_padded_region(view))
                count = torch.tensor(5, device=self.DEVICE, dtype=torch.int32)
                expected = _eager_fill_padded_rows(view, count, fill_value)
                recorder = _ComparisonDtypeRecorder()
                with recorder:
                    if helper is _mask_topk_ids_padded_region:
                        helper(view, count, fill_value=fill_value)
                    else:
                        helper(view, count)
                self.assertEqual(recorder.input_dtypes, [(count.dtype, count.dtype)])
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


@unittest.skipUnless(torch.cuda.is_available(), "EP remap fusion needs a GPU")
@unittest.skipIf(_IS_HIP, "CUDA EP remap fusion is disabled under AITER/HIP")
class TestEpRemapFusionRuntime(CustomTestCase):
    DEVICE = "cuda"

    def setUp(self):
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cuda")
        )
        self.enterContext(envs.SGLANG_DISABLE_FUSED_EP_REMAP.override(False))
        self.enterContext(envs.SGLANG_ENABLE_EP_INTERLEAVED_EXPERTS.override(False))
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_ep_size=2,
                moe_ep_rank=1,
                moe_tp_size=1,
                moe_tp_rank=0,
                dwdp_size=1,
                tp_group=None,
            )
        )
        self.enterContext(get_exec().moe.override(ep_join_mode=None))
        self.enterContext(
            patch.object(
                standard_mod,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            )
        )
        self.enterContext(
            patch.object(
                moe_utils_mod,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            )
        )
        self.enterContext(
            patch.object(
                ep_remap_mod,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            )
        )
        self.enterContext(
            patch.object(dp_attn_mod, "is_dp_max_padding", return_value=False)
        )
        self.enterContext(
            patch.object(ep_remap_mod, "is_dp_max_padding", return_value=False)
        )
        self.enterContext(
            patch.object(
                topk_mod, "is_expert_remapping_enabled", return_value=False
            )
        )
        self.enterContext(
            patch.object(
                ep_remap_mod, "get_global_experts_capturer", return_value=None
            )
        )
        self.enterContext(
            patch.object(
                ep_remap_mod,
                "get_global_expert_distribution_recorder",
                return_value=SimpleNamespace(
                    recording=False,
                    get_current_pass_count_buffer=lambda: None,
                    on_select_experts=lambda **kw: None,
                ),
            )
        )
        self.enterContext(
            patch.object(
                topk_mod,
                "get_global_expert_distribution_recorder",
                return_value=SimpleNamespace(
                    recording=False,
                    get_current_pass_count_buffer=lambda: None,
                    on_select_experts=lambda **kw: None,
                ),
            )
        )
        self.enterContext(
            patch.object(
                standard_mod,
                "use_symmetric_memory",
                lambda *a, **kw: contextlib.nullcontext(),
            )
        )
        self.enterContext(
            patch.object(
                topk_mod,
                "use_symmetric_memory",
                lambda *a, **kw: contextlib.nullcontext(),
            )
        )

    def test_select_experts_ep_remap_equivalence(self):
        with patch.object(
            moe_fused_gate_mod.moe_route_warp, "covered", return_value=False
        ):
            for scoring, shared in (
                ("softmax", 0),
                ("softmax", 1),
                ("sigmoid", 0),
                ("sqrtsoftplus", 1),
            ):
                with self.subTest(scoring=scoring, shared=shared):
                    num_experts, m_rows = 128, 8
                    total_topk = 8 + shared
                    hidden = torch.randn(
                        (m_rows, 64), dtype=torch.bfloat16, device=self.DEVICE
                    )
                    logits = torch.randn(
                        (m_rows, num_experts), dtype=torch.bfloat16, device=self.DEVICE
                    )
                    bias = (
                        None
                        if scoring == "softmax"
                        else torch.randn(
                            (num_experts,), dtype=torch.bfloat16, device=self.DEVICE
                        )
                        * 0.1
                    )
                    non_pad = torch.tensor([5], dtype=torch.int32, device=self.DEVICE)
                    cfg = TopKConfig(
                        top_k=total_topk,
                        renormalize=True,
                        scoring_func=scoring,
                        correction_bias=bias,
                        shared_experts=SharedExpertsConfig(
                            num_fused_shared_experts=shared
                        ),
                        runner_backend=MoeRunnerBackend.TRITON,
                        layer_id=0,
                    )
                    runner_cfg = MoeRunnerConfig(
                        num_experts=num_experts + shared,
                        num_local_experts=(num_experts // 2) + shared,
                        top_k=total_topk,
                        num_fused_shared_experts=shared,
                        runner_backend=MoeRunnerBackend.TRITON,
                    )
                    with envs.SGLANG_DISABLE_FUSED_EP_REMAP.override(True):
                        ref_topk = select_experts(
                            hidden, logits, cfg, num_token_non_padded=non_pad
                        )
                        ref_disp = StandardDispatcher(runner_cfg).dispatch(
                            hidden, ref_topk
                        )
                    treat_topk = select_experts(
                        hidden, logits, cfg, num_token_non_padded=non_pad
                    )
                    self.assertIsInstance(treat_topk, StandardTopKOutputLocalMapped)
                    d_treat = StandardDispatcher(runner_cfg)
                    treat_disp = d_treat.dispatch(hidden, treat_topk)
                    treat_disp2 = d_treat.dispatch(hidden, treat_topk)
                    self.assertTrue(
                        torch.equal(
                            treat_disp.topk_output.topk_ids[:5],
                            ref_disp.topk_output.topk_ids[:5],
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            treat_disp2.topk_output.topk_ids[:5],
                            ref_disp.topk_output.topk_ids[:5],
                        )
                    )
                    torch.testing.assert_close(
                        treat_disp.topk_output.topk_weights[:5],
                        ref_disp.topk_output.topk_weights[:5],
                        atol=1e-6,
                        rtol=1e-5,
                    )
                    self.assertTrue(torch.all(treat_disp.topk_output.topk_ids[5:] == -1))
                    self.assertTrue(
                        torch.all(treat_disp.topk_output.topk_weights[5:] == 0.0)
                    )

    def test_select_experts_ep_remap_cuda_graph_replay(self):
        m_rows, num_experts, topk = 64, 128, 8
        hidden = torch.randn((m_rows, 32), dtype=torch.bfloat16, device=self.DEVICE)
        logits = torch.randn(
            (m_rows, num_experts), dtype=torch.bfloat16, device=self.DEVICE
        )
        non_pad = torch.tensor([32], dtype=torch.int32, device=self.DEVICE)
        cfg = TopKConfig(
            top_k=topk,
            renormalize=True,
            scoring_func="softmax",
            runner_backend=MoeRunnerBackend.TRITON,
            layer_id=0,
        )
        runner_cfg = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_experts // 2,
            top_k=topk,
            num_fused_shared_experts=0,
            runner_backend=MoeRunnerBackend.TRITON,
        )
        disp = StandardDispatcher(runner_cfg)
        out_ids = torch.empty((m_rows, topk), dtype=torch.int32, device=self.DEVICE)
        out_w = torch.empty((m_rows, topk), dtype=torch.float32, device=self.DEVICE)

        def step():
            tk = select_experts(hidden, logits, cfg, num_token_non_padded=non_pad)
            d = disp.dispatch(hidden, tk)
            out_ids.copy_(d.topk_output.topk_ids)
            out_w.copy_(d.topk_output.topk_weights)

        s = torch.cuda.Stream(device=self.DEVICE)
        s.wait_stream(torch.cuda.current_stream(self.DEVICE))
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream(self.DEVICE).wait_stream(s)
        torch.cuda.synchronize(self.DEVICE)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            step()
        torch.cuda.synchronize(self.DEVICE)

        for v in (48, 1, 64, 0):
            non_pad.fill_(v)
            g.replay()
            torch.cuda.synchronize(self.DEVICE)
            with envs.SGLANG_DISABLE_FUSED_EP_REMAP.override(True):
                tk_ref = select_experts(
                    hidden, logits, cfg, num_token_non_padded=non_pad
                )
                d_ref = StandardDispatcher(runner_cfg).dispatch(hidden, tk_ref)
            self.assertTrue(torch.equal(out_ids, d_ref.topk_output.topk_ids))
            torch.testing.assert_close(
                out_w[:v], d_ref.topk_output.topk_weights[:v], atol=1e-6, rtol=1e-5
            )
            self.assertTrue(torch.all(out_ids[v:] == -1))
            self.assertTrue(torch.all(out_w[v:] == 0.0))

    def test_custom_local_expert_mapping_matches_unfused_from_global_logits(self):
        num_experts, m_rows, topk = 128, 4, 8
        hidden = torch.randn((m_rows, 32), dtype=torch.bfloat16, device=self.DEVICE)
        logits = torch.full(
            (m_rows, num_experts), -20.0, dtype=torch.bfloat16, device=self.DEVICE
        )
        # Rows [0, 1] select global [0..7]; rows [2, 3] select global [64..71].
        logits[:2, :topk] = 10.0
        logits[2:, 64 : 64 + topk] = 10.0

        # Global-indexed custom map: [0..63] -> [63..0], [64..127] -> -1.
        custom_map = torch.full((num_experts,), -1, dtype=torch.int32, device=self.DEVICE)
        custom_map[:64] = torch.arange(63, -1, -1, dtype=torch.int32, device=self.DEVICE)

        runner_cfg = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=64,
            top_k=topk,
            num_fused_shared_experts=0,
            runner_backend=MoeRunnerBackend.TRITON,
        )
        cfg_custom = TopKConfig(
            top_k=topk,
            renormalize=True,
            scoring_func="softmax",
            runner_backend=MoeRunnerBackend.TRITON,
            use_custom_local_expert_mapping=True,
            layer_id=0,
        )
        d_ctrl = StandardDispatcher(runner_cfg)
        d_ctrl.local_expert_mapping = custom_map
        d_treat = StandardDispatcher(runner_cfg)
        d_treat.local_expert_mapping = custom_map

        with envs.SGLANG_DISABLE_FUSED_EP_REMAP.override(True):
            tk_ctrl = select_experts(hidden, logits, cfg_custom)
            self.assertNotIsInstance(tk_ctrl, StandardTopKOutputLocalMapped)
            ctrl_disp = d_ctrl.dispatch(hidden, tk_ctrl)

        tk_treat = select_experts(hidden, logits, cfg_custom)
        self.assertNotIsInstance(tk_treat, StandardTopKOutputLocalMapped)
        treat_disp = d_treat.dispatch(hidden, tk_treat)

        self.assertTrue(
            torch.equal(treat_disp.topk_output.topk_ids, ctrl_disp.topk_output.topk_ids)
        )
        torch.testing.assert_close(
            treat_disp.topk_output.topk_weights,
            ctrl_disp.topk_output.topk_weights,
            atol=1e-6,
            rtol=1e-5,
        )
        # Global [0..7] -> custom_map[0..7] == [63..56]; global [64..71] -> -1 (dropped).
        expected_live = torch.arange(63, 63 - topk, -1, dtype=torch.int32, device=self.DEVICE)
        for row in (0, 1):
            self.assertEqual(
                set(treat_disp.topk_output.topk_ids[row].tolist()),
                set(expected_live.tolist()),
            )
        self.assertTrue(torch.all(treat_disp.topk_output.topk_ids[2:] == -1))

        # Mismatch guard: omitting use_custom_local_expert_mapping fails loudly.
        cfg_default = TopKConfig(
            top_k=topk,
            renormalize=True,
            scoring_func="softmax",
            runner_backend=MoeRunnerBackend.TRITON,
            layer_id=0,
        )
        tk_local = select_experts(hidden, logits, cfg_default)
        self.assertIsInstance(tk_local, StandardTopKOutputLocalMapped)
        with self.assertRaisesRegex(ValueError, "custom local_expert_mapping"):
            d_treat.dispatch(hidden, tk_local)


if __name__ == "__main__":
    unittest.main()
