"""CP-prefill must not make unsharded decode lose or duplicate its sums.

Build the real attention/FFN boundaries under CP2, then exercise their fused
and fallback consumers on CPU. Only collectives and GPU norm kernels are
replaced, with exact arithmetic for identical per-rank contributions.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.cp.cp_decode_attn_tp import (
    CpDecodeAttnTpContext,
    can_defer_cp_decode_attn_tp,
    configure_cp_decode_attn_tp,
)
from sglang.srt.layers.layer_boundary import SumGroup, declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.residual import batch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.boundary_fixtures import build_stages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Norm(torch.nn.Module):
    supports_non_attn_tp_allreduce = True

    def __init__(self, fused):
        super().__init__()
        self.fused = fused
        self.fused_sums = 0

    def forward(self, value, residual=None, post_residual_addition=None):
        if residual is None:
            return value.clone()
        result = value + residual
        return result, result.clone()

    def forward_with_allreduce_fusion(self, value, residual, *, use_attn_tp_group):
        if not self.fused:
            return None
        # CP2 uses the full-TP MoE workspace, not the size-one attention group.
        if use_attn_tp_group:
            raise AssertionError("decode sum sent to the prefill attention group")
        self.fused_sums += 1
        result = value * 2 + residual
        return result, result.clone()


class TestCpDecodeDispatch(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # The lazily initialized strategy is a process singleton, not a bag.
        self.stack.enter_context(patch("sglang.srt.layers.cp.base._STRATEGY", None))
        self.stack.enter_context(
            get_context().override_server_args(
                tp_size=2,
                attn_cp_size=2,
                enable_prefill_cp=True,
                cp_strategy="interleave",
                enable_cp_decode_attn_tp=True,
                device="cuda",
                boundary_reduction="rs+rsv",
                flashinfer_allreduce_fusion_backend="auto",
                # Seed checkpoint metadata at the real process-config boundary.
                _model_config=SimpleNamespace(
                    attention_arch=AttentionArch.MLA,
                    hf_config=SimpleNamespace(
                        architectures=["GlmMoeDsaForCausalLM"], index_topk=2048
                    ),
                ),
            )
        )
        self.unfused_sums = 0

        def all_reduce(value):
            self.unfused_sums += 1
            return value * 2

        full = SimpleNamespace(
            ranks=[0, 1], world_size=2, rank_in_group=0, all_reduce=all_reduce
        )
        single = SimpleNamespace(ranks=[0], world_size=1, rank_in_group=0)
        self.stack.enter_context(
            get_parallel().override(
                tp_group=full,
                attn_cp_group=full,
                attn_tp_group=single,
                moe_tp_group=full,
                moe_ep_group=single,
                moe_dp_group=full,
                attn_cp_rank=0,
                attn_tp_rank=0,
            )
        )
        # Substitute GPU-kernel availability only; boundary selection is real.
        self.stack.enter_context(
            patch_communicator("flashinfer_ar_fusion_applies", return_value=True)
        )

    def test_decode_and_verify_complete_each_sum_once(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                norms = [Norm(fused) for _ in range(3)]
                attn, ffn, following = build_stages(
                    (declare_attn(ordinary_tp_group=SumGroup.TP), norms[0]),
                    (declare_ffn(sparse=True), norms[1]),
                    (declare_attn(ordinary_tp_group=SumGroup.TP), norms[2]),
                    terminal=True,
                )
                self.assertTrue(can_defer_cp_decode_attn_tp())
                ordinary = ffn.plan.paths[BatchVariant.ORDINARY]
                cp = ffn.plan.paths[BatchVariant.CONTEXT_PARALLEL]
                self.assertTrue(ordinary.exit.may_defer_sum)
                self.assertFalse(cp.exit.may_defer_sum)
                self.assertIs(ordinary.entry.declared_sum, SumGroup.ATTN_CP)
                self.assertIsNone(cp.entry.declared_sum)
                _, original_ffn, _ = build_stages(
                    (declare_attn(), Norm(fused)),
                    (declare_ffn(sparse=True), Norm(fused)),
                    (declare_attn(), Norm(fused)),
                    terminal=True,
                )
                cp_edge = ffn.plan.edges[BatchVariant.CONTEXT_PARALLEL]
                original_cp_edge = original_ffn.plan.edges[
                    BatchVariant.CONTEXT_PARALLEL
                ]
                for field in ("incoming", "outgoing", "attn_input_adapter"):
                    self.assertEqual(
                        getattr(cp_edge, field), getattr(original_cp_edge, field)
                    )
                for field in ("gather", "take_back", "reduce_scatter"):
                    self.assertIs(
                        getattr(cp_edge.cp_moves, field),
                        getattr(original_cp_edge.cp_moves, field),
                    )
                # Each construction creates a fresh group accessor closure.
                self.assertIs(
                    cp_edge.cp_moves.reduce_scatter_group(),
                    original_cp_edge.cp_moves.reduce_scatter_group(),
                )

                for mode in (
                    ForwardMode.DECODE,
                    ForwardMode.TARGET_VERIFY,
                    ForwardMode.DRAFT_EXTEND_V2,
                    ForwardMode.EXTEND,
                ):
                    fb = SimpleNamespace(
                        forward_mode=mode,
                        input_ids=torch.arange(2),
                        attn_cp_metadata=None,
                        can_run_tbo=False,
                        # One real extend token (plus padding) cannot use CP2.
                        extend_seq_lens_cpu=[1],
                    )
                    batch.start(fb)
                    initial = torch.arange(8).reshape(2, 4).float()
                    hidden = attn.prepare(initial, fb)
                    hidden = attn.finish(hidden / 2, fb)
                    hidden = ffn.prepare(hidden, fb)
                    torch.testing.assert_close(hidden, initial * 2, rtol=0, atol=0)
                    with ffn.exit(fb) as output:
                        partial = hidden / 2
                    hidden = output.finish(partial)
                    hidden = following.prepare(hidden, fb)
                    torch.testing.assert_close(hidden, initial * 4, rtol=0, atol=0)

                self.assertEqual(sum(n.fused_sums for n in norms), 8 if fused else 0)
                if not fused:
                    self.assertEqual(self.unfused_sums, 8)

    def test_unmodified_attention_does_not_leave_a_decode_sum(self):
        _, ffn = build_stages(
            (declare_attn(), Norm(False)),
            (declare_ffn(sparse=True), Norm(False)),
            terminal=True,
        )
        self.assertIsNone(ffn.plan.paths[BatchVariant.ORDINARY].entry.declared_sum)
        single = get_parallel().attn_tp_group
        full = get_parallel().tp_group
        four = SimpleNamespace(ranks=[0, 1, 2, 3], world_size=4, rank_in_group=0)
        for name, changes in (
            ("disabled", dict(enable_cp_decode_attn_tp=False)),
            (
                "no_cp",
                dict(
                    attn_cp_size=1,
                    attn_tp_size=2,
                    attn_cp_group=single,
                    attn_tp_group=full,
                ),
            ),
            (
                "attention_dp",
                dict(attn_cp_size=1, attn_dp_size=2, attn_cp_group=single),
            ),
            (
                "hybrid_cp_tp",
                dict(
                    tp_size=4,
                    attn_tp_size=2,
                    moe_tp_size=4,
                    tp_group=four,
                    attn_tp_group=full,
                    moe_tp_group=four,
                ),
            ),
            ("sequence_parallel", dict(enable_layernorm_sp=True)),
        ):
            with self.subTest(topology=name), get_parallel().override(**changes):
                self.assertFalse(can_defer_cp_decode_attn_tp())

    def test_every_forward_mode_agrees_with_weight_dispatch(self):
        """Only CP-sharded rows keep full weights; every other mode owes a sum.

        Include short/padded extends, mixed batches, empty graph dummies and
        stale CP metadata on non-extend modes. Adding a ForwardMode must not
        silently leave the boundary and projection on different layouts.
        """
        from sglang.srt.layers.linear import RowParallelLinear

        _, ffn = build_stages(
            (declare_attn(ordinary_tp_group=SumGroup.TP), Norm(False)),
            (declare_ffn(sparse=True), Norm(False)),
            terminal=True,
        )
        layer = RowParallelLinear(
            8, 4, bias=False, reduce_results=False, parallel_group="attn_tp"
        )
        ctx = CpDecodeAttnTpContext()
        for mode in ForwardMode:
            for tokens in (0, 1, 2, 8):
                for metadata in (None, object()):
                    with self.subTest(mode=mode, tokens=tokens, metadata=metadata):
                        fb = SimpleNamespace(
                            forward_mode=mode,
                            input_ids=torch.arange(max(tokens, 2)),
                            extend_seq_lens_cpu=[tokens],
                            attn_cp_metadata=metadata,
                        )
                        cp_rows = mode in (ForwardMode.EXTEND, ForwardMode.MIXED) and (
                            tokens >= 2 or metadata is not None
                        )
                        variant = ffn.plan.variant_for(fb)
                        self.assertIs(
                            variant,
                            BatchVariant.CONTEXT_PARALLEL
                            if cp_rows
                            else BatchVariant.ORDINARY,
                        )
                        # Ordinary rows must never take the CP MoE gather.
                        self.assertEqual(
                            ffn.plan.edges[variant].cp_moves is not None, cp_rows
                        )
                        with ctx.maybe_use_decode_attn_tp(
                            fb, [layer], defer_all_reduce=True
                        ):
                            self.assertEqual(ctx.use_decode_attn_tp, not cp_rows)
                            self.assertEqual(layer.weight.shape[1], 8 if cp_rows else 4)
                            self.assertIs(
                                ffn.plan.path_for(fb).entry.declared_sum,
                                None if cp_rows else SumGroup.ATTN_CP,
                            )

    def test_norm_without_group_selection_falls_back_to_exactly_one_sum(self):
        """Having a fusion method alone must not opt into the MoE/CP group."""
        norm = Norm(True)
        norm.supports_non_attn_tp_allreduce = False
        attn, ffn = build_stages(
            (declare_attn(ordinary_tp_group=SumGroup.TP), Norm(False)),
            (declare_ffn(sparse=True), norm),
            terminal=True,
        )
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            attn_cp_metadata=None,
            input_ids=torch.arange(2),
            can_run_tbo=False,
        )
        batch.start(fb)
        initial = torch.arange(8).reshape(2, 4).float()
        hidden = attn.prepare(initial, fb)
        hidden = attn.finish(hidden / 2, fb)
        hidden = ffn.prepare(hidden, fb)
        torch.testing.assert_close(hidden, initial * 2, rtol=0, atol=0)
        self.assertEqual(self.unfused_sums, 1)
        self.assertEqual(norm.fused_sums, 0)

    def test_invalid_shards_fail_before_mutating_weights(self):
        from sglang.srt.layers.linear import RowParallelLinear

        ctx = CpDecodeAttnTpContext()
        for dim, shape in ((0, (3, 8)), (1, (4, 7))):
            with self.subTest(dim=dim), self.assertRaisesRegex(ValueError, "divisible"):
                ctx._slice(torch.empty(shape), dim)

        # The padded scale tensor divides evenly across CP2, but K/2=384
        # cuts through a packed group of four 128-column scales.
        layer = RowParallelLinear(
            768, 4, bias=False, reduce_results=False, parallel_group="attn_tp"
        )
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.ones(4, 2, dtype=torch.int32), requires_grad=False
        )
        pointers = (layer.weight.data_ptr(), layer.weight_scale_inv.data_ptr())
        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE, attn_cp_metadata=None)
        with self.assertRaisesRegex(ValueError, "multiple of 512"):
            with ctx.maybe_use_decode_attn_tp(fb, [layer]):
                self.fail("invalid packed K shard reached the forward")
        self.assertEqual(
            (layer.weight.data_ptr(), layer.weight_scale_inv.data_ptr()), pointers
        )
        self.assertEqual(layer.input_size_per_partition, 768)
        self.assertEqual(ctx._slice_cache, {})

    def test_deferred_reduction_rejected_at_configuration(self):
        from sglang.srt.layers.linear import RowParallelLinear

        layer = RowParallelLinear(
            8, 4, bias=False, reduce_results=True, parallel_group="attn_tp"
        )
        with self.assertRaisesRegex(ValueError, "reduce_results=False"):
            configure_cp_decode_attn_tp(layer, defer_all_reduce=True)

    def test_online_updates_reject_before_writes_or_cache_population(self):
        """Reject on every rank at construction, even before graph warmup."""
        from sglang.srt.layers.linear import RowParallelLinear
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            WeightUpdater,
        )

        layer = RowParallelLinear(
            8, 4, bias=False, reduce_results=False, parallel_group="attn_tp"
        )
        configure_cp_decode_attn_tp(layer, defer_all_reduce=True)
        model = torch.nn.Sequential(layer)
        with torch.no_grad():
            layer.weight.fill_(3)
        original = layer.weight.detach().clone()
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *a, **kw: self.fail("updated model fields"),
            recapture_cuda_graph=lambda: self.fail("recaptured graph"),
            get_model_runner=lambda: self.fail("accessed runner"),
        )
        for method, args in (
            (
                "update_weights_from_tensor",
                ([("0.weight", torch.zeros_like(original))], "direct"),
            ),
            (
                "load_weights_from_distributed",
                ([("0.weight", torch.zeros_like(original))],),
            ),
            ("update_weights_from_disk", ("unused", "auto")),
            ("update_weights_from_ipc", (SimpleNamespace(),)),
        ):
            with self.subTest(method=method):
                ok, message = getattr(updater, method)(*args)
                self.assertFalse(ok)
                self.assertIn("CP decode attention TP", message)
                torch.testing.assert_close(layer.weight, original, rtol=0, atol=0)

    def test_same_width_different_peers_does_not_fuse(self):
        from sglang.srt.layers.layer_boundary.fusions.allreduce import ffn_input_fusions

        _, ffn = build_stages(
            (declare_attn(ordinary_tp_group=SumGroup.TP), Norm(True)),
            (declare_ffn(sparse=True), Norm(True)),
            terminal=True,
        )
        self.assertIn(
            SumGroup.ATTN_CP,
            [fusion.completes for fusion in ffn_input_fusions(ffn.plan)],
        )
        different_peers = SimpleNamespace(ranks=[2, 3], world_size=2, rank_in_group=0)
        with get_parallel().override(moe_tp_group=different_peers):
            self.assertNotIn(
                SumGroup.ATTN_CP,
                [fusion.completes for fusion in ffn_input_fusions(ffn.plan)],
            )

    def test_weight_dispatch_restores_after_decode_and_exception(self):
        from sglang.srt.layers.linear import RowParallelLinear

        layer = RowParallelLinear(
            8, 4, bias=False, reduce_results=False, parallel_group="attn_tp"
        )
        original = layer.weight.data_ptr()
        ctx = CpDecodeAttnTpContext()
        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE, attn_cp_metadata=None)
        for defer in (False, True):
            with self.subTest(defer=defer):
                with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                    with ctx.maybe_use_decode_attn_tp(
                        fb, [layer], defer_all_reduce=defer
                    ):
                        self.assertEqual(tuple(layer.weight.shape), (4, 4))
                        self.assertEqual(layer.use_decode_attn_tp, not defer)
                        raise RuntimeError("consumer failed")
                self.assertEqual(layer.weight.data_ptr(), original)
                self.assertEqual(tuple(layer.weight.shape), (4, 8))
                self.assertEqual(layer.input_size_per_partition, 8)
                self.assertFalse(layer.use_decode_attn_tp)
        fb.forward_mode = ForwardMode.EXTEND
        fb.attn_cp_metadata = object()
        with ctx.maybe_use_decode_attn_tp(fb, [layer], defer_all_reduce=True):
            self.assertEqual(layer.weight.data_ptr(), original)
            self.assertFalse(ctx.use_decode_attn_tp)


if __name__ == "__main__":
    unittest.main()
