"""Synthetic HC regressions through the serving module; no model checkpoint."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.elementwise.hc_sum_state import HCSumState, SmallTSumHCShell
from sglang.srt.layers.hyperconnection import HyperConnectionConfig
from sglang.srt.layers.hyperconnection_packed import (
    PackedGatedResidual,
    create_gated_residual,
    prepare_packed_hc_weights,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class _HCSequential(torch.nn.Sequential):
    def post_load_weights_from_preprocessed(self):
        prepare_packed_hc_weights(self, weights_preprocessed=True)


def _checkpoint(h, lowrank, dtype, seed=311):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shapes = {
        "input_mix_weight_down.weight": (lowrank, 4 * h),
        "block_inject_weight.weight": (4, 4 * h),
        "input_mix_weight_up.weight": (4 * h, lowrank),
        "hc_norm.weight": (4 * h,),
    }
    return {
        name: torch.randn(shape, dtype=dtype, generator=generator)
        * (0.15 if name == "hc_norm.weight" else 0.02)
        for name, shape in shapes.items()
    }


def _layer(h, lowrank, dtype, *, combine=True, eps=1e-6, meta=False):
    layer = create_gated_residual(
        HyperConnectionConfig(
            hc_count=4,
            hidden_size=h,
            hc_lowrank=lowrank,
            hc_per_branch_norm=True,
            rms_norm_eps=eps,
            params_dtype=dtype,
        ),
        use_combine=combine,
    )
    if not meta:
        layer.to(device="cuda", dtype=dtype)
    return layer.eval().requires_grad_(False)


def _bootstrap_sum_state(layer, residual):
    if not layer.can_use_sum_state(residual):
        raise ValueError("HC sum-state weights/input are not prepared or supported")
    return layer._sum_operators[int(residual.shape[0] > 24)].bootstrap(residual)


def _native_checkpoint(checkpoint):
    down = checkpoint["input_mix_weight_down.weight"]
    inject = checkpoint["block_inject_weight.weight"]
    norm = checkpoint["hc_norm.weight"]
    lowrank, width = down.shape
    h = width // 4
    npad = (lowrank + 4 + 7) // 8 * 8
    folded = torch.zeros((npad, width), dtype=down.dtype, device=down.device)
    folded[: lowrank + 4] = (torch.cat((down, inject)).float() * (1 + norm.float())).to(
        down.dtype
    )
    return {
        "input_mix_weight_down.weight": folded.view(npad, 4, h)
        .permute(1, 0, 2)
        .contiguous(),
        "hc_norm.weight": norm.view(4, h).T.contiguous(),
        "input_mix_weight_up.weight": checkpoint["input_mix_weight_up.weight"],
    }


def _reference_mix(state, checkpoint, eps):
    """Independent math from checkpoint weights, not production packed buffers.

    Preserve the documented storage casts: folded weights, normalized residual,
    Down output, and (for large T) raw logits and the pointwise Up/Mix chain.
    """
    residual, sums = state
    t, width = residual.shape
    h = width // 4
    dtype = residual.dtype
    weights = {name: value.to(residual.device) for name, value in checkpoint.items()}
    lowrank = weights["input_mix_weight_down.weight"].shape[0]
    gamma = 1.0 + weights["hc_norm.weight"].float()
    folded = (
        torch.cat(
            (
                weights["input_mix_weight_down.weight"],
                weights["block_inject_weight.weight"],
            )
        ).float()
        * gamma
    ).to(dtype)
    grouped = residual.float().view(t, 4, h)
    inv = torch.rsqrt(sums / h + eps)
    raw = sum(
        F.linear(grouped[:, branch], folded[:, branch * h : (branch + 1) * h].float())
        * inv[:, branch : branch + 1]
        for branch in range(4)
    )
    if t > 24:
        raw = raw.to(dtype).float()
    down = F.silu(raw[:, :lowrank] / 4).to(dtype)
    alpha = 2 * torch.sigmoid(raw[:, lowrank:] / 4)
    norm = (grouped * inv[:, :, None] * gamma.view(4, h)).to(dtype)
    up = weights["input_mix_weight_up.weight"]
    if t <= 24:
        logits = F.linear(down.float(), up.float()).view(t, 4, h)
        mixed = (torch.sigmoid(logits) * norm.float()).mean(1).to(dtype)
    else:
        logits = F.linear(down, up).view(t, 4, h)
        mixed = (torch.sigmoid(logits) * norm).mean(1)
    return mixed, alpha


def _reference_unfused_mix(residual, checkpoint, eps):
    """HC math with RMS scale on activations, independent of folded weights."""
    t, width = residual.shape
    grouped = residual.float().view(t, 4, width // 4)
    weights = {
        name: value.to(device=residual.device, dtype=torch.float32)
        for name, value in checkpoint.items()
    }
    norm = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    norm = (norm.flatten(1) * (1 + weights["hc_norm.weight"])).to(residual.dtype)
    down = F.silu(F.linear(norm.float(), weights["input_mix_weight_down.weight"]) / 4)
    logits = F.linear(
        down.to(residual.dtype).float(), weights["input_mix_weight_up.weight"]
    ).view_as(grouped)
    mixed = (torch.sigmoid(logits) * norm.float().view_as(grouped)).mean(1)
    alpha = 2 * torch.sigmoid(
        F.linear(norm.float(), weights["block_inject_weight.weight"]) / 4
    )
    return mixed.to(residual.dtype), alpha


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "HC sum-state projections require SM100-family GPUs",
)
class TestHCSumState(CustomTestCase):
    def setUp(self):
        old = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.generator = torch.Generator(device="cuda").manual_seed(197)
        self.hc_parallel = SimpleNamespace(
            attn_dp_size=1, attn_cp_size=1, moe_ep_size=2
        )
        self.hc_tp = SimpleNamespace(input_scattered=False)
        self.hc_a2a = Mock()
        self.hc_a2a.is_none.return_value = True
        for name, value in (
            ("get_parallel", self.hc_parallel),
            ("get_attn_tp_context", self.hc_tp),
            ("get_moe_a2a_backend", self.hc_a2a),
        ):
            self.enterContext(
                patch("sglang.srt.models.qwen4_exp." + name, return_value=value)
            )

    @torch.no_grad()
    def test_qwen_attention_to_ffn_sum_handoff(self):
        from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
        from sglang.srt.models.qwen4_exp import Qwen4ExpLayerExtensionMixin

        checkpoint = _checkpoint(512, 64, torch.bfloat16)
        layer = Qwen4ExpLayerExtensionMixin()
        layer.ple = None
        layer.attn_hyper_connection = _layer(512, 64, torch.bfloat16)
        layer.mlp_hyper_connection = _layer(512, 64, torch.bfloat16)
        for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
            hc.load_state_dict(checkpoint)
            hc.prepare_sum_state_weights()
        for t in (1, 24, 25, 40):
            with self.subTest(t=t):
                prepared = _bootstrap_sum_state(
                    layer.attn_hyper_connection,
                    self.tensor((t, 4 * 512), torch.bfloat16),
                )
                state = GatedResidualState(
                    lambda value: value,
                    layer._attn_mix,
                    layer._ffn_mix,
                    layer._attn_combine,
                    layer._ffn_combine,
                )
                with layer.hc_inputs(prepared, None) as inputs:
                    mixed, residual = state.read_attn_input(prepared.residual)
                    attn_context = state.sum_context
                    self.check_mix(mixed, attn_context, checkpoint)
                    attn_core = self.tensor(mixed.shape, mixed.dtype)
                    mixed, residual = state.update_and_read_ffn_input(
                        attn_core, residual
                    )
                    self.check_apply(attn_context, attn_core, inputs.ffn_input)
                    self.assertIs(state.sum_context.state, inputs.ffn_input)
                    self.check_mix(mixed, state.sum_context, checkpoint)
                    ffn_context = state.sum_context
                    ffn_core = self.tensor(mixed.shape, mixed.dtype)
                    result = state.residual_ops().ffn_update.update(ffn_core, residual)
                    self.check_apply(ffn_context, ffn_core, inputs.output)
                    self.assertIs(result, inputs.output.residual)
                    self.assertIsNone(state.sum_context)
                self.assertIsNone(layer._hc_inputs)

    def tensor(self, shape, dtype):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=self.generator)

    @torch.no_grad()
    def test_guarded_handoff_bootstraps_after_in_place_change(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpLayerExtensionMixin

        h, lowrank, dtype = 512, 64, torch.bfloat16
        checkpoint = _checkpoint(h, lowrank, dtype)
        layer = Qwen4ExpLayerExtensionMixin()
        layer.ple = None
        layer.attn_hyper_connection = _layer(h, lowrank, dtype)
        layer.attn_hyper_connection.load_state_dict(checkpoint)
        # CPU tests cover the mode predicates; exercise their real bootstrap
        # fallback on both dispatch sides with one in-place-changing mode.
        self.hc_a2a.is_none.return_value = False
        for t in (24, 25):
            with self.subTest(t=t):
                residual = self.tensor((t, 4 * h), dtype)
                stale = _bootstrap_sum_state(layer.attn_hyper_connection, residual)
                residual.add_(1)
                with layer.hc_inputs(stale, None):
                    mixed, context = layer._attn_mix(residual)
                self.assertIsNot(context.state, stale)
                torch.testing.assert_close(
                    context.state.sum_sq,
                    residual.float().view(t, 4, h).square().sum(-1),
                    rtol=3e-6,
                    atol=1e-5,
                )
                self.check_mix(mixed, context, checkpoint)

    @torch.no_grad()
    def test_deterministic_mode(self):
        """Flag-enabled HC is repeatable at fixed T, including Graph and slices.

        Small/large-T projection rounding is unchanged; this is not a claim of
        bitwise equality across that dispatch boundary or whole-model determinism.
        """
        from sglang.srt.batch_invariant_ops import (
            disable_batch_invariant_mode,
            enable_batch_invariant_mode,
            is_batch_invariant_mode_enabled,
        )

        self.assertFalse(is_batch_invariant_mode_enabled())
        execution = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=True)
        )
        cases = []
        with patch("sglang.srt.runtime_context.get_exec", return_value=execution):
            for dtype in (torch.bfloat16, torch.float16):
                for h, lowrank in ((512, 64), (2560, 320)):
                    checkpoint = _checkpoint(h, lowrank, dtype)
                    layer = _layer(h, lowrank, dtype)
                    layer.load_state_dict(checkpoint)
                    for t in (0, 1, 24, 25, 80):
                        state = _bootstrap_sum_state(
                            layer, self.tensor((t, 4 * h), dtype)
                        )
                        # Reference runs before installing the framework overrides.
                        expected = _reference_mix(
                            state, checkpoint, layer.config.rms_norm_eps
                        )
                        cases.append(
                            (layer, state, self.tensor((t, h), dtype), expected)
                        )

        enable_batch_invariant_mode()
        self.addCleanup(disable_batch_invariant_mode)
        for layer, state, core, expected in cases:
            t, h = core.shape
            with self.subTest(dtype=core.dtype, h=h, t=t):

                def step():
                    mixed, context = layer.mix(state)
                    return mixed, context, layer.combine(core, context)

                def values(mixed, context, output):
                    return (
                        mixed,
                        context.buffers.alpha,
                        output.residual,
                        output.sum_sq,
                    )

                mixed, context, output = step()
                self.assertEqual(context.buffers.partial_sums.shape, (t, 4, 10))
                self.check_mix(mixed, context, None, expected=expected)
                self.check_apply(context, core, output)
                reference = [
                    x.view(torch.uint8).clone() for x in values(mixed, context, output)
                ]
                for _ in range(3):
                    mixed, context, output = step()
                    for actual, wanted in zip(
                        values(mixed, context, output), reference
                    ):
                        self.assertTrue(torch.equal(actual.view(torch.uint8), wanted))
                if t in (24, 25):
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        mixed, context, output = step()
                    for _ in range(3):
                        context.buffers.partial_sums.fill_(float("nan"))
                        output.sum_sq.fill_(float("nan"))
                        graph.replay()
                        for actual, wanted in zip(
                            values(mixed, context, output), reference
                        ):
                            self.assertTrue(
                                torch.equal(actual.view(torch.uint8), wanted)
                            )
                    # Attention TP can slice a large-T Mix down to small-T Apply.
                    sliced = context.slice_rows(1, 2)
                    sliced.buffers.output.sum_sq.fill_(float("nan"))
                    sliced.buffers.partial_sums.fill_(float("nan"))
                    written = sliced.combine(core.tensor_split(2)[1])
                    self.check_apply(sliced, core.tensor_split(2)[1], written)
                    for _ in range(3):
                        sliced.combine(core.tensor_split(2)[1])
                        self.assertTrue(
                            torch.equal(
                                written.sum_sq.view(torch.uint8),
                                reference[-1].tensor_split(2)[1],
                            )
                        )

    def check_mix(self, mixed, context, checkpoint, *, expected=None):
        if expected is None:
            expected = _reference_mix(context.state, checkpoint, context.operator.eps)
        expected, alpha = expected
        if not context.operator.deterministic:
            self.assertIsNone(context.buffers.partial_sums)
        bf16 = mixed.dtype == torch.bfloat16
        torch.testing.assert_close(
            mixed,
            expected,
            rtol=0.02 if bf16 else 0.003,
            atol=0.004 if bf16 else 0.0005,
        )
        if isinstance(context.operator, SmallTSumHCShell):
            alpha_tol = dict(rtol=3e-5, atol=3e-6)
        else:
            # Independent FP32 accumulation can straddle a BF16/FP16 midpoint
            # before the raw-logit cast; allow one storage ULP after sigmoid.
            alpha_tol = dict(rtol=3e-5, atol=0.004 if bf16 else 0.0005)
        torch.testing.assert_close(context.buffers.alpha, alpha, **alpha_tol)

    def check_apply(self, context, core, output):
        t, h = core.shape
        # FP64 multiply/add followed by FP32 models the fused FP32 update.
        # The sum observes this update BEFORE casting the stored residual.
        update = (
            context.state.residual.double().view(t, 4, h)
            + context.buffers.alpha.double()[:, :, None] * core.double()[:, None, :]
        ).float()
        self.assertIsInstance(output, HCSumState)
        torch.testing.assert_close(
            output.residual, update.to(core.dtype).flatten(1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            output.sum_sq,
            update.double().square().sum(-1).float(),
            rtol=3e-6,
            atol=1e-5,
        )

    def step(self, layer, inputs, checkpoint):
        mixed, context = layer.mix(inputs)
        if mixed.shape[0] <= 24:
            self.assertIsInstance(context.operator, SmallTSumHCShell)
        else:
            self.assertNotIsInstance(context.operator, SmallTSumHCShell)
        self.check_mix(mixed, context, checkpoint)
        torch.testing.assert_close(
            context.buffers.output.sum_sq,
            torch.zeros_like(context.state.sum_sq),
            rtol=0,
            atol=0,
        )
        core = self.tensor(mixed.shape, mixed.dtype)  # Explicit non-identity core.
        output = layer.combine(core, context)
        self.check_apply(context, core, output)
        return output, context

    @torch.no_grad()
    def test_numerics_dispatch_and_empty(self):
        for dtype in (torch.bfloat16, torch.float16):
            # Both shapes use a CTA-aligned, trailing alpha-only Inject tile.
            for h, lowrank, tokens in (
                (2560, 320, (0, 1, 8, 16, 24, 25, 80)),
                (2048, 192, (3, 25)),
            ):
                checkpoint = _checkpoint(h, lowrank, dtype)
                layer = _layer(h, lowrank, dtype)
                layer.load_state_dict(checkpoint)
                for t in tokens:
                    with self.subTest(dtype=dtype, h=h, lowrank=lowrank, t=t):
                        residual = self.tensor((t, 4 * h), dtype)
                        saved = residual.clone()
                        output, context = self.step(layer, residual, checkpoint)
                        # Reuse the existing cases, including both dispatch sides.
                        # This oracle normalizes stored activations before GEMM;
                        # it does not reproduce the optimized weight folding.
                        expected = _reference_unfused_mix(
                            residual, checkpoint, layer.config.rms_norm_eps
                        )
                        for actual, reference in zip(
                            (context.buffers.mixed, context.buffers.alpha), expected
                        ):
                            torch.testing.assert_close(
                                actual,
                                reference,
                                rtol=0.02 if dtype == torch.bfloat16 else 0.003,
                                atol=0.004 if dtype == torch.bfloat16 else 0.0005,
                            )
                        bootstrap = _bootstrap_sum_state(layer, output.residual)
                        # For normal-range values rounded with unit roundoff u,
                        # |sum(round(x)^2) - sum(x^2)| <= (2u + u^2) * sum(x^2).
                        # Allow the two FP32 reductions' existing 3e-6 budgets.
                        u = torch.finfo(dtype).eps / 2
                        torch.testing.assert_close(
                            bootstrap.sum_sq,
                            output.sum_sq,
                            rtol=2 * u + u * u + 6e-6,
                            atol=2e-5,
                        )
                        torch.testing.assert_close(
                            context.state.sum_sq,
                            residual.float().view(t, 4, h).square().sum(-1),
                            rtol=3e-6,
                            atol=1e-5,
                        )
                        torch.testing.assert_close(residual, saved, rtol=0, atol=0)

    @torch.no_grad()
    def test_internal_buffers_keep_external_input_guards(self):
        """Internally allocated outputs do not make external sums/core trusted."""
        h, lowrank, dtype = 512, 64, torch.bfloat16
        layer = _layer(h, lowrank, dtype)
        layer.load_state_dict(_checkpoint(h, lowrank, dtype))
        for t in (24, 25):
            residual = self.tensor((t, 4 * h), dtype)
            state = _bootstrap_sum_state(layer, residual)
            misaligned = torch.empty((t * 4 + 4,), device="cuda", dtype=torch.float32)
            for invalid in (
                state.sum_sq.to(dtype),
                state.sum_sq[:, :3],
                misaligned[4:].view(t, 4),
            ):
                with self.subTest(t=t, shape=invalid.shape, dtype=invalid.dtype):
                    with self.assertRaisesRegex(ValueError, "statistics metadata"):
                        layer.mix(state._replace(sum_sq=invalid))

            mixed, context = layer.mix(state)
            for invalid in (mixed.float(), mixed[:, :-1]):
                with self.subTest(
                    t=t, core_shape=invalid.shape, core_dtype=invalid.dtype
                ):
                    with self.assertRaisesRegex(ValueError, "block_output"):
                        layer.combine(invalid, context)
            output = layer.combine(mixed, context)
            self.check_apply(context, mixed, output)

    @torch.no_grad()
    def test_caller_buffers_keep_weight_alias_and_metadata_guards(self):
        """An output view into packed weights must fail before any writable launch."""
        h, lowrank, dtype = 512, 64, torch.bfloat16
        layer = _layer(h, lowrank, dtype)
        layer.load_state_dict(_checkpoint(h, lowrank, dtype))
        for t in (24, 25):
            state = _bootstrap_sum_state(layer, self.tensor((t, 4 * h), dtype))
            operator = layer._sum_operators[int(t > 24)]
            buffers = operator.allocate_buffers(t)
            weight_output = layer._hc_packed_down.view(-1)[: t * 4 * h].view(t, 4 * h)
            invalid = buffers._replace(
                output=buffers.output._replace(residual=weight_output)
            )
            with self.subTest(t=t):
                with self.assertRaisesRegex(ValueError, "alias"):
                    operator.mix(state, invalid)
                with self.assertRaisesRegex(ValueError, "alias"):
                    operator.combine(state, invalid, buffers.mixed)
                with self.assertRaisesRegex(ValueError, "metadata"):
                    operator.mix(state, buffers._replace(mixed=buffers.mixed.float()))
                partials = (
                    None
                    if operator.deterministic
                    else torch.empty((t, 4, 10), device="cuda", dtype=torch.float32)
                )
                with self.assertRaisesRegex(ValueError, "deterministic mode"):
                    operator.mix(state, buffers._replace(partial_sums=partials))

    @torch.no_grad()
    def test_state_handoff_invalidation_and_final_mix(self):
        h, lowrank, dtype = 512, 64, torch.bfloat16
        checkpoint = _checkpoint(h, lowrank, dtype)
        first = _layer(h, lowrank, dtype)
        second = _layer(h, lowrank, dtype, eps=1e-4)
        first.load_state_dict(checkpoint)
        second.load_state_dict(checkpoint)
        residual = self.tensor((25, 4 * h), dtype)
        output, _ = self.step(first, residual, checkpoint)
        selected = HCSumState(output.residual[:24], output.sum_sq[:24])
        next_output, context = self.step(second, selected, checkpoint)
        self.assertIs(context.state, selected)
        expanded = HCSumState(
            torch.cat((next_output.residual, output.residual[24:])),
            torch.cat((next_output.sum_sq, output.sum_sq[24:])),
        )
        output, context = self.step(first, expanded, checkpoint)
        self.assertIs(context.state, expanded)
        # Passing a changed Tensor (e.g. PLE) must bootstrap fresh statistics.
        changed = output.residual + 0.125
        _, context = self.step(second, changed, checkpoint)
        torch.testing.assert_close(
            context.state.sum_sq,
            changed.float().view(25, 4, h).square().sum(-1),
            rtol=3e-6,
            atol=1e-5,
        )
        final = _layer(h, lowrank, dtype, combine=False)
        final.load_state_dict({name: checkpoint[name] for name in final.state_dict()})
        self.assertFalse(final._packed_sum_weights)
        from sglang.srt.models.qwen4_exp import _MixStreams

        from_state = _MixStreams(final, output)(output.residual)
        from_tensor, saved = final.mix(output.residual)
        self.assertIsInstance(saved, tuple)
        self.assertEqual(len(saved), 3)
        self.assertIsNone(saved[2])
        torch.testing.assert_close(from_state, from_tensor, rtol=0, atol=0)

    @torch.no_grad()
    def test_boundary_slice_and_prepared_handoff(self):
        from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState

        h, lowrank = 512, 64
        for dtype in (torch.bfloat16, torch.float16):
            checkpoint = _checkpoint(h, lowrank, dtype)
            layer = _layer(h, lowrank, dtype)
            consumer = _layer(h, lowrank, dtype)
            layer.load_state_dict(checkpoint)
            consumer.load_state_dict(checkpoint)
            for tokens in (6, 26, 50):
                with self.subTest(dtype=dtype, tokens=tokens):
                    residual = self.tensor((tokens, 4 * h), dtype)
                    boundary = GatedResidualState(
                        expand=lambda value: value,
                        attn_mix=layer.mix_prepared,
                        ffn_mix=consumer.mix_prepared,
                        attn_combine=lambda core, ctx: layer.combine_and_prepare(
                            core, ctx
                        )[0],
                        ffn_combine=lambda core, ctx: consumer.combine_and_prepare(
                            core, ctx
                        )[0],
                    )
                    _, saved = boundary.read_attn_input(residual)
                    with patch(
                        "sglang.srt.layers.layer_boundary.residual.gated.get_parallel",
                        return_value=SimpleNamespace(attn_tp_rank=1, attn_tp_size=2),
                    ):
                        sliced = boundary.slice_residual_attn_tp(saved)
                    context = boundary.sum_context
                    self.assertEqual(context.buffers.alpha.data_ptr() % 32, 16)
                    core = self.tensor((tokens // 2, h), dtype)
                    written = boundary.apply_attn_combine(core, sliced)
                    output = context.buffers.output
                    self.assertIs(written, output.residual)
                    self.check_apply(context, core, output)
                    mixed, following = consumer.mix_prepared(written, output)
                    torch.testing.assert_close(
                        following.state.sum_sq, output.sum_sq, rtol=0, atol=0
                    )
                    self.check_mix(mixed, following, checkpoint)

    @torch.no_grad()
    def test_graph_replay_with_changed_inputs(self):
        h, lowrank = 512, 64
        for dtype in (torch.bfloat16, torch.float16):
            checkpoint = _checkpoint(h, lowrank, dtype)
            layer = _layer(h, lowrank, dtype)
            layer.load_state_dict(checkpoint)
            for t in (24, 25):
                with self.subTest(dtype=dtype, t=t):
                    residual = self.tensor((t, 4 * h), dtype)
                    state = _bootstrap_sum_state(layer, residual)
                    core = self.tensor((t, h), dtype)

                    def step():
                        mixed, context = layer.mix(state)
                        return mixed, context, layer.combine(core, context)

                    for _ in range(3):
                        step()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        mixed, context, output = step()
                    for _ in range(3):
                        residual.copy_(self.tensor(residual.shape, dtype))
                        state.sum_sq.copy_(
                            residual.float().view(t, 4, h).square().sum(-1)
                        )
                        core.copy_(self.tensor(core.shape, dtype))
                        output.sum_sq.fill_(float("nan"))
                        graph.replay()
                        torch.cuda.synchronize()
                        self.check_mix(mixed, context, checkpoint)
                        self.check_apply(context, core, output)

    @torch.no_grad()
    def test_deterministic_apply_partials(self):
        from sglang.kernels.ops.elementwise.hc_apply_sum import (
            PARTIALS_PER_BRANCH,
            hc_apply_sum,
        )

        for dtype in (torch.bfloat16, torch.float16):
            for h in (512, 2560, 16384):
                residual = self.tensor((80, 4 * h), dtype)
                core = self.tensor((80, h), dtype)
                alpha = torch.rand((80, 4), device="cuda", generator=self.generator) * 2
                previous_batch = None
                for t in (0, 1, 24, 25, 80):
                    with self.subTest(dtype=dtype, h=h, t=t):
                        out = torch.empty_like(residual[:t])
                        sums = torch.empty((t, 4), device="cuda", dtype=torch.float32)
                        partials = torch.empty(
                            (t, 4, PARTIALS_PER_BRANCH),
                            device="cuda",
                            dtype=torch.float32,
                        )
                        context = SimpleNamespace(
                            state=HCSumState(residual[:t], sums),
                            buffers=SimpleNamespace(alpha=alpha[:t]),
                        )

                        def step():
                            return hc_apply_sum(
                                core[:t],
                                residual[:t],
                                alpha[:t],
                                out=out,
                                sum_sq=sums,
                                partial_sums=partials,
                            )

                        expected_bits = None
                        for _ in range(3):
                            sums.fill_(float("nan"))
                            partials.fill_(float("nan"))
                            step()
                            self.check_apply(context, core[:t], HCSumState(out, sums))
                            bits = (out.view(torch.int16), sums.view(torch.int32))
                            if expected_bits is None:
                                expected_bits = tuple(x.clone() for x in bits)
                            for actual, expected in zip(bits, expected_bits):
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                        if t:
                            if previous_batch is not None:
                                for actual, expected in zip(bits, previous_batch):
                                    torch.testing.assert_close(
                                        actual[: expected.shape[0]],
                                        expected,
                                        rtol=0,
                                        atol=0,
                                    )
                            previous_batch = expected_bits
                        if t == 25:
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                step()
                            for _ in range(3):
                                sums.fill_(float("nan"))
                                partials.fill_(float("nan"))
                                graph.replay()
                                for actual, expected in zip(bits, expected_bits):
                                    torch.testing.assert_close(
                                        actual, expected, rtol=0, atol=0
                                    )

                        atomic_out = torch.empty_like(out)
                        atomic_sums = torch.zeros_like(sums)
                        hc_apply_sum(
                            core[:t],
                            residual[:t],
                            alpha[:t],
                            out=atomic_out,
                            sum_sq=atomic_sums,
                        )
                        torch.testing.assert_close(out, atomic_out, rtol=0, atol=0)
                        torch.testing.assert_close(
                            sums, atomic_sums, rtol=3e-6, atol=1e-5
                        )

        with self.assertRaisesRegex(ValueError, "partial_sums"):
            hc_apply_sum(
                core,
                residual,
                alpha,
                out=out,
                sum_sq=sums,
                partial_sums=partials[..., :9],
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            hc_apply_sum(
                core,
                residual,
                alpha,
                out=out,
                sum_sq=partials.view(-1)[: 80 * 4].view(80, 4),
                partial_sums=partials,
            )

    def check_storage(self, layer, checkpoint):
        small, large = layer._sum_operators
        parameters = dict(layer.named_parameters())
        expected = _native_checkpoint(checkpoint)
        self.assertEqual(set(parameters), set(expected))
        for name, parameter in parameters.items():
            self.assertTrue(parameter.is_cuda)
            torch.testing.assert_close(parameter.cpu(), expected[name], rtol=0, atol=0)
        for view, name in (
            (small.weights.down_inject, "input_mix_weight_down.weight"),
            (large.weights.matrix, "input_mix_weight_down.weight"),
            (small.weights.norm_permuted, "hc_norm.weight"),
            (large.norm_weight, "hc_norm.weight"),
            (large.up_weight, "input_mix_weight_up.weight"),
        ):
            self.assertEqual(view.data_ptr(), parameters[name].data_ptr())
        h = layer.hidden_size
        lowrank = layer.config.hc_lowrank
        torch.testing.assert_close(
            small.weights.up[:, :lowrank]
            .view(h, 4, lowrank)
            .transpose(0, 1)
            .reshape(4 * h, lowrank)
            .cpu(),
            checkpoint["input_mix_weight_up.weight"],
            rtol=0,
            atol=0,
        )

    @torch.no_grad()
    def test_checkpoint_round_trip_and_shared_storage(self):
        h, lowrank = 512, 64
        for dtype in (torch.bfloat16, torch.float16):
            checkpoint = _checkpoint(h, lowrank, dtype)
            checkpoint["hc_norm.weight"][0] = -1
            layer = _layer(h, lowrank, dtype)
            layer.load_state_dict(checkpoint)
            self.check_storage(layer, checkpoint)
            state = layer.state_dict()
            native = _native_checkpoint(checkpoint)
            self.assertEqual(set(state), set(native))
            for name, value in state.items():
                torch.testing.assert_close(value.cpu(), native[name], rtol=0, atol=0)
            restored = _layer(h, lowrank, dtype)
            restored.load_state_dict(state)
            self.check_storage(restored, checkpoint)
            for t in (24, 25):
                self.step(restored, self.tensor((t, 4 * h), dtype), checkpoint)
            # A cold native loader copies registered tensors directly, then
            # prepares local views once; it must not fold them again.
            restored = _layer(h, lowrank, dtype)
            replacement = _checkpoint(h, lowrank, dtype, seed=419)
            native_replacement = _native_checkpoint(replacement)
            for name, tensor in restored.state_dict().items():
                tensor.copy_(native_replacement[name])
            self.assertTrue(restored._sum_weights_dirty)
            restored.prepare_sum_state_weights(force=True)
            self.check_storage(restored, replacement)
            for t in (24, 25):
                self.step(restored, self.tensor((t, 4 * h), dtype), replacement)

    @torch.no_grad()
    def test_initial_weight_loaders_and_reload_rejection(self):
        h, lowrank = 512, 64
        for dtype in (torch.bfloat16, torch.float16):
            checkpoint = _checkpoint(h, lowrank, dtype)
            layer = _layer(h, lowrank, dtype)
            parameters = dict(layer.named_parameters())
            for name, value in reversed(tuple(checkpoint.items())):
                target = name.replace(
                    "block_inject_weight.weight", "input_mix_weight_down.weight"
                )
                parameter = parameters[target]
                parameter.weight_loader(parameter, value)
            layer.prepare_sum_state_weights()
            self.check_storage(layer, checkpoint)
            for t in (24, 25):
                self.step(layer, self.tensor((t, 4 * h), dtype), checkpoint)
            before = {name: value.clone() for name, value in layer.state_dict().items()}
            parameter = layer.input_mix_weight_down.weight
            with self.assertRaisesRegex(ValueError, "Online weight updates"):
                parameter.weight_loader(
                    parameter, checkpoint["input_mix_weight_down.weight"]
                )
            with self.assertRaisesRegex(ValueError, "Online weight updates"):
                layer.load_state_dict(checkpoint)
            for name, value in layer.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    @torch.no_grad()
    def test_ipc_restore_rebinds_without_repacking(self):
        from types import SimpleNamespace

        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        h, lowrank, dtype = 512, 64, torch.bfloat16
        checkpoint = _checkpoint(h, lowrank, dtype)
        source = _HCSequential(_layer(h, lowrank, dtype))
        source[0].load_state_dict(checkpoint)
        entries = {
            name: {"tensor": value.detach(), "is_param": True}
            for name, value in source.named_parameters()
        }
        entries.update(
            {
                name: {"tensor": value, "is_param": False}
                for name, value in source.named_buffers()
            }
        )
        loader = IpcModelLoader(SimpleNamespace())
        loader._resolve_engine_quant = Mock(return_value=("", None))
        loader._fetch_from_cache = Mock(return_value={"entries": entries, "pid": None})
        loader._start_daemon_liveness_watchdog = Mock()
        # Mapping itself is tested by the CPU transport tests; this exercises
        # the real loader after mapping, with synthetic shared CUDA buffers.
        loader._transport_backend = SimpleNamespace(
            name="mapped", import_tensor=lambda entry: entry["tensor"]
        )
        with (
            patch(
                "sglang.srt.weight_cache.ipc_loader._initialize_model",
                side_effect=lambda *args: _HCSequential(
                    _layer(h, lowrank, dtype, meta=True)
                ),
            ),
            patch(
                "sglang.srt.model_loader.loader._get_quantization_config",
                return_value=None,
            ),
            patch.object(
                PackedGatedResidual,
                "_fold_checkpoint_weights",
                side_effect=AssertionError("IPC must not repack shared weights"),
            ),
        ):
            imported = loader.load_model(
                model_config=SimpleNamespace(dtype=dtype),
                device_config=SimpleNamespace(device=torch.device("cuda", 0)),
            )
        self.check_storage(imported[0], checkpoint)
        self.assertEqual(
            imported[0]._hc_packed_down.data_ptr(), source[0]._hc_packed_down.data_ptr()
        )
        for t in (1, 25):
            self.step(imported[0], self.tensor((t, 4 * h), dtype), checkpoint)

    @torch.no_grad()
    def test_native_export_and_offload_residency(self):
        from sglang.srt.model_executor.model_runner_components.weight_exporter import (
            WeightExporter,
        )
        from sglang.srt.utils.offloader import _get_resident_parameter_ids

        layer = _layer(512, 64, torch.bfloat16)
        layer.load_state_dict(_checkpoint(512, 64, torch.bfloat16))
        group = object()
        exporter = WeightExporter(
            tp_rank=0,
            tp_size=1,
            gpu_id=0,
            get_model_path=lambda: "",
            get_model=lambda: layer,
            _weights_send_group={"native_33001_0": group},
        )
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.broadcast"),
            patch("torch.distributed.distributed_c10d.destroy_process_group"),
        ):
            success, message = exporter.send_weights_to_remote_instance(
                "127.0.0.1", "33001", "native"
            )
        self.assertTrue(success, message)
        parameters = list(layer.parameters())
        self.assertEqual(
            _get_resident_parameter_ids(layer), {id(p) for p in parameters}
        )

        # Native device placement is honored; CUDA shells are rebuilt on onload.
        native = {
            name: value.detach().cpu().clone()
            for name, value in layer.state_dict().items()
        }
        layer.cpu()
        self.assertTrue(all(p.device.type == "cpu" for p in layer.parameters()))
        self.assertIsNone(layer._sum_operators)
        layer.cuda()
        self.assertTrue(all(p.is_cuda for p in layer.parameters()))
        self.assertIsNotNone(layer._sum_operators)
        for name, value in layer.state_dict().items():
            torch.testing.assert_close(value.cpu(), native[name], rtol=0, atol=0)

        invalid = dict(layer.state_dict())
        # Same logical shape, but interleaving is not dense: must not silently
        # allocate an RMS copy and leave the Graph bound to stale storage.
        invalid["hc_norm.weight"] = invalid["hc_norm.weight"].T.contiguous().T
        self.assertFalse(invalid["hc_norm.weight"].is_contiguous())
        fresh = _layer(512, 64, torch.bfloat16, meta=True)
        with self.assertRaisesRegex(ValueError, "RMS parameter must be contiguous"):
            fresh.load_state_dict(invalid, assign=True)


if __name__ == "__main__":
    unittest.main()
