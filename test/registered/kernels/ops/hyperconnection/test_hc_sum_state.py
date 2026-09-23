"""Synthetic HC regressions through the serving module; no model checkpoint."""

import io
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.elementwise.hc_sum_state import HCSumState, SmallTSumHCShell
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


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
    layer = GatedResidual(
        HyperConnectionConfig(
            hc_count=4,
            hidden_size=h,
            hc_lowrank=lowrank,
            hc_per_branch_norm=True,
            rms_norm_eps=eps,
            params_dtype=dtype,
        ),
        use_combine=combine,
        packed_weights=True,
    )
    if not meta:
        layer.to(device="cuda", dtype=dtype)
    return layer.eval().requires_grad_(False)


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

    def tensor(self, shape, dtype):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=self.generator)

    def check_mix(self, mixed, context, checkpoint):
        expected, alpha = _reference_mix(
            context.state, checkpoint, context.operator.eps
        )
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
            # L=320 uses alpha-only; L=288 also exercises the direct epilogue.
            for h, lowrank, tokens in (
                (2560, 320, (0, 1, 8, 16, 24, 25, 80)),
                (2048, 288, (3, 25)),
            ):
                checkpoint = _checkpoint(h, lowrank, dtype)
                layer = _layer(h, lowrank, dtype)
                layer.load_state_dict(checkpoint)
                for t in tokens:
                    with self.subTest(dtype=dtype, h=h, lowrank=lowrank, t=t):
                        residual = self.tensor((t, 4 * h), dtype)
                        saved = residual.clone()
                        _, context = self.step(layer, residual, checkpoint)
                        torch.testing.assert_close(
                            context.state.sum_sq,
                            residual.float().view(t, 4, h).square().sum(-1),
                            rtol=3e-6,
                            atol=1e-5,
                        )
                        torch.testing.assert_close(residual, saved, rtol=0, atol=0)

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
        from_state, _ = final.mix(output)
        from_tensor, saved = final.mix(output.residual)
        self.assertIsInstance(saved, tuple)
        self.assertEqual(len(saved), 2)
        torch.testing.assert_close(from_state, from_tensor, rtol=0, atol=0)

    @torch.no_grad()
    def test_graph_replay_resets_sums_with_changed_inputs(self):
        h, lowrank = 512, 64
        for dtype in (torch.bfloat16, torch.float16):
            checkpoint = _checkpoint(h, lowrank, dtype)
            layer = _layer(h, lowrank, dtype)
            layer.load_state_dict(checkpoint)
            for t in (24, 25):
                with self.subTest(dtype=dtype, t=t):
                    residual = self.tensor((t, 4 * h), dtype)
                    state = layer.bootstrap_sum_state(residual)
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

    def check_storage(self, layer, checkpoint):
        small, large = layer._sum_operators
        for matrix in (small.weights.down_inject, large.weights.matrix):
            self.assertEqual(
                matrix.untyped_storage().data_ptr(),
                layer._hc_packed_down.untyped_storage().data_ptr(),
            )
        for norm in (small.weights.norm_permuted, large.norm_weight):
            self.assertEqual(
                norm.untyped_storage().data_ptr(),
                layer._hc_packed_norm.untyped_storage().data_ptr(),
            )
        for name in (
            "input_mix_weight_down.weight",
            "block_inject_weight.weight",
            "hc_norm.weight",
        ):
            self.assertEqual(dict(layer.named_parameters())[name].device.type, "cpu")
        # Check the physical pack against the checkpoint, including gamma=0.
        h = layer.hidden_size
        lowrank = layer.config.hc_lowrank
        folded = (
            torch.cat(
                (
                    checkpoint["input_mix_weight_down.weight"],
                    checkpoint["block_inject_weight.weight"],
                )
            ).float()
            * (1 + checkpoint["hc_norm.weight"].float())
        ).to(layer.params_dtype)
        torch.testing.assert_close(
            layer._hc_packed_down[:, : lowrank + 4]
            .permute(1, 0, 2)
            .reshape(lowrank + 4, 4 * h)
            .cpu(),
            folded,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            layer._hc_packed_norm.cpu().view(h, 4).T.flatten(),
            checkpoint["hc_norm.weight"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            large.up_weight.data_ptr(), layer.input_mix_weight_up.weight.data_ptr()
        )
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
        h, lowrank = 512, 8
        for dtype in (torch.bfloat16, torch.float16):
            checkpoint = _checkpoint(h, lowrank, dtype)
            checkpoint["hc_norm.weight"][0] = -1
            layer = _layer(h, lowrank, dtype)
            layer.load_state_dict(checkpoint)
            self.check_storage(layer, checkpoint)
            state = layer.state_dict()
            self.assertEqual(set(state), set(checkpoint))
            for name, value in state.items():
                torch.testing.assert_close(
                    value.cpu(), checkpoint[name], rtol=0, atol=0
                )
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer.seek(0)
            restored = _layer(h, lowrank, dtype)
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            self.check_storage(restored, checkpoint)
            for t in (24, 25):
                self.step(restored, self.tensor((t, 4 * h), dtype), checkpoint)
            # Presharded/sharded loaders copy tensors without invoking the
            # weight_loader or load_state_dict hooks. Force must refresh them.
            replacement = _checkpoint(h, lowrank, dtype, seed=419)
            for name, tensor in restored.state_dict().items():
                tensor.copy_(replacement[name])
            self.assertFalse(restored._sum_weights_dirty)
            restored.prepare_sum_state_weights(force=True)
            self.check_storage(restored, replacement)
            for t in (24, 25):
                self.step(restored, self.tensor((t, 4 * h), dtype), replacement)

    @torch.no_grad()
    def test_weight_loader_and_direct_updates(self):
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            _model_load_weights_direct,
        )

        h, lowrank, dtype = 512, 8, torch.bfloat16
        checkpoint = _checkpoint(h, lowrank, dtype)
        model = torch.nn.Module()
        model.hc = _layer(h, lowrank, dtype)
        # Real parameter loaders, with RMS last, followed by one transaction end.
        parameters = dict(model.hc.named_parameters())
        for name, value in checkpoint.items():
            parameters[name].weight_loader(parameters[name], value)
        model.hc.prepare_sum_state_weights()
        self.check_storage(model.hc, checkpoint)
        model.other = torch.nn.Parameter(
            torch.zeros(2, device="cuda"), requires_grad=False
        )
        model.other.weight_loader = Mock(
            side_effect=AssertionError("non-HC loader must not be called")
        )
        pair = {
            name: checkpoint[name] * 0.75
            for name in ("input_mix_weight_down.weight", "block_inject_weight.weight")
        }
        _model_load_weights_direct(
            model,
            [("other", torch.ones_like(model.other))]
            + [("hc." + name, value) for name, value in pair.items()],
        )
        checkpoint.update(pair)
        self.check_storage(model.hc, checkpoint)
        torch.testing.assert_close(
            model.other, torch.ones_like(model.other), rtol=0, atol=0
        )
        model.other.weight_loader.assert_not_called()
        checkpoint["hc_norm.weight"] = checkpoint["hc_norm.weight"] + 0.125
        _model_load_weights_direct(
            model, [("hc.hc_norm.weight", checkpoint["hc_norm.weight"])]
        )
        self.check_storage(model.hc, checkpoint)
        # Up's ordinary nn.Linear state_dict load bypasses its weight_loader;
        # the parent hook must still refresh the small-T derived layout.
        up_name = "input_mix_weight_up.weight"
        checkpoint[up_name] = checkpoint[up_name] * 2
        model.hc.load_state_dict({up_name: checkpoint[up_name]}, strict=False)
        self.check_storage(model.hc, checkpoint)
        for t in (24, 25):
            self.step(model.hc, self.tensor((t, 4 * h), dtype), checkpoint)
        with self.assertRaisesRegex(ValueError, "Down and Inject"):
            model.hc.load_state_dict(
                {"block_inject_weight.weight": pair["block_inject_weight.weight"]},
                strict=False,
            )
        with self.assertRaisesRegex(ValueError, "Down and Inject"):
            _model_load_weights_direct(
                model,
                [
                    (
                        "hc.input_mix_weight_down.weight",
                        pair["input_mix_weight_down.weight"] * 2,
                    )
                ],
            )
        # A failed partial direct transaction is recovered with a complete pair.
        _model_load_weights_direct(
            model, [("hc." + name, value) for name, value in pair.items()]
        )
        self.check_storage(model.hc, checkpoint)
        self.step(model.hc, self.tensor((25, 4 * h), dtype), checkpoint)

    @torch.no_grad()
    def test_ipc_restore_rebinds_without_repacking(self):
        from types import SimpleNamespace

        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        h, lowrank, dtype = 512, 8, torch.bfloat16
        checkpoint = _checkpoint(h, lowrank, dtype)
        source = torch.nn.Sequential(_layer(h, lowrank, dtype))
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
                side_effect=lambda *args: torch.nn.Sequential(
                    _layer(h, lowrank, dtype, meta=True)
                ),
            ),
            patch(
                "sglang.srt.model_loader.loader._get_quantization_config",
                return_value=None,
            ),
            patch.object(
                GatedResidual,
                "_prepare_packed_sum_state_weights",
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
    def test_nccl_staging_preserves_checkpoint_storage(self):
        from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
            broadcast_weights,
        )

        # Exercise real H2D/D2H copies, but stub the collective itself: no
        # process group or peer is needed to test the new staging contract.
        model = _layer(512, 8, torch.bfloat16)
        checkpoint = _checkpoint(512, 8, torch.bfloat16)
        model.load_state_dict(checkpoint)
        before = [(p.device, p.data_ptr()) for p in model.parameters()]
        group = object()
        for is_src in (True, False):
            calls = []

            def broadcast(tensor, *, src, group):
                self.assertTrue(tensor.is_cuda)
                self.assertEqual(src, 0)
                calls.append(tensor.clone())
                if not is_src:
                    tensor.fill_(len(calls))

            with (
                patch("torch.distributed.get_backend", return_value="nccl"),
                patch(
                    "torch.distributed.broadcast", side_effect=broadcast
                ) as collective,
            ):
                broadcast_weights(
                    model, group=group, device=torch.device("cuda", 0), is_src=is_src
                )
            parameters = list(model.parameters())
            self.assertEqual(collective.call_count, len(parameters))
            for index, (parameter, sent, call) in enumerate(
                zip(parameters, calls, collective.call_args_list), 1
            ):
                self.assertIs(call.kwargs["group"], group)
                self.assertEqual(sent.shape, parameter.shape)
                self.assertEqual(sent.dtype, parameter.dtype)
                expected = (
                    sent.cpu() if is_src else torch.full_like(parameter, index).cpu()
                )
                torch.testing.assert_close(parameter.cpu(), expected, rtol=0, atol=0)
            self.assertEqual(before, [(p.device, p.data_ptr()) for p in parameters])


if __name__ == "__main__":
    unittest.main()
