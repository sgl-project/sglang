"""FlashInfer HC sum adapter: native single-storage weights, the initial-load
lifecycle and its write guards, Apply-context slicing, construction-time
selection, and the Qwen4-Exp prepared-sum gates. FlashInfer itself is replaced
by a CPU stand-in for the flashinfer.hyperconnection API it promises."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers import hyperconnection_flashinfer as hc_fi
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.layers.hyperconnection_flashinfer import (
    FlashInferGatedResidual,
    FlashInferGatedResidualState,
    FlashInferHCCheckpoint,
    bind_flashinfer_hc,
    create_gated_residual,
    install_flashinfer_hc_load_guard,
    zero_flashinfer_hc_padding,
)
from sglang.srt.layers.offloader import _get_resident_parameter_ids
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp_mtp import _mtp_param_name
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 8
# ceil8(L + 4) > L + 4, so the packed Down has padding rows.
LOWRANK = 64
WIDTH = 4 * HIDDEN
CONFIG = HyperConnectionConfig(
    hc_count=4, hidden_size=HIDDEN, hc_lowrank=LOWRANK, hc_per_branch_norm=True
)
HC = "layers.0.attn_hyper_connection"


def _fake_flashinfer(calls, support_reason=None):
    """The flashinfer.hyperconnection API the adapter imports, on CPU."""
    api = types.ModuleType("flashinfer.hyperconnection")

    def hc_sum_support_reason(*, hidden_size, lowrank, dtype, device, rms_eps=1e-6):
        return support_reason

    def hc_sum_weight_shapes(hidden_size, lowrank):
        return (4, (lowrank + 4 + 7) // 8 * 8, hidden_size), (hidden_size, 4)

    def pack_hc_sum_weights_into(down, inject, norm, *, out_down, out_norm):
        calls.append(("pack", down, inject, norm, out_down, out_norm))
        out_down.fill_(1)
        out_norm.copy_(norm.T)

    def bind_hc_sum_weights(packed_down, up, packed_norm, *, lowrank, rms_eps):
        calls.append(("bind", packed_down, up, packed_norm))
        return SimpleNamespace(packed_down=packed_down, up=up, norm=packed_norm)

    class HCSumRuntime:
        def __init__(self, plan, *, deterministic=False):
            self.plan = plan
            self.deterministic = deterministic

    api.hc_sum_support_reason = hc_sum_support_reason
    api.hc_sum_weight_shapes = hc_sum_weight_shapes
    api.pack_hc_sum_weights_into = pack_hc_sum_weights_into
    api.bind_hc_sum_weights = bind_hc_sum_weights
    api.HCSumRuntime = HCSumRuntime
    return api


def _install_flashinfer(test, api):
    package = types.ModuleType("flashinfer")
    package.hyperconnection = api
    patcher = patch.dict(
        sys.modules, {"flashinfer": package, "flashinfer.hyperconnection": api}
    )
    patcher.start()
    test.addCleanup(patcher.stop)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Registered before the HC module, so load_state_dict writes it first.
        self.embed = nn.Linear(2, 2, bias=False)
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attn_hyper_connection = FlashInferGatedResidual(CONFIG)
        install_flashinfer_hc_load_guard(self)

    @property
    def hc(self):
        return self.layers[0].attn_hyper_connection


def _raw_checkpoint():
    return {
        f"{HC}.input_mix_weight_down.weight": torch.randn(LOWRANK, WIDTH),
        f"{HC}.block_inject_weight.weight": torch.randn(4, WIDTH),
        f"{HC}.hc_norm.weight": torch.randn(WIDTH),
    }


def _load_up(model):
    param = model.hc.input_mix_weight_up.weight
    param.weight_loader(param, torch.randn_like(param))


def _storage(module):
    return {name: tensor.data_ptr() for name, tensor in module.state_dict().items()}


class TestFlashInferHCWeights(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        _install_flashinfer(self, _fake_flashinfer(self.calls))

    def test_raw_checkpoint_packs_once_into_the_native_parameters(self):
        model = _Model()
        _load_up(model)
        storage = _storage(model.hc)
        raw = _raw_checkpoint()
        flat_norm = raw[f"{HC}.hc_norm.weight"]
        expected_norm = flat_norm.to(torch.bfloat16).view(4, HIDDEN)
        checkpoint = FlashInferHCCheckpoint(model)
        for name, tensor in raw.items():
            self.assertTrue(checkpoint.stage(name, tensor))
        # Loaders may reuse the buffer a tensor came in.
        flat_norm.zero_()
        loaded = checkpoint.finish()

        [(_, down, inject, norm, out_down, out_norm)] = self.calls
        self.assertEqual({t.device.type for t in (down, inject, norm)}, {"cpu"})
        # The flat checkpoint norm is branch-major: [4 * H] is [4, H].
        torch.testing.assert_close(norm, expected_norm, rtol=0, atol=0)
        torch.testing.assert_close(
            model.hc.packed_norm_weight, expected_norm.T, rtol=0, atol=0
        )
        # One storage: packed into the parameters, which never move.
        self.assertEqual(out_down.data_ptr(), storage["packed_down_weight"])
        self.assertEqual(out_norm.data_ptr(), storage["packed_norm_weight"])
        self.assertEqual(_storage(model.hc), storage)
        self.assertEqual(
            set(storage),
            {"packed_down_weight", "packed_norm_weight", "input_mix_weight_up.weight"},
        )
        self.assertEqual(
            loaded, {f"{HC}.packed_down_weight", f"{HC}.packed_norm_weight"}
        )

        bind_flashinfer_hc(model)
        _, packed_down, up, packed_norm = self.calls[-1]
        self.assertEqual(
            [packed_down.data_ptr(), up.data_ptr(), packed_norm.data_ptr()],
            [
                storage["packed_down_weight"],
                storage["input_mix_weight_up.weight"],
                storage["packed_norm_weight"],
            ],
        )

    def test_partial_raw_checkpoint_is_rejected_without_packing(self):
        model = _Model()
        _load_up(model)
        checkpoint = FlashInferHCCheckpoint(model)
        for name, tensor in _raw_checkpoint().items():
            if "block_inject_weight" not in name:
                checkpoint.stage(name, tensor)
        with self.assertRaisesRegex(ValueError, "inject"):
            checkpoint.finish()
        self.assertEqual(self.calls, [])

    def test_missing_up_is_rejected_before_packing(self):
        model = _Model()
        checkpoint = FlashInferHCCheckpoint(model)
        for name, tensor in _raw_checkpoint().items():
            checkpoint.stage(name, tensor)
        with self.assertRaisesRegex(ValueError, "Up weight"):
            checkpoint.finish()
        self.assertEqual(self.calls, [])
        _load_up(model)
        checkpoint.finish()
        self.assertEqual(len(self.calls), 1)
        param = model.hc.input_mix_weight_up.weight
        before = param.detach().clone()
        with self.assertRaisesRegex(ValueError, "Online weight updates"):
            param.weight_loader(param, torch.zeros_like(param))
        torch.testing.assert_close(param, before, rtol=0, atol=0)

    def test_dtype_conversion_before_or_during_raw_loading(self):
        for before_staging in (True, False):
            with self.subTest(before_staging=before_staging):
                model = _Model()
                if before_staging:
                    model.to(dtype=torch.float16)
                _load_up(model)
                checkpoint = FlashInferHCCheckpoint(model)
                for name, tensor in _raw_checkpoint().items():
                    checkpoint.stage(name, tensor)
                if not before_staging:
                    model.to(dtype=torch.float16)
                checkpoint.finish()
                _, down, inject, norm, out_down, out_norm = self.calls[-1]
                self.assertEqual(
                    {t.dtype for t in (down, inject, norm, out_down, out_norm)},
                    {torch.float16},
                )
                self.assertEqual({t.device.type for t in (down, inject, norm)}, {"cpu"})

    def test_native_checkpoint_is_never_folded(self):
        model = _Model()
        _load_up(model)
        native = {
            name: torch.randn_like(param, dtype=torch.float32)
            for name, param in model.hc.named_parameters()
            if name.startswith("packed_")
        }
        checkpoint = FlashInferHCCheckpoint(model)
        for name, tensor in native.items():
            param = getattr(model.hc, name)
            param.weight_loader(param, tensor)
        checkpoint.finish()
        self.assertEqual(self.calls, [])
        for name, tensor in native.items():
            expected = tensor.to(torch.bfloat16)
            if name == "packed_down_weight":
                expected[:, LOWRANK + 4 :] = 0
            torch.testing.assert_close(
                getattr(model.hc, name), expected, rtol=0, atol=0
            )

        mixed = _Model()
        param = mixed.hc.packed_norm_weight
        param.weight_loader(param, native["packed_norm_weight"])
        name, tensor = next(iter(_raw_checkpoint().items()))
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            FlashInferHCCheckpoint(mixed).stage(name, tensor)

    def test_writes_after_the_initial_load_are_rejected_before_mutation(self):
        model = _Model()
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.5)
        state = {name: torch.ones_like(t) for name, t in model.state_dict().items()}
        before = {name: t.clone() for name, t in model.state_dict().items()}
        # Raw tensors pack only through load_weights, even in an initial load.
        with self.assertRaisesRegex(ValueError, "load_weights"):
            model.load_state_dict({**state, **_raw_checkpoint()}, strict=False)

        bind_flashinfer_hc(model)
        with self.assertRaisesRegex(ValueError, "Online weight updates"):
            FlashInferHCCheckpoint(model)
        with self.assertRaisesRegex(ValueError, "Online weight updates"):
            model.load_state_dict(state)
        # The root hook runs before the embedding, registered first, is written.
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertEqual(
            _unsupported_derived_weight_cache_error(model),
            FlashInferGatedResidual._derived_weight_cache_error,
        )

    def test_native_checkpoint_is_rejected_after_falling_back_to_legacy(self):
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module()])
        model.model.layers[0].attn_hyper_connection = GatedResidual(CONFIG)
        checkpoint = FlashInferHCCheckpoint(model)
        tensor = torch.zeros(1)
        for leaf in ("packed_down_weight", "packed_norm_weight"):
            with self.subTest(leaf=leaf):
                with self.assertRaisesRegex(ValueError, "native FlashInfer HC"):
                    checkpoint.stage(f"model.{HC}.{leaf}", tensor)
                with self.assertRaisesRegex(ValueError, "native FlashInfer HC"):
                    list(
                        checkpoint.route(
                            [(f"mtp.{HC}.{leaf}", tensor)], param_name=_mtp_param_name
                        )
                    )

    def test_bind_follows_parameters_replaced_after_construction(self):
        # The IPC weight cache swaps in new Parameters without Module._apply.
        model = _Model()
        replacement = torch.zeros_like(model.hc.packed_down_weight)
        model.hc.packed_down_weight = nn.Parameter(replacement, requires_grad=False)
        bind_flashinfer_hc(model)
        _, packed_down, _, _ = self.calls[-1]
        self.assertEqual(packed_down.data_ptr(), replacement.data_ptr())

    def test_offloader_keeps_the_bound_parameters_resident(self):
        model = _Model()
        model.hc.bind()
        self._assert_resident(model)
        # A conversion moves the weights off the bound plan's storage.
        model.to(torch.float16)
        self._assert_resident(model)
        model.hc.bind()
        _, packed_down, up, packed_norm = self.calls[-1]
        hc = model.hc
        self.assertEqual(
            [t.data_ptr() for t in (packed_down, up, packed_norm)],
            [
                p.data_ptr()
                for p in (
                    hc.packed_down_weight,
                    hc.input_mix_weight_up.weight,
                    hc.packed_norm_weight,
                )
            ],
        )

    def _assert_resident(self, model):
        resident = _get_resident_parameter_ids(model)
        self.assertLessEqual({id(param) for param in model.hc.parameters()}, resident)
        self.assertNotIn(id(model.embed.weight), resident)

    def test_padding_after_down_and_inject_rows_is_zeroed(self):
        model = _Model()
        with torch.no_grad():
            model.hc.packed_down_weight.fill_(1)
        zero_flashinfer_hc_padding(model)
        rows = model.hc.packed_down_weight.float().sum(dim=(0, 2))
        live = LOWRANK + 4
        self.assertTrue(bool((rows[:live] == 4 * HIDDEN).all()))
        self.assertTrue(bool((rows[live:] == 0).all()))

    def test_unbound_mix_is_rejected_during_capture(self):
        model = _Model()
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "must be bound"):
                model.hc.mix(torch.zeros(1, WIDTH))
        self.assertEqual(self.calls, [])


class _Context:
    """FlashInfer's HCApplyContext contract: it validates the residual it is
    applied to and slices its own rows."""

    def __init__(self, residual):
        self.residual = residual

    def with_residual(self, residual):
        if (residual.data_ptr(), residual.shape) != (
            self.residual.data_ptr(),
            self.residual.shape,
        ):
            raise ValueError("HC Apply residual differs from its Mix")
        return self

    def slice_rows(self, size, rank):
        return _Context(self.residual.tensor_split(size)[rank])


class TestFlashInferHCState(CustomTestCase):
    def test_attention_tp_slice_carries_the_context_rows(self):
        applied = []

        def mix(residual):
            return residual[:, :HIDDEN], _Context(residual)

        def combine(block_output, context):
            applied.append(context)
            return context.residual + 0

        state = FlashInferGatedResidualState(
            expand=lambda hidden_states: hidden_states,
            attn_mix=mix,
            ffn_mix=mix,
            attn_combine=combine,
            ffn_combine=combine,
        )
        ops = state.residual_ops()
        streams = torch.arange(4 * WIDTH, dtype=torch.float32).reshape(4, WIDTH)
        parallel = SimpleNamespace(attn_tp_size=2, attn_tp_rank=1)
        with patch.object(hc_fi, "get_parallel", return_value=parallel):
            _, residual = ops.attn_readout.read(streams, None)
            sliced = ops.attn_update.slice_residual_attn_tp(residual)
        torch.testing.assert_close(sliced, streams[2:], rtol=0, atol=0)
        block_output = torch.zeros(2, HIDDEN)
        with self.assertRaises(ValueError):
            ops.attn_update.update(block_output, residual)

        _, residual = ops.ffn_readout.update_and_read(
            ops.attn_update, block_output, sliced, None
        )
        self.assertEqual(applied[0].residual.data_ptr(), sliced.data_ptr())
        ops.ffn_update.update(block_output, residual)
        # The context holds the invocation's buffers; the write-back drops it.
        self.assertIsNone(state.context)


class TestFlashInferHCSelection(CustomTestCase):
    def test_real_compile_gate(self):
        with patch.object(
            hc_fi,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(enable_torch_compile=True)
            ),
        ):
            self.assertIsNotNone(hc_fi._torch_compile_reason())
        with patch.object(hc_fi, "get_exec", side_effect=ValueError("not published")):
            self.assertIsNone(hc_fi._torch_compile_reason())

    def _create(
        self,
        *,
        support_reason=None,
        api=True,
        sm=(10, 0),
        compiles=None,
        deterministic=False,
        **kwargs,
    ):
        modules = {"flashinfer": None, "flashinfer.hyperconnection": None}
        if api:
            fake = _fake_flashinfer([], support_reason)
            package = types.ModuleType("flashinfer")
            package.hyperconnection = fake
            modules = {"flashinfer": package, "flashinfer.hyperconnection": fake}
        with (
            patch.dict(sys.modules, modules),
            patch("torch.cuda.is_available", return_value=sm is not None),
            patch("torch.cuda.get_device_capability", return_value=sm),
            patch("torch.cuda.current_device", return_value=0),
            patch.object(hc_fi, "_torch_compile_reason", return_value=compiles),
            patch.object(hc_fi, "_deterministic_inference", return_value=deterministic),
        ):
            return create_gated_residual(CONFIG, **kwargs)

    def test_opt_in_selects_flashinfer_only_where_supported(self):
        opt_in = envs.SGLANG_ENABLE_FLASHINFER_HC_SUM
        with self.subTest("not opted in"), opt_in.override(False):
            self.assertIs(type(self._create(api=False)), GatedResidual)
        with opt_in.override(True):
            # Only FlashInfer's own refusal needs its API; the others never import it.
            for name, kwargs in (
                ("mix only", dict(api=False, use_combine=False)),
                ("not SM100", dict(api=False, sm=(9, 0))),
                ("no GPU", dict(api=False, sm=None)),
                ("torch.compile", dict(api=False, sm=(10, 3), compiles="on")),
                ("batch invariance", dict(api=False, deterministic=True)),
                ("FlashInfer declines", dict(support_reason="H % 512 != 0")),
            ):
                with self.subTest(name):
                    self.assertIs(type(self._create(**kwargs)), GatedResidual)
            with self.subTest("supported"):
                self.assertIs(type(self._create()), FlashInferGatedResidual)

    def test_eligible_opt_in_without_the_flashinfer_api_fails(self):
        with envs.SGLANG_ENABLE_FLASHINFER_HC_SUM.override(True):
            with self.assertRaisesRegex(ImportError, "FLASHINFER_HC_SUM"):
                self._create(api=False)
        api = _fake_flashinfer([])
        del api.HCSumRuntime
        _install_flashinfer(self, api)
        with self.assertRaisesRegex(ImportError, "Missing:.*HCSumRuntime"):
            hc_fi._flashinfer_hc()


class _Layer(qwen4_exp.Qwen4ExpLayerExtensionMixin):
    def __init__(self, flashinfer_hc):
        self._flashinfer_hc = flashinfer_hc
        self._hc_norms = None


class TestQwen4ExpPreparedSums(CustomTestCase):
    def setUp(self):
        super().setUp()
        _install_flashinfer(self, _fake_flashinfer([]))
        self.flashinfer_reader = FlashInferGatedResidual(CONFIG)
        self.legacy_mixer = GatedResidual(CONFIG, use_combine=False)

    def _held(
        self, layer, next_reader=None, *, dp=1, cp=1, ep=1, a2a=False, scattered=False
    ):
        parallel = SimpleNamespace(attn_dp_size=dp, attn_cp_size=cp, moe_ep_size=ep)
        with (
            patch.object(qwen4_exp, "get_parallel", return_value=parallel),
            patch.object(
                qwen4_exp,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=scattered),
            ),
            patch.object(
                qwen4_exp,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: not a2a),
            ),
            layer.hc_norms(None, next_reader) as norms,
        ):
            return layer._hc_norms is norms, norms

    def test_prepared_sums_follow_rows_only_without_row_movement(self):
        flashinfer, legacy = _Layer(True), _Layer(False)
        # EP without all-to-all keeps every row on every rank.
        self.assertTrue(self._held(flashinfer, ep=2)[0])
        self.assertFalse(self._held(legacy, ep=2)[0])
        for movement in (dict(dp=2), dict(cp=2), dict(a2a=True), dict(scattered=True)):
            for layer in (flashinfer, legacy):
                with self.subTest(movement=movement, flashinfer=layer._flashinfer_hc):
                    self.assertFalse(self._held(layer, **movement)[0])

    def test_sums_reach_only_a_flashinfer_reader(self):
        flashinfer, legacy = _Layer(True), _Layer(False)
        _, norms = self._held(flashinfer, self.flashinfer_reader)
        self.assertTrue(norms.next_reads_output)
        # The final mix is legacy: it must not receive an HCSumState.
        _, norms = self._held(flashinfer, self.legacy_mixer)
        self.assertFalse(norms.next_reads_output)
        self.assertIsNone(self._held(legacy, self.flashinfer_reader)[1].next_norm)
        self.assertIs(
            self._held(legacy, self.legacy_mixer)[1].next_norm,
            self.legacy_mixer.hc_norm,
        )

    def test_ple_invalidates_carried_sums_before_mix(self):
        layer = _Layer(True)
        layer.ple = object()
        residual = torch.zeros(2, WIDTH)
        updated = residual + 1
        layer._add_ple_embedding = lambda value: updated
        layer._hc_norms = qwen4_exp._HCSums(attn_input=object())
        seen = []
        layer.attn_hyper_connection = SimpleNamespace(
            mix=lambda value, prepared: seen.append((value, prepared))
        )
        layer._attn_mix_sums(residual)
        self.assertIs(seen[0][0], updated)
        self.assertIsNone(seen[0][1])


class TestFlashInferMTPNames(CustomTestCase):
    def test_raw_hc_names_follow_parent_mtp_mapping(self):
        for leaf in (
            "input_mix_weight_down.weight",
            "block_inject_weight.weight",
            "hc_norm.weight",
        ):
            expected = f"model.layers.0.attn_hyper_connection.{leaf}"
            self.assertEqual(
                _mtp_param_name(f"mtp.layers.0.attn_hyper_connection.{leaf}"), expected
            )
            self.assertEqual(
                _mtp_param_name(f"mtp.layers.0.self_attn.attn_hyper_connection.{leaf}"),
                expected,
            )
            self.assertIsNone(_mtp_param_name(expected))


if __name__ == "__main__":
    unittest.main()
