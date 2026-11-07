"""CPU contracts for the optional fused FlashInfer Kimi decode consumer.

Load the actual small adapter and consumer methods without server/CUDA imports.
The FI callable is a stand-in: these tests exercise preparation, views, dispatch
and state-update ownership, not native GPU numerics or FlashInfer selection.
"""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[7]
LINEAR = ROOT / "python/sglang/srt/layers/attention/linear"
ADAPTER_NAME = "sglang.srt.layers.attention.linear.kernels.kda_flashinfer_fused"


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


adapter = _load_file(ADAPTER_NAME, LINEAR / "kernels/kda_flashinfer_fused.py")
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _source_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _complete_api(
    x,
    weight,
    conv_state,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state_indices,
    state,
    output_gate,
    norm_weight,
    lower_bound,
    norm_eps,
    *,
    backend,
    state_indices_mode,
):
    raise AssertionError("capability probing must not launch a kernel")


def _make_inputs():
    batch, heads, width, slots = 3, 12, 128, 5
    seg = heads * width
    packed = torch.zeros(batch, 4 * seg + heads, dtype=torch.bfloat16)
    # Production projection outputs use padded row strides for qkv/gate/beta.
    x, gate, beta = (
        packed[:, : 3 * seg],
        packed[:, 3 * seg : 4 * seg],
        packed[:, -heads:],
    )
    a = torch.zeros(batch, seg, dtype=torch.bfloat16)
    conv = torch.zeros(slots, 2, 3, 3 * seg, dtype=torch.bfloat16)[:, 1]
    state = torch.zeros(slots, 2, heads, width, width, dtype=torch.float32)[:, 1]
    indices = torch.tensor([1, 3, -1], dtype=torch.int32)
    layer = SimpleNamespace(
        layer_id=0,
        head_k_dim=width,
        conv_weights=torch.arange(3 * seg * 4, dtype=torch.float32).view(3 * seg, 4),
        A_log=torch.zeros(heads),
        dt_bias=torch.zeros(seg),
        bias=None,
        lower_bound=-5.0,
        _k3_onorm_gate=gate,
        _k3_onorm_consumed=False,
    )
    model = SimpleNamespace(
        attn=layer,
        local_num_heads=heads,
        o_norm=SimpleNamespace(weight=torch.ones(width), eps=1e-5),
    )
    cache = SimpleNamespace(
        conv=[conv],
        temporal=state,
        replayssm_d=None,
        replayssm_k=None,
        replayssm_g=None,
    )
    backend = SimpleNamespace(
        state_layer_cache=lambda _: cache,
        forward_metadata=SimpleNamespace(
            query_start_loc=None, mamba_cache_indices=indices
        ),
        _track_mamba_state_decode=Mock(),
        _fused_override_notice="flashinfer",
    )
    return model, backend, (x, a, beta, conv, state, indices, gate)


def _prepare(model, kernel, *, backend="flashinfer", speculative=False):
    namespace = {
        "torch": torch,
        "_is_hip": False,
        "_is_npu": False,
        "get_exec": lambda: SimpleNamespace(
            mamba=SimpleNamespace(
                linear_attn_backend="triton",
                linear_attn_decode_backend=backend,
                mamba_ssm_dtype=None,
            )
        ),
        "get_spec": lambda: SimpleNamespace(
            speculative_algorithm="DSPARK" if speculative else None
        ),
        "SpeculativeAlgorithm": SimpleNamespace(
            from_string=lambda value: SimpleNamespace(is_none=lambda: value is None)
        ),
        "get_platform": lambda: SimpleNamespace(is_cuda=True),
        "rank0_log": Mock(),
    }
    prepare = _source_function(
        ROOT / "python/sglang/srt/models/kimi_k3.py", "_prepare_fused_decode", namespace
    )
    native_namespace = SimpleNamespace(SUPPORTED_HEADS={3, 6, 12})
    with (
        patch.dict(
            sys.modules,
            {
                ADAPTER_NAME: adapter,
                "sglang.kernels.ops.attention": SimpleNamespace(
                    kda_fused_decode=native_namespace
                ),
            },
        ),
        patch.object(adapter, "_get_fused_kda_decode", return_value=kernel) as probe,
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
    ):
        prepare(model)
    return probe


def _decode(model, backend, inputs):
    native_covered = _source_function(
        ROOT / "python/sglang/kernels/ops/attention/kda_fused_decode.py",
        "covered",
        {"torch": torch, "SUPPORTED_HEADS": {3, 6, 12}, "_CONV_STATE_W": 3},
    )
    native = Mock(return_value=torch.empty(1, 3, 12, 128))
    namespace = {
        "torch": torch,
        "kda_fused_decode": SimpleNamespace(
            covered=native_covered, kda_fused_decode=native
        ),
        "rank0_log": Mock(),
    }
    decode = _source_function(LINEAR / "kda_backend.py", "forward_decode", namespace)
    x, a, beta, *_ = inputs
    return (
        decode,
        native,
        namespace["rank0_log"],
        (backend, model.attn, object(), x, a, beta.unsqueeze(0)),
    )


class TestFlashInferFusedKDAConsumer(CustomTestCase):
    def test_default_spec_and_old_api_do_not_prepare_extra_weights(self):
        for backend, speculative, kernel in (
            ("triton", False, Mock()),
            ("flashinfer", True, Mock()),
            ("flashinfer", False, None),
        ):
            with self.subTest(backend=backend, speculative=speculative, kernel=kernel):
                model, _, _ = _make_inputs()
                probe = _prepare(
                    model, kernel, backend=backend, speculative=speculative
                )
                self.assertIsNone(model.attn._k3_flashinfer_fused_decode)
                self.assertEqual(
                    probe.call_count, int(backend == "flashinfer" and not speculative)
                )

    def test_prepared_consumer_preserves_views_and_tracks_once(self):
        model, backend, inputs = _make_inputs()
        kernel = Mock(return_value=torch.zeros(1, 3, 12, 128))
        _prepare(model, kernel)
        prepared = model.attn._k3_flashinfer_fused_decode
        self.assertEqual(prepared.heads, 12)
        self.assertEqual(prepared.head_dim, 128)
        self.assertEqual(prepared.seg, 1536)
        expected = model.attn.conv_weights.t().reshape(4, 3, -1).permute(1, 0, 2)
        torch.testing.assert_close(prepared.weight, expected)
        self.assertTrue(prepared.weight.is_contiguous())
        decode, native, log, args = _decode(model, backend, inputs)
        result = decode(*args)
        self.assertIs(result, kernel.return_value)
        native.assert_not_called()
        kernel.assert_called_once()
        call = kernel.call_args.kwargs
        self.assertEqual(call["backend"], "auto")
        self.assertEqual(call["state_indices_mode"], "unique_or_null")
        self.assertEqual(
            call["conv_state"].stride(), inputs[3].transpose(-1, -2).stride()
        )
        self.assertEqual(call["conv_state"].data_ptr(), inputs[3].data_ptr())
        self.assertIs(call["state"], inputs[4])
        self.assertIs(call["state_indices"], inputs[5])
        self.assertEqual(call["raw_beta"].stride(1), inputs[2].stride(0))
        self.assertEqual(call["output_gate"].stride(0), inputs[6].stride(0))
        self.assertTrue(model.attn._k3_onorm_consumed)
        backend._track_mamba_state_decode.assert_called_once()
        self.assertIn("FlashInfer (backend=auto)", log.call_args.args[0])

    def test_rejected_layout_or_missing_api_keeps_native_consumer(self):
        for missing_api in (False, True):
            model, backend, inputs = _make_inputs()
            kernel = Mock()
            _prepare(model, None if missing_api else kernel)
            if not missing_api:
                x, _, *tail = inputs
                # Unit last stride still satisfies native covered(), but FI
                # requires the entire raw_gate tensor to be contiguous.
                a = torch.zeros(3, 1537, dtype=torch.bfloat16)[:, :1536]
                inputs = (x, a, *tail)
            decode, native, log, args = _decode(model, backend, inputs)
            self.assertIs(decode(*args), native.return_value)
            native.assert_called_once()
            kernel.assert_not_called()
            self.assertTrue(model.attn._k3_onorm_consumed)
            backend._track_mamba_state_decode.assert_called_once()
            message = log.call_args.args[0]
            self.assertIn("native CUDA JIT", message)
            self.assertIn("layout:", message)
            if missing_api:
                self.assertIn("not prepared", message)
            else:
                self.assertIn("raw_gate", message)

    def test_unsupported_layout_reason_names_the_rejected_input(self):
        model, _, inputs = _make_inputs()
        _prepare(model, Mock())
        prepared = model.attn._k3_flashinfer_fused_decode
        self.assertIsNone(prepared.unsupported_layout_reason(*inputs))
        x, a, b, conv, state, indices, gate = inputs
        self.assertIn(
            "raw_gate",
            prepared.unsupported_layout_reason(
                x, a.float(), b, conv, state, indices, gate
            ),
        )
        self.assertIn(
            "cache indices",
            prepared.unsupported_layout_reason(
                x, a, b, conv, state, indices.long(), gate
            ),
        )
        self.assertIn(
            "2-D",
            prepared.unsupported_layout_reason(x[0], a, b, conv, state, indices, gate),
        )

    def test_launch_failure_is_not_retried_after_state_mutation(self):
        model, backend, inputs = _make_inputs()

        def partial_update_then_fail(**kwargs):
            kwargs["state"][1, 0, 0, 0] = 7
            raise RuntimeError("launch failure")

        _prepare(model, partial_update_then_fail)
        decode, native, _, args = _decode(model, backend, inputs)
        with self.assertRaisesRegex(RuntimeError, "launch failure"):
            decode(*args)
        native.assert_not_called()
        self.assertFalse(model.attn._k3_onorm_consumed)
        backend._track_mamba_state_decode.assert_not_called()
        self.assertEqual(inputs[4][1, 0, 0, 0].item(), 7)

    def test_incompatible_api_and_conv_bias_fail_before_launch(self):
        old_api = SimpleNamespace(fused_kda_decode=lambda x: x)
        adapter._get_fused_kda_decode.cache_clear()
        with patch.dict(sys.modules, {"flashinfer.kda_decode": old_api}):
            self.assertIsNone(adapter._get_fused_kda_decode())
        adapter._get_fused_kda_decode.cache_clear()
        model, _, _ = _make_inputs()
        model.attn.bias = torch.zeros(3 * 1536)
        kernel = Mock()
        _prepare(model, kernel)
        self.assertIsNone(model.attn._k3_flashinfer_fused_decode)
        kernel.assert_not_called()

    def test_partial_flashinfer_build_is_rejected_during_prepare(self):
        for available in (True, False):
            adapter._get_fused_kda_decode.cache_clear()
            module = SimpleNamespace(
                fused_kda_decode=_complete_api,
                _FUSED_KDA_DECODE_AVAILABLE=available,
            )
            with patch.dict(sys.modules, {"flashinfer.kda_decode": module}):
                self.assertIs(
                    adapter._get_fused_kda_decode(),
                    _complete_api if available else None,
                )
        adapter._get_fused_kda_decode.cache_clear()

    def test_transient_probe_failure_does_not_disable_later_preparation(self):
        """One import failure must not pin every later layer to native decode."""
        model, _, _ = _make_inputs()
        _prepare(model, None, backend="triton")
        layer = model.attn
        adapter._get_fused_kda_decode.cache_clear()
        self.addCleanup(adapter._get_fused_kda_decode.cache_clear)
        healthy = SimpleNamespace(fused_kda_decode=_complete_api)
        with patch.object(
            adapter,
            "import_module",
            side_effect=[RuntimeError("transient FlashInfer probe failure"), healthy],
        ):
            prepare_args = (layer._k3_fused_decode_args, layer.dt_bias)
            prepare_kwargs = dict(has_conv_bias=False, lower_bound=layer.lower_bound)
            self.assertIsNone(
                adapter.prepare_fused_kda_decode(*prepare_args, **prepare_kwargs)
            )
            prepared = adapter.prepare_fused_kda_decode(*prepare_args, **prepare_kwargs)
        self.assertIsNotNone(prepared)
        self.assertIs(prepared.kernel, _complete_api)

    def test_misaligned_views_are_rejected_without_launch(self):
        for position in (0, 1, 4, 5):
            model, _, inputs = _make_inputs()
            kernel = Mock()
            _prepare(model, kernel)
            tensor = inputs[position]
            misaligned = torch.zeros(tensor.numel() + 1, dtype=tensor.dtype)[1:].view(
                tensor.shape
            )
            self.assertNotEqual(misaligned.data_ptr() % 16, 0)
            args = list(inputs)
            args[position] = misaligned
            self.assertFalse(model.attn._k3_flashinfer_fused_decode.covered(*args))
            kernel.assert_not_called()

    def test_existing_slot_validator_rejects_duplicate_live_slots(self):
        validate = _source_function(
            ROOT
            / "python/sglang/srt/layers/attention/mamba/replay_state_indices_validator.py",
            "validate_replay_state_indices_cpu",
            {"torch": torch},
        )
        options = dict(valid_bs=2, total_bs=3, num_state_slots=5)
        validate(torch.tensor([1, 3, -1]), **options)
        validate(torch.tensor([0, 3, -1]), **options)  # capture/idle null slot
        with self.assertRaisesRegex(AssertionError, "unique slots"):
            validate(torch.tensor([1, 1, -1]), **options)


if __name__ == "__main__":
    unittest.main()
