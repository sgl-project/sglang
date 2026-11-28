"""Unit tests for activation steering vector registry, hooks, and request wiring."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from safetensors.torch import save_file

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.steering import (
    SteeringScope,
    SteeringVectorManager,
    SteeringVectorSpec,
    decode_steering_vector_spec,
    resolve_decoder_layers,
)
from sglang.test.test_utils import CustomTestCase


class _IdentityLinearLayer(nn.Module):
    """Deterministic toy decoder layer: h -> h + bias_val."""

    def __init__(self, hidden_size: int, bias_val: float = 0.0, fail: bool = False):
        super().__init__()
        self.bias_val = bias_val
        self.fail = fail
        self.weight = nn.Parameter(torch.eye(hidden_size), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, forward_batch=None) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("simulated layer failure")
        return hidden_states @ self.weight + self.bias_val


class _ToyCausalLM(nn.Module):
    """Toy CausalLM matching SGLang's model.layers convention."""

    def __init__(self, num_layers: int, hidden_size: int, fail_layer: int = -1):
        super().__init__()
        self.shared_experts_config = None
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                _IdentityLinearLayer(
                    hidden_size, bias_val=float(i + 1), fail=(i == fail_layer)
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, forward_batch=None) -> torch.Tensor:
        x = hidden_states
        for layer in self.model.layers:
            x = layer(x, forward_batch=forward_batch)
        return x


class _TupleDecoderLayer(nn.Module):
    """Toy layer matching classic SGLang (hidden_states, residual) tuple return."""

    def __init__(self, add_val: float):
        super().__init__()
        self.add_val = add_val

    def forward(self, hidden_states: torch.Tensor, residual=None, forward_batch=None):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        return torch.full_like(hidden_states, self.add_val), residual


class _TupleToyModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_TupleDecoderLayer(add_val=float(i + 1)) for i in range(num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, forward_batch=None) -> torch.Tensor:
        residual = None
        x = hidden_states
        for layer in self.layers:
            x, residual = layer(x, residual=residual, forward_batch=forward_batch)
        return x + residual


class _ResidualStreamLayer(nn.Module):
    """Toy layer using SGLang's ResidualStream protocol and strict stream.check()."""

    def __init__(self, add_val: float, owed: bool = False):
        super().__init__()
        self.add_val = add_val
        self.owed = owed

    def forward(self, hidden_states, forward_batch):
        stream = residual_batch.stream_of(forward_batch)
        stream.check(hidden_states)
        if stream.pending is not None:
            val = stream.pending.complete()
            written = val if stream.residual is None else stream.residual + val
            stream.write(written)
        else:
            stream.write(hidden_states)
        contribution = torch.full_like(stream.residual, self.add_val)
        if self.owed:
            # Simulates an unreduced TP output whose completion sums 2x partial.
            unreduced = UnreducedOutput(
                partial=contribution,
                complete_sum=lambda p: p * 2.0,
            )
            return stream.record(unreduced, PLAIN_ADD)
        return stream.record(contribution, PLAIN_ADD)


class _ResidualStreamToyModel(nn.Module):
    def __init__(self, num_layers: int, owed_layer: int = -1):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                _ResidualStreamLayer(add_val=float(i + 1), owed=(i == owed_layer))
                for i in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, forward_batch) -> torch.Tensor:
        residual_batch.start(forward_batch)
        x = hidden_states
        for layer in self.model.layers:
            x = layer(x, forward_batch)
        stream = residual_batch.stream_of(forward_batch)
        stream.check(x)
        val = stream.pending.complete()
        out = stream.residual + val
        residual_batch.clear(forward_batch)
        return out


class TestSteeringVectorSpec(CustomTestCase):
    def test_decode_valid_forms(self):
        from_str = decode_steering_vector_spec("honesty")
        self.assertEqual(
            from_str,
            SteeringVectorSpec(name="honesty", scale=1.0, scope="all", layers=None),
        )

        from_dict = decode_steering_vector_spec(
            {
                "name": " refusal ",
                "scale": -1.5,
                "scope": "decode",
                "layers": [14, 10, 14],
            }
        )
        self.assertEqual(from_dict.name, "refusal")
        self.assertEqual(from_dict.scale, -1.5)
        self.assertEqual(from_dict.scope, SteeringScope.DECODE.value)
        self.assertEqual(from_dict.layers, (10, 14))

        from_struct = decode_steering_vector_spec(from_dict)
        self.assertEqual(from_struct, from_dict)

    def test_decode_rejects_invalid_inputs(self):
        invalid_cases = [
            "",
            "   ",
            {},
            {"name": ""},
            {"name": "ok", "unknown": 1},
            {"name": "ok", "scale": float("nan")},
            {"name": "ok", "scale": float("inf")},
            {"name": "ok", "scale": True},
            {"name": "ok", "scope": "invalid"},
            {"name": "ok", "layers": []},
            {"name": "ok", "layers": [-1]},
            {"name": "ok", "layers": [True]},
            {"name": "ok", "layers": "12"},
            123,
        ]
        for bad in invalid_cases:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    decode_steering_vector_spec(bad)

    def test_phase_and_radix_cache_keys(self):
        all_spec = SteeringVectorSpec(name="v", scale=1.5, scope="all", layers=(2, 4))
        prefill_spec = SteeringVectorSpec(name="v", scale=1.5, scope="prefill")
        decode_spec = SteeringVectorSpec(name="v", scale=1.5, scope="decode")
        zero_spec = SteeringVectorSpec(name="v", scale=0.0, scope="all")

        self.assertTrue(all_spec.applies_to_phase(is_decode=False))
        self.assertTrue(all_spec.applies_to_phase(is_decode=True))
        self.assertTrue(prefill_spec.applies_to_phase(is_decode=False))
        self.assertFalse(prefill_spec.applies_to_phase(is_decode=True))
        self.assertFalse(decode_spec.applies_to_phase(is_decode=False))
        self.assertTrue(decode_spec.applies_to_phase(is_decode=True))
        self.assertFalse(zero_spec.applies_to_phase(is_decode=False))
        self.assertFalse(zero_spec.applies_to_phase(is_decode=True))

        # Prefill-affecting specs must isolate radix cache; decode-only & zero do not.
        self.assertIsNotNone(all_spec.radix_cache_key())
        self.assertIsNotNone(prefill_spec.radix_cache_key())
        self.assertNotEqual(all_spec.radix_cache_key(), prefill_spec.radix_cache_key())
        self.assertIsNone(decode_spec.radix_cache_key())
        self.assertIsNone(zero_spec.radix_cache_key())


class TestSteeringVectorManager(CustomTestCase):
    def test_register_validate_and_unregister(self):
        mgr = SteeringVectorManager(hidden_size=4, num_hidden_layers=4)
        v1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        v2 = torch.tensor([-0.5, 0.5, 0.0, 1.0])

        entry = mgr.register(
            "persona", {1: v1, "model.layers.3": v2}, default_scale=0.5
        )
        self.assertEqual(entry.layer_ids, (1, 3))
        self.assertEqual(mgr.list_vectors(), ["persona"])
        self.assertIn("persona", mgr)
        self.assertEqual(len(mgr), 1)

        # Mutating original tensor must not affect registered copy.
        v1.zero_()
        self.assertTrue(
            torch.equal(
                mgr.get("persona").vectors[1], torch.tensor([1.0, 2.0, 3.0, 4.0])
            )
        )

        # Duplicate registration rejected unless overwrite=True.
        with self.assertRaises(ValueError):
            mgr.register("persona", {0: torch.ones(4)})
        mgr.register("persona", {0: torch.ones(4)}, overwrite=True)
        self.assertEqual(mgr.get("persona").layer_ids, (0,))

        mgr.unregister("persona")
        self.assertNotIn("persona", mgr)
        with self.assertRaises(KeyError):
            mgr.unregister("persona")

    def test_register_validation_errors(self):
        mgr = SteeringVectorManager(hidden_size=4, num_hidden_layers=3)

        with self.assertRaises(ValueError):
            mgr.register("", {0: torch.ones(4)})
        with self.assertRaises(ValueError):
            mgr.register("empty", {})
        with self.assertRaises(ValueError):
            mgr.register("bad_layer_neg", {-1: torch.ones(4)})
        with self.assertRaises(ValueError):
            mgr.register("bad_layer_oob", {3: torch.ones(4)})
        with self.assertRaises(ValueError):
            mgr.register("bad_shape", {0: torch.ones(5)})
        with self.assertRaises(ValueError):
            mgr.register("bad_rank", {0: torch.ones(1, 4)})
        with self.assertRaises(ValueError):
            mgr.register("bad_dtype", {0: torch.ones(4, dtype=torch.int64)})
        with self.assertRaises(ValueError):
            mgr.register("nan_vec", {0: torch.tensor([1.0, float("nan"), 0.0, 0.0])})
        with self.assertRaises(ValueError):
            mgr.register("inf_vec", {0: torch.tensor([1.0, float("inf"), 0.0, 0.0])})
        with self.assertRaises(ValueError):
            mgr.register("bad_scale", {0: torch.ones(4)}, default_scale=float("inf"))
        with self.assertRaises(TypeError):
            mgr.register("not_tensor", {0: [1.0, 2.0, 3.0, 4.0]})

    def test_load_from_safetensors_and_pt_files(self):
        mgr = SteeringVectorManager(
            hidden_size=4, num_hidden_layers=3, dtype=torch.float32
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            st_path = Path(tmpdir) / "vec.safetensors"
            save_file(
                {
                    "layer.0": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float16),
                    "model.layers.2": torch.tensor(
                        [0.0, 2.0, 0.0, 0.0], dtype=torch.float16
                    ),
                },
                str(st_path),
            )
            loaded_st = mgr.load_from_file("from_st", st_path)
            self.assertEqual(loaded_st.layer_ids, (0, 2))
            self.assertEqual(loaded_st.vectors[0].dtype, torch.float32)

            pt_2d_path = Path(tmpdir) / "matrix.pt"
            mat = torch.zeros(3, 4)
            mat[1] = torch.tensor([0.0, 0.0, 3.0, 0.0])
            torch.save(mat, pt_2d_path)
            loaded_pt = mgr.load_from_file("from_pt", pt_2d_path)
            self.assertEqual(loaded_pt.layer_ids, (1,))
            self.assertTrue(
                torch.equal(loaded_pt.vectors[1], torch.tensor([0.0, 0.0, 3.0, 0.0]))
            )

            bad_ext = Path(tmpdir) / "vec.json"
            bad_ext.write_text("{}")
            with self.assertRaises(ValueError):
                mgr.load_from_file("bad", bad_ext)
            with self.assertRaises(FileNotFoundError):
                mgr.load_from_file("missing", Path(tmpdir) / "nope.safetensors")

    def test_exact_numerical_delta_single_and_multi_layer(self):
        hidden_size = 4
        model = _ToyCausalLM(num_layers=3, hidden_size=hidden_size)
        mgr = SteeringVectorManager(hidden_size=hidden_size, num_hidden_layers=3)

        v0 = torch.tensor([0.5, -1.0, 2.0, 0.25])
        v2 = torch.tensor([1.0, 1.0, -0.5, 0.0])
        mgr.register("steer_a", {0: v0, 2: v2})

        x = torch.randn(3, hidden_size)
        baseline = model(x)

        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            steering_vectors=[
                SteeringVectorSpec(name="steer_a", scale=2.0),
                SteeringVectorSpec(name="steer_a", scale=2.0),
                SteeringVectorSpec(name="steer_a", scale=2.0),
            ],
            steering_is_decode=[True, True, True],
            input_ids=torch.zeros(3, dtype=torch.long),
        )
        with mgr.forward_context(model, fb) as plan:
            self.assertIsNotNone(plan)
            # Only layers 0 and 2 should have hooks; layer 1 must have zero hooks.
            self.assertEqual(len(model.model.layers[0]._forward_hooks), 1)
            self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)
            self.assertEqual(len(model.model.layers[2]._forward_hooks), 1)
            steered = model(x, forward_batch=fb)

        # Hooks are always cleaned up on exit.
        for layer in model.model.layers:
            self.assertEqual(len(layer._forward_hooks), 0)

        expected = baseline + 2.0 * (v0 + v2)
        self.assertTrue(torch.allclose(steered, expected, atol=1e-6))

        # Layer subset filtering via spec.layers applies only selected layers.
        fb_subset = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            steering_vectors=[
                SteeringVectorSpec(name="steer_a", scale=-0.5, layers=(2,)),
            ]
            * 3,
            steering_is_decode=[True, True, True],
            input_ids=torch.zeros(3, dtype=torch.long),
        )
        with mgr.forward_context(model, fb_subset):
            steered_subset = model(x, forward_batch=fb_subset)
        self.assertTrue(torch.allclose(steered_subset, baseline - 0.5 * v2, atol=1e-6))

    def test_selective_mixed_batch_and_padding(self):
        hidden_size = 4
        model = _ToyCausalLM(num_layers=2, hidden_size=hidden_size)
        mgr = SteeringVectorManager(hidden_size=hidden_size, num_hidden_layers=2)
        va = torch.tensor([1.0, 0.0, -1.0, 2.0])
        vb = torch.tensor([0.0, 3.0, 1.0, -0.5])
        mgr.register("a", {1: va})
        mgr.register("b", {1: vb})

        # 3 requests with extend lengths [2, 3, 2] = 7 real tokens + 2 padded tokens = 9.
        x = torch.randn(9, hidden_size)
        baseline = model(x)

        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[2, 3, 2],
            input_ids=torch.zeros(9, dtype=torch.long),
            steering_vectors=[
                SteeringVectorSpec(name="a", scale=1.5),
                None,
                SteeringVectorSpec(name="b", scale=-2.0),
            ],
            steering_is_decode=[False, False, False],
        )
        with mgr.forward_context(model, fb):
            out = model(x, forward_batch=fb)

        # Req 0 (tokens 0..2): shifted by +1.5 * va
        self.assertTrue(torch.allclose(out[0:2], baseline[0:2] + 1.5 * va, atol=1e-6))
        # Req 1 (tokens 2..5): unsteered -> exact match
        self.assertTrue(torch.equal(out[2:5], baseline[2:5]))
        # Req 2 (tokens 5..7): shifted by -2.0 * vb
        self.assertTrue(torch.allclose(out[5:7], baseline[5:7] - 2.0 * vb, atol=1e-6))
        # Trailing padding (tokens 7..9): untouched
        self.assertTrue(torch.equal(out[7:9], baseline[7:9]))

    def test_prefill_vs_decode_scoping_and_mixed_phase_batch(self):
        hidden_size = 4
        model = _ToyCausalLM(num_layers=2, hidden_size=hidden_size)
        mgr = SteeringVectorManager(hidden_size=hidden_size, num_hidden_layers=2)
        v = torch.tensor([2.0, -1.0, 0.5, 4.0])
        mgr.register("v", {0: v})

        x = torch.randn(4, hidden_size)
        baseline = model(x)

        # Decode-only spec during pure prefill -> no-op, zero hooks registered.
        fb_prefill = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[4],
            input_ids=torch.zeros(4, dtype=torch.long),
            steering_vectors=[SteeringVectorSpec(name="v", scale=1.0, scope="decode")],
            steering_is_decode=[False],
        )
        with mgr.forward_context(model, fb_prefill) as plan:
            self.assertIsNone(plan)
            self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
            out_prefill = model(x, forward_batch=fb_prefill)
        self.assertTrue(torch.equal(out_prefill, baseline))

        # Prefill-only spec during pure decode -> no-op, zero hooks registered.
        fb_decode = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            input_ids=torch.zeros(4, dtype=torch.long),
            steering_vectors=[SteeringVectorSpec(name="v", scale=1.0, scope="prefill")]
            * 4,
            steering_is_decode=[True, True, True, True],
        )
        with mgr.forward_context(model, fb_decode) as plan:
            self.assertIsNone(plan)
            out_decode = model(x, forward_batch=fb_decode)
        self.assertTrue(torch.equal(out_decode, baseline))

        # Mixed batch: req 0 is prefill (3 tokens, scope="prefill"),
        # req 1 is decode (1 token, scope="decode"). Both apply to their own tokens!
        fb_mixed = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            extend_seq_lens_cpu=[3, 1],
            input_ids=torch.zeros(4, dtype=torch.long),
            steering_vectors=[
                SteeringVectorSpec(name="v", scale=1.0, scope="prefill"),
                SteeringVectorSpec(name="v", scale=-3.0, scope="decode"),
            ],
            steering_is_decode=[False, True],
        )
        with mgr.forward_context(model, fb_mixed):
            out_mixed = model(x, forward_batch=fb_mixed)
        self.assertTrue(torch.allclose(out_mixed[0:3], baseline[0:3] + v, atol=1e-6))
        self.assertTrue(
            torch.allclose(out_mixed[3:4], baseline[3:4] - 3.0 * v, atol=1e-6)
        )

    def test_zero_overhead_when_disabled_and_cleanup_on_exception(self):
        hidden_size = 4
        model = _ToyCausalLM(num_layers=2, hidden_size=hidden_size, fail_layer=1)
        mgr = SteeringVectorManager(hidden_size=hidden_size, num_hidden_layers=2)
        mgr.register("v", {0: torch.ones(hidden_size)})

        x = torch.randn(2, hidden_size)

        # Disabled batch (steering_vectors=None) installs zero hooks.
        fb_none = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            steering_vectors=None,
            input_ids=torch.zeros(2, dtype=torch.long),
        )
        with mgr.forward_context(model, fb_none) as plan:
            self.assertIsNone(plan)
            self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

        # Exception mid-forward must still remove all installed hooks.
        fb_fail = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            steering_vectors=[SteeringVectorSpec(name="v", scale=1.0)] * 2,
            steering_is_decode=[True, True],
            input_ids=torch.zeros(2, dtype=torch.long),
        )
        with self.assertRaisesRegex(RuntimeError, "simulated layer failure"):
            with mgr.forward_context(model, fb_fail):
                self.assertEqual(len(model.model.layers[0]._forward_hooks), 1)
                model(x, forward_batch=fb_fail)

        for layer in model.model.layers:
            self.assertEqual(len(layer._forward_hooks), 0)

    def test_tuple_output_and_residual_stream_models(self):
        hidden_size = 4
        v = torch.tensor([1.0, -2.0, 0.5, 3.0])
        mgr = SteeringVectorManager(hidden_size=hidden_size, num_hidden_layers=3)
        mgr.register("v", {1: v})

        x = torch.randn(2, hidden_size)
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            steering_vectors=[SteeringVectorSpec(name="v", scale=2.0)] * 2,
            steering_is_decode=[True, True],
            input_ids=torch.zeros(2, dtype=torch.long),
            residual_stream=None,
        )

        # Classic (hidden_states, residual) tuple model.
        tuple_model = _TupleToyModel(num_layers=3)
        tuple_base = tuple_model(x)
        with mgr.forward_context(tuple_model, fb):
            tuple_steered = tuple_model(x, forward_batch=fb)
        self.assertTrue(torch.allclose(tuple_steered, tuple_base + 2.0 * v, atol=1e-6))

        # SGLang ResidualStream model with plain contribution.
        rs_model = _ResidualStreamToyModel(num_layers=3)
        rs_base = rs_model(x, fb)
        with mgr.forward_context(rs_model, fb):
            rs_steered = rs_model(x, fb)
        self.assertTrue(torch.allclose(rs_steered, rs_base + 2.0 * v, atol=1e-6))

        # SGLang ResidualStream model with OwedOutput (pending all-reduce):
        # delta must be added once to stream.residual, NOT doubled by complete_sum!
        owed_model = _ResidualStreamToyModel(num_layers=3, owed_layer=1)
        owed_base = owed_model(x, fb)
        with mgr.forward_context(owed_model, fb):
            owed_steered = owed_model(x, fb)
        self.assertTrue(torch.allclose(owed_steered, owed_base + 2.0 * v, atol=1e-6))

    def test_resolve_decoder_layers_conventions(self):
        m = nn.Module()
        m.transformer = nn.Module()
        m.transformer.h = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.assertEqual(len(resolve_decoder_layers(m)), 2)

        bad_model = nn.Linear(4, 4)
        with self.assertRaises(ValueError):
            resolve_decoder_layers(bad_model)


class TestSteeringRequestAndRunnerIntegration(CustomTestCase):
    def test_generate_req_input_single_and_batch_normalization(self):
        single = GenerateReqInput(
            text="hello",
            steering_vector={"name": "honesty", "scale": 1.25, "scope": "decode"},
        )
        single.normalize_batch_and_arguments()
        self.assertEqual(
            single.steering_vector,
            SteeringVectorSpec(name="honesty", scale=1.25, scope="decode"),
        )

        # Single request rejects a list of specs.
        bad_single = GenerateReqInput(text="hello", steering_vector=["honesty"])
        with self.assertRaises(ValueError):
            bad_single.normalize_batch_and_arguments()

        # Batch broadcast + parallel sampling expansion.
        batch_broadcast = GenerateReqInput(
            text=["p1", "p2"],
            sampling_params={"n": 2},
            steering_vector="honesty",
        )
        batch_broadcast.normalize_batch_and_arguments()
        self.assertEqual(len(batch_broadcast.steering_vector), 4)
        self.assertEqual(batch_broadcast[0].steering_vector.name, "honesty")
        self.assertEqual(batch_broadcast[3].steering_vector.name, "honesty")

        # Per-item batch list with mixed None and dict specs.
        batch_mixed = GenerateReqInput(
            text=["p1", "p2"],
            steering_vector=[{"name": "v1", "scale": 2.0}, None],
        )
        batch_mixed.normalize_batch_and_arguments()
        self.assertEqual(batch_mixed[0].steering_vector.name, "v1")
        self.assertEqual(batch_mixed[0].steering_vector.scale, 2.0)
        self.assertIsNone(batch_mixed[1].steering_vector)

        # Mismatched batch length rejected.
        bad_batch = GenerateReqInput(
            text=["p1", "p2"],
            steering_vector=["v1"],
        )
        with self.assertRaises(ValueError):
            bad_batch.normalize_batch_and_arguments()

    def test_tokenized_req_msgpack_roundtrip(self):
        spec = SteeringVectorSpec(
            name="persona", scale=-0.75, scope="all", layers=(4, 8)
        )
        req = TokenizedGenerateReqInput(
            rid="req-1",
            input_text="hi",
            input_ids=array("q", [1, 2, 3]),
            mm_inputs=None,
            input_embeds=None,
            token_type_ids=None,
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            steering_vector=spec,
        )
        decoded = msgpack_decode(msgpack_encode(req))
        self.assertEqual(decoded.steering_vector, spec)

    def test_openai_protocol_parses_steering_vector(self):
        chat_req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            steering_vector={"name": "honesty", "scale": 1.5, "scope": "decode"},
        )
        self.assertEqual(chat_req.steering_vector["name"], "honesty")

        comp_req = CompletionRequest(
            model="test-model",
            prompt="hello",
            steering_vector="concise",
        )
        self.assertEqual(comp_req.steering_vector, "concise")

    def test_req_radix_cache_extra_key_isolation(self):
        spec_all = SteeringVectorSpec(name="honesty", scale=1.5, scope="all")
        spec_diff_scale = SteeringVectorSpec(name="honesty", scale=2.0, scope="all")
        spec_decode = SteeringVectorSpec(name="honesty", scale=1.5, scope="decode")

        req_plain = Req(
            rid="r0",
            origin_input_text="hi",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        req_steered = Req(
            rid="r1",
            origin_input_text="hi",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
            steering_vector=spec_all,
        )
        req_diff_scale = Req(
            rid="r2",
            origin_input_text="hi",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
            steering_vector=spec_diff_scale,
        )
        req_decode = Req(
            rid="r3",
            origin_input_text="hi",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(),
            steering_vector=spec_decode,
        )

        self.assertIsNone(req_plain.extra_key)
        self.assertIsNotNone(req_steered.extra_key)
        self.assertNotEqual(req_steered.extra_key, req_plain.extra_key)
        self.assertNotEqual(req_steered.extra_key, req_diff_scale.extra_key)
        # Decode-only steering does not modify prompt KV cache, so it shares prefix cache.
        self.assertIsNone(req_decode.extra_key)

    def test_graph_runners_disable_graph_for_steered_batches(self):
        fb_steered = SimpleNamespace(
            steering_vectors=[SteeringVectorSpec(name="v", scale=1.0)],
            replace_embeds=None,
            forward_mode=ForwardMode.DECODE,
        )

        decode_runner = object.__new__(DecodeCudaGraphRunner)
        decode_runner.model_runner = SimpleNamespace(model=SimpleNamespace())
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.megakernel_can_run_graph",
            return_value=True,
        ):
            self.assertFalse(decode_runner.can_run_graph(fb_steered))

        prefill_runner = object.__new__(PrefillCudaGraphRunner)
        self.assertFalse(prefill_runner.can_run_graph(fb_steered))

        cpu_runner = object.__new__(CPUGraphRunner)
        self.assertFalse(cpu_runner.can_run_graph(fb_steered))

    def test_model_runner_forward_raw_wiring_and_missing_manager_error(self):
        hidden_size = 4
        model = _ToyCausalLM(num_layers=2, hidden_size=hidden_size)
        x = torch.randn(2, hidden_size)
        v = torch.tensor([1.0, 2.0, -1.0, 0.5])

        runner = object.__new__(ModelRunner)
        runner.model = model
        runner.model_config = SimpleNamespace(
            hidden_size=hidden_size, num_hidden_layers=2
        )
        runner.device = "cpu"
        runner.dtype = torch.float32
        runner.steering_manager = None
        runner.attn_backend = None
        runner.decode_cuda_graph_runner = None
        runner.prefill_cuda_graph_runner = None
        runner.hisparse_coordinator = None
        runner.uocr_greedy_graph = None
        runner._prepare_eager_forward_batch = Mock()
        runner._maybe_execute_deferred_mamba_cow_and_clear = Mock()
        runner.eager_runner = SimpleNamespace(
            execute=lambda fb, pp_proxy_tensors=None: model(x, forward_batch=fb)
        )

        fb_unsteered = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=torch.zeros(2, dtype=torch.long),
            req_pool_indices=torch.zeros(2, dtype=torch.long),
            seq_lens=torch.ones(2, dtype=torch.long),
            out_cache_loc=torch.zeros(2, dtype=torch.long),
            seq_lens_sum=2,
        )
        fb_steered = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=torch.zeros(2, dtype=torch.long),
            req_pool_indices=torch.zeros(2, dtype=torch.long),
            seq_lens=torch.ones(2, dtype=torch.long),
            out_cache_loc=torch.zeros(2, dtype=torch.long),
            seq_lens_sum=2,
            steering_vectors=[SteeringVectorSpec(name="v", scale=1.5), None],
            steering_is_decode=[True, True],
        )

        with (
            patch(
                "sglang.srt.model_executor.model_runner.get_model",
                return_value=SimpleNamespace(
                    override=lambda **_: contextlib.nullcontext()
                ),
            ),
            patch(
                "sglang.srt.model_executor.model_runner.get_global_dwdp_manager",
                return_value=None,
            ),
        ):
            # Unsteered works with steering_manager=None (zero overhead).
            out_base = runner._forward_raw(fb_unsteered, pp_proxy_tensors=None)
            self.assertTrue(torch.equal(out_base.logits_output, model(x)))

            # Requesting steering without a registered manager raises RuntimeError.
            with self.assertRaisesRegex(RuntimeError, "steering_manager"):
                runner._forward_raw(fb_steered, pp_proxy_tensors=None)

            # Registering a vector initializes steering_manager and steers req 0 only.
            runner.register_steering_vector("v", {1: v})
            out_steered = runner._forward_raw(fb_steered, pp_proxy_tensors=None)
            self.assertTrue(
                torch.allclose(
                    out_steered.logits_output[0],
                    out_base.logits_output[0] + 1.5 * v,
                    atol=1e-6,
                )
            )
            self.assertTrue(
                torch.equal(out_steered.logits_output[1], out_base.logits_output[1])
            )

            # Unregistering removes the vector.
            runner.unregister_steering_vector("v")
            with self.assertRaises(KeyError):
                runner._forward_raw(fb_steered, pp_proxy_tensors=None)

    def test_forward_batch_has_steering_and_padding_preserves_alignment(self):
        fb_plain = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=torch.zeros(2, dtype=torch.long),
            req_pool_indices=torch.zeros(2, dtype=torch.long),
            seq_lens=torch.ones(2, dtype=torch.long),
            out_cache_loc=torch.zeros(2, dtype=torch.long),
            seq_lens_sum=2,
        )
        self.assertFalse(fb_plain.has_steering())

        spec = SteeringVectorSpec(name="v", scale=1.0)
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=torch.zeros(2, dtype=torch.long),
            req_pool_indices=torch.zeros(2, dtype=torch.long),
            seq_lens=torch.ones(2, dtype=torch.long),
            out_cache_loc=torch.zeros(2, dtype=torch.long),
            seq_lens_sum=2,
            positions=torch.zeros(2, dtype=torch.long),
            steering_vectors=[spec, None],
            steering_is_decode=[True, False],
        )
        self.assertTrue(fb.has_steering())

        mock_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(
                get_cpu_graph_seq_len_fill_value=lambda: 1,
                get_cuda_graph_seq_len_fill_value=lambda: 1,
            )
        )
        fb._pad_inputs_to_size(model_runner=mock_runner, num_tokens=4, bs=4)
        self.assertEqual(fb.steering_vectors, [spec, None, None, None])
        self.assertEqual(fb.steering_is_decode, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
