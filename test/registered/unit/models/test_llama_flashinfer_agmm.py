from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/models/llama_flashinfer_agmm.py"
LLAMA_SOURCE = ROOT / "python/sglang/srt/models/llama.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "_test_llama_flashinfer_agmm", SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load llama_flashinfer_agmm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def named_object(name: str, **attributes):
    cls = type(name, (), {})
    value = cls()
    for key, attribute in attributes.items():
        setattr(value, key, attribute)
    return value


class FakeWeight:
    def __init__(self, shape, dtype="bfloat16", is_cuda=True):
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda


def make_linear(name, shape, **attributes):
    defaults = {
        "weight": FakeWeight(shape),
        "bias": None,
        "quant_method": named_object("UnquantizedLinearMethod"),
    }
    defaults.update(attributes)
    return named_object(name, **defaults)


def make_model():
    layers = []
    for _ in range(80):
        qkv = make_linear(
            "QKVParallelLinear",
            (2560, 8192),
            tp_size=4,
            q_proj_shard_size=2048,
            kv_proj_shard_size=256,
            v_proj_shard_size=256,
            gather_output=False,
        )
        o_proj = make_linear(
            "RowParallelLinear",
            (8192, 2048),
            tp_size=4,
            input_is_parallel=True,
            reduce_results=True,
        )
        gate_up = make_linear(
            "MergedColumnParallelLinear",
            (14336, 8192),
            tp_size=4,
            gather_output=False,
        )
        down = make_linear(
            "RowParallelLinear",
            (8192, 7168),
            tp_size=4,
            input_is_parallel=True,
            reduce_results=True,
        )
        attention = types.SimpleNamespace(
            qkv_proj=qkv,
            o_proj=o_proj,
            q_size=2048,
            kv_size=256,
        )
        mlp = types.SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
        layers.append(
            named_object("LlamaDecoderLayer", self_attn=attention, mlp=mlp)
        )
    return types.SimpleNamespace(
        pp_group=types.SimpleNamespace(
            world_size=1, is_first_rank=True, is_last_rank=True
        ),
        start_layer=0,
        end_layer=80,
        layers_to_capture=[],
        config=types.SimpleNamespace(
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
        ),
        layers=layers,
        norm=types.SimpleNamespace(hidden_size=8192),
    )


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_only_the_validated_row_count_routes(self):
        self.assertEqual(self.module._row_partition(4096), 1024)
        for rows in (1, 512, 1024, 2048, 8192):
            self.assertIsNone(self.module._row_partition(rows))

    def test_only_plain_extend_batches_route(self):
        extend = types.SimpleNamespace(
            is_extend=lambda: True,
            is_target_verify=lambda: False,
        )
        batch = types.SimpleNamespace(forward_mode=extend, can_run_tbo=False)
        self.assertIsNone(self.module._forward_mode_reason(batch))

        batch.can_run_tbo = True
        self.assertEqual(
            self.module._forward_mode_reason(batch), "two_batch_overlap"
        )
        batch.can_run_tbo = False
        extend.is_target_verify = lambda: True
        self.assertEqual(self.module._forward_mode_reason(batch), "target_verify")
        extend.is_target_verify = lambda: False
        extend.is_extend = lambda: False
        self.assertEqual(self.module._forward_mode_reason(batch), "not_extend")

    def test_exact_model_contract_and_fail_closed_mutations(self):
        torch_stub = types.SimpleNamespace(bfloat16="bfloat16")
        model = make_model()
        self.assertIsNone(self.module._model_contract_reason(model, torch_stub))

        model_id, token = self.module._bind_model_contract(model, torch_stub)
        self.assertEqual(model_id, id(model))
        self.assertIs(
            getattr(model, self.module._MODEL_TOKEN_ATTRIBUTE), token
        )
        with self.assertRaisesRegex(RuntimeError, "already owns"):
            self.module._bind_model_contract(model, torch_stub)

        model = make_model()
        model.config.hidden_size = 4096
        self.assertEqual(
            self.module._model_contract_reason(model, torch_stub),
            "config_hidden_size",
        )
        model.config.hidden_size = 8192
        model.layers[17].self_attn.qkv_proj.weight.dtype = "float16"
        self.assertEqual(
            self.module._model_contract_reason(model, torch_stub),
            "layer_17_weight_dtype",
        )

    def test_prepared_api_signature_is_exact(self):
        def accepted(inp, w, group, *, backend="auto", verbose=False):
            return None

        self.module._validate_prepare_signature(accepted)

        def missing_backend(inp, w, group, *, verbose=False):
            return None

        with self.assertRaisesRegex(RuntimeError, "incompatible signature"):
            self.module._validate_prepare_signature(missing_backend)

    def test_route_source_contains_no_explicit_device_synchronization(self):
        source = inspect.getsource(self.module)
        self.assertNotIn(".synchronize(", source)


class ServerArgumentTests(unittest.TestCase):
    def test_llama_reads_true_sp_flag_from_parallel_config(self):
        source = LLAMA_SOURCE.read_text()
        self.assertIn(
            "get_parallel().config.enable_flashinfer_agmm_true_sp", source
        )
        self.assertNotIn(
            "get_parallel().enable_flashinfer_agmm_true_sp", source
        )

    def test_true_sp_flag_parses_from_cli_and_yaml(self):
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        defaults = parser.parse_args(["--model", "dummy"])
        self.assertFalse(defaults.enable_flashinfer_agmm_true_sp)

        cli = parser.parse_args(
            ["--model", "dummy", "--enable-flashinfer-agmm-true-sp"]
        )
        self.assertTrue(cli.enable_flashinfer_agmm_true_sp)

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "server.yaml"
            config.write_text("enable-flashinfer-agmm-true-sp: true\n")
            merged = ConfigArgumentMerger(parser).merge_config_with_args(
                ["--config", str(config), "--model", "dummy"]
            )
            yaml_args = parser.parse_args(merged)

        self.assertTrue(yaml_args.enable_flashinfer_agmm_true_sp)


class FakePackedWeight:
    shape = (8192, 2560)

    def is_contiguous(self):
        return True


class FakeSourceWeight:
    def __init__(self):
        self.packed = FakePackedWeight()

    def t(self):
        return self

    def contiguous(self):
        return self.packed


class PreparedBindingTests(unittest.TestCase):
    def test_binding_is_prepared_once_for_one_weight_and_row_shape(self):
        module = load_module()
        calls = []
        launches = []

        def prepare(inp, w, group, *, backend="auto", verbose=False):
            calls.append((inp, w, group, backend, verbose))

            def launch(current):
                launches.append(current)
                return ("output", current)

            return launch

        route = object.__new__(module.LlamaFlashInferAgmmTrueSP)
        route._packed_weights = {}
        route._bindings = {}
        route._prepare_all_gather_matmul = prepare
        weight = FakeSourceWeight()
        qkv = types.SimpleNamespace(weight=weight)
        group = object()
        first = types.SimpleNamespace(shape=(1024, 8192))
        second = types.SimpleNamespace(shape=(1024, 8192))

        self.assertEqual(
            route._prepared_qkv(first, qkv, group), ("output", first)
        )
        self.assertEqual(
            route._prepared_qkv(second, qkv, group), ("output", second)
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][3:], ("auto", False))
        self.assertEqual(launches, [first, second])


if __name__ == "__main__":
    unittest.main()
