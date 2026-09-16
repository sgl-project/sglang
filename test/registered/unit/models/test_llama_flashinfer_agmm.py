from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/models/llama_flashinfer_agmm.py"
LLAMA_SOURCE = ROOT / "python/sglang/srt/models/llama.py"


def load_module():
    spec = importlib.util.spec_from_file_location("_test_llama_flashinfer_agmm", SOURCE)
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


def make_model(tp_size=4):
    q_size = 8192 // tp_size
    kv_size = 1024 // tp_size
    intermediate_size = 28672 // tp_size
    layers = []
    for _ in range(80):
        qkv = make_linear(
            "QKVParallelLinear",
            (q_size + 2 * kv_size, 8192),
            tp_size=tp_size,
            q_proj_shard_size=q_size,
            kv_proj_shard_size=kv_size,
            v_proj_shard_size=kv_size,
            gather_output=False,
        )
        o_proj = make_linear(
            "RowParallelLinear",
            (8192, q_size),
            tp_size=tp_size,
            input_is_parallel=True,
            reduce_results=True,
        )
        gate_up = make_linear(
            "MergedColumnParallelLinear",
            (2 * intermediate_size, 8192),
            tp_size=tp_size,
            gather_output=False,
        )
        down = make_linear(
            "RowParallelLinear",
            (8192, intermediate_size),
            tp_size=tp_size,
            input_is_parallel=True,
            reduce_results=True,
        )
        attention = types.SimpleNamespace(
            qkv_proj=qkv,
            o_proj=o_proj,
            q_size=q_size,
            kv_size=kv_size,
        )
        mlp = types.SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
        layers.append(named_object("LlamaDecoderLayer", self_attn=attention, mlp=mlp))
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
        tp4 = self.module._topology_for_tp_size(4)
        tp8 = self.module._topology_for_tp_size(8)
        self.assertEqual(self.module._row_partition(4096, tp4), 1024)
        self.assertEqual(self.module._row_partition(4096, tp8), 512)
        for topology in (tp4, tp8):
            for rows in (1, 512, 1024, 2048, 8192):
                self.assertIsNone(self.module._row_partition(rows, topology))

    def test_exact_tp4_and_tp8_topology_shapes(self):
        tp4 = self.module._topology_for_tp_size(4)
        self.assertEqual(
            (
                tp4.local_rows,
                tp4.q_size,
                tp4.kv_size,
                tp4.packed_qkv_n,
                tp4.intermediate_size,
                tp4.ranks,
            ),
            (1024, 2048, 256, 2560, 7168, (0, 1, 2, 3)),
        )
        tp8 = self.module._topology_for_tp_size(8)
        self.assertEqual(
            (
                tp8.local_rows,
                tp8.q_size,
                tp8.kv_size,
                tp8.packed_qkv_n,
                tp8.intermediate_size,
                tp8.ranks,
            ),
            (512, 1024, 128, 1280, 3584, tuple(range(8))),
        )
        with self.assertRaisesRegex(RuntimeError, "size 4 or 8"):
            self.module._topology_for_tp_size(2)

    def test_only_plain_extend_batches_route(self):
        extend = types.SimpleNamespace(
            is_extend=lambda: True,
            is_target_verify=lambda: False,
        )
        batch = types.SimpleNamespace(forward_mode=extend, can_run_tbo=False)
        self.assertIsNone(self.module._forward_mode_reason(batch))

        batch.can_run_tbo = True
        self.assertEqual(self.module._forward_mode_reason(batch), "two_batch_overlap")
        batch.can_run_tbo = False
        extend.is_target_verify = lambda: True
        self.assertEqual(self.module._forward_mode_reason(batch), "target_verify")
        extend.is_target_verify = lambda: False
        extend.is_extend = lambda: False
        self.assertEqual(self.module._forward_mode_reason(batch), "not_extend")

    def test_exact_model_contract_and_fail_closed_mutations(self):
        torch_stub = types.SimpleNamespace(bfloat16="bfloat16")
        for tp_size in (4, 8):
            topology = self.module._topology_for_tp_size(tp_size)
            model = make_model(tp_size)
            self.assertIsNone(
                self.module._model_contract_reason(model, torch_stub, topology)
            )

            model_id, token = self.module._bind_model_contract(
                model, torch_stub, topology
            )
            self.assertEqual(model_id, id(model))
            self.assertIs(getattr(model, self.module._MODEL_TOKEN_ATTRIBUTE), token)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                self.module._bind_model_contract(model, torch_stub, topology)

        model = make_model()
        topology = self.module._topology_for_tp_size(4)
        model.config.hidden_size = 4096
        self.assertEqual(
            self.module._model_contract_reason(model, torch_stub, topology),
            "config_hidden_size",
        )
        model.config.hidden_size = 8192
        model.layers[17].self_attn.qkv_proj.weight.dtype = "float16"
        self.assertEqual(
            self.module._model_contract_reason(model, torch_stub, topology),
            "layer_17_weight_dtype",
        )

        self.assertEqual(
            self.module._model_contract_reason(
                make_model(4), torch_stub, self.module._topology_for_tp_size(8)
            ),
            "layer_0_weight_shapes",
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

    def test_runtime_config_selects_tp4_or_tp8(self):
        exec_config = types.SimpleNamespace(
            graph=types.SimpleNamespace(disable_cuda_graph=True),
            kernel=types.SimpleNamespace(attention_backend="flashinfer"),
        )
        schedule = types.SimpleNamespace(
            disable_overlap_schedule=True,
            chunked_prefill_size=4096,
            max_running_requests=1,
        )
        memory = types.SimpleNamespace(disable_radix_cache=True)
        for tp_size, local_rows in ((4, 1024), (8, 512)):
            parallel = types.SimpleNamespace(
                tp_size=tp_size,
                attn_tp_size=tp_size,
                attn_dp_size=1,
                attn_cp_size=1,
                pp_size=1,
            )
            with (
                patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
                patch("sglang.srt.runtime_context.get_exec", return_value=exec_config),
                patch("sglang.srt.runtime_context.get_schedule", return_value=schedule),
                patch("sglang.srt.runtime_context.get_memory", return_value=memory),
            ):
                topology = (
                    self.module.LlamaFlashInferAgmmTrueSP._validate_runtime_config()
                )
            self.assertEqual(topology.tp_size, tp_size)
            self.assertEqual(topology.local_rows, local_rows)

        parallel.tp_size = 2
        parallel.attn_tp_size = 2
        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch("sglang.srt.runtime_context.get_exec", return_value=exec_config),
            patch("sglang.srt.runtime_context.get_schedule", return_value=schedule),
            patch("sglang.srt.runtime_context.get_memory", return_value=memory),
            self.assertRaisesRegex(RuntimeError, "size 4 or 8"),
        ):
            self.module.LlamaFlashInferAgmmTrueSP._validate_runtime_config()


class ServerArgumentTests(unittest.TestCase):
    def test_llama_reads_true_sp_flag_from_parallel_config(self):
        source = LLAMA_SOURCE.read_text()
        self.assertIn("get_parallel().enable_flashinfer_agmm_true_sp", source)
        self.assertNotIn("get_parallel().config.enable_flashinfer_agmm_true_sp", source)
        self.assertNotIn("get_server_args().enable_flashinfer_agmm_true_sp", source)

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
    def __init__(self, packed_qkv_n):
        self.shape = (8192, packed_qkv_n)

    def is_contiguous(self):
        return True


class FakeSourceWeight:
    def __init__(self, packed_qkv_n):
        self.packed = FakePackedWeight(packed_qkv_n)

    def t(self):
        return self

    def contiguous(self):
        return self.packed


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def is_contiguous(self):
        return True

    def new_empty(self, *shape):
        return FakeTensor(shape)


class FakeCoordinator:
    def __init__(self):
        self.all_gathers = []
        self.reduce_scatters = []

    def all_gather_into_tensor(self, output, local):
        self.all_gathers.append((output.shape, local.shape))

    def reduce_scatter_tensor(self, output, partial):
        self.reduce_scatters.append((output.shape, partial.shape))


class PreparedBindingTests(unittest.TestCase):
    def test_binding_is_prepared_once_for_each_supported_topology(self):
        for tp_size in (4, 8):
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
            route._topology = module._topology_for_tp_size(tp_size)
            route._packed_weights = {}
            route._bindings = {}
            route._prepare_all_gather_matmul = prepare
            weight = FakeSourceWeight(route._topology.packed_qkv_n)
            qkv = types.SimpleNamespace(weight=weight)
            group = object()
            first = types.SimpleNamespace(shape=(route._topology.local_rows, 8192))
            second = types.SimpleNamespace(shape=(route._topology.local_rows, 8192))

            self.assertEqual(route._prepared_qkv(first, qkv, group), ("output", first))
            self.assertEqual(
                route._prepared_qkv(second, qkv, group), ("output", second)
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][3:], ("auto", False))
            self.assertEqual(launches, [first, second])

    def test_row_collectives_use_the_bound_topology(self):
        module = load_module()
        for tp_size, local_rows in ((4, 1024), (8, 512)):
            route = object.__new__(module.LlamaFlashInferAgmmTrueSP)
            route._topology = module._topology_for_tp_size(tp_size)
            coordinator = FakeCoordinator()

            gathered = route._all_gather_rows(
                coordinator, FakeTensor((local_rows, 8192))
            )
            scattered = route._reduce_scatter_rows(
                coordinator, FakeTensor((4096, 8192))
            )

            self.assertEqual(gathered.shape, (4096, 8192))
            self.assertEqual(scattered.shape, (local_rows, 8192))
            self.assertEqual(
                coordinator.all_gathers,
                [((4096, 8192), (local_rows, 8192))],
            )
            self.assertEqual(
                coordinator.reduce_scatters,
                [((local_rows, 8192), (4096, 8192))],
            )


if __name__ == "__main__":
    unittest.main()
