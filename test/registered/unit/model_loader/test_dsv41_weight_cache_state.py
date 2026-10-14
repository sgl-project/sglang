# SPDX-License-Identifier: Apache-2.0
"""CPU checks for processed-weight mapping; CUDA recovery is tested separately."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.srt.weight_cache.model_state import (
    CONTRACT,
    STATE_ENTRY,
    TENSOR_PREFIX,
    export_model_state,
    model_state_contract,
    restore_model_state,
    validate_model_state,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 4), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(2, 2), requires_grad=False)
        self.cached_parts = None
        self.ape_converted = False
        self.quant_method = SimpleNamespace(use_mxfp8=False, weight_block_size=[32, 32])


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = ToyLayer()


def config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41"), dtype=torch.float32
    )


def processed_state():
    model = ToyModel()
    # Post-loading transformed shape/dtype differs from the constructor.
    model.layer.weight = nn.Parameter(
        torch.arange(8).reshape(2, 4).to(torch.int8), requires_grad=False
    )
    model.layer.scale = nn.Parameter(
        torch.arange(4).reshape(1, 4).float(), requires_grad=False
    )
    model.layer.scale.format_ue8m0 = True
    model.layer.cached_parts = (model.layer.weight[:, :2], model.layer.scale)
    model.layer.ape_converted = True
    model.layer.quant_method.use_mxfp8 = True
    tensors = {k: (v, True) for k, v in model.state_dict().items()}
    manifest = export_model_state(model, tensors)
    entries = {k: {"tensor": v, "is_param": kind} for k, (v, kind) in tensors.items()}
    entries[STATE_ENTRY] = {"model_state": manifest}
    return model, tensors, manifest, entries


class TestDsv41WeightCacheState(CustomTestCase):
    def test_daemon_initializes_dense_gemm_backends(self):
        from sglang.srt.weight_cache.daemon import _initialize_gemm_configs

        with (
            patch(
                "sglang.srt.layers.quantization.fp8_utils.initialize_fp8_gemm_config"
            ) as fp8,
            patch(
                "sglang.srt.layers.quantization.fp4_utils.initialize_fp4_gemm_config"
            ) as fp4,
            patch(
                "sglang.srt.layers.quantization.unquant.initialize_bf16_gemm_config"
            ) as bf16,
        ):
            _initialize_gemm_configs()
        fp8.assert_called_once_with()
        fp4.assert_called_once_with()
        bf16.assert_called_once_with()

    def test_router_config_rebinds_to_imported_bias(self):
        model = ToyModel()
        bias = nn.Parameter(torch.ones(4), requires_grad=False)
        gate = nn.Module()
        gate.e_score_correction_bias = bias
        topk_config = SimpleNamespace(correction_bias=torch.empty(4, device="meta"))
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate = gate
        layer.mlp.topk = SimpleNamespace(topk_config=topk_config)
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([layer])
        tensors = {k: (v, True) for k, v in model.state_dict().items()}
        manifest = export_model_state(model, tensors)
        restore_model_state(model, manifest, {})
        self.assertIs(topk_config.correction_bias, bias)

    def test_daemon_exports_shared_nonpersistent_buffer_aliases(self):
        from sglang.srt.weight_cache.daemon import WeightCacheDaemon

        model = ToyModel()
        model.other = ToyLayer()
        shared = torch.ones(3)
        model.layer.register_buffer("freqs_cis", shared, persistent=False)
        model.other.register_buffer("freqs_cis", shared, persistent=False)
        daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
        daemon.model = model
        daemon.gpu_id = 0
        daemon.state_entries = {}
        daemon.config = SimpleNamespace(model_state_contract="")
        backend = SimpleNamespace(
            name="test",
            prepare_export=lambda tensors: {
                name: {"tensor": tensor, "is_param": kind}
                for name, (tensor, kind) in tensors.items()
            },
        )
        with patch(
            "sglang.srt.weight_cache.daemon.choose_daemon_transport_backend",
            return_value=backend,
        ):
            daemon._export_state()
        self.assertEqual(
            daemon.state_entries["layer.freqs_cis"]["tensor"].data_ptr(),
            shared.data_ptr(),
        )
        self.assertEqual(
            daemon.state_entries["other.freqs_cis"]["tensor"].data_ptr(),
            shared.data_ptr(),
        )

    def test_processed_tensors_map_without_reprocessing(self):
        source, _, _, entries = processed_state()
        loader = IpcModelLoader(
            LoadConfig(model_loader_extra_config={"dsv41_weight_cache": True})
        )
        loader._transport_backend = SimpleNamespace(import_tensor=lambda e: e["tensor"])
        with patch(
            "sglang.srt.weight_cache.ipc_loader._initialize_model",
            side_effect=lambda *args: ToyModel(),
        ):
            model = loader._load_zero_copy_mode(config(), None, entries, None)
        self.assertEqual(model.layer.weight.dtype, torch.int8)
        self.assertEqual(model.layer.scale.shape, (1, 4))
        self.assertEqual(model.layer.weight.data_ptr(), source.layer.weight.data_ptr())
        self.assertEqual(model.layer.scale.data_ptr(), source.layer.scale.data_ptr())
        self.assertTrue(model.layer.scale.format_ue8m0)
        self.assertTrue(model.layer.ape_converted)
        self.assertTrue(model.layer.quant_method.use_mxfp8)
        self.assertTrue(
            torch.equal(model.layer.cached_parts[0], source.layer.cached_parts[0])
        )

    def test_missing_contract_fails_before_mapping(self):
        _, _, _, entries = processed_state()
        entries.pop(STATE_ENTRY)
        loader = IpcModelLoader(
            LoadConfig(model_loader_extra_config={"dsv41_weight_cache": True})
        )
        with patch(
            "sglang.srt.weight_cache.ipc_loader._initialize_model",
            side_effect=lambda *args: ToyModel(),
        ):
            with self.assertRaisesRegex(ValueError, "missing"):
                loader._load_zero_copy_mode(config(), None, entries, None)

    def test_bad_tensor_schema_is_rejected(self):
        _, _, manifest, entries = processed_state()
        manifest["schema"]["layer.scale"][0] = [99]
        loader = IpcModelLoader(
            LoadConfig(model_loader_extra_config={"dsv41_weight_cache": True})
        )
        loader._transport_backend = SimpleNamespace(import_tensor=lambda e: e["tensor"])
        with patch(
            "sglang.srt.weight_cache.ipc_loader._initialize_model",
            side_effect=lambda *args: ToyModel(),
        ):
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                loader._load_zero_copy_mode(config(), None, entries, None)

    def test_class_drift_rejected(self):
        _, _, manifest, _ = processed_state()
        manifest["objects"]["layer"] = "wrong.Layer"
        with self.assertRaisesRegex(ValueError, "class mismatch"):
            validate_model_state(ToyModel(), manifest)

    def test_mhc_seam_does_not_duplicate_state_dict(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

        layers = []
        for _ in range(2):
            layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
            nn.Module.__init__(layer)
            layer.hc_pre_from_prev_sublayer = True
            layer.engram = None
            layer.register_parameter("test_weight", nn.Parameter(torch.ones(1)))
            layers.append(layer)
        layers[0].bind_next(layers[1])
        self.assertIs(layers[0]._next_layer, layers[1])
        self.assertNotIn("_next_layer", layers[0]._modules)
        self.assertEqual(list(layers[0].state_dict()), ["test_weight"])

    def test_meta_hasher_skips_tokenizer_and_request_history(self):
        from sglang.srt.layers.engram import EngramHasher, EngramLayout
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", weight_cache_mode="client"), role="tokenizer"
        )
        layout = EngramLayout(
            max_ngram_size=3,
            layer_ids=(1, 14),
            num_embeddings=(20, 20),
            primes=(((11,), (13,)), ((17,), (19,))),
            n_heads=1,
            head_dim=2,
        )
        with (
            torch.device("meta"),
            patch(
                "sglang.srt.layers.engram.get_tokenizer",
                side_effect=AssertionError("tokenizer rebuilt"),
            ),
        ):
            hasher = EngramHasher.from_config(SimpleNamespace(vocab_size=20), layout)
        self.assertTrue(hasher.token_map.is_meta)
        self.assertIsNone(hasher.history)
        self.assertEqual(hasher.max_ngram_size, 3)

    def test_weight_layout_fingerprint_changes_with_backend(self):
        from sglang.srt.weight_cache.model_state import layout_fingerprint

        cfg = SimpleNamespace(
            hf_config=SimpleNamespace(to_dict=lambda: {"model_type": "deepseek_v41"})
        )
        a = layout_fingerprint(
            cfg, SimpleNamespace(fp8_gemm_runner_backend="deep_gemm")
        )
        b = layout_fingerprint(
            cfg, SimpleNamespace(fp8_gemm_runner_backend="flashinfer_cutedsl")
        )
        self.assertNotEqual(a, b)

    def test_weight_layout_fingerprint_rejects_recovery_source_drift(self):
        from pathlib import Path

        from sglang.srt.weight_cache.model_state import layout_fingerprint

        cfg = SimpleNamespace(
            hf_config=SimpleNamespace(to_dict=lambda: {"model_type": "deepseek_v41"})
        )
        args = SimpleNamespace()
        baseline = layout_fingerprint(cfg, args)
        read_bytes = Path.read_bytes
        for source in (
            "weight_cache/daemon.py",
            "weight_cache/ipc_loader.py",
            "model_executor/model_runner_components/load_model_utils.py",
            "layers/quantization/fp4_utils.py",
            "layers/quantization/unquant.py",
        ):
            with self.subTest(source=source):

                def changed_source(path):
                    data = read_bytes(path)
                    return (
                        data + b"\n# different checkout\n"
                        if path.as_posix().endswith(source)
                        else data
                    )

                with patch.object(Path, "read_bytes", changed_source):
                    self.assertNotEqual(baseline, layout_fingerprint(cfg, args))

    def test_daemon_parser_does_not_consume_tp_alias(self):
        from sglang.srt.weight_cache.daemon import _parse_worker_cli_args

        worker, remaining = _parse_worker_cli_args(
            ["--tp", "4", "--model-path", "/model"]
        )
        self.assertIsNone(worker.tp_rank)
        self.assertIsNone(worker.gpu_id)
        self.assertEqual(remaining, ["--tp", "4", "--model-path", "/model"])

    def test_private_host_backing_cannot_be_exported(self):
        from sglang.srt.weight_cache.model_state import host_table_exports

        model = ToyModel()
        model.layer.host_table = SimpleNamespace(fd=None)
        with self.assertRaisesRegex(ValueError, "shareable host storage"):
            host_table_exports(model, {})

    def test_daemon_json_opt_in_is_accepted(self):
        self.assertEqual(
            model_state_contract(config(), '{"dsv41_weight_cache":true}'), CONTRACT
        )

    def test_experiment_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "experimental"):
            model_state_contract(config(), {})

    def test_plain_derived_tensors_are_not_in_manifest(self):
        _, tensors, manifest, _ = processed_state()
        self.assertTrue(any(k.startswith(TENSOR_PREFIX) for k in tensors))
        # Protocol metadata contains no tensor payloads.
        import json

        json.dumps(manifest)


class TestDraftCacheIsolation(CustomTestCase):
    def test_only_target_selects_ipc(self):
        from sglang.srt.configs.load_config import LoadFormat
        from sglang.srt.model_executor.model_runner_components.load_model_utils import (
            build_load_config,
            maybe_enable_ipc_weight_cache,
        )
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        reset_context()
        self.addCleanup(reset_context)
        args = ServerArgs(model_path="dummy", weight_cache_mode="client")
        publish(args, role="tokenizer")
        common = dict(
            server_args=args,
            tp_rank=0,
            remote_instance_weight_transporter_engine=None,
            remote_instance_weight_transporter_session_id=None,
            draft_model_idx=None,
            weight_cache_mode="client",
            weight_cache_socket="/tmp/target.sock",
        )
        target = build_load_config(**common)
        draft = build_load_config(**common, is_draft_worker=True)
        maybe_enable_ipc_weight_cache(load_config=target)
        maybe_enable_ipc_weight_cache(load_config=draft)
        self.assertEqual(target.load_format, LoadFormat.IPC_CACHE)
        self.assertNotEqual(draft.load_format, LoadFormat.IPC_CACHE)
        self.assertEqual(draft.weight_cache_mode, "off")
        self.assertIsNone(draft.weight_cache_socket)
        self.assertEqual(target.weight_cache_socket, "/tmp/target.sock")


if __name__ == "__main__":
    unittest.main()
