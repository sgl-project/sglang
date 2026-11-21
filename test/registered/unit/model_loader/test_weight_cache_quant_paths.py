"""
CPU-only unit tests for quantized checkpoints on the weight-cache path.

- Engine: a weight-cache client (LoadFormat.IPC_CACHE) gets the weight-cache
  loader for pre-quantized ModelOpt checkpoints, and refuses ModelOpt
  quantization at load time, which only ModelOptModelLoader runs.
- Daemon: WeightCacheDaemon and launch_weight_cache_daemons read resolved
  server arguments, so model overrides reach the daemon's CacheConfig.

The checkpoints are config files only, laid out like ModelOpt exports and
gpt-oss-20b. GPU-dependent facts are set as on a B200: the platform, the mxfp4
entry a CUDA build registers in QUANTIZATION_METHODS, and the device UUID.
"""

import contextlib
import json
import os
import shutil
import tempfile
import unittest
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.model_loader import loader as loader_mod
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.weight_cache import daemon as daemon_mod
from sglang.srt.weight_cache.ipc_loader import WeightCacheModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _write_checkpoint(directory: str, config: dict, extra_files=None) -> str:
    os.makedirs(directory)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(config, f)
    for name, content in (extra_files or {}).items():
        with open(os.path.join(directory, name), "w") as f:
            json.dump(content, f)
    return directory


class _CheckpointCase(CustomTestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="wc_quant_paths_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def checkpoint(self, name, config, extra_files=None):
        return _write_checkpoint(os.path.join(self.root, name), config, extra_files)


# ---- Engine: get_model_loader for ModelOpt checkpoints ----


_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
    "torch_dtype": "bfloat16",
}


def _hf_quant_config(quantization: dict) -> dict:
    """ModelOpt's hf_quant_config.json, written next to config.json."""
    return {
        "producer": {"name": "modelopt", "version": "0.29.0"},
        "quantization": quantization,
    }


# (case, config.json quantization_config or None, hf_quant_config.json or None,
#  the quantization ModelConfig detects)
_MODELOPT_CHECKPOINTS = [
    (
        "nvfp4_hf_quant_config",
        None,
        _hf_quant_config(
            {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": "FP8",
                "group_size": 16,
                "exclude_modules": ["lm_head"],
            }
        ),
        "modelopt_fp4",
    ),
    (
        "fp8_hf_quant_config",
        None,
        _hf_quant_config(
            {
                "quant_algo": "FP8",
                "kv_cache_quant_algo": "FP8",
                "group_size": 16,
                "exclude_modules": ["lm_head"],
            }
        ),
        "modelopt_fp8",
    ),
    (
        "nvfp4_config_json",  # quantization_config inside config.json
        {
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "targets": ["Linear"],
                }
            },
            "ignore": ["lm_head", "model.embed_tokens"],
            "quant_algo": "NVFP4",
            "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
            "producer": {"name": "modelopt", "version": "0.46.0"},
            "quant_method": "modelopt",
        },
        None,
        "modelopt_fp4",
    ),
    (
        "mixed_hf_quant_config",  # FP8 dense layers + NVFP4 experts
        None,
        _hf_quant_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": "FP8",
                "quantized_layers": {
                    "model.layers.0.mlp.down_proj": {"quant_algo": "FP8"},
                    "model.layers.1.mlp.experts.0.down_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                },
            }
        ),
        "modelopt_mixed",
    ),
]


def _engine_load_config(weight_cache_mode: str) -> LoadConfig:
    """The engine's LoadConfig once ModelRunner has switched it onto the weight
    cache."""
    load_config = LoadConfig(
        load_format="auto",
        weight_cache_mode=weight_cache_mode,
        weight_cache_socket="/nonexistent/sglang_weight_cache.sock",
    )
    with mock.patch.object(
        load_model_utils,
        "get_model",
        lambda: SimpleNamespace(weight_cache_mode=weight_cache_mode),
    ):
        load_model_utils.maybe_enable_ipc_weight_cache(load_config=load_config)
    return load_config


class TestModelOptCheckpointLoader(_CheckpointCase):
    def _model_configs(self):
        for case, quantization_config, hf_quant, expected in _MODELOPT_CHECKPOINTS:
            config = dict(_LLAMA)
            if quantization_config is not None:
                config["quantization_config"] = quantization_config
            extra = {"hf_quant_config.json": hf_quant} if hf_quant else None
            path = self.checkpoint(case, config, extra)
            yield case, path, ModelConfig(model_path=path), expected

    def test_client_gets_the_weight_cache_loader(self):
        """A weight-cache engine loads a pre-quantized ModelOpt checkpoint with
        the weight-cache loader; only a disk load takes ModelOptModelLoader."""
        for case, path, model_config, expected in self._model_configs():
            with self.subTest(checkpoint=case):
                self.assertEqual(model_config.quantization, expected)
                self.assertTrue(model_config._is_already_quantized())
                for mode in ("client", "daemon"):
                    load_config = _engine_load_config(mode)
                    self.assertEqual(load_config.load_format, LoadFormat.IPC_CACHE)
                    loader = loader_mod.get_model_loader(
                        load_config=load_config, model_config=model_config
                    )
                    self.assertIsInstance(loader, WeightCacheModelLoader)
                    self.assertEqual(loader.weight_cache_mode, mode)
                    self.assertEqual(loader._fallback_load_format, LoadFormat.AUTO)
                    self.assertEqual(
                        loader.socket_path, load_config.weight_cache_socket
                    )
                # From disk (a daemon, or an engine without the weight cache)
                # the checkpoint still takes ModelOptModelLoader.
                disk = loader_mod.get_model_loader(
                    load_config=LoadConfig(load_format="auto"),
                    model_config=model_config,
                )
                self.assertIs(type(disk), loader_mod.ModelOptModelLoader)

    def test_nothing_modelopt_specific_is_skipped(self):
        """A client bypasses ModelOptModelLoader, so for a pre-quantized
        checkpoint that loader must stay DefaultModelLoader.load_model, which
        the client's disk fallback runs."""
        device_config = DeviceConfig("cpu")
        for case, path, model_config, _ in self._model_configs():
            with self.subTest(checkpoint=case):
                calls = []

                def default_load(loader, *, model_config, device_config):
                    calls.append((type(loader), loader.load_config.load_format))
                    return "model"

                with mock.patch.object(
                    loader_mod.DefaultModelLoader, "load_model", default_load
                ):
                    modelopt = loader_mod.ModelOptModelLoader(
                        LoadConfig(load_format="auto")
                    )
                    self.assertEqual(
                        modelopt.load_model(
                            model_config=model_config, device_config=device_config
                        ),
                        "model",
                    )
                    client = loader_mod.get_model_loader(
                        load_config=_engine_load_config("client"),
                        model_config=model_config,
                    )
                    self.assertEqual(
                        client._fallback_load(model_config, device_config), "model"
                    )
                self.assertEqual(
                    calls,
                    [
                        (loader_mod.ModelOptModelLoader, LoadFormat.AUTO),
                        (loader_mod.DefaultModelLoader, LoadFormat.AUTO),
                    ],
                )

    def test_modelopt_quantization_at_load_is_refused(self):
        """A client refuses a checkpoint that ModelOpt would quantize while
        loading (calibration, checkpoint restore or export), instead of serving
        the daemon's weights or loading it unquantized in the disk fallback."""
        plain = self.checkpoint("bf16", dict(_LLAMA))
        restore = LoadConfig(
            load_format="auto",
            modelopt_checkpoint_restore_path=os.path.join(self.root, "restore"),
        )
        cases = [
            # unified flag on an unquantized checkpoint: calibration
            ("modelopt_fp8", LoadConfig(load_format="auto")),
            # modelopt_fp4 is converted online unless a workflow is requested
            ("modelopt_fp4", restore),
        ]
        for quantization, disk_config in cases:
            with self.subTest(quantization=quantization):
                model_config = ModelConfig(model_path=plain, quantization=quantization)
                self.assertFalse(model_config._is_already_quantized())
                self.assertIs(
                    type(
                        loader_mod.get_model_loader(
                            load_config=disk_config, model_config=model_config
                        )
                    ),
                    loader_mod.ModelOptModelLoader,
                )
                client_config = _engine_load_config("client")
                client_config.modelopt_config = disk_config.modelopt_config
                with self.assertRaisesRegex(ValueError, "cannot quantize"):
                    loader_mod.get_model_loader(
                        load_config=client_config, model_config=model_config
                    )

    def test_other_choices_unchanged(self):
        """Online modelopt_fp4 and unquantized checkpoints are not refused."""
        plain = self.checkpoint("bf16", dict(_LLAMA))
        # Online modelopt_fp4 (a BF16 checkpoint quantized at load) keeps its
        # paths: DefaultModelLoader from disk, the weight cache as a client.
        online = ModelConfig(model_path=plain, quantization="modelopt_fp4")
        self.assertFalse(online._is_already_quantized())
        self.assertIs(
            type(
                loader_mod.get_model_loader(
                    load_config=LoadConfig(load_format="auto"), model_config=online
                )
            ),
            loader_mod.DefaultModelLoader,
        )
        self.assertIsInstance(
            loader_mod.get_model_loader(
                load_config=_engine_load_config("client"), model_config=online
            ),
            WeightCacheModelLoader,
        )
        # Unquantized: unchanged.
        self.assertIsInstance(
            loader_mod.get_model_loader(
                load_config=_engine_load_config("client"),
                model_config=ModelConfig(model_path=plain),
            ),
            WeightCacheModelLoader,
        )


# ---- Daemon: resolved server arguments ----


# gpt-oss-20b's config.json with small sizes: mxfp4 experts and no torch_dtype,
# so dtype "auto" means float16 (ModelConfig casts float32 configs to float16).
_GPT_OSS = {
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "attention_bias": True,
    "experts_per_token": 4,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 128,
    "initial_context_length": 4096,
    "intermediate_size": 128,
    "layer_types": ["sliding_attention", "full_attention"],
    "max_position_embeddings": 131072,
    "num_attention_heads": 4,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "num_local_experts": 4,
    "quantization_config": {
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "quant_method": "mxfp4",
    },
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
        "truncate": False,
    },
    "rope_theta": 150000,
    "sliding_window": 128,
    "swiglu_limit": 7.0,
    "tie_word_embeddings": False,
    "vocab_size": 1024,
}

# Server arguments WeightCacheDaemon.__init__ copies into same-named attributes.
_DAEMON_FIELDS = (
    "model_path",
    "tp_size",
    "pp_size",
    "ep_size",
    "moe_dp_size",
    "enable_dp_lm_head",
    "attn_cp_size",
    "moe_dense_tp_size",
    "moe_a2a_backend",
    "deepep_mode",
    "load_format",
    "dtype",
    "quantization",
    "model_loader_extra_config",
    "trust_remote_code",
    "revision",
)


def _device_uuid(device_id: int = 0) -> str:
    """CudaPlatform.get_device_uuid on an 8-GPU node:
    torch.cuda.get_device_properties(device_id).uuid, which refuses an
    ordinal outside the visible devices."""
    if not 0 <= device_id < 8:
        raise AssertionError("Invalid device id")
    return f"GPU-00000000-0000-0000-0000-00000000000{device_id}"


@contextlib.contextmanager
def _b200():
    """A B200 node as resolution and the daemon see it."""
    with (
        override_platform(
            is_cuda=True,
            is_sm100=True,
            is_sm90=False,
            is_sm120=False,
            is_blackwell=True,
        ),
        # A CUDA build registers mxfp4 (layers/quantization/__init__.py).
        mock.patch.dict(QUANTIZATION_METHODS, {"mxfp4": Mxfp4Config}),
        mock.patch.object(daemon_mod.current_platform, "get_device_uuid", _device_uuid),
    ):
        yield


def _daemon_model_config(daemon) -> ModelConfig:
    """The ModelConfig WeightCacheDaemon.load() builds from its fields."""
    return ModelConfig(
        model_path=daemon.model_path,
        trust_remote_code=daemon.trust_remote_code,
        revision=daemon.revision,
        dtype=daemon.dtype,
        quantization=daemon.quantization,
    )


class TestDaemonReadsResolvedArgs(_CheckpointCase):
    def test_daemon_builds_the_engine_dtype(self):
        """The daemon reads resolved arguments, so model overrides (here the
        gpt-oss mxfp4 dtype) give it the engine's ModelConfig dtype."""
        path = self.checkpoint("gpt_oss", _GPT_OSS)
        with _b200():
            server_args = ServerArgs(model_path=path, device="cuda")
            self.assertEqual(server_args.dtype, "auto")  # what the operator gave
            daemon = daemon_mod.WeightCacheDaemon(
                server_args=server_args, gpu_id=0, tp_rank=0, pp_rank=0
            )
            view = resolving_view(server_args)
            # The gpt-oss override for mxfp4 (and SM100's MoE runner).
            self.assertEqual(view.dtype, "bfloat16")
            self.assertEqual(view.moe_runner_backend, "flashinfer_mxfp4")
            for field in _DAEMON_FIELDS:
                self.assertEqual(getattr(daemon, field), getattr(view, field), field)
            # The dtype both sides put into CacheConfig (str(model_config.dtype)):
            # the engine's ModelConfig comes from ModelConfig.from_server_args.
            engine = ModelConfig.from_server_args(server_args)
            self.assertEqual(engine.dtype, torch.bfloat16)
            self.assertEqual(_daemon_model_config(daemon).dtype, engine.dtype)
            self.assertEqual(engine.quantization, "mxfp4")
            # Unresolved, "auto" gives float16 here, so the checks above can fail.
            self.assertEqual(
                ModelConfig(model_path=path, dtype="auto").dtype, torch.float16
            )

    def test_launcher_hands_children_a_resolved_record(self):
        """The standalone launcher resolves before it spawns, and a child reads
        that record without resolving it again."""
        path = self.checkpoint("gpt_oss", _GPT_OSS)
        spawned = []

        class _Spawned(Exception):
            pass

        def spawn(server_args, **kwargs):
            # What multiprocessing's spawn start method sends to the child.
            spawned.append((bytes(ForkingPickler.dumps(server_args)), kwargs))
            raise _Spawned

        cleaned = []
        with (
            _b200(),
            mock.patch.object(daemon_mod, "spawn_weight_cache_daemon", spawn),
            mock.patch.object(
                daemon_mod,
                "cleanup_stale_daemon_files",
                lambda uuid, force=False: cleaned.append(uuid),
            ),
        ):
            server_args = ServerArgs(model_path=path, device="cuda", tp_size=2)
            with self.assertRaises(_Spawned):
                daemon_mod.launch_weight_cache_daemons(
                    server_args, dist_init_method="tcp://127.0.0.1:29500"
                )
            self.assertEqual(cleaned, [_device_uuid(0), _device_uuid(1)])
            ((payload, kwargs),) = spawned
            self.assertEqual(
                kwargs,
                {
                    "gpu_id": 0,
                    "tp_rank": 0,
                    "pp_rank": 0,
                    "dist_init_method": "tcp://127.0.0.1:29500",
                },
            )
            child_args = ForkingPickler.loads(payload)
            self.assertEqual(resolving_view(child_args).dtype, "bfloat16")
            # The child reads the record as resolved and does not resolve it
            # again (resolution is not idempotent over its own output).
            with mock.patch(
                "sglang.srt.arg_groups.pipeline.run_resolution_pipeline",
                side_effect=AssertionError("resolved twice"),
            ):
                daemon = daemon_mod.WeightCacheDaemon(
                    server_args=child_args, gpu_id=0, tp_rank=0, pp_rank=0
                )
            self.assertEqual(daemon.dtype, "bfloat16")
            self.assertEqual(daemon.tp_size, 2)
            self.assertEqual(_daemon_model_config(daemon).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
