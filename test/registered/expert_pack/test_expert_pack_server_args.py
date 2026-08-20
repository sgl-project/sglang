from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.server_args import LOAD_FORMAT_CHOICES, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _kimi_config(**overrides):
    values = {
        "model_type": "kimi_linear",
        "num_hidden_layers": 93,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "first_k_dense_replace": 1,
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestExpertPackServerArgs(unittest.TestCase):
    def _args(self, root: Path, hf_config, **overrides) -> ServerArgs:
        model_path = root / "model-meta"
        model_path.mkdir()
        pack_path = root / "model.expert-pack"
        pack_path.write_bytes(b"pack")
        manifest_path = root / "model.expert-pack.manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")

        args = ServerArgs(model_path="dummy")
        args.model_path = str(model_path)
        args.model_config = SimpleNamespace(hf_config=hf_config)
        args.load_format = "expert_pack"
        args.model_loader_extra_config = {
            "pack_path": str(pack_path),
            "manifest_path": str(manifest_path),
        }
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def test_expert_pack_remains_a_public_load_format(self):
        self.assertIn("expert_pack", LOAD_FORMAT_CHOICES)

    def test_valid_kimi_config_applies_execution_invariants(self):
        with tempfile.TemporaryDirectory() as value:
            args = self._args(Path(value), _kimi_config())
            with self.assertLogs(
                "sglang.srt.arg_groups.expert_pack_hook", level="INFO"
            ) as logs:
                args._handle_expert_pack()

        self.assertTrue(args.disable_cuda_graph)
        self.assertTrue(args.disable_shared_experts_fusion)
        self.assertIn("expert_pack selected", "\n".join(logs.output))

    def test_all_startup_errors_are_reported_together(self):
        with tempfile.TemporaryDirectory() as value:
            args = self._args(
                Path(value),
                _kimi_config(num_experts=1, num_experts_per_token=1),
                tp_size=2,
                dp_size=3,
                ep_size=4,
                enforce_shared_experts_fusion=True,
                cuda_graph_backend_decode="full",
            )
            args.model_loader_extra_config = {}
            with self.assertRaisesRegex(
                ValueError, "Invalid expert_pack configuration"
            ) as raised:
                args._handle_expert_pack()

        message = str(raised.exception)
        for expected in (
            "--tp-size) must be 1, got 2",
            "--dp-size) must be 1, got 3",
            "--ep-size) must be 1, got 4",
            "--enforce-shared-experts-fusion",
            "decode CUDA graph backend",
            "pack_path is required",
            "Kimi-K3 requires manifest_path",
            "Kimi-K3 num_experts must be 896, got 1",
            "Kimi-K3 num_experts_per_token must be 16, got 1",
        ):
            self.assertIn(expected, message)

    def test_deepseek_disables_incompatible_wo_a_kernel(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source_path = root / "source.gguf"
            source_path.write_bytes(b"source")
            args = self._args(
                root,
                SimpleNamespace(model_type="deepseek_v4"),
            )
            args.model_loader_extra_config.update(
                {
                    "source_path": str(source_path),
                    "source_sha256": "1" * 64,
                    "ollama_manifest_sha256": "2" * 64,
                    "config_sha256": "3" * 64,
                }
            )
            with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(True):
                args._handle_expert_pack()
                self.assertFalse(envs.SGLANG_OPT_FP8_WO_A_GEMM.get())

    def test_other_load_formats_are_unchanged(self):
        args = ServerArgs(model_path="dummy")
        args._handle_expert_pack()
        self.assertFalse(args.disable_cuda_graph)
        self.assertFalse(args.disable_shared_experts_fusion)

    def test_raw_kimi_gguf_is_prepared_inside_server_startup(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            gguf = root / "KIMI-K3-00001-of-00001.gguf"
            gguf.write_bytes(b"gguf")
            model_dir = root / "model-meta"
            model_dir.mkdir()
            pack = root / "model.expert-pack"
            pack.write_bytes(b"pack")
            manifest = root / "model.expert-pack.manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            stats = root / "stats.json"

            args = ServerArgs(model_path="dummy")
            args.model_path = str(gguf)
            args.tokenizer_path = str(gguf)
            args.load_format = "expert_pack"
            args.model_loader_extra_config = {"read_splits": 1}
            args.model_config = SimpleNamespace(hf_config=_kimi_config())

            def prepare(server_args, loader_config):
                server_args.model_path = str(model_dir)
                server_args.tokenizer_path = str(model_dir)
                loader_config.update(
                    pack_path=str(pack),
                    manifest_path=str(manifest),
                    stats_path=str(stats),
                )

            with patch(
                "sglang.srt.model_loader.expert_pack_runtime."
                "prepare_raw_kimi_server_args",
                side_effect=prepare,
            ) as prepare_mock:
                args._handle_expert_pack()

        prepare_mock.assert_called_once()
        self.assertEqual(args.model_path, str(model_dir))
        self.assertEqual(args.tokenizer_path, str(model_dir))
        self.assertEqual(args.model_loader_extra_config["pack_path"], str(pack))
        self.assertEqual(args.model_loader_extra_config["manifest_path"], str(manifest))
        self.assertEqual(args.model_loader_extra_config["read_splits"], 1)

    def test_suffixless_ollama_deepseek_blob_is_prepared_inside_server_startup(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            gguf = root / f"sha256-{'a' * 64}"
            gguf.write_bytes(b"gguf")
            artifact_dir = root / "artifact"
            model_dir = artifact_dir / "model-meta"
            model_dir.mkdir(parents=True)
            config = model_dir / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            pack = root / "DeepSeek-V4-Flash.expert-pack"
            pack.write_bytes(b"pack")
            manifest = root / "DeepSeek-V4-Flash.expert-pack.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "source": {"sha256": "1" * 64},
                        "model": {
                            "model_identity_sha256": "2" * 64,
                            "config_sha256": "3" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = ServerArgs(model_path="dummy")
            args.model_path = str(gguf)
            args.tokenizer_path = str(gguf)
            args.load_format = "expert_pack"
            args.model_loader_extra_config = {"read_splits": 2}
            args.model_config = SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v4")
            )

            with (
                patch(
                    "sglang.srt.model_loader.expert_pack_runtime."
                    "_deepseek_artifact_dir_for_source",
                    return_value=artifact_dir,
                ),
                patch(
                    "gguf.GGUFReader",
                    return_value=SimpleNamespace(
                        fields={
                            "general.architecture": SimpleNamespace(
                                contents=lambda: "deepseek4"
                            )
                        }
                    ),
                ),
                patch(
                    "sglang.srt.model_loader.expert_pack_runtime."
                    "_prepare_deepseek_model_metadata",
                    return_value=config,
                ),
                patch(
                    "sglang.srt.model_loader.expert_pack_runtime."
                    "_prepare_deepseek_pack",
                    return_value=(pack, manifest),
                ),
            ):
                args._handle_expert_pack()

        self.assertEqual(args.model_path, str(model_dir))
        self.assertEqual(args.tokenizer_path, str(model_dir))
        self.assertEqual(args.model_loader_extra_config["pack_path"], str(pack))
        self.assertEqual(args.model_loader_extra_config["manifest_path"], str(manifest))
        self.assertEqual(args.model_loader_extra_config["source_path"], str(gguf))
        self.assertEqual(args.model_loader_extra_config["read_splits"], 2)


if __name__ == "__main__":
    unittest.main()
