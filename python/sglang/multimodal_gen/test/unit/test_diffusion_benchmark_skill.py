import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_benchmark_module(temp_root: Path):
    multimodal_gen_root = Path(__file__).resolve().parents[2]
    script_path = (
        multimodal_gen_root
        / ".claude"
        / "skills"
        / "sglang-diffusion-benchmark-profile"
        / "scripts"
        / "bench_diffusion_denoise.py"
    )
    fake_env = types.ModuleType("diffusion_skill_env")
    fake_env.ensure_dir = lambda path: (
        Path(path).mkdir(parents=True, exist_ok=True) or Path(path)
    )
    fake_env.get_assets_dir = lambda _root: temp_root / "assets"
    fake_env.get_output_dir = lambda _kind, _root: temp_root / "outputs"
    fake_env.get_repo_root = lambda: temp_root / "repo"
    fake_env.pick_idle_gpus = lambda count: list(range(count))

    spec = importlib.util.spec_from_file_location(
        "test_bench_diffusion_denoise", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"diffusion_skill_env": fake_env}):
        spec.loader.exec_module(module)
    return module


class TestDiffusionBenchmarkSkill(unittest.TestCase):
    def test_nightly_presets_remain_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = _load_benchmark_module(Path(tmpdir))
            repo_root = Path(__file__).resolve().parents[5]
            module.NIGHTLY_CONFIG_PATH = (
                repo_root
                / "scripts"
                / "ci"
                / "utils"
                / "diffusion"
                / "comparison_configs.json"
            )

            self.assertEqual(module.validate_nightly_alignment(), 0)

    def test_recent_model_presets_are_eager_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = _load_benchmark_module(Path(tmpdir))

            expected = {
                "longcat-image",
                "sana-video",
                "lingbot-video-moe",
                "cosmos3-edge-t2i",
                "cosmos3-super-t2i-distilled",
                "ltx25",
                "ltx25-diffusion-decoder",
            }
            self.assertTrue(expected.issubset(module.MODELS))

            eager_cmd = module.build_sglang_cmd("longcat-image")
            self.assertNotIn("--enable-torch-compile", eager_cmd)
            self.assertIn("--enable-prompt-rewrite=false", eager_cmd)

            compiled_cmd = module.build_sglang_cmd("longcat-image", torch_compile=True)
            self.assertIn("--enable-torch-compile", compiled_cmd)

            h3_cmd = module.build_sglang_cmd("minimax-h3-t2va", torch_compile=True)
            self.assertNotIn("--enable-torch-compile", h3_cmd)

    def test_isolated_cache_cleanup_writes_zero_residual_ledger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            cache_root = temp_root / "model-caches"
            cache_dir = module._prepare_model_cache(
                cache_root, "longcat-image", "baseline"
            )

            weight_path = cache_dir / "huggingface" / "hub" / "model.safetensors"
            weight_path.parent.mkdir(parents=True)
            weight_path.write_bytes(b"weights")
            env = module._model_cache_env(cache_dir)
            self.assertTrue(env["HF_HOME"].startswith(str(cache_dir)))
            self.assertTrue(env["HF_XET_CACHE"].startswith(str(cache_dir)))
            self.assertTrue(env["TRANSFORMERS_CACHE"].startswith(str(cache_dir)))
            self.assertTrue(env["MODELSCOPE_CACHE"].startswith(str(cache_dir)))

            ledger_path = temp_root / "artifacts" / "cleanup.jsonl"
            record = module._cleanup_model_cache(
                cache_root,
                cache_dir,
                ledger_path,
                "longcat-image",
                "baseline",
                "success",
            )

            self.assertFalse(cache_dir.exists())
            self.assertEqual(record["before"]["weight_file_count"], 1)
            self.assertEqual(record["after"]["file_count"], 0)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["exit_reason"], "success")
            self.assertEqual(ledger["after"]["weight_file_count"], 0)

    def test_isolated_cache_refuses_to_reuse_existing_run_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            cache_root = temp_root / "model-caches"
            module._prepare_model_cache(cache_root, "sana-video", "baseline")

            with self.assertRaises(FileExistsError):
                module._prepare_model_cache(cache_root, "sana-video", "baseline")

    def test_interrupted_run_cleans_isolated_cache_in_finally(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            cache_root = temp_root / "model-caches"
            output_dir = temp_root / "outputs"
            output_dir.mkdir()

            with (
                patch.object(
                    module, "_run_benchmark_once_impl", side_effect=KeyboardInterrupt
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                module.run_benchmark_once(
                    "sana-video",
                    "baseline",
                    output_dir,
                    model_cache_root=cache_root,
                    cleanup_model_cache=True,
                )

            self.assertFalse((cache_root / "sana-video-baseline").exists())
            ledger = json.loads(
                (output_dir / "cleanup.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(ledger["exit_reason"], "interrupted")
            self.assertEqual(ledger["after"]["weight_file_count"], 0)

    def test_failed_run_is_recorded_as_error_and_cleaned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            cache_root = temp_root / "model-caches"
            output_dir = temp_root / "outputs"
            output_dir.mkdir()

            with (
                patch.object(
                    module, "_run_benchmark_once_impl", side_effect=RuntimeError("boom")
                ),
                self.assertRaisesRegex(RuntimeError, "boom"),
            ):
                module.run_benchmark_once(
                    "sana-video",
                    "baseline",
                    output_dir,
                    model_cache_root=cache_root,
                    cleanup_model_cache=True,
                )

            self.assertFalse((cache_root / "sana-video-baseline").exists())
            ledger = json.loads(
                (output_dir / "cleanup.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(ledger["exit_reason"], "error")


if __name__ == "__main__":
    unittest.main()
