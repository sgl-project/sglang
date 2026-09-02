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


def _load_skill_env_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / ".claude"
        / "skills"
        / "sglang-diffusion-benchmark-profile"
        / "scripts"
        / "diffusion_skill_env.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_diffusion_skill_env", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDiffusionBenchmarkSkill(unittest.TestCase):
    def test_skill_env_prefers_own_worktree_over_installed_package(self):
        module = _load_skill_env_module()
        installed = types.ModuleType("sglang")
        installed.__file__ = "/sgl-workspace/sglang/python/sglang/__init__.py"

        with patch.dict(sys.modules, {"sglang": installed}):
            repo_root = module.get_repo_root()

        self.assertEqual(repo_root, Path(__file__).resolve().parents[5])

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
                "longcat-image-edit",
                "longcat-image-edit-turbo",
                "qwen-edit-base",
                "qwen-image-layered",
                "stable-diffusion-3.5-medium",
                "sana-video",
                "sana-wm-bidirectional",
                "sana-wm-streaming",
                "lingbot-video-moe",
                "lingbot-world",
                "lingbot-world-v2",
                "fastwan21-t2v-1.3b",
                "wan22-t2v-nvfp4",
                "krea2-turbo",
                "krea2-raw",
                "ideogram4-fast",
                "ideogram4-instant",
                "longlive2-t2v",
                "longlive2-i2v",
                "fast-hunyuan",
                "turbowan21-t2v-1.3b",
                "helios-mid",
                "helios-distilled",
                "joy-echo",
                "cosmos3-edge-t2i",
                "cosmos3-super-t2v-cfg2tp2",
                "cosmos3-super-i2v",
                "cosmos3-super-t2i-distilled",
                "ltx25",
                "ltx25-diffusion-decoder",
            }
            self.assertTrue(expected.issubset(module.MODELS))

            eager_cmd = module.build_sglang_cmd("longcat-image")
            self.assertNotIn("--enable-torch-compile", eager_cmd)
            self.assertIn("--enable-prompt-rewrite=false", eager_cmd)
            self.assertIn("--quality=lossless", eager_cmd)

            compiled_cmd = module.build_sglang_cmd("longcat-image", torch_compile=True)
            self.assertIn("--enable-torch-compile", compiled_cmd)

            longcat_edit_cmd = module.build_sglang_cmd("longcat-image-edit")
            self.assertIn(
                "--model-path=meituan-longcat/LongCat-Image-Edit",
                longcat_edit_cmd,
            )
            self.assertTrue(
                any(arg.startswith("--image-path=") for arg in longcat_edit_cmd)
            )
            self.assertIn("--enable-prompt-rewrite=false", longcat_edit_cmd)

            longcat_edit_bcg_cmd = module.build_sglang_cmd(
                "longcat-image-edit", breakable_cuda_graph=True
            )
            resolution_index = longcat_edit_bcg_cmd.index("--warmup-resolutions")
            self.assertEqual(longcat_edit_bcg_cmd[resolution_index + 1], "1264x848")

            longcat_edit_turbo_cmd = module.build_sglang_cmd("longcat-image-edit-turbo")
            self.assertIn(
                "--model-path=meituan-longcat/LongCat-Image-Edit-Turbo",
                longcat_edit_turbo_cmd,
            )

            layered_cmd = module.build_sglang_cmd("qwen-image-layered")
            self.assertIn("--model-path=Qwen/Qwen-Image-Layered", layered_cmd)
            self.assertIn("--num-frames=4", layered_cmd)

            sd35_cmd = module.build_sglang_cmd("stable-diffusion-3.5-medium")
            self.assertIn(
                "--model-path=stabilityai/stable-diffusion-3.5-medium-diffusers",
                sd35_cmd,
            )
            self.assertIn("stable-diffusion-3.5-medium", module.GATED_MODELS)

            h3_cmd = module.build_sglang_cmd("minimax-h3-t2va", torch_compile=True)
            self.assertNotIn("--enable-torch-compile", h3_cmd)

            fastwan_cmd = module.build_sglang_cmd("fastwan21-t2v-1.3b")
            self.assertIn("--num-frames=61", fastwan_cmd)
            self.assertIn("--num-inference-steps=3", fastwan_cmd)
            self.assertIn("--dit-layerwise-offload=false", fastwan_cmd)

            wan_nvfp4_cmd = module.build_sglang_cmd("wan22-t2v-nvfp4")
            self.assertIn(
                "--model-path=nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4",
                wan_nvfp4_cmd,
            )
            self.assertIn("--num-frames=81", wan_nvfp4_cmd)
            self.assertIn("--dit-layerwise-offload=false", wan_nvfp4_cmd)
            self.assertEqual(module.required_gpus_for_model("wan22-t2v-nvfp4"), 1)

            krea_raw_cmd = module.build_sglang_cmd("krea2-raw")
            self.assertIn("--num-inference-steps=50", krea_raw_cmd)
            self.assertIn("--guidance-scale=4.5", krea_raw_cmd)

            cosmos_i2v_cmd = module.build_sglang_cmd("cosmos3-super-i2v")
            self.assertIn(
                "--model-path=nvidia/Cosmos3-Super-Image2Video", cosmos_i2v_cmd
            )
            self.assertIn("--num-gpus=2", cosmos_i2v_cmd)
            self.assertIn("--tp-size=2", cosmos_i2v_cmd)
            self.assertIn("--num-frames=81", cosmos_i2v_cmd)

            cosmos_cfg_cmd = module.build_sglang_cmd("cosmos3-super-t2v-cfg2tp2")
            self.assertIn("--model-path=nvidia/Cosmos3-Super", cosmos_cfg_cmd)
            self.assertIn("--num-gpus=4", cosmos_cfg_cmd)
            self.assertIn("--tp-size=2", cosmos_cfg_cmd)

            sana_wm_dense_cmd = module.build_sglang_cmd("sana-wm-bidirectional")
            self.assertIn(
                "--model-path=Efficient-Large-Model/SANA-WM_bidirectional",
                sana_wm_dense_cmd,
            )
            self.assertIn("--num-inference-steps=20", sana_wm_dense_cmd)
            self.assertNotIn("--streaming", sana_wm_dense_cmd)

            sana_wm_streaming_cmd = module.build_sglang_cmd("sana-wm-streaming")
            self.assertIn("--streaming", sana_wm_streaming_cmd)
            self.assertIn("--refiner-chunked", sana_wm_streaming_cmd)
            self.assertIn("--action=w-16,wl-16,l-16", sana_wm_streaming_cmd)

            lingbot_world_cmd = module.build_sglang_cmd("lingbot-world")
            self.assertIn(
                "--model-path=robbyant/lingbot-world-fast-diffusers",
                lingbot_world_cmd,
            )
            self.assertIn("--num-frames=9", lingbot_world_cmd)
            self.assertIn("--warmup-mode=off", lingbot_world_cmd)
            self.assertIn("--config=", " ".join(lingbot_world_cmd))
            self.assertEqual(
                module.MODELS["lingbot-world"]["config_overrides"]["actions"],
                [["w"] for _ in range(9)],
            )

            lingbot_world_v2_cmd = module.build_sglang_cmd("lingbot-world-v2")
            self.assertIn(
                "--model-path=robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
                lingbot_world_v2_cmd,
            )
            self.assertIn("--num-frames=9", lingbot_world_v2_cmd)
            self.assertIn("--num-inference-steps=4", lingbot_world_v2_cmd)

            ideogram_cmd = module.build_sglang_cmd("ideogram4-instant")
            self.assertFalse(
                any(arg.startswith("--num-inference-steps") for arg in ideogram_cmd)
            )

            longlive_i2v_cmd = module.build_sglang_cmd("longlive2-i2v")
            self.assertIn("--num-frames=61", longlive_i2v_cmd)
            self.assertTrue(
                any(arg.startswith("--image-path=") for arg in longlive_i2v_cmd)
            )

            joy_echo_cmd = module.build_sglang_cmd("joy-echo")
            self.assertIn("--num-gpus=2", joy_echo_cmd)
            self.assertIn("--ulysses-degree=2", joy_echo_cmd)
            config_arg = next(
                arg for arg in joy_echo_cmd if arg.startswith("--config=")
            )
            config = json.loads(Path(config_arg.removeprefix("--config=")).read_text())
            self.assertFalse(config["enable_memory_bank"])

    def test_quality_and_bcg_comparators_are_explicit_and_exclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = _load_benchmark_module(Path(tmpdir))

            high_cmd = module.build_sglang_cmd("longcat-image", quality="high")
            self.assertIn("--quality=high", high_cmd)
            self.assertNotIn("--enable-breakable-cuda-graph", high_cmd)

            extra_high_cmd = module.build_sglang_cmd(
                "longcat-image", quality="extra-high"
            )
            self.assertIn("--quality=extra-high", extra_high_cmd)
            self.assertNotIn("--enable-breakable-cuda-graph", extra_high_cmd)

            bcg_cmd = module.build_sglang_cmd(
                "longcat-image",
                breakable_cuda_graph=True,
                bcg_text_buckets=[256, 512],
            )
            self.assertIn("--enable-breakable-cuda-graph", bcg_cmd)
            self.assertEqual(
                bcg_cmd[bcg_cmd.index("--warmup-resolutions") + 1], "1024x1024"
            )
            bucket_index = bcg_cmd.index("--bcg-text-buckets")
            self.assertEqual(
                bcg_cmd[bucket_index + 1 : bucket_index + 3], ["256", "512"]
            )

            sana_video_bcg_cmd = module.build_sglang_cmd(
                "sana-video", breakable_cuda_graph=True
            )
            self.assertEqual(
                sana_video_bcg_cmd[sana_video_bcg_cmd.index("--warmup-num-frames") + 1],
                "17",
            )

            for _, quality, breakable_cuda_graph in module.QUALITY_BCG_ABBA_MATRIX:
                module.build_sglang_cmd(
                    "longcat-image",
                    quality=quality,
                    breakable_cuda_graph=breakable_cuda_graph,
                    bcg_text_buckets=[256, 512] if breakable_cuda_graph else None,
                )

            with self.assertRaisesRegex(ValueError, "comparators"):
                module.build_sglang_cmd(
                    "longcat-image",
                    torch_compile=True,
                    breakable_cuda_graph=True,
                )
            with self.assertRaisesRegex(ValueError, "requires"):
                module.build_sglang_cmd("longcat-image", bcg_text_buckets=[256])

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

    def test_isolated_cache_seeds_read_only_hf_cache_with_writable_overlay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            seed_root = temp_root / "shared-hf"
            source_model = seed_root / "hub" / "models--org--model"
            source_weight = source_model / "snapshots" / "abc" / "model.safetensors"
            source_weight.parent.mkdir(parents=True)
            source_weight.write_bytes(b"shared weights")
            source_ref = source_model / "refs" / "main"
            source_ref.parent.mkdir()
            source_ref.write_text("abc")

            cache_root = temp_root / "model-caches"
            cache_dir = module._prepare_model_cache(
                cache_root,
                "sana-video",
                "baseline",
                seed_model_cache_roots=[seed_root],
            )
            seeded_model = cache_dir / "huggingface" / "hub" / "models--org--model"
            self.assertTrue(seeded_model.is_dir())
            self.assertFalse(seeded_model.is_symlink())
            seeded_weight = seeded_model / "snapshots" / "abc" / "model.safetensors"
            self.assertTrue(seeded_weight.is_symlink())
            self.assertEqual(
                seeded_weight.read_bytes(),
                b"shared weights",
            )

            new_blob = seeded_model / "blobs" / "downloaded"
            new_blob.parent.mkdir(exist_ok=True)
            new_blob.write_bytes(b"new download")
            (seeded_model / "refs" / "main").write_text("new-revision")
            self.assertEqual(new_blob.read_bytes(), b"new download")
            self.assertEqual(source_ref.read_text(), "abc")

            module._cleanup_model_cache(
                cache_root,
                cache_dir,
                temp_root / "cleanup.jsonl",
                "sana-video",
                "baseline",
                "success",
            )
            self.assertFalse(cache_dir.exists())
            self.assertEqual(source_weight.read_bytes(), b"shared weights")

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

    def test_zero_exit_without_artifacts_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            output_dir = temp_root / "outputs"
            output_dir.mkdir()

            with patch.object(module.subprocess, "Popen") as popen:
                popen.return_value.stdout = iter(())
                popen.return_value.wait.return_value = 0
                result = module._run_benchmark_once_impl(
                    "sana-video",
                    "missing-artifacts",
                    output_dir,
                    warmup=False,
                    cuda_visible_devices="0",
                )

            command = popen.call_args.args[0]
            self.assertIn("--output-path", command)
            self.assertIn("--output-file-name", command)
            self.assertTrue(result["error"])
            self.assertEqual(
                result["missing_artifacts"], ["perf dump", "generated output"]
            )

    def test_mesh_artifacts_are_accepted_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            output_dir = temp_root / "outputs"
            output_dir.mkdir()

            def finish_run():
                (output_dir / "hunyuan3d-shape_mesh-output.json").write_text(
                    json.dumps({"total_duration_ms": 1000, "steps": []}),
                    encoding="utf-8",
                )
                (output_dir / "hunyuan3d-shape-mesh-output.obj").write_bytes(
                    b"v 0 0 0\n"
                )
                return 0

            with patch.object(module.subprocess, "Popen") as popen:
                popen.return_value.stdout = iter(())
                popen.return_value.wait.side_effect = finish_run
                result = module._run_benchmark_once_impl(
                    "hunyuan3d-shape",
                    "mesh-output",
                    output_dir,
                    warmup=False,
                    cuda_visible_devices="0",
                )

            self.assertFalse(result["error"])
            self.assertEqual(
                result["output_artifacts"],
                [str(output_dir / "hunyuan3d-shape-mesh-output.obj")],
            )
            self.assertEqual(len(result["output_sha256"]), 1)

    def test_high_bcg_rejects_quality_fusion_mounted_after_capture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            output_dir = temp_root / "outputs"
            output_dir.mkdir()

            with patch.object(module.subprocess, "Popen") as popen:
                popen.return_value.stdout = iter(
                    (
                        "[Diffusion BCG] captured 3 segment(s)\n",
                        "Mounted LTX-2 fused RMSNorm+modulate for quality=high\n",
                    )
                )
                popen.return_value.wait.return_value = 0
                result = module._run_benchmark_once_impl(
                    "longcat-image",
                    "bcg-high",
                    output_dir,
                    warmup=False,
                    quality="high",
                    breakable_cuda_graph=True,
                    cuda_visible_devices="0",
                )

            self.assertTrue(result["error"])
            self.assertEqual(
                result["bcg_invalid_signals"],
                [module.BCG_LATE_QUALITY_FUSION_SIGNAL],
            )

    def test_extra_high_bcg_rejects_quality_fusion_mounted_after_capture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            output_dir = temp_root / "outputs"
            output_dir.mkdir()

            with patch.object(module.subprocess, "Popen") as popen:
                popen.return_value.stdout = iter(
                    (
                        "[Diffusion BCG] captured 3 segment(s)\n",
                        "Mounted Qwen fused added-QKV for quality=extra-high\n",
                    )
                )
                popen.return_value.wait.return_value = 0
                result = module._run_benchmark_once_impl(
                    "longcat-image",
                    "bcg-extra-high",
                    output_dir,
                    warmup=False,
                    quality="extra-high",
                    breakable_cuda_graph=True,
                    cuda_visible_devices="0",
                )

            self.assertTrue(result["error"])
            self.assertEqual(
                result["bcg_invalid_signals"],
                [module.BCG_LATE_QUALITY_FUSION_SIGNAL],
            )

    def test_quality_bcg_matrix_reuses_one_gpu_set_and_cleans_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            module = _load_benchmark_module(temp_root)
            cache_root = temp_root / "model-caches"
            output_dir = temp_root / "outputs"
            output_dir.mkdir()
            calls = []

            def fake_run(model_key, label, _output_dir, **kwargs):
                calls.append((model_key, label, kwargs))
                cache_dir = kwargs["model_cache_dir"]
                weight_path = cache_dir / "hub" / "model.safetensors"
                weight_path.parent.mkdir(parents=True, exist_ok=True)
                weight_path.write_bytes(b"weights")
                return {"model": model_key, "label": label, "error": False}

            with patch.object(module, "_run_benchmark_once_impl", side_effect=fake_run):
                results = module.run_quality_bcg_matrix(
                    "sana-video",
                    "h200",
                    output_dir,
                    model_cache_root=cache_root,
                    cleanup_model_cache=True,
                )

            self.assertEqual(len(results), 12)
            self.assertEqual(
                [
                    (call[2]["quality"], call[2]["breakable_cuda_graph"])
                    for call in calls
                ],
                [
                    (quality, breakable_cuda_graph)
                    for _, quality, breakable_cuda_graph in module.QUALITY_BCG_ABBA_MATRIX
                ],
            )
            self.assertEqual({call[2]["cuda_visible_devices"] for call in calls}, {"0"})
            self.assertEqual(
                {call[2]["model_cache_dir"] for call in calls},
                {calls[0][2]["model_cache_dir"]},
            )
            self.assertFalse(calls[0][2]["model_cache_dir"].exists())
            ledger = json.loads(
                (output_dir / "cleanup.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(ledger["exit_reason"], "success")
            self.assertEqual(ledger["before"]["weight_file_count"], 1)
            self.assertEqual(ledger["after"]["weight_file_count"], 0)

    def test_quality_bcg_matrix_rejects_output_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = _load_benchmark_module(Path(tmpdir))
            results = [
                {
                    "quality": "lossless",
                    "breakable_cuda_graph": False,
                    "output_sha256": ["eager"],
                    "error": False,
                },
                {
                    "quality": "lossless",
                    "breakable_cuda_graph": True,
                    "output_sha256": ["bcg"],
                    "error": False,
                },
            ]

            module._validate_quality_bcg_output_hashes(results)

            self.assertFalse(results[0]["error"])
            self.assertTrue(results[1]["error"])
            self.assertEqual(
                results[1]["output_hash_error"],
                "BCG lossless output hash differs from eager",
            )


if __name__ == "__main__":
    unittest.main()
