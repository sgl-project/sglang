import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def load_auto_tune_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "python" / "sglang" / "auto_tune.py"
    spec = importlib.util.spec_from_file_location("sglang_auto_tune_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


auto_tune = load_auto_tune_module()
RuntimeInfo = auto_tune.RuntimeInfo
main = auto_tune.main
parse_args = auto_tune.parse_args
resolve_output_path = auto_tune.resolve_output_path
run_tuning = auto_tune.run_tuning
select_quick_search_space = auto_tune.select_quick_search_space
validate_output_config = auto_tune.validate_output_config


class FakeCommonUtils:
    @staticmethod
    def get_model_config(
        model_path, tp_size, ep_size, disable_shared_experts_fusion=False
    ):
        return {
            "architecture": "FakeMoeForCausalLM",
            "num_experts": 8,
            "topk": 2,
            "hidden_size": 1024,
            "shard_intermediate_size": 2048,
            "dtype": "torch.bfloat16",
            "block_shape": None,
        }

    @staticmethod
    def get_config_filename(*args, **kwargs):
        return "E=8,N=1024,device_name=Fake_GPU,dtype=fp8_w8a8.json"

    @staticmethod
    def get_configs_compute_bound():
        return [
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 3,
            },
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 64,
                "num_warps": 8,
                "num_stages": 5,
            },
        ]


class FakeTuner:
    @staticmethod
    def get_configs_compute_bound():
        return FakeCommonUtils.get_configs_compute_bound()

    @staticmethod
    def tune_fused_moe_triton(**kwargs):
        FakeTuner.kwargs = kwargs
        return {"output_file": kwargs["output_file"]}


class TestAutoTune(unittest.TestCase):
    def test_parse_tp_aliases_and_batch_sizes(self):
        args = parse_args(
            [
                "--model-path",
                "fake/model",
                "--tp",
                "2",
                "--batch-size",
                "1,8",
                "--batch-size",
                "32",
            ]
        )
        self.assertEqual(args.tp_size, 2)
        self.assertEqual(args.batch_sizes, [1, 8, 32])

        args = parse_args(["--model-path", "fake/model", "--tp-size", "4"])
        self.assertEqual(args.tp_size, 4)

    def test_invalid_tp_ep_fails_clearly(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_args(
                ["--model-path", "fake/model", "--tp-size", "3", "--ep-size", "2"]
            )
        self.assertIn("--tp-size must be divisible by --ep-size", stderr.getvalue())

    def test_output_path_uses_sglang_config_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = resolve_output_path(tmp_dir, "config.json", "3.4.0")

            self.assertEqual(
                output_path,
                Path(tmp_dir).resolve() / "configs" / "triton_3_4_0" / "config.json",
            )

    def test_quick_search_space_is_reduced_and_deterministic(self):
        search_space = []
        for block_m in [16, 32, 64, 128, 256]:
            search_space.append(
                {
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 3,
                }
            )

        selected = select_quick_search_space(search_space, max_configs=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["BLOCK_SIZE_M"], 16)
        self.assertEqual(selected[1]["BLOCK_SIZE_M"], 128)

    def test_dry_run_resolves_model_config_without_tuning(self):
        runtime = RuntimeInfo(
            device_name="Fake GPU",
            torch_version="2.test",
            cuda_version="12.test",
            triton_version="3.4.0",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()
            with (
                patch.object(
                    auto_tune, "_load_common_utils", return_value=FakeCommonUtils
                ),
                patch.object(auto_tune, "_collect_runtime_info", return_value=runtime),
                patch.object(
                    auto_tune,
                    "run_tuning",
                    side_effect=AssertionError("dry-run should not tune"),
                ),
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "--model-path",
                        "fake/model",
                        "--tp",
                        "1",
                        "--dtype",
                        "fp8_w8a8",
                        "--quick",
                        "--dry-run",
                        "--output-dir",
                        tmp_dir,
                    ]
                )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Dry run", output)
        self.assertIn("FakeMoeForCausalLM", output)
        self.assertIn("search: quick, 1 candidates", output)
        self.assertIn("runtime loader validation: skipped", output)
        self.assertIn("SGLANG_MOE_CONFIG_DIR=", output)
        self.assertIn("configs/triton_3_4_0", output)

    def test_run_tuning_passes_resolved_output_and_quick_search_space(self):
        runtime = RuntimeInfo(
            device_name="Fake GPU",
            torch_version="2.test",
            cuda_version="12.test",
            triton_version="3.4.0",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = parse_args(
                [
                    "--model-path",
                    "fake/model",
                    "--tp",
                    "1",
                    "--dtype",
                    "fp8_w8a8",
                    "--quick",
                    "--batch-size",
                    "1,8",
                    "--output-dir",
                    tmp_dir,
                ]
            )
            with (
                patch.object(
                    auto_tune, "_load_common_utils", return_value=FakeCommonUtils
                ),
                patch.object(auto_tune, "_load_tuner", return_value=FakeTuner),
                patch.object(auto_tune, "_collect_runtime_info", return_value=runtime),
                patch.object(auto_tune, "validate_output_config", return_value=[1, 8]),
            ):
                plan = auto_tune.resolve_tuning_plan(args)
                result = run_tuning(args, plan)

        self.assertEqual(result["output_file"], str(plan.output_file))
        self.assertTrue(result["load_validated"])
        self.assertEqual(result["load_validation_keys"], [1, 8])
        self.assertEqual(FakeTuner.kwargs["model"], "fake/model")
        self.assertEqual(FakeTuner.kwargs["batch_sizes"], [1, 8])
        self.assertEqual(FakeTuner.kwargs["output_file"], str(plan.output_file))
        self.assertEqual(len(FakeTuner.kwargs["search_space"]), 1)
        self.assertTrue(FakeTuner.kwargs["tune"])

    def test_validate_output_config_checks_runtime_loader_path(self):
        runtime = RuntimeInfo(
            device_name="Fake GPU",
            torch_version="2.test",
            cuda_version="12.test",
            triton_version="3.4.0",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = parse_args(
                [
                    "--model-path",
                    "fake/model",
                    "--tp",
                    "1",
                    "--dtype",
                    "fp8_w8a8",
                    "--batch-size",
                    "1,8",
                    "--output-dir",
                    tmp_dir,
                ]
            )
            with (
                patch.object(
                    auto_tune, "_load_common_utils", return_value=FakeCommonUtils
                ),
                patch.object(auto_tune, "_collect_runtime_info", return_value=runtime),
            ):
                plan = auto_tune.resolve_tuning_plan(args)

            plan.output_file.parent.mkdir(parents=True, exist_ok=True)
            plan.output_file.write_text(
                '{"1": {"BLOCK_SIZE_M": 16}, "8": {"BLOCK_SIZE_M": 32}}'
            )

            captured = {}

            def fake_get_moe_configs(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                captured["config_dir"] = auto_tune.os.environ.get(
                    "SGLANG_MOE_CONFIG_DIR"
                )
                return {1: {"BLOCK_SIZE_M": 16}, 8: {"BLOCK_SIZE_M": 32}}

            def cache_clear():
                captured["cache_clears"] = captured.get("cache_clears", 0) + 1

            fake_get_moe_configs.cache_clear = cache_clear

            class FakeRuntimeLoader:
                get_moe_configs = staticmethod(fake_get_moe_configs)

            with (
                patch.object(
                    auto_tune,
                    "_load_runtime_config_loader",
                    return_value=FakeRuntimeLoader,
                ),
                patch.object(
                    auto_tune,
                    "_install_runtime_server_args",
                    return_value=lambda: None,
                ),
            ):
                keys = validate_output_config(plan)

        self.assertEqual(keys, [1, 8])
        self.assertEqual(captured["config_dir"], str(plan.output_root))
        self.assertEqual(captured["args"][:3], (8, 1024, "fp8_w8a8"))
        self.assertEqual(captured["kwargs"]["per_channel_quant"], False)
        self.assertGreaterEqual(captured["cache_clears"], 2)

    def test_missing_ray_dependency_has_actionable_error(self):
        missing_ray = ModuleNotFoundError("No module named 'ray'", name="ray")

        with (
            patch.object(auto_tune.importlib, "import_module", side_effect=missing_ray),
            self.assertRaisesRegex(RuntimeError, "requires Ray"),
        ):
            auto_tune._load_tuner()

    def test_runtime_error_is_reported_without_traceback(self):
        stderr = io.StringIO()
        with (
            patch.object(
                auto_tune, "resolve_tuning_plan", side_effect=RuntimeError("boom")
            ),
            redirect_stderr(stderr),
        ):
            exit_code = main(["--model-path", "fake/model"])

        self.assertEqual(exit_code, 1)
        self.assertIn("error: boom", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
