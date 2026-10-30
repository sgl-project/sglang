"""The MoE runner --lora-backend triton_v2 picks when --moe-runner-backend is auto."""

import inspect
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.lora_hook import resolve_lora_moe_runner_backend
from sglang.srt.arg_groups.model_override_base import ResolvedView
from sglang.srt.lora.lora_registry import LoRARef
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MOE_TARGETS = ["gate_proj", "up_proj", "down_proj"]
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _adapter(directory: str, name: str, target_modules) -> str:
    path = os.path.join(directory, name)
    os.makedirs(path)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump({"target_modules": target_modules, "r": 16, "lora_alpha": 32}, f)
    return path


def _resolve(
    *,
    lora_paths,
    quantization=None,
    is_fp4_experts=False,
    experts=128,
    moe_runner_backend="auto",
    lora_backend="triton_v2",
    enable_lora=None,
    lora_target_modules=None,
):
    """Run the pass on a record whose ModelConfig is the memoized fixture."""
    model_config = SimpleNamespace(
        quantization=quantization,
        is_fp4_experts=is_fp4_experts,
        hf_text_config=SimpleNamespace(num_experts=experts),
        hf_config=SimpleNamespace(),
    )
    record = SimpleNamespace(
        model_path="model",
        disable_hybrid_swa_memory=False,
        moe_runner_backend=moe_runner_backend,
        lora_backend=lora_backend,
        enable_lora=enable_lora,
        lora_paths=lora_paths,
        lora_target_modules=lora_target_modules,
        _model_config=model_config,
    )
    return resolve_lora_moe_runner_backend(ResolvedView(record))


class TestLoraMoeRunnerDefault(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.moe_adapter = _adapter(self._tmp.name, "moe", MOE_TARGETS)
        self.attention_adapter = _adapter(
            self._tmp.name, "attention", ATTENTION_TARGETS
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_family_default_vendor(self):
        cases = [
            (dict(quantization=None), "lora_cutedsl"),
            (dict(quantization="fp8"), "lora_cutedsl"),
            (dict(quantization="mxfp8"), "lora_cutedsl"),
            (dict(quantization="modelopt_fp4"), "lora_marlin"),
            (dict(quantization="fp8", is_fp4_experts=True), "lora_marlin"),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                self.assertEqual(
                    _resolve(lora_paths=[self.moe_adapter], **kwargs),
                    {"moe_runner_backend": expected},
                )

    def test_unserved_quantization_stays_auto(self):
        self.assertEqual(
            _resolve(lora_paths=[self.moe_adapter], quantization="awq"), {}
        )

    def test_explicit_runner_and_other_backends_untouched(self):
        for kwargs in (
            dict(moe_runner_backend="lora_triton"),
            dict(moe_runner_backend="flashinfer_trtllm"),
            dict(lora_backend="triton"),
            dict(enable_lora=False),
        ):
            with self.subTest(**kwargs):
                self.assertEqual(_resolve(lora_paths=[self.moe_adapter], **kwargs), {})

    def test_dense_models_and_adapters_off_the_experts_stay_auto(self):
        self.assertEqual(_resolve(lora_paths=[self.moe_adapter], experts=0), {})
        self.assertEqual(_resolve(lora_paths=[self.attention_adapter]), {})
        # The adapters' union decides, as the manager's target set does.
        self.assertEqual(
            _resolve(lora_paths=[self.attention_adapter, self.moe_adapter]),
            {"moe_runner_backend": "lora_cutedsl"},
        )

    def test_explicit_targets_win_over_the_adapters(self):
        self.assertEqual(
            _resolve(
                lora_paths=[self.moe_adapter], lora_target_modules=["q_proj", "o_proj"]
            ),
            {},
        )
        self.assertEqual(
            _resolve(lora_paths=[self.attention_adapter], lora_target_modules=["all"]),
            {"moe_runner_backend": "lora_cutedsl"},
        )
        self.assertEqual(
            _resolve(
                lora_paths=[],
                enable_lora=True,
                lora_target_modules=["gate_up_proj", "down_proj"],
            ),
            {"moe_runner_backend": "lora_cutedsl"},
        )

    def test_every_lora_paths_form(self):
        forms = [
            [f"moe={self.moe_adapter}"],
            {"moe": self.moe_adapter},
            [{"lora_name": "moe", "lora_path": self.moe_adapter}],
            [LoRARef(lora_name="moe", lora_path=self.moe_adapter)],
        ]
        for index, lora_paths in enumerate(forms):
            with self.subTest(form=index):
                self.assertEqual(
                    _resolve(lora_paths=lora_paths),
                    {"moe_runner_backend": "lora_cutedsl"},
                )

    def test_pipeline_runs_the_pass_before_the_auto_consumers(self):
        from sglang.srt.arg_groups.pipeline import run_resolution_pipeline

        source = inspect.getsource(run_resolution_pipeline)
        pass_at = source.index(
            "run_post_process_pass(server_args, resolve_lora_moe_runner_backend)"
        )
        for consumer in (
            "apply_muse_spark_env_defaults(",
            "run_hook(handle_model_specific_adjustments,",
            "run_hook(handle_moe_kernel_config,",
        ):
            self.assertLess(pass_at, source.index(consumer), consumer)


if __name__ == "__main__":
    unittest.main()
