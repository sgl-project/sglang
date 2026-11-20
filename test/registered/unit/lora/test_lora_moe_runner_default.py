"""The MoE runner --lora-backend triton_v2 picks when --moe-runner-backend is auto."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from transformers import PretrainedConfig

from sglang.srt.arg_groups.lora_hook import (
    check_lora_moe_runner_args,
    resolve_lora_moe_runner_backend,
)
from sglang.srt.arg_groups.model_override_base import ResolvedView
from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.moe.families import family_of_config
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="weekly", runner_config="cpu")

MOE_TARGETS = ["gate_proj", "up_proj", "down_proj"]
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# compressed-tensors (weights, input_activations) quantization args.
BLOCK_FP8 = (
    dict(strategy="block", block_structure=[128, 128]),
    dict(strategy="group", group_size=128, dynamic=True),
)
CHANNEL_FP8 = (dict(strategy="channel"), dict(strategy="token", dynamic=True))
TENSOR_FP8 = (dict(strategy="tensor"), dict(strategy="tensor", dynamic=False))


def _compressed_tensors(*groups) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "config_groups": {
            f"group_{index}": {
                "targets": ["Linear"],
                "weights": {"type": "float", "num_bits": 8, **weights},
                "input_activations": {"type": "float", "num_bits": 8, **activations},
            }
            for index, (weights, activations) in enumerate(groups)
        },
    }


def _adapter(directory: str, name: str, target_modules) -> str:
    path = Path(directory) / name
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"target_modules": target_modules})
    )
    return str(path)


def _args(
    moe_runner_backend="auto",
    lora_backend="triton_v2",
    *,
    quantization=None,
    resolved_quantization=None,
    is_fp4_experts=False,
    hf_quant_config=None,
    compression_config=None,
    hf_text_config=None,
    experts=128,
    **fields,
) -> ServerArgs:
    """`quantization` is the flag; ModelConfig resolves it (or the checkpoint's
    method when it is unset) to `resolved_quantization`, default the same."""
    args = ServerArgs(
        model_path="dummy",
        quantization=quantization,
        lora_backend=lora_backend,
        moe_runner_backend=moe_runner_backend,
        **fields,
    )
    text_config = (
        hf_text_config
        if hf_text_config is not None
        else PretrainedConfig(num_experts=experts)
    )
    args._model_config = SimpleNamespace(
        quantization=resolved_quantization or quantization,
        is_fp4_experts=is_fp4_experts,
        hf_quant_config=hf_quant_config or {},
        hf_text_config=text_config,
        hf_config=PretrainedConfig(
            quantization_config=hf_quant_config,
            text_config=text_config,
            compression_config=compression_config,
        ),
    )
    return args


def _resolve(**kwargs) -> dict:
    return resolve_lora_moe_runner_backend(ResolvedView(_args(**kwargs)))


def _check(backend, **kwargs):
    check_lora_moe_runner_args(_args(backend, enable_lora=True, **kwargs))


class TestLoraMoeRunnerDefault(CustomTestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.moe_adapter = _adapter(directory, "moe", MOE_TARGETS)
        self.attention_adapter = _adapter(directory, "attention", ATTENTION_TARGETS)

    def assert_runner(self, backend, **kwargs):
        self.assertEqual(
            _resolve(**kwargs), {"moe_runner_backend": backend} if backend else {}
        )

    def test_family_default_vendor(self):
        cases = [
            (dict(quantization=None), "lora_cutedsl"),
            (dict(quantization="fp8"), "lora_marlin"),
            (dict(quantization="mxfp8"), "lora_marlin"),
            # Packed NVFP4 experts leave MiniMax's override to pick the runner.
            (
                dict(
                    resolved_quantization="mxfp8",
                    hf_quant_config=dict(
                        quant_method="mxfp8",
                        moe_quant_algo="NVFP4",
                        moe_quant_format="nvfp4-pack-quantized",
                    ),
                ),
                None,
            ),
            (dict(quantization="modelopt_fp4"), "lora_marlin"),
            (dict(quantization="fp8", is_fp4_experts=True), "lora_marlin"),
            (dict(quantization="w8a8_fp8"), "lora_marlin"),
            (dict(quantization="awq"), None),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                self.assert_runner(expected, lora_paths=[self.moe_adapter], **kwargs)

    def test_compressed_tensors_family_from_the_config_groups(self):
        int4 = dict(type="int", num_bits=4, strategy="group", group_size=128)
        cases = (
            (_compressed_tensors(BLOCK_FP8), "fp8"),
            (_compressed_tensors(CHANNEL_FP8), "fp8_channel"),
            (_compressed_tensors(TENSOR_FP8), "fp8_tensor"),
            (_compressed_tensors(BLOCK_FP8, CHANNEL_FP8), None),
            (_compressed_tensors((int4, {})), None),
            ({}, None),
        )
        for config, family in cases:
            with self.subTest(config=config):
                args = _args(quantization="compressed-tensors", hf_quant_config=config)
                self.assertEqual(family_of_config(args._model_config), family)

    def test_compressed_tensors_defaults_to_marlin(self):
        for field, quant_args in (
            ("hf_quant_config", BLOCK_FP8),
            ("compression_config", TENSOR_FP8),
        ):
            with self.subTest(field=field):
                self.assert_runner(
                    "lora_marlin",
                    lora_paths=[self.moe_adapter],
                    quantization="compressed-tensors",
                    **{field: _compressed_tensors(quant_args)},
                )

    def test_explicit_runners_admit_the_served_quantizations(self):
        for kwargs in (
            dict(quantization="fp8"),
            dict(quantization="fp8", is_fp4_experts=True),
            dict(quantization="mxfp8"),
            dict(quantization="w8a8_fp8"),
            dict(quantization="compressed-tensors"),
            # ModelOpt's CLI name resolves to the checkpoint's NVFP4 method.
            dict(quantization="modelopt", resolved_quantization="modelopt_fp4"),
            dict(resolved_quantization="modelopt_mixed"),
        ):
            with self.subTest(**kwargs):
                _check("lora_cutedsl", **kwargs)
        # With no flag, the checkpoint's own method is checked.
        with self.assertRaisesRegex(ValueError, "cannot serve awq MoE weights"):
            _check("lora_cutedsl", resolved_quantization="awq")

    def test_gates_targets_and_lora_paths_forms(self):
        moe, attention = [self.moe_adapter], [self.attention_adapter]
        dense_nemotron_h = NemotronHConfig(layers_block_type=["mlp"])
        moe_nemotron_h = NemotronHConfig(layers_block_type=["moe"])
        dense_gemma4, moe_gemma4 = (
            PretrainedConfig(
                model_type="gemma4_text", enable_moe_block=enabled, num_experts=128
            )
            for enabled in (False, True)
        )
        cases = [
            (dict(lora_paths=moe, moe_runner_backend="lora_triton"), None),
            (dict(lora_paths=moe, lora_backend="triton"), None),
            (dict(lora_paths=moe, enable_lora=False), None),
            (dict(lora_paths=moe, experts=0), None),
            # Dense NemotronH (n_routed_experts defaults to 8) and Gemma4 configs.
            (dict(lora_paths=moe, hf_text_config=dense_nemotron_h), None),
            (dict(lora_paths=moe, hf_text_config=moe_nemotron_h), "lora_cutedsl"),
            (dict(lora_paths=moe, hf_text_config=dense_gemma4), None),
            (dict(lora_paths=moe, hf_text_config=moe_gemma4), "lora_cutedsl"),
            (dict(lora_paths=attention, quantization="fp8"), None),
            # The adapters' union decides, as the manager's target set does.
            (dict(lora_paths=attention + moe), "lora_cutedsl"),
            # --lora-target-modules wins over the adapters.
            (dict(lora_paths=moe, lora_target_modules=["q_proj", "o_proj"]), None),
            (dict(lora_paths=attention, lora_target_modules=["all"]), "lora_cutedsl"),
            (
                dict(
                    enable_lora=True, lora_target_modules=["gate_up_proj", "down_proj"]
                ),
                "lora_cutedsl",
            ),
            (dict(lora_paths=[f"moe={self.moe_adapter}"]), "lora_cutedsl"),
            (dict(lora_paths={"moe": self.moe_adapter}), "lora_cutedsl"),
            (
                dict(lora_paths=[{"lora_name": "moe", "lora_path": self.moe_adapter}]),
                "lora_cutedsl",
            ),
            (
                dict(lora_paths=[LoRARef(lora_name="moe", lora_path=self.moe_adapter)]),
                "lora_cutedsl",
            ),
        ]
        for index, (kwargs, expected) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_runner(expected, **kwargs)

    def test_adapter_without_a_target_list_does_not_decide_the_runner(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        for name, config in (("missing", {}), ("null", {"target_modules": None})):
            with self.subTest(name=name):
                path = Path(directory) / name
                path.mkdir()
                (path / "adapter_config.json").write_text(json.dumps(config))
                self.assert_runner(None, lora_paths=[str(path)])
                self.assert_runner(
                    "lora_cutedsl", lora_paths=[str(path), self.moe_adapter]
                )


if __name__ == "__main__":
    unittest.main()
