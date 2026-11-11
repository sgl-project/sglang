"""The MoE runner --lora-backend triton_v2 picks when --moe-runner-backend is auto."""

import inspect
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.lora_hook import (
    check_lora_moe_runner_args,
    resolve_lora_moe_runner_backend,
)
from sglang.srt.arg_groups.model_override_base import ResolvedView
from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MOE_TARGETS = ["gate_proj", "up_proj", "down_proj"]
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# compressed-tensors (weights, input_activations) quantization args.
BLOCK_FP8 = (
    dict(strategy="block", block_structure=[128, 128]),
    dict(strategy="group", group_size=128, dynamic=True),
)
CHANNEL_FP8 = (dict(strategy="channel"), dict(strategy="token", dynamic=True))
TENSOR_FP8 = (dict(strategy="tensor"), dict(strategy="tensor", dynamic=False))
# sglang-native (quant_method fp8) config.json quantization_config forms.
NATIVE_BLOCK_FP8 = {"quant_method": "fp8", "weight_block_size": [128, 128]}
# AutoFP8 keeps the head in BF16; no expert path can match that entry.
NATIVE_TENSOR_FP8 = {
    "quant_method": "fp8",
    "activation_scheme": "static",
    "ignored_layers": ["lm_head"],
}


def _compressed_tensors(*groups) -> dict:
    """config.json quantization_config with one group per (weights, activations)."""
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
    path = os.path.join(directory, name)
    os.makedirs(path)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump({"target_modules": target_modules, "r": 16, "lora_alpha": 32}, f)
    return path


def _model_config(
    quantization=None, is_fp4_experts=False, hf_quant_config=None, experts=128
):
    """The memoized ModelConfig fixture the hook reads."""
    return SimpleNamespace(
        quantization=quantization,
        is_fp4_experts=is_fp4_experts,
        hf_quant_config=hf_quant_config or {},
        hf_text_config=SimpleNamespace(num_experts=experts),
        hf_config=SimpleNamespace(),
    )


def _resolve(
    *,
    lora_paths,
    quantization=None,
    is_fp4_experts=False,
    hf_quant_config=None,
    experts=128,
    moe_runner_backend="auto",
    lora_backend="triton_v2",
    enable_lora=None,
    lora_target_modules=None,
):
    """Run the pass on a record whose ModelConfig is the memoized fixture."""
    record = SimpleNamespace(
        model_path="model",
        disable_hybrid_swa_memory=False,
        moe_runner_backend=moe_runner_backend,
        lora_backend=lora_backend,
        enable_lora=enable_lora,
        lora_paths=lora_paths,
        lora_target_modules=lora_target_modules,
        _model_config=_model_config(
            quantization, is_fp4_experts, hf_quant_config, experts
        ),
    )
    return resolve_lora_moe_runner_backend(ResolvedView(record))


def _explicit_runner(
    moe_runner_backend, targets=MOE_TARGETS, **model_config
) -> ServerArgs:
    """Server args that ask for a lora_* runner on an expert-targeted adapter."""
    args = ServerArgs(
        model_path="dummy",
        enable_lora=True,
        lora_backend="triton_v2",
        moe_runner_backend=moe_runner_backend,
        max_lora_rank=8,
        lora_target_modules=targets,
    )
    args._model_config = _model_config(**model_config)
    return args


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
            (
                dict(quantization="fp8", hf_quant_config=NATIVE_BLOCK_FP8),
                "lora_marlin",
            ),
            (dict(quantization="mxfp8"), "lora_marlin"),
            (dict(quantization="modelopt_fp4"), "lora_marlin"),
            (dict(quantization="fp8", is_fp4_experts=True), "lora_marlin"),
            (dict(quantization="w8a8_fp8"), "lora_marlin"),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                self.assertEqual(
                    _resolve(lora_paths=[self.moe_adapter], **kwargs),
                    {"moe_runner_backend": expected},
                )

    def test_fp8_families_default_to_marlin(self):
        """Every supported FP8 family serves W8A16 through Marlin by default
        (BF16 activations), like the FP4 families' W4A16; the resolution does
        not depend on the device."""
        cases = [
            dict(quantization="fp8", hf_quant_config=NATIVE_BLOCK_FP8),
            dict(quantization="mxfp8"),
            dict(quantization="w8a8_fp8"),
            dict(
                quantization="compressed-tensors",
                hf_quant_config=_compressed_tensors(BLOCK_FP8),
            ),
            dict(
                quantization="compressed-tensors",
                hf_quant_config=_compressed_tensors(CHANNEL_FP8),
            ),
            dict(
                quantization="compressed-tensors",
                hf_quant_config=_compressed_tensors(TENSOR_FP8),
            ),
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                self.assertEqual(
                    _resolve(lora_paths=[self.moe_adapter], **kwargs),
                    {"moe_runner_backend": "lora_marlin"},
                )

    def test_explicit_lora_runners_are_kept(self):
        """The pass never replaces an explicit lora_* runner by the family
        default, and every FP8 family still admits each runner it has a
        provider for behind it."""
        families = [
            dict(quantization="fp8", hf_quant_config=NATIVE_BLOCK_FP8),
            dict(quantization="mxfp8"),
            dict(quantization="w8a8_fp8"),
            dict(
                quantization="compressed-tensors",
                hf_quant_config=_compressed_tensors(CHANNEL_FP8),
            ),
            dict(
                quantization="compressed-tensors",
                hf_quant_config=_compressed_tensors(TENSOR_FP8),
            ),
        ]
        for kwargs in families:
            for backend in ("lora_cutedsl", "lora_triton", "lora_marlin"):
                with self.subTest(backend=backend, **kwargs):
                    self.assertEqual(
                        _resolve(
                            lora_paths=[self.moe_adapter],
                            moe_runner_backend=backend,
                            **kwargs,
                        ),
                        {},
                    )
                    check_lora_moe_runner_args(_explicit_runner(backend, **kwargs))

    def test_compressed_tensors_family_from_the_config_groups(self):
        cases = [
            ("block", _compressed_tensors(BLOCK_FP8), "lora_marlin"),
            ("channel", _compressed_tensors(CHANNEL_FP8), "lora_marlin"),
            ("tensor", _compressed_tensors(TENSOR_FP8), "lora_marlin"),
            (
                "tensor_dynamic",
                _compressed_tensors(
                    (dict(strategy="tensor"), dict(strategy="tensor", dynamic=True))
                ),
                "lora_marlin",
            ),
            # Mixed groups and non-FP8 weights stay auto.
            ("mixed", _compressed_tensors(BLOCK_FP8, CHANNEL_FP8), None),
            (
                "int4",
                _compressed_tensors(
                    (dict(type="int", num_bits=4, strategy="group", group_size=128), {})
                ),
                None,
            ),
            ("unquantized_config", {}, None),
        ]
        for name, hf_quant_config, backend in cases:
            with self.subTest(name):
                self.assertEqual(
                    _resolve(
                        lora_paths=[self.moe_adapter],
                        quantization="compressed-tensors",
                        hf_quant_config=hf_quant_config,
                    ),
                    {"moe_runner_backend": backend} if backend else {},
                )

    def test_native_fp8_needs_weight_blocks(self):
        """quant_method fp8 without weight_block_size (AutoFP8 checkpoints) and
        --quantization fp8 at load scale the experts per tensor, which no
        provider reads: both admission paths refuse; the block layout and
        the mixed FP8/MXFP4 checkpoints pass, and dense adapters never ask."""
        for name, hf_quant_config in (("autofp8", NATIVE_TENSOR_FP8), ("online", {})):
            with self.subTest(name):
                with self.assertRaisesRegex(
                    ValueError, "triton_v2 does not serve per-tensor FP8 MoE experts"
                ):
                    _resolve(
                        lora_paths=[self.moe_adapter],
                        quantization="fp8",
                        hf_quant_config=hf_quant_config,
                    )
                with self.assertRaisesRegex(
                    ValueError, "lora_triton does not serve per-tensor FP8 MoE experts"
                ):
                    check_lora_moe_runner_args(
                        _explicit_runner(
                            "lora_triton",
                            quantization="fp8",
                            hf_quant_config=hf_quant_config,
                        )
                    )
        check_lora_moe_runner_args(
            _explicit_runner(
                "lora_triton", quantization="fp8", hf_quant_config=NATIVE_BLOCK_FP8
            )
        )
        check_lora_moe_runner_args(
            _explicit_runner("lora_marlin", quantization="fp8", is_fp4_experts=True)
        )
        self.assertEqual(
            _resolve(lora_paths=[self.attention_adapter], quantization="fp8"), {}
        )

    def test_fp8_layer_exclusions_defer_to_the_layer_admission(self):
        """ignored_layers, modules_to_not_convert and SGLANG_FP8_IGNORED_LAYERS
        may leave the routed experts unquantized, which the global metadata
        cannot tell: both paths admit, and the layer admission settles each
        layer (TestDeferredLayerAdmission). Head and embedding entries name
        no expert path, so they keep the refusal."""
        ignored = {
            **NATIVE_TENSOR_FP8,
            "ignored_layers": ["lm_head", "model.layers.0.mlp.experts"],
        }
        not_converted = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["mlp.experts"],
        }
        for name, hf_quant_config, env in (
            ("ignored_layers", ignored, ""),
            ("modules_to_not_convert", not_converted, ""),
            ("env_layer", {}, "model.layers.3.mlp"),
            ("env_experts", NATIVE_TENSOR_FP8, "lm_head, experts"),
        ):
            with self.subTest(name), envs.SGLANG_FP8_IGNORED_LAYERS.override(env):
                self.assertEqual(
                    _resolve(
                        lora_paths=[self.moe_adapter],
                        quantization="fp8",
                        hf_quant_config=hf_quant_config,
                    ),
                    # Excluded experts load BF16; the engine serves those from
                    # the checkpoint layout whatever the runner (never tiles).
                    {"moe_runner_backend": "lora_marlin"},
                )
                check_lora_moe_runner_args(
                    _explicit_runner(
                        "lora_triton",
                        quantization="fp8",
                        hf_quant_config=hf_quant_config,
                    )
                )
        with envs.SGLANG_FP8_IGNORED_LAYERS.override("lm_head,model.embed_tokens"):
            with self.assertRaisesRegex(ValueError, "per-tensor FP8 MoE experts"):
                _resolve(lora_paths=[self.moe_adapter], quantization="fp8")

    def test_explicit_runner_without_routed_experts_skips_the_expert_check(self):
        """A dense per-tensor FP8 model with dense adapters keeps an explicitly
        selected, unused lora_* runner."""
        check_lora_moe_runner_args(
            _explicit_runner(
                "lora_triton", ATTENTION_TARGETS, quantization="fp8", experts=0
            )
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


class TestDeferredLayerAdmission(CustomTestCase):
    """What the hook defers to: family_of_layer binds experts a
    native FP8 config left unquantized as the bf16 family and refuses
    per-tensor FP8 experts, so a partial exclusion admits no such layer."""

    def test_excluded_experts_bind_as_bf16(self):
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.lora.moe.families import family_of_layer

        layer = torch.nn.Module()
        layer.quant_method = object.__new__(UnquantizedFusedMoEMethod)
        layer.w13_weight = torch.zeros(2, 256, 256, dtype=torch.bfloat16)
        layer.w2_weight = torch.zeros(2, 256, 128, dtype=torch.bfloat16)
        self.assertEqual(family_of_layer(layer), "bf16")

    def test_per_tensor_experts_are_refused(self):
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
        from sglang.srt.lora.moe.families import family_of_layer

        layer = torch.nn.Module()
        layer.quant_method = object.__new__(Fp8MoEMethod)
        layer.quant_method.block_quant = False
        layer.params_dtype = torch.bfloat16
        with self.assertRaisesRegex(
            NotImplementedError, "per-tensor layers are unsupported"
        ):
            family_of_layer(layer)


if __name__ == "__main__":
    unittest.main()
