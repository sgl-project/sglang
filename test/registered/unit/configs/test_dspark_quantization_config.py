"""Exercise DSpark draft quantization through real local configuration entry points.

No weights or accelerator kernels are loaded. The complete ServerArgs resolution
cases use a small Llama target to keep hardware-specific GLM DSA setup outside
this CPU suite; the GLM case separately exercises the ModelConfig entry point.
"""

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import GlmMoeDsaConfig

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.arg_groups.serving_hook import handle_missing_default_values
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import EnvField, envs
from sglang.srt.server_args import prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSparkQuantizationConfig(CustomTestCase):
    def setUp(self):
        # Resolution updates both os.environ and EnvField's explicit-None state.
        environ = dict(os.environ)
        fields = {
            name: field
            for klass in reversed(type(envs).__mro__)
            for name, field in vars(klass).items()
            if isinstance(field, EnvField)
        }
        none_flags = {name: field._set_to_none for name, field in fields.items()}

        def restore_environment():
            os.environ.clear()
            os.environ.update(environ)
            for name, value in none_flags.items():
                fields[name]._set_to_none = value

        self.addCleanup(restore_environment)
        envs.SGLANG_RAGGED_VERIFY_MODE.set("static")

        # prepare_server_args configures launcher logging; keep it test-local.
        root_logger = logging.getLogger()
        handlers, level = root_logger.handlers[:], root_logger.level

        def restore_logging():
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                if handler not in handlers:
                    handler.close()
            for handler in handlers:
                root_logger.addHandler(handler)
            root_logger.setLevel(level)

        self.addCleanup(restore_logging)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target = self.root / "target"
        self.draft = self.root / "draft"
        self.target.mkdir()
        self.draft.mkdir()
        self._write(
            self.target / "config.json",
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "dtype": "bfloat16",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 128,
            },
        )
        self._write(
            self.target / "quant_model_description.json",
            {"model.layers.0.self_attn.q_proj.weight": "W8A8"},
        )
        self.raw_draft = {
            "architectures": ["DSparkDraftModel"],
            "speculators_model_type": "dspark",
            "dtype": "bfloat16",
            "block_size": 8,
            "aux_hidden_state_layer_ids": [1, 3],
            "draft_vocab_size": 128,
            "mask_token_id": 127,
            "markov_rank": 4,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "transformer_layer_config": {
                "model_type": "qwen3",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 3,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "vocab_size": 128,
                "max_position_embeddings": 128,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
                "layer_types": ["full_attention"] * 3,
                "tie_word_embeddings": False,
            },
        }
        self._write(self.draft / "config.json", self.raw_draft)

    @staticmethod
    def _write(path, value):
        path.write_text(json.dumps(value))

    def _args(self, draft_quantization="unquant", dtype="auto", target=None):
        # cuda is only a configuration value. No CUDA platform or capability
        # is mocked, and the generic target resolution does not run kernels.
        argv = [
            "--model-path",
            str(target or self.target),
            "--device",
            "cuda",
            "--quantization",
            "modelslim",
            "--dtype",
            dtype,
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            str(self.draft),
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--random-seed",
            "42",
        ]
        if draft_quantization is not None:
            argv += ["--speculative-draft-model-quantization", draft_quantization]
        return prepare_server_args(argv)

    def _resolved_args(self, **kwargs):
        args = self._args(**kwargs)
        args.resolve_once()
        return args

    def _draft_config(self, args, path=None):
        # TpModelWorker supplies the draft path explicitly in production too.
        return ModelConfig.from_server_args(
            args, model_path=str(path or self.draft), is_draft_model=True
        )

    def test_unquant_isolates_bf16_draft_without_changing_target(self):
        for dtype in ("auto", "bfloat16"):
            with self.subTest(dtype=dtype):
                args = self._resolved_args(dtype=dtype)
                resolved = resolving_view(args)
                self.assertEqual(resolved.quantization, "modelslim")
                self.assertIsNone(resolved.speculative_draft_model_quantization)
                self.assertTrue(resolved._speculative_draft_quantization_explicitly_set)
                self.assertEqual(resolved.speculative_num_draft_tokens, 9)
                target = ModelConfig.from_server_args(args)
                before = target.hf_config.to_dict()
                draft = self._draft_config(args)

                self.assertEqual(target.quantization, "modelslim")
                self.assertEqual(target.dtype, torch.bfloat16)
                self.assertIsNone(draft.quantization)
                self.assertEqual(draft.dtype, torch.bfloat16)
                self.assertEqual(draft.hf_config.architectures, ["DSparkDraftModel"])
                self.assertEqual(draft.hf_config.target_layer_ids, [0, 2])
                self.assertTrue(draft.is_draft_quantization_explicit)
                self.assertEqual(target.hf_config.to_dict(), before)
                self.assertEqual(resolved.quantization, "modelslim")

    def test_omitted_draft_quantization_inherits_modelslim(self):
        args = self._resolved_args(draft_quantization=None)
        resolved = resolving_view(args)
        self.assertEqual(resolved.speculative_draft_model_quantization, "modelslim")
        self.assertFalse(resolved._speculative_draft_quantization_explicitly_set)
        draft = self._draft_config(args)
        self.assertEqual(draft.quantization, "modelslim")
        self.assertEqual(draft.dtype, torch.bfloat16)
        self.assertFalse(draft.is_draft_quantization_explicit)

    def test_unquant_does_not_force_bfloat16_over_explicit_dtype(self):
        args = self._resolved_args(dtype="float16")
        draft = self._draft_config(args)
        self.assertIsNone(draft.quantization)
        self.assertEqual(draft.dtype, torch.float16)
        target = ModelConfig.from_server_args(args)
        self.assertEqual(target.quantization, "modelslim")
        self.assertEqual(target.dtype, torch.float16)

    def test_unquant_does_not_suppress_draft_hf_quantization_metadata(self):
        for key in ("quantization_config", "compression_config"):
            with self.subTest(key=key):
                config = {**self.raw_draft, key: {"quant_method": "fp8"}}
                self._write(self.draft / "config.json", config)
                args = self._resolved_args()
                self.assertIsNone(
                    resolving_view(args).speculative_draft_model_quantization
                )
                self.assertEqual(self._draft_config(args).quantization, "fp8")
                self.assertEqual(
                    ModelConfig.from_server_args(args).quantization, "modelslim"
                )

    def test_unquant_does_not_suppress_standalone_hf_quantization_metadata(self):
        self._write(
            self.draft / "hf_quant_config.json",
            {"quantization": {"quant_algo": "FP8"}},
        )
        args = self._resolved_args()
        self.assertIsNone(resolving_view(args).speculative_draft_model_quantization)
        self.assertEqual(self._draft_config(args).quantization, "modelopt_fp8")

    def test_draft_does_not_autodetect_modelslim_description(self):
        self._write(
            self.draft / "quant_model_description.json",
            {"model.layers.0.self_attn.q_proj.weight": "W8A8"},
        )
        args = self._resolved_args()
        self.assertIsNone(self._draft_config(args).quantization)
        self.assertEqual(ModelConfig.from_server_args(args).quantization, "modelslim")

    def test_saved_draft_keeps_bfloat16_and_capture_ids_on_reload(self):
        args = self._resolved_args()
        draft = self._draft_config(args)
        saved = self.root / "saved"
        draft.hf_config.save_pretrained(saved)
        reloaded = self._draft_config(args, saved)
        self.assertIsNone(reloaded.quantization)
        self.assertEqual(reloaded.dtype, torch.bfloat16)
        self.assertEqual(reloaded.hf_config.target_layer_ids, [0, 2])
        self.assertEqual(reloaded.hf_config.aux_hidden_state_layer_ids, [1, 3])

    def test_glm_model_config_entry_accepts_isolated_dspark_draft(self):
        glm = self.root / "glm"
        GlmMoeDsaConfig(
            architectures=["GlmMoeDsaForCausalLM"],
            dtype="bfloat16",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=128,
            max_position_embeddings=128,
        ).save_pretrained(glm)
        self._write(
            glm / "quant_model_description.json",
            {"model.layers.0.self_attn.q_a_proj.weight": "W8A8"},
        )
        args = self._args(target=glm)
        # This is the actual default-resolution step. The remaining GLM DSA
        # pipeline queries hardware, so this case only claims ModelConfig
        # coverage; the other cases above cover the full generic pipeline.
        handle_missing_default_values(args)
        target = ModelConfig.from_server_args(args)
        draft = self._draft_config(args)
        self.assertEqual(target.hf_config.architectures, ["GlmMoeDsaForCausalLM"])
        self.assertEqual(target.quantization, "modelslim")
        self.assertEqual(target.dtype, torch.bfloat16)
        self.assertIsNone(draft.quantization)
        self.assertEqual(draft.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
