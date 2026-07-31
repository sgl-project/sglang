# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
    BagelEditPipelineConfig,
    BagelPipelineConfig,
    BagelThinkingPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.bagel import BagelSamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.bagel_pipeline import (
    BagelEditPipeline,
    BagelPipeline,
    BagelThinkingPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel import (
    BagelBeforeDenoisingStage,
    BagelEditBeforeDenoisingStage,
    BagelInputValidationStage,
    BagelThinkingBeforeDenoisingStage,
    validate_bagel_special_tokens,
)


class _FakeTokenizer:
    token_ids = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
    }
    unk_token_id = 0
    unk_token = "<unk>"

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_ids.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.token_ids:
            return [self.token_ids[text]]
        if text == "first":
            return [101]
        if text == "second prompt":
            return [201, 202]
        return [17, 23]


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_calls = []
        self.pack_context_calls = []

    def build_context(
        self,
        conditional_input_ids,
        unconditional_input_ids,
        *,
        height,
        width,
        start_of_image_token_id,
        end_of_image_token_id,
    ):
        context = SimpleNamespace(
            conditional_input_ids=conditional_input_ids.clone(),
            unconditional_input_ids=unconditional_input_ids,
            height=height,
            width=width,
            start_of_image_token_id=start_of_image_token_id,
            end_of_image_token_id=end_of_image_token_id,
        )
        self.context_calls.append(context)
        return context

    def pack_contexts(self, contexts):
        packed = SimpleNamespace(
            contexts=list(contexts),
            batch_size=len(contexts),
            height=contexts[0].height,
            width=contexts[0].width,
            start_of_image_token_id=contexts[0].start_of_image_token_id,
            end_of_image_token_id=contexts[0].end_of_image_token_id,
        )
        self.pack_context_calls.append(packed)
        return packed


def _server_args(config: BagelPipelineConfig | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=config or BagelPipelineConfig(),
        enable_cfg_parallel=False,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        use_fsdp_inference=False,
        enable_torch_compile=False,
        dit_layerwise_offload=False,
        layerwise_offload_components=None,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        cache_dit_config=None,
        quantization=None,
        lora_path=None,
        comfyui_mode=False,
        revision=None,
        model_paths={},
        model_loaded={
            "transformer": True,
            "vae": True,
            "tokenizer": True,
            "scheduler": True,
        },
    )


class TestBagelBeforeDenoisingStage(unittest.TestCase):
    def _make_batch(self, seed: int = 42, steps: int = 4) -> Req:
        params = BagelSamplingParams(
            prompt="Doraemon is eating dorayaki",
            height=32,
            width=48,
            seed=seed,
            num_inference_steps=steps,
            save_output=False,
        )
        batch = Req(sampling_params=params)
        batch.generator = [torch.Generator("cpu").manual_seed(seed)]
        batch.seeds = [seed]
        return batch

    def _make_stage(self):
        stage = BagelBeforeDenoisingStage.__new__(BagelBeforeDenoisingStage)
        stage.transformer = _FakeTransformer()
        stage.tokenizer = _FakeTokenizer()
        stage.scheduler = FlowMatchEulerDiscreteScheduler(
            shift=1.0,
            preserve_sample_dtype=True,
        )
        stage._special_token_ids = None
        stage._component_residency_manager = None
        stage._registered_stage_name = "bagel_before_denoising_stage"
        stage.server_args = _server_args()
        return stage

    def test_input_validation_normalizes_generic_cuda_default_to_cpu(self) -> None:
        batch = self._make_batch(seed=42)
        batch.generator_device = "cuda"
        batch.generator = []

        BagelInputValidationStage()._generate_seeds(batch, _server_args())

        self.assertEqual(batch.generator_device, "cpu")
        self.assertEqual(len(batch.generator), 1)
        self.assertEqual(batch.generator[0].device.type, "cpu")
        self.assertEqual(batch.generator[0].initial_seed(), 42)

    def test_scheduler_sample_dtype_preservation_is_opt_in(self) -> None:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
        scheduler.set_timesteps(
            sigmas=[1.0],
            timesteps=[1.0],
            device="cpu",
        )
        sample = torch.ones(2, dtype=torch.float32)
        model_output = torch.ones(2, dtype=torch.bfloat16)

        next_latents = scheduler.step(
            model_output=model_output,
            timestep=scheduler.timesteps[0],
            sample=sample,
            return_dict=False,
        )[0]

        self.assertFalse(scheduler.config.preserve_sample_dtype)
        self.assertEqual(next_latents.dtype, torch.bfloat16)

    def test_exact_n_step_shifted_schedule_and_request_isolation(self) -> None:
        stage = self._make_stage()
        args = _server_args()
        batch_a = self._make_batch(seed=42, steps=4)
        batch_b = self._make_batch(seed=7, steps=4)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            output_a = stage.forward(batch_a, args)
            output_b = stage.forward(batch_b, args)

        expected = BagelBeforeDenoisingStage.build_shifted_schedule(
            4, 3.0, torch.device("cpu")
        )
        torch.testing.assert_close(
            expected, torch.tensor([1.0, 0.9, 0.75, 0.5], dtype=torch.float32)
        )
        torch.testing.assert_close(output_a.timesteps.cpu(), expected)
        torch.testing.assert_close(
            output_a.scheduler.sigmas,
            torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0], dtype=torch.float32),
        )
        self.assertEqual(len(output_a.timesteps), 4)
        self.assertEqual(len(output_a.scheduler.sigmas), 5)
        self.assertEqual(output_a.scheduler.sigmas[-1].item(), 0.0)
        self.assertEqual(output_a.scheduler.shift, 1.0)
        self.assertTrue(output_a.scheduler.config.preserve_sample_dtype)
        self.assertIsNot(output_a.scheduler, stage.scheduler)
        self.assertIsNot(output_a.scheduler, output_b.scheduler)
        self.assertIsNot(
            output_a.extra["bagel_context"], output_b.extra["bagel_context"]
        )
        self.assertEqual(tuple(output_a.latents.shape), (6, 64))
        self.assertEqual(output_a.latents.dtype, torch.float32)
        self.assertEqual(output_a.raw_latent_shape, (1, 6, 64))
        self.assertFalse(output_a.do_classifier_free_guidance)
        context = output_a.extra["bagel_context"]
        self.assertEqual(context.start_of_image_token_id, 151652)
        self.assertEqual(context.end_of_image_token_id, 151653)
        model_output = torch.full_like(
            output_a.latents,
            1.0 / 3.0,
            dtype=torch.bfloat16,
        )
        expected_next_latents = (
            output_a.latents
            + (output_a.scheduler.sigmas[1] - output_a.scheduler.sigmas[0])
            * model_output
        )
        next_latents = output_a.scheduler.step(
            model_output=model_output,
            timestep=output_a.timesteps[0],
            sample=output_a.latents,
            return_dict=False,
        )[0]
        self.assertEqual(next_latents.dtype, torch.float32)
        torch.testing.assert_close(
            next_latents,
            expected_next_latents,
            rtol=0,
            atol=0,
        )

    def test_generator_from_input_validation_is_used_without_replacement(self) -> None:
        stage = self._make_stage()
        args = _server_args()
        first = self._make_batch(seed=123)
        second = self._make_batch(seed=123)
        first_generator = first.generator[0]

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            stage.forward(first, args)
            stage.forward(second, args)

        self.assertIs(first.generator[0], first_generator)
        torch.testing.assert_close(first.latents, second.latents)

    def test_taylorseer_state_is_request_owned(self) -> None:
        stage = self._make_stage()
        args = _server_args()
        first = self._make_batch(seed=11, steps=4)
        second = self._make_batch(seed=22, steps=4)
        first.enable_taylorseer = True
        second.enable_taylorseer = True

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            stage.forward(first, args)
            stage.forward(second, args)

        first_state = first.extra["bagel_taylorseer_context"]
        second_state = second.extra["bagel_taylorseer_context"]
        self.assertIsNot(first_state, second_state)
        self.assertIsNot(first_state.conditional, first_state.unconditional)
        self.assertIsNone(first_state.secondary_unconditional)
        self.assertEqual(first_state.num_steps, 4)
        self.assertEqual(first_state.conditional.num_layers, 28)

    def test_nonempty_negative_prompt_fails_fast(self) -> None:
        stage = self._make_stage()
        batch = self._make_batch()
        batch.negative_prompt = "low quality"

        with self.assertRaisesRegex(ValueError, "negative_prompt"):
            stage.forward(batch, _server_args())

    def test_invalid_scalar_seed_and_multiple_outputs_fail_fast(self) -> None:
        stage = self._make_stage()
        args = _server_args()
        mutations = (
            ("seed", [1], "one scalar seed"),
            ("num_outputs_per_prompt", 2, "num_outputs_per_prompt=1"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                batch = self._make_batch()
                setattr(batch, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    stage.forward(batch, args)

    def test_dynamic_batch_prefills_and_draws_each_request_in_order(self) -> None:
        stage = self._make_stage()
        batch = self._make_batch(seed=11)
        batch.prompt = ["first", "second prompt"]
        batch.extra["dynamic_batch_seeds"] = [11, 22]
        batch.seeds = [11, 22]
        batch.generator = [
            torch.Generator("cpu").manual_seed(seed) for seed in batch.seeds
        ]

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            output = stage.forward(batch, _server_args())

        transformer = stage.transformer
        self.assertEqual(len(transformer.context_calls), 2)
        self.assertEqual(len(transformer.pack_context_calls), 1)
        torch.testing.assert_close(
            transformer.context_calls[0].conditional_input_ids,
            torch.tensor([151644, 101, 151645]),
        )
        torch.testing.assert_close(
            transformer.context_calls[1].conditional_input_ids,
            torch.tensor([151644, 201, 202, 151645]),
        )
        self.assertIs(output.extra["bagel_context"], transformer.pack_context_calls[0])
        self.assertEqual(tuple(output.latents.shape), (2, 6, 64))
        self.assertEqual(output.raw_latent_shape, (2, 6, 64))
        self.assertEqual(output.n_tokens, 6)

        expected = torch.stack(
            [
                torch.randn(
                    6,
                    64,
                    generator=torch.Generator("cpu").manual_seed(seed),
                )
                for seed in (11, 22)
            ]
        )
        torch.testing.assert_close(output.latents, expected, rtol=0, atol=0)

    def test_single_item_prompt_list_uses_the_t2i_context_path(self) -> None:
        stage = self._make_stage()
        batch = self._make_batch(seed=11)
        batch.prompt = ["first"]

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            output = stage.forward(batch, _server_args())

        self.assertEqual(len(stage.transformer.context_calls), 1)
        self.assertEqual(len(stage.transformer.pack_context_calls), 0)
        torch.testing.assert_close(
            stage.transformer.context_calls[0].conditional_input_ids,
            torch.tensor([151644, 101, 151645]),
        )
        self.assertEqual(tuple(output.latents.shape), (6, 64))
        self.assertEqual(output.raw_latent_shape, (1, 6, 64))

    def test_dynamic_batch_metadata_must_align_with_prompts(self) -> None:
        stage = self._make_stage()
        args = _server_args()

        cases = (
            (
                lambda batch: setattr(batch, "prompt", ["first", "second prompt"]),
                "dynamic_batch_seeds metadata",
            ),
            (
                lambda batch: (
                    setattr(batch, "prompt", ["first", "second prompt"]),
                    batch.extra.update({"dynamic_batch_seeds": [42]}),
                ),
                "one integer per prompt",
            ),
            (
                lambda batch: (
                    setattr(batch, "prompt", ["first", "second prompt"]),
                    batch.extra.update({"dynamic_batch_seeds": [42, 7]}),
                ),
                "one validated seed per prompt",
            ),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                batch = self._make_batch()
                mutate(batch)
                with self.assertRaisesRegex(ValueError, message):
                    stage.forward(batch, args)

    def test_dynamic_batch_generator_seed_must_match_each_request(self) -> None:
        stage = self._make_stage()
        batch = self._make_batch(seed=11)
        batch.prompt = ["first", "second prompt"]
        batch.extra["dynamic_batch_seeds"] = [11, 22]
        batch.seeds = [11, 22]
        batch.generator = [
            torch.Generator("cpu").manual_seed(11),
            torch.Generator("cpu").manual_seed(99),
        ]

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            with self.assertRaisesRegex(ValueError, "at index 1"):
                stage.forward(batch, _server_args())

    def test_non_t2i_variants_reject_direct_batched_prompts(self) -> None:
        batch = self._make_batch()
        batch.prompt = ["first", "second prompt"]

        self.assertTrue(BagelBeforeDenoisingStage.allows_dynamic_batching)
        for stage_type in (
            BagelThinkingBeforeDenoisingStage,
            BagelEditBeforeDenoisingStage,
        ):
            with self.subTest(stage=stage_type.__name__):
                self.assertFalse(stage_type.allows_dynamic_batching)
                stage = stage_type.__new__(stage_type)
                with self.assertRaisesRegex(ValueError, "pure T2I only"):
                    stage._request_prompts(batch)


class TestBagelLoaderContract(unittest.TestCase):
    def test_default_scheduler_preserves_official_fp32_euler_state(self) -> None:
        pipeline = BagelPipeline.__new__(BagelPipeline)
        pipeline.model_path = "must-not-be-resolved"
        pipeline.memory_usages = {}
        modules = {
            "transformer": _FakeTransformer(),
            "vae": torch.nn.Identity(),
            "tokenizer": _FakeTokenizer(),
        }

        with patch.object(
            BagelPipeline,
            "_resolve_checkpoint",
            side_effect=AssertionError("snapshot resolution must not run"),
        ):
            loaded = pipeline.load_modules(_server_args(), modules)

        self.assertTrue(loaded["scheduler"].config.preserve_sample_dtype)

    def test_fully_injected_modules_do_not_resolve_snapshot(self) -> None:
        pipeline = BagelPipeline.__new__(BagelPipeline)
        pipeline.model_path = "must-not-be-resolved"
        pipeline.memory_usages = {}
        transformer = _FakeTransformer()
        tokenizer = _FakeTokenizer()
        modules = {
            "transformer": transformer,
            "vae": torch.nn.Identity(),
            "tokenizer": tokenizer,
            "scheduler": FlowMatchEulerDiscreteScheduler(shift=1.0),
        }

        with patch.object(
            BagelPipeline,
            "_resolve_checkpoint",
            side_effect=AssertionError("snapshot resolution must not run"),
        ):
            loaded = pipeline.load_modules(_server_args(), modules)

        self.assertEqual(set(loaded), set(modules))
        self.assertIs(loaded["transformer"], transformer)
        self.assertEqual(
            set(pipeline.memory_usages),
            {"transformer", "vae", "tokenizer", "scheduler"},
        )
        self.assertEqual(pipeline.memory_usages["transformer"], 0.0)

    def test_editing_fully_injected_modules_do_not_resolve_snapshot(self) -> None:
        pipeline = BagelEditPipeline.__new__(BagelEditPipeline)
        pipeline.model_path = "must-not-be-resolved"
        pipeline.memory_usages = {}
        modules = {
            "transformer": _FakeTransformer(),
            "vae": torch.nn.Identity(),
            "image_encoder": torch.nn.Identity(),
            "tokenizer": _FakeTokenizer(),
            "scheduler": FlowMatchEulerDiscreteScheduler(shift=1.0),
        }
        args = _server_args(BagelEditPipelineConfig())

        with patch.object(
            BagelEditPipeline,
            "_resolve_checkpoint",
            side_effect=AssertionError("snapshot resolution must not run"),
        ):
            loaded = pipeline.load_modules(args, modules)

        self.assertEqual(set(loaded), set(modules))
        self.assertEqual(
            set(pipeline.memory_usages),
            {"transformer", "vae", "image_encoder", "tokenizer", "scheduler"},
        )

    def test_thinking_fully_injected_modules_do_not_resolve_snapshot(self) -> None:
        pipeline = BagelThinkingPipeline.__new__(BagelThinkingPipeline)
        pipeline.model_path = "must-not-be-resolved"
        pipeline.memory_usages = {}
        modules = {
            "transformer": _FakeTransformer(),
            "vae": torch.nn.Identity(),
            "tokenizer": _FakeTokenizer(),
            "scheduler": FlowMatchEulerDiscreteScheduler(shift=1.0),
        }
        args = _server_args(BagelThinkingPipelineConfig())

        with patch.object(
            BagelThinkingPipeline,
            "_resolve_checkpoint",
            side_effect=AssertionError("snapshot resolution must not run"),
        ):
            loaded = pipeline.load_modules(args, modules)

        self.assertEqual(set(loaded), set(modules))
        self.assertTrue(args.pipeline_config.dit_config.load_lm_head)

    def test_runtime_capability_gates_report_all_invalid_modes(self) -> None:
        args = _server_args()
        args.enable_cfg_parallel = True
        args.enable_torch_compile = True
        args.dit_cpu_offload = True

        with self.assertRaisesRegex(
            ValueError,
            "CFG parallel.*DiT CPU offload.*torch.compile",
        ):
            BagelPipeline._validate_runtime_capabilities(args)

    def test_runtime_capability_allows_tp_two_and_rejects_unvalidated_sizes(
        self,
    ) -> None:
        args = _server_args()
        args.tp_size = 2

        BagelPipeline._validate_runtime_capabilities(args)

        args.tp_size = 3
        with self.assertRaisesRegex(ValueError, "TP size 3"):
            BagelPipeline._validate_runtime_capabilities(args)

    def test_checkpoint_requires_all_markers_and_bagel_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "bagel",
                        "architectures": ["BagelForConditionalGeneration"],
                        "llm_config": {"hidden_size": 1},
                    }
                ),
                encoding="utf-8",
            )
            (root / "llm_config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2",
                        "hidden_size": 3584,
                        "intermediate_size": 18944,
                        "num_hidden_layers": 28,
                        "num_attention_heads": 28,
                        "num_key_value_heads": 4,
                        "vocab_size": 152064,
                    }
                ),
                encoding="utf-8",
            )
            (root / "ema.safetensors").touch()

            with self.assertRaisesRegex(FileNotFoundError, "ae.safetensors"):
                BagelPipeline._validate_checkpoint(temp_dir)

            (root / "ae.safetensors").touch()
            config = BagelPipeline._validate_checkpoint(temp_dir)
            self.assertEqual(config["model_type"], "bagel")
            self.assertEqual(config["llm_config"]["hidden_size"], 3584)

            (root / "config.json").write_text(
                json.dumps({"model_type": "unrelated"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "BAGEL architecture"):
                BagelPipeline._validate_checkpoint(temp_dir)

    def test_legacy_config_fallback_still_requires_exact_llm_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text(
                '{"name": ["BAGEL-7B-MoT"],}', encoding="utf-8"
            )
            llm_config = {
                "model_type": "qwen2",
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "vocab_size": 152064,
            }
            (root / "llm_config.json").write_text(
                json.dumps(llm_config), encoding="utf-8"
            )
            (root / "ema.safetensors").touch()
            (root / "ae.safetensors").touch()

            config = BagelPipeline._validate_checkpoint(temp_dir)
            self.assertEqual(config["model_type"], "bagel")
            self.assertEqual(config["llm_config"], llm_config)

            llm_config["hidden_size"] = 4096
            (root / "llm_config.json").write_text(
                json.dumps(llm_config), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hidden_size"):
                BagelPipeline._validate_checkpoint(temp_dir)

    def test_legacy_config_fallback_rejects_unrecognized_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text(
                '{"name": ["not-bagel",],}', encoding="utf-8"
            )
            (root / "llm_config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2",
                        "hidden_size": 3584,
                        "intermediate_size": 18944,
                        "num_hidden_layers": 28,
                        "num_attention_heads": 28,
                        "num_key_value_heads": 4,
                        "vocab_size": 152064,
                    }
                ),
                encoding="utf-8",
            )
            (root / "ema.safetensors").touch()
            (root / "ae.safetensors").touch()

            with self.assertRaisesRegex(ValueError, "Invalid BAGEL config.json"):
                BagelPipeline._validate_checkpoint(temp_dir)

    def test_tokenizer_load_uses_validated_staging_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text(
                '{"name": ["BAGEL-7B-MoT"],}', encoding="utf-8"
            )
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
                encoding="utf-8",
            )
            llm_config = {
                "model_type": "qwen2",
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "vocab_size": 152064,
            }
            sentinel = object()

            def load_staged_tokenizer(staged_path: str, **kwargs):
                self.assertNotEqual(Path(staged_path), root)
                self.assertEqual(
                    json.loads((Path(staged_path) / "config.json").read_text()),
                    llm_config,
                )
                self.assertTrue((Path(staged_path) / "tokenizer.json").is_file())
                self.assertTrue((Path(staged_path) / "tokenizer_config.json").is_file())
                self.assertTrue(kwargs["local_files_only"])
                self.assertFalse(kwargs["trust_remote_code"])
                return sentinel

            with patch(
                "sglang.multimodal_gen.runtime.pipelines.bagel_pipeline."
                "AutoTokenizer.from_pretrained",
                side_effect=load_staged_tokenizer,
            ):
                tokenizer = BagelPipeline._load_tokenizer(
                    temp_dir, {"llm_config": llm_config}
                )

            self.assertIs(tokenizer, sentinel)

    def test_special_token_validation_rejects_wrong_checkpoint_id(self) -> None:
        tokenizer = _FakeTokenizer()
        tokenizer.token_ids = dict(tokenizer.token_ids)
        tokenizer.token_ids["<|vision_start|>"] = 99

        with self.assertRaisesRegex(ValueError, "checkpoint ID 151652"):
            validate_bagel_special_tokens(tokenizer)


if __name__ == "__main__":
    unittest.main()
