# SPDX-License-Identifier: Apache-2.0
"""Ovis batching, conditioning, and unsupported-feature admission contracts."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ovis_image import (
    OvisImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.ovis_image import OvisImageSamplingParams
from sglang.test.test_utils import CustomTestCase

# The diffusion lane provides Diffusers and native multimodal dependencies.


class TestOvisImageConfig(CustomTestCase):
    @torch.no_grad()
    def test_cfg_text_length_changes_preserve_negative_conditioning(self):
        """Both CFG branches must honor length changes without stale cached negatives."""
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

        from sglang.multimodal_gen.configs.pipeline_configs.ovis_image import (
            OVIS_IMAGE_SYSTEM_PROMPT,
        )
        from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
        from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
        from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
            TextEncodingStage,
        )
        from sglang.multimodal_gen.runtime.server_args import (
            server_args as server_args_module,
        )

        # A local HF tokenizer/model keeps this regression independent of checkpoint
        # downloads while exercising the real tokenization, encoding, and cache paths.
        words = list(
            dict.fromkeys(
                ["[PAD]", "[UNK]", *OVIS_IMAGE_SYSTEM_PROMPT.split(), "red", "blue"]
            )
        )
        tokenizer_backend = Tokenizer(
            WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
        )
        tokenizer_backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            chat_template="{{ messages[0]['content'] }}",
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            model_config = Qwen3Config(
                vocab_size=len(words),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                pad_token_id=0,
            )
            model_config._attn_implementation = "sdpa"
            encoder = Qwen3Model(model_config).eval()
        target_device = get_local_torch_device()
        encoder = encoder.to(target_device)
        encoder.uses_sglang_forward_context = False
        config = OvisImagePipelineConfig()
        args = SimpleNamespace(pipeline_config=config, encoder_parallel="replicate")
        cache = ConditioningCache(max_bytes=1024 * 1024)
        observed = []
        # Scope the diffusion configuration boundary and restore any existing value.
        with patch.object(server_args_module, "_global_server_args", args):
            stage = TextEncodingStage([encoder], [tokenizer])
            for length in (64, 256, 64, 1):
                batch = Req(
                    sampling_params=OvisImageSamplingParams(
                        prompt=" ".join(["red"] * 300),
                        negative_prompt=" ".join(["blue"] * 40),
                        max_sequence_length=length,
                    ),
                    max_sequence_length=length,
                )
                with cache.scope():
                    encoded = stage.forward(batch, args)
                for branch in ("prompt", "negative_prompt"):
                    actual = getattr(encoded, f"{branch}_embeds")[0]
                    self.assertEqual(tuple(actual.shape), (1, length, 16))
                    self.assertEqual(getattr(encoded, f"{branch}_seq_lens"), [[length]])
                    self.assertTrue(getattr(encoded, f"{branch}_embeds_mask")[0].all())
                    prompt = (
                        batch.prompt if branch == "prompt" else batch.negative_prompt
                    )
                    inputs = config.tokenize_prompt(
                        [prompt], tokenizer, {"max_length": length}
                    )
                    inputs = inputs.to(target_device)
                    reference = encoder(**inputs, use_cache=False).last_hidden_state
                    expected = (reference * inputs.attention_mask.unsqueeze(-1))[:, 28:]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                observed.append(encoded)
            for branch in ("prompt", "negative_prompt"):
                torch.testing.assert_close(
                    getattr(observed[0], f"{branch}_embeds")[0],
                    getattr(observed[2], f"{branch}_embeds")[0],
                    atol=0,
                    rtol=0,
                )
            # The preferred negative cache contains distinct lengths and actually
            # reuses the restored request; tensor equality above guards its content.
            self.assertEqual(cache.stats()["entries"], 3)
            self.assertGreaterEqual(cache.stats()["hits"], 1)
        cache.clear()

    def test_hf_encoder_numerics_survive_checkpoint_and_json_updates(self):
        from transformers import Qwen3Config

        from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig

        self.assertFalse(Qwen3TextConfig().preserve_hf_numerics)
        config = OvisImagePipelineConfig()
        encoder = config.text_encoder_configs[0]
        self.assertTrue(encoder.preserve_hf_numerics)
        encoder.update_model_arch(Qwen3Config().to_dict())
        self.assertTrue(encoder.preserve_hf_numerics)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.json"
            path.write_text(json.dumps({"text_encoder_configs": [{"prefix": "qwen3"}]}))
            config.load_from_json(str(path))
        config.update_config_from_dict(
            {"pipeline_config.vae_sp": False}, prefix="pipeline_config"
        )
        self.assertTrue(config.text_encoder_configs[0].preserve_hf_numerics)

    def test_request_dimensions_match_packed_latent_grid(self):
        """Ovis must reject dimensions that would be truncated or create empty latents."""
        for name in ("height", "width"):
            for value in (520, 8, 0, -16, True, 16.0):
                with (
                    self.subTest(dimension=name, value=value),
                    self.assertRaisesRegex(ValueError, f"{name}.*multiple of 16"),
                ):
                    OvisImageSamplingParams(**{name: value})
        # None remains available for the shared request-default resolution path.
        OvisImageSamplingParams(height=None, width=None)

    def test_non_square_requested_size_survives_latent_pack_and_unpack(self):
        config = OvisImagePipelineConfig()
        batch = OvisImageSamplingParams(height=80, width=112)
        latent_shape = config.prepare_latent_shape(batch, batch_size=2, num_frames=1)
        self.assertEqual(latent_shape, (2, 16, 10, 14))
        latent = torch.arange(2 * 16 * 10 * 14).reshape(latent_shape).float()
        packed = config.maybe_pack_latents(latent, batch_size=2, batch=batch)
        self.assertEqual(packed.shape, (2, 35, 64))
        unpacked = config.post_denoising_loop(packed, batch)
        torch.testing.assert_close(unpacked, latent, atol=0, rtol=0)
        vae_scale = config.vae_config.arch_config.vae_scale_factor
        self.assertEqual(
            tuple(size * vae_scale for size in unpacked.shape[-2:]),
            (batch.height, batch.width),
        )

    def test_parallel_cfg_preserves_official_bfloat16_arithmetic(self):
        """Equivalent formulas round differently; both CFG paths need the official order."""
        config = OvisImagePipelineConfig()
        batch = SimpleNamespace(
            do_classifier_free_guidance=True,
            cfg_normalization=0,
            guidance_rescale=0,
        )
        policy = config.cfg_policy.build(batch, {}, {}, {})
        positive = torch.tensor([1.1, 0.3, -0.78], dtype=torch.bfloat16)
        negative = torch.tensor([0.9, -0.4, 0.36], dtype=torch.bfloat16)
        expected = negative + 5 * (positive - negative)
        legacy = 5 * positive - 4 * negative
        self.assertFalse(torch.equal(expected, legacy))
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                actual = policy.combine(
                    [positive, negative], batch, 5, config, cfg_parallel=parallel
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_vae_parallel_decode_tracks_constructor_json_and_cli_controls(self):
        """Disabling VAE SP must also disable decode collectives after every config load."""
        config = OvisImagePipelineConfig()
        self.assertFalse(config.vae_config.use_parallel_decode)
        explicit = OvisImagePipelineConfig(vae_sp=True, vae_tiling=True)
        self.assertTrue(explicit.vae_config.use_parallel_decode)
        config.update_pipeline_config({"vae_sp": True, "vae_tiling": True})
        self.assertTrue(config.vae_config.use_parallel_decode)
        config.update_config_from_dict({"vae_sp": False})
        self.assertFalse(config.vae_config.use_parallel_decode)
        config.update_config_from_dict(
            {"pipeline_config.vae_sp": True}, prefix="pipeline_config"
        )
        self.assertTrue(config.vae_config.use_parallel_decode)
        config.vae_sp = False
        with patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "false"}):
            config.validate_server_args(self._normalized_server_args())
        self.assertFalse(config.vae_config.use_parallel_decode)

    def test_multiple_outputs_keep_both_cfg_branches_in_prompt_order(self):
        """Sample order [A, A, B, B] must also govern every conditioning field."""
        config = OvisImagePipelineConfig()
        positive = torch.arange(12).reshape(2, 3, 2).float()
        negative = positive + 100
        positive_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        negative_mask = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool)
        batch = SimpleNamespace(
            num_outputs_per_prompt=2,
            do_classifier_free_guidance=True,
            prompt_embeds=[positive],
            negative_prompt_embeds=[negative],
            prompt_embeds_mask=[positive_mask],
            negative_prompt_embeds_mask=[negative_mask],
            prompt_attention_mask=[positive_mask.long()],
            negative_attention_mask=[negative_mask.long()],
            prompt_seq_lens=[[3, 3]],
            negative_prompt_seq_lens=[[3, 3]],
        )
        config.expand_conditioning_to_sample_batch(batch)
        order = torch.tensor([0, 0, 1, 1])
        for prefix, original, mask, attention_name in (
            ("prompt", positive, positive_mask, "prompt_attention_mask"),
            ("negative_prompt", negative, negative_mask, "negative_attention_mask"),
        ):
            with self.subTest(branch=prefix):
                torch.testing.assert_close(
                    getattr(batch, f"{prefix}_embeds")[0], original[order]
                )
                torch.testing.assert_close(
                    getattr(batch, f"{prefix}_embeds_mask")[0], mask[order]
                )
                torch.testing.assert_close(
                    getattr(batch, attention_name)[0], mask.long()[order]
                )
                self.assertEqual(getattr(batch, f"{prefix}_seq_lens"), [[3] * 4])

    def test_no_cfg_and_single_output_do_not_require_negative_metadata(self):
        config = OvisImagePipelineConfig()
        embedding = torch.arange(6).reshape(1, 3, 2).float()
        batch = SimpleNamespace(
            num_outputs_per_prompt=2,
            do_classifier_free_guidance=False,
            prompt_embeds=[embedding],
            prompt_seq_lens=[[3]],
        )
        config.expand_conditioning_to_sample_batch(batch)
        torch.testing.assert_close(batch.prompt_embeds[0], embedding[[0, 0]])
        self.assertEqual(batch.prompt_seq_lens, [[3, 3]])
        batch.num_outputs_per_prompt = 1
        original = batch.prompt_embeds[0].clone()
        config.expand_conditioning_to_sample_batch(batch)
        torch.testing.assert_close(batch.prompt_embeds[0], original)

    def test_rope_uses_full_sequences_and_conditioning_dtype(self):
        """Scheduler latents stay replicated; text and image positions reach DiT whole."""
        config = OvisImagePipelineConfig()
        batch = SimpleNamespace(height=80, width=112)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                prompt = torch.zeros(4, 3, 8, dtype=dtype)
                positions = config.get_freqs_cis(
                    prompt, 112, 80, "cpu", lambda ids: ids, batch, [3] * 4
                )
                self.assertEqual(positions.dtype, dtype)
                torch.testing.assert_close(
                    positions[:3],
                    torch.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2]], dtype=dtype),
                )
                self.assertEqual(positions.shape, (3 + 5 * 7, 3))
                torch.testing.assert_close(
                    positions[-1], torch.tensor([0, 4, 6], dtype=dtype)
                )
                latent = torch.zeros(4, 35, 64, dtype=dtype)
                sharded, was_sharded = config.shard_latents_for_sp(batch, latent)
                torch.testing.assert_close(sharded, latent)
                self.assertFalse(was_sharded)
                self.assertEqual(config.get_latent_dtype(dtype), dtype)
        with self.assertRaisesRegex(ValueError, "fixed-length text"):
            config.get_freqs_cis(
                prompt, 112, 80, "cpu", lambda ids: ids, batch, [3, 2, 3, 3]
            )

    def test_request_rejects_unimplemented_conditioning_and_caches(self):
        """Unsupported requests must fail before silently producing a text-only image."""
        for kwargs, error in (
            ({"task_type": "I2I"}, "text-to-image"),
            ({"image_path": "input.png"}, "image or video"),
            ({"video_path": "input.mp4"}, "image or video"),
            ({"num_frames": 2}, "num_frames=1"),
            ({"enable_teacache": True}, "enable_teacache"),
            ({"enable_spectrum": True}, "enable_spectrum"),
            ({"enable_cache_dit": True}, "enable_cache_dit"),
            ({"max_sequence_length": 257}, "max_sequence_length"),
            ({"max_sequence_length": 0}, "max_sequence_length"),
            ({"max_sequence_length": True}, "max_sequence_length"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, error):
                OvisImageSamplingParams(**kwargs)
        # Quality is also used for numerical kernel choices; it alone is not a cache request.
        OvisImageSamplingParams(quality="high", max_sequence_length=1)
        OvisImageSamplingParams(max_sequence_length=256)
        OvisImageSamplingParams(max_sequence_length=None)

    @staticmethod
    def _normalized_server_args(**overrides):
        return SimpleNamespace(
            **{
                "model_path": "ATH-MaaS/Ovis-Image-7B",
                "comfyui_mode": False,
                "quantization": None,
                "component_quantizations": {},
                "nunchaku_config": None,
                "kv_cache_quant_config": SimpleNamespace(enabled=False),
                "lora_path": None,
                "enable_torch_compile": False,
                "enable_breakable_cuda_graph": False,
                "cache_dit_config": None,
                **overrides,
            }
        )

    def test_server_rejects_unimplemented_weights_and_execution_modes(self):
        config = OvisImagePipelineConfig()
        with patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "false"}):
            config.validate_server_args(self._normalized_server_args())
            for kwargs, error in (
                ({"model_path": "model.safetensors"}, "model directory"),
                ({"comfyui_mode": True}, "model directory"),
                ({"quantization": "fp8"}, "quantization"),
                ({"component_quantizations": {"transformer": "fp8"}}, "quantization"),
                ({"nunchaku_config": object()}, "quantization"),
                (
                    {"kv_cache_quant_config": SimpleNamespace(enabled=True)},
                    "quantization",
                ),
                ({"lora_path": "adapter"}, "LoRA"),
                ({"enable_torch_compile": True}, "torch.compile"),
                ({"enable_breakable_cuda_graph": True}, "CUDA graphs"),
                ({"cache_dit_config": {}}, "Cache-DiT"),
            ):
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    config.validate_server_args(self._normalized_server_args(**kwargs))
        with (
            patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "true"}),
            self.assertRaisesRegex(ValueError, "Cache-DiT"),
        ):
            config.validate_server_args(self._normalized_server_args())

    def test_server_rejects_incompatible_attention_head_partitions(self):
        config = OvisImagePipelineConfig()
        # A three-way DiT split fits its 24 heads, but not the Qwen encoder.
        with self.assertRaisesRegex(ValueError, "Qwen attention heads"):
            config.validate_server_args(self._normalized_server_args(tp_size=3))
        with self.assertRaisesRegex(ValueError, "DiT attention heads"):
            config.validate_server_args(
                self._normalized_server_args(tp_size=2, ulysses_degree=5)
            )
        for tp_size, ulysses_degree in ((1, 1), (2, 1), (4, 1), (2, 2), (1, 3)):
            with self.subTest(tp_size=tp_size, ulysses_degree=ulysses_degree):
                config.validate_server_args(
                    self._normalized_server_args(
                        tp_size=tp_size, ulysses_degree=ulysses_degree
                    )
                )

    def test_explicit_graph_request_is_rejected_before_server_normalization(self):
        """BCG normalization must not silently erase an unsupported Ovis request."""
        from sglang.multimodal_gen.configs.pipeline_configs.flux import (
            FluxPipelineConfig,
        )
        from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        with (
            patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "false"}),
            patch(
                "sglang.multimodal_gen.runtime.server_args.server_args.current_platform",
                CpuPlatform(),
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.auto_tune.current_platform",
                CpuPlatform(),
            ),
        ):
            kwargs = {
                "model_path": "ATH-MaaS/Ovis-Image-7B",
                "pipeline_config": OvisImagePipelineConfig(),
                "num_gpus": 1,
                "performance_mode": "manual",
                "attention_backend": "torch_sdpa",
            }
            with self.assertRaisesRegex(ValueError, "Ovis-Image.*CUDA graphs"):
                ServerArgs(**kwargs, enable_breakable_cuda_graph=True)
            defaults = ServerArgs(**kwargs, enable_breakable_cuda_graph=False)
            self.assertFalse(defaults.enable_breakable_cuda_graph)
            self.assertIsNone(defaults.nunchaku_config)
            # Keep the pre-existing warning/disable behavior for another model
            # whose config has not opted in to rejecting unsupported graphs.
            other = ServerArgs(
                model_path="black-forest-labs/FLUX.1-schnell",
                pipeline_config=FluxPipelineConfig(),
                num_gpus=1,
                performance_mode="manual",
                attention_backend="torch_sdpa",
                enable_breakable_cuda_graph=True,
            )
            self.assertFalse(other.enable_breakable_cuda_graph)


@unittest.skipUnless(
    os.environ.get("MODEL_PATH"), "set MODEL_PATH for the checkpoint tokenizer oracle"
)
class TestOvisImageCheckpointTokenizer(CustomTestCase):
    def test_hf_encoder_numerics_survive_actual_checkpoint_metadata(self):
        path = Path(os.environ["MODEL_PATH"]) / "text_encoder" / "config.json"
        encoder = OvisImagePipelineConfig().text_encoder_configs[0]
        encoder.update_model_arch(json.loads(path.read_text()))
        self.assertEqual(encoder.hidden_size, 2048)
        self.assertTrue(encoder.preserve_hf_numerics)

    def test_empty_and_truncated_prompts_match_official_chat_tokenization(self):
        """The public tokenizer fixes chat rendering and the 28-token prefix contract."""
        from diffusers.pipelines.ovis_image.pipeline_ovis_image import (
            OvisImagePipeline as ReferencePipeline,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["MODEL_PATH"], subfolder="tokenizer", local_files_only=True
        )
        prompts = ["", "A striped cat in a garden. " * 200]
        reference = SimpleNamespace(
            tokenizer=tokenizer,
            system_prompt=(
                "Describe the image by detailing the color, quantity, text, shape, size, "
                "texture, spatial relationships of the objects and background: "
            ),
        )
        messages = ReferencePipeline._get_messages(reference, prompts)
        expected = tokenizer(
            messages,
            padding="max_length",
            truncation=True,
            max_length=284,
            return_tensors="pt",
            add_special_tokens=False,
        )
        actual = OvisImagePipelineConfig().tokenize_prompt(
            prompts, tokenizer, {"max_length": 256}
        )
        torch.testing.assert_close(actual.input_ids, expected.input_ids)
        torch.testing.assert_close(actual.attention_mask, expected.attention_mask)
        self.assertEqual(actual.input_ids.shape, (2, 284))
        self.assertTrue((actual.input_ids[0, :28] == actual.input_ids[1, :28]).all())
        self.assertTrue(actual.attention_mask[1].all())
        self.assertFalse(actual.attention_mask[0].all())


if __name__ == "__main__":
    unittest.main()
