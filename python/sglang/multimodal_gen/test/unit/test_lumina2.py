# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Lumina-Image-2.0 config/weight plumbing.

These run without a GPU or a checkpoint download. They cover the failure modes
that are invisible until a real load: architecture fields silently not picked up
from the checkpoint's config.json, and weight-name mappings that resolve to
params the model does not have.

LUMINA2_CHECKPOINT_CONFIG below is verbatim
Alpha-VLLM/Lumina-Image-2.0 transformer/config.json.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.multimodal_gen.configs.models.dits.lumina2 import (
    Lumina2ArchConfig,
    Lumina2Config,
)
from sglang.multimodal_gen.configs.pipeline_configs.lumina2 import (
    LUMINA2_MAX_TEXT_LEN,
    LUMINA2_SYSTEM_PROMPT,
    Lumina2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lumina2 import Lumina2SamplingParams
from sglang.multimodal_gen.registry import (
    _PIPELINE_REGISTRY,
    _discover_and_register_pipelines,
    _get_config_info,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.models.dits.lumina2 import (
    ADALN_EMBED_DIM,
    FP32SiluAndMul,
    Lumina2RMSNorm,
    Lumina2Transformer2DModel,
)
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    FeedForward as ZImageFeedForward,
)
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    RopeEmbedder as ZImageRopeEmbedder,
)
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    TimestepEmbedder as ZImageTimestepEmbedder,
)
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    ZImageAttention,
)
from sglang.multimodal_gen.runtime.pipelines.lumina2 import Lumina2Pipeline
from sglang.multimodal_gen.runtime.server_args import (
    get_global_server_args,
    set_global_server_args,
)
from sglang.test.test_utils import CustomTestCase

LUMINA2_CHECKPOINT_CONFIG = {
    "_class_name": "Lumina2Transformer2DModel",
    "_diffusers_version": "0.33.0.dev0",
    "axes_dim_rope": [32, 32, 32],
    "axes_lens": [300, 512, 512],
    "cap_feat_dim": 2304,
    "ffn_dim_multiplier": None,
    "hidden_size": 2304,
    "in_channels": 16,
    "multiple_of": 256,
    "norm_eps": 1e-05,
    "num_attention_heads": 24,
    "num_kv_heads": 8,
    "num_layers": 26,
    "num_refiner_layers": 2,
    "out_channels": None,
    "patch_size": 2,
    "sample_size": 128,
    "scaling_factor": 1.0,
}


def _global_server_args_or_none():
    """The installed global, or None. conftest's fixture is a pytest mechanism,
    so a direct ``python test_lumina2.py`` run has none and the getter raises."""
    try:
        return get_global_server_args()
    except ValueError:
        return None


def build_meta_model(config: Lumina2Config) -> Lumina2Transformer2DModel:
    """Instantiate the DiT on the meta device (no weights, no GPU)."""
    # Under pytest, extend the conftest fixture's stub rather than replacing it,
    # so a layer that starts reading some other ServerArgs field gets fixed once
    # in the fixture instead of here.
    prev_args = _global_server_args_or_none()
    if prev_args is None:
        overridden = SimpleNamespace(attention_backend="torch_sdpa", comfyui_mode=False)
    else:
        overridden = copy.copy(prev_args)
        overridden.attention_backend = "torch_sdpa"
    fake_tp_group = SimpleNamespace(world_size=1, rank_in_group=0)
    try:
        set_global_server_args(overridden)
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                return_value=fake_tp_group,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits.zimage.get_tp_world_size",
                return_value=1,
            ),
        ):
            with torch.device("meta"):
                return Lumina2Transformer2DModel(config, {})
    finally:
        set_global_server_args(prev_args)


class TestSharedZImagePrimitives(CustomTestCase):
    def test_lumina_reuses_zimage_primitives_rather_than_forking_them(self):
        """Lumina-2 and Z-Image are the same joint-DiT family, so the shared
        blocks live in zimage.py -- the elder sibling -- matching this repo's
        convention (causal_wanvideo <- wanvideo, longlive2 <- causal_wanvideo).

        A local fork of any of these would drift from Z-Image silently, which
        is how GQA support came to be missing from one of two copies of the
        masked attention path.
        """
        import sglang.multimodal_gen.runtime.models.dits.lumina2 as lumina2_mod
        import sglang.multimodal_gen.runtime.models.dits.zimage as zimage_mod

        self.assertIs(lumina2_mod.ZImageAttention, ZImageAttention)
        self.assertIs(lumina2_mod.FeedForward, ZImageFeedForward)
        self.assertIs(lumina2_mod.RopeEmbedder, ZImageRopeEmbedder)
        self.assertIs(lumina2_mod.TimestepEmbedder, ZImageTimestepEmbedder)

        # Lumina's adaLN width is NOT Z-Image's. Importing zimage's
        # ADALN_EMBED_DIM would build a silently wrong model -- no shape error
        # at construction, just the wrong modulation width in three places.
        self.assertEqual(ADALN_EMBED_DIM, 1024)
        self.assertNotEqual(ADALN_EMBED_DIM, zimage_mod.ADALN_EMBED_DIM)

    def test_fp32_silu_and_mul_matches_lumina_reference(self):
        x = torch.linspace(-8, 8, 64, dtype=torch.float32).reshape(2, 32)
        x = x.to(torch.bfloat16)
        gate, value = x.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate.float()).to(gate.dtype) * value

        actual = FP32SiluAndMul()(x)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_lumina_model_selects_lumina_precision_policies(self):
        model = build_meta_model(Lumina2Config())
        block = model.layers[0]

        self.assertIsInstance(block.attention, ZImageAttention)
        self.assertIsInstance(block.attention.norm_q, Lumina2RMSNorm)
        self.assertIsInstance(block.attention.norm_k, Lumina2RMSNorm)
        self.assertFalse(block.attention.enable_zimage_qk_fusion)
        self.assertIsInstance(block.feed_forward, ZImageFeedForward)
        self.assertIsInstance(block.feed_forward.act, FP32SiluAndMul)
        self.assertIsInstance(model.rope_embedder, ZImageRopeEmbedder)

    def test_lumina_disables_usp_for_every_replicated_block_group(self):
        model = build_meta_model(Lumina2Config())

        for blocks in (model.context_refiner, model.noise_refiner, model.layers):
            self.assertTrue(blocks)
            for block in blocks:
                self.assertTrue(block.attention.attn.skip_sequence_parallel)

    def test_rope_phase_is_evaluated_in_fp64_for_lumina_only(self):
        """diffusers transformer_lumina2.py:245 builds the phase in fp64 and
        takes cos/sin there; rounding to fp32 first moves a few table entries by
        a bf16 ulp. Z-Image shares this embedder and must keep fp32."""
        axes_dims, axes_lens = (32, 32, 32), (300, 512, 512)
        theta = 10000.0

        cos64, sin64 = ZImageRopeEmbedder.precompute_freqs(
            axes_dims, axes_lens, theta=theta, freqs_dtype=torch.float64
        )
        for i, (d, e) in enumerate(zip(axes_dims, axes_lens)):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            phase = torch.outer(torch.arange(e, dtype=torch.float64), freqs)
            self.assertTrue(torch.equal(cos64[i], phase.cos().float()))
            self.assertTrue(torch.equal(sin64[i], phase.sin().float()))
            self.assertEqual(cos64[i].dtype, torch.float32)

        # If the two agreed, Lumina's opt-in would be untested and Z-Image's
        # default unprotected.
        cos32, _ = ZImageRopeEmbedder.precompute_freqs(
            axes_dims, axes_lens, theta=theta
        )
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(cos32, cos64)))

    def test_rms_norm_rounds_before_the_affine_multiply(self):
        """diffusers normalization.py:553-562 casts to the weight dtype before
        applying the weight. Hoisting that multiply into fp32 reads as a strict
        accuracy win and silently drifts from the reference in every block."""
        torch.manual_seed(0)
        dim = 64
        norm = Lumina2RMSNorm(dim).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(dim, dtype=torch.bfloat16))
        x = torch.randn(4, 12, dim, dtype=torch.bfloat16)

        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + norm.variance_epsilon)
        expected = normalized.to(norm.weight.dtype) * norm.weight

        actual = norm(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

        # Keeps the assertion above honest: a revert cannot pass by coincidence.
        fp32_affine = (normalized * norm.weight.float()).to(torch.bfloat16)
        self.assertFalse(torch.equal(fp32_affine, expected))


class TestLumina2ArchConfig(CustomTestCase):
    def test_every_checkpoint_config_key_is_a_declared_field(self):
        """ArchConfig routes undeclared keys into extra_attrs, where nothing
        reads them. A key landing there means update_model_arch silently drops
        the checkpoint's value and the model is built from a stale default."""
        declared = Lumina2ArchConfig.__dataclass_fields__
        undeclared = sorted(
            key
            for key in LUMINA2_CHECKPOINT_CONFIG
            if not key.startswith("_") and key not in declared
        )
        self.assertEqual(undeclared, [])

    def test_update_model_arch_applies_checkpoint_values(self):
        config = Lumina2Config()
        config.update_model_arch(LUMINA2_CHECKPOINT_CONFIG)
        arch = config.arch_config

        self.assertEqual(arch.extra_attrs.get("num_kv_heads"), None)
        self.assertEqual(arch.num_kv_heads, 8)
        self.assertEqual(arch.num_refiner_layers, 2)
        self.assertEqual(arch.patch_size, 2)
        self.assertEqual(tuple(arch.axes_dim_rope), (32, 32, 32))
        self.assertIsNone(arch.ffn_dim_multiplier)
        # out_channels is null in the checkpoint and backfilled from in_channels.
        self.assertEqual(arch.out_channels, 16)
        self.assertEqual(arch.num_channels_latents, 16)

    def test_non_default_checkpoint_values_reach_the_model(self):
        """Guards the failure this whole file exists for: a variant checkpoint
        whose values differ from our defaults must actually change the model."""
        variant = dict(LUMINA2_CHECKPOINT_CONFIG, num_kv_heads=12, num_refiner_layers=3)
        config = Lumina2Config()
        config.update_model_arch(variant)
        model = build_meta_model(config)

        self.assertEqual(len(model.noise_refiner), 3)
        self.assertEqual(len(model.context_refiner), 3)
        # 24 heads * 96 head_dim + 2 * (12 kv heads * 96) = 2304 + 2304
        self.assertEqual(
            tuple(model.layers[0].attention.to_qkv.weight.shape), (4608, 2304)
        )

    def test_axes_dim_rope_must_sum_to_head_dim(self):
        config = Lumina2Config()
        with self.assertRaisesRegex(ValueError, "axes_dim_rope"):
            config.update_model_arch(
                dict(LUMINA2_CHECKPOINT_CONFIG, axes_dim_rope=[32, 32, 16])
            )

    def test_ffn_width_follows_the_checkpoint(self):
        model = build_meta_model(Lumina2Config())
        # round_up(4 * 2304, 256) = 9216, fused gate+up -> 18432 rows.
        self.assertEqual(
            tuple(model.layers[0].feed_forward.w13.weight.shape), (18432, 2304)
        )
        self.assertEqual(
            tuple(model.layers[0].feed_forward.w2.weight.shape), (2304, 9216)
        )

    def test_fused_qkv_width_follows_the_checkpoint(self):
        """The default-config counterpart to the GQA variant checked above. A
        fusion can resolve by name and still be sized wrong, which surfaces much
        later inside the loader's cat."""
        model = build_meta_model(Lumina2Config())
        # (24 q + 8 kv + 8 kv) * 96 head_dim.
        self.assertEqual(
            tuple(model.layers[0].attention.to_qkv.weight.shape), (3840, 2304)
        )
        # patch_size**2 * in_channels = 4 * 16.
        self.assertEqual(tuple(model.x_embedder.weight.shape), (2304, 64))


class TestLumina2WeightMapping(CustomTestCase):
    def test_block_weights_map_onto_real_params(self):
        mapping = get_param_names_mapping(Lumina2ArchConfig().param_names_mapping)
        weights = [
            ("layers.0.attn.to_q.weight", torch.full((2, 2), 1.0)),
            ("layers.0.attn.to_k.weight", torch.full((2, 2), 2.0)),
            ("layers.0.attn.to_v.weight", torch.full((2, 2), 3.0)),
            ("layers.0.feed_forward.linear_1.weight", torch.full((2, 2), 4.0)),
            ("layers.0.feed_forward.linear_3.weight", torch.full((2, 2), 5.0)),
            ("layers.0.feed_forward.linear_2.weight", torch.full((2, 2), 6.0)),
            ("layers.0.norm1.linear.weight", torch.full((2, 2), 7.0)),
            ("layers.0.norm1.norm.weight", torch.full((2,), 8.0)),
            ("layers.0.norm2.weight", torch.full((2,), 9.0)),
            ("context_refiner.0.norm1.weight", torch.full((2,), 10.0)),
            ("norm_out.linear_1.weight", torch.full((2, 2), 11.0)),
            ("norm_out.linear_2.weight", torch.full((2, 2), 12.0)),
            (
                "time_caption_embed.timestep_embedder.linear_1.weight",
                torch.full((2, 2), 13.0),
            ),
        ]
        mapped, _ = hf_to_custom_state_dict(iter(weights), mapping)

        # q/k/v fuse in order, gate/up fuse in order.
        torch.testing.assert_close(
            mapped["layers.0.attention.to_qkv.weight"],
            torch.cat([w for _, w in weights[:3]], dim=0),
        )
        torch.testing.assert_close(
            mapped["layers.0.feed_forward.w13.weight"],
            torch.cat([w for _, w in weights[3:5]], dim=0),
        )
        for name in (
            "layers.0.feed_forward.w2.weight",
            "layers.0.adaLN_modulation.1.weight",
            "layers.0.attention_norm1.weight",
            "layers.0.attention_norm2.weight",
            "context_refiner.0.attention_norm1.weight",
            "norm_out.adaLN_modulation.1.weight",
            "norm_out.linear.weight",
            "time_caption_embed.timestep_embedder.mlp.0.weight",
        ):
            self.assertIn(name, mapped)

        # A rule resolving to a name the model lacks is dropped at load time and
        # leaves that layer at its initialized values, silently.
        model = build_meta_model(Lumina2Config())
        self.assertEqual(sorted(set(mapped) - set(model.state_dict())), [])

    def test_packed_modules_mapping_covers_every_fusion(self):
        """Omitting a fusion here silently quantizes a layer the user asked to
        keep, since --quantization-ignored-layers names checkpoint weights."""
        model = build_meta_model(Lumina2Config())
        mapping = Lumina2Transformer2DModel.packed_modules_mapping
        leaf_modules = {name.rsplit(".", 2)[-2] for name in model.state_dict()}

        # Both fusions are unconditional for Lumina.
        for fused in ("to_qkv", "w13"):
            self.assertIn(fused, leaf_modules)
            self.assertIn(fused, mapping)

        # Shards must be names the checkpoint actually uses. Z-Image's w1/w3
        # appear in no Lumina key and would resolve to nothing.
        pattern_text = " ".join(Lumina2ArchConfig().param_names_mapping)
        for shards in mapping.values():
            for shard in shards:
                self.assertIn(shard, pattern_text)


class TestLumina2PipelineConfig(CustomTestCase):
    def test_registry_resolves_model_path(self):
        # get_model_info would fetch model_index.json and, when that fails,
        # still pass by falling back to these same classes. __wrapped__ skips
        # lru_cache(maxsize=1) rather than evicting the entry other tests share.
        # The patch is for the detector path, which pulls model_index.json
        # before running detectors; the exact-path match returns before that.
        resolve = _get_config_info.__wrapped__
        with patch(
            "sglang.multimodal_gen.registry.maybe_download_model_index",
            return_value={},
        ):
            for path in (
                "Alpha-VLLM/Lumina-Image-2.0",  # exact hf_model_paths entry
                "some-org/lumina2-finetune",  # model_detectors fallback
            ):
                info = resolve(path)
                self.assertIsNotNone(info, path)
                self.assertIs(info.pipeline_config_cls, Lumina2PipelineConfig)
                self.assertIs(info.sampling_param_cls, Lumina2SamplingParams)

    def test_pipeline_name_matches_hf_class_name(self):
        """_class_name in model_index.json is how a checkpoint finds this
        pipeline; renaming pipeline_name drops Lumina onto the generic path."""
        _discover_and_register_pipelines()
        self.assertIs(
            _PIPELINE_REGISTRY.get("Lumina2Pipeline"),
            Lumina2Pipeline,
        )

    def test_sequence_parallel_sharding_is_disabled(self):
        """The DiT has no SP handling, so sharding latents across ranks would
        silently produce a wrong image rather than fail."""
        config = Lumina2PipelineConfig()
        latents = torch.zeros(1, 16, 128, 128)
        sharded, did_shard = config.shard_latents_for_sp(None, latents)
        self.assertFalse(did_shard)
        self.assertIs(sharded, latents)
        self.assertIs(config.gather_latents_for_sp(latents), latents)

    def test_conditioning_expands_to_the_sample_batch(self):
        """Conditioning must reach one row per sample, not per prompt; see
        Lumina2PipelineConfig.expand_conditioning_to_sample_batch."""
        config = Lumina2PipelineConfig()
        batch = SimpleNamespace(
            prompt="a cat",
            num_outputs_per_prompt=4,
            prompt_embeds=[torch.zeros(1, 256, 2304)],
            negative_prompt_embeds=[torch.zeros(1, 256, 2304)],
            prompt_attention_mask=[torch.ones(1, 256, dtype=torch.long)],
            negative_attention_mask=[torch.ones(1, 256, dtype=torch.long)],
        )

        config.expand_conditioning_to_sample_batch(batch)

        for field in (
            "prompt_embeds",
            "negative_prompt_embeds",
            "prompt_attention_mask",
            "negative_attention_mask",
        ):
            (tensor,) = getattr(batch, field)
            self.assertEqual(tensor.shape[0], 4, field)

    def test_conditioning_expansion_is_a_no_op_for_a_single_output(self):
        config = Lumina2PipelineConfig()
        embeds = [torch.zeros(2, 256, 2304)]
        batch = SimpleNamespace(
            prompt=["a cat", "a dog"],
            num_outputs_per_prompt=1,
            prompt_embeds=embeds,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_attention_mask=None,
        )

        config.expand_conditioning_to_sample_batch(batch)

        self.assertIs(batch.prompt_embeds, embeds)

    def test_tokenizer_is_pinned_to_right_padding(self):
        """A tokenizer arriving with left padding must be corrected in place."""
        config = Lumina2PipelineConfig()

        def fake_tokenizer(prompts, **kwargs):
            return SimpleNamespace(to=lambda _device: {})

        fake_tokenizer.padding_side = "left"
        config.tokenize_prompt(["a"], fake_tokenizer, {})
        self.assertEqual(fake_tokenizer.padding_side, "right")

    def test_caption_length_is_capped_to_the_rope_table(self):
        config = Lumina2PipelineConfig()
        captured = {}

        def fake_tokenizer(prompts, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(to=lambda _device: {})

        # One below the row count; see Lumina2PipelineConfig.tokenize_prompt.
        axis0 = config.dit_config.arch_config.axes_lens[0]
        config.tokenize_prompt(["a"], fake_tokenizer, {"max_length": 512})
        self.assertEqual(captured["max_length"], axis0 - 1)

        config.tokenize_prompt(["a"], fake_tokenizer, {})
        self.assertEqual(captured["max_length"], LUMINA2_MAX_TEXT_LEN)

    def test_prompt_preprocessing_matches_diffusers(self):
        config = Lumina2PipelineConfig()
        (preprocess,) = config.get_preprocess_text_funcs(is_negative=False)
        (negative_preprocess,) = config.get_preprocess_text_funcs(is_negative=True)

        # Transcribed from diffusers pipeline_lumina2.py:288, identical in
        # v0.33.0, v0.37.0 (the pinned one) and main:
        #   prompt = [system_prompt + " <Prompt Start> " + p for p in prompt]
        # Spelled out, not built from LUMINA2_PROMPT_SEPARATOR, so editing the
        # separator turns this red instead of agreeing with itself.
        self.assertEqual(
            preprocess("a cat"),
            LUMINA2_SYSTEM_PROMPT + " <Prompt Start> " + "a cat",
        )
        self.assertEqual(preprocess(""), LUMINA2_SYSTEM_PROMPT + " <Prompt Start> ")
        self.assertIsNone(negative_preprocess)

    def test_renorm_cfg_matches_the_conditional_norm(self):
        config = Lumina2PipelineConfig()
        noise_pred_cond = torch.randn(1, 16, 8, 8)
        noise_pred = noise_pred_cond * 3.0

        out = config.postprocess_cfg_noise(None, noise_pred, noise_pred_cond)

        torch.testing.assert_close(
            torch.norm(out, dim=-1), torch.norm(noise_pred_cond, dim=-1)
        )

    def test_renorm_matches_the_diffusers_formula_bitwise(self):
        """Transcribed from diffusers pipeline_lumina2.py:750-758:

            cond_norm  = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
            noise_norm = torch.norm(noise_pred,      dim=-1, keepdim=True)
            noise_pred = noise_pred * (cond_norm / noise_norm)

        The property test above only pins that the conditional norm is
        restored; this pins the formula, so a different rescale that happens
        to preserve the norm would still turn it red.
        """
        config = Lumina2PipelineConfig()
        torch.manual_seed(0)
        cond = torch.randn(2, 1024, 16)
        combined = torch.randn(2, 1024, 16)

        expected = combined * (
            torch.norm(cond, dim=-1, keepdim=True)
            / torch.norm(combined, dim=-1, keepdim=True)
        )
        actual = config.postprocess_cfg_noise(None, combined, cond)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_renorm_survives_an_all_zero_prediction(self):
        """clamp_min(1e-12) on the divisor. Bitwise identical to diffusers on
        real input (verified above); it only differs where diffusers would
        divide by zero."""
        config = Lumina2PipelineConfig()
        out = config.postprocess_cfg_noise(
            None, torch.zeros(1, 8, 16), torch.randn(1, 8, 16)
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_negating_before_cfg_equals_negating_after(self):
        """The DiT returns ``-output`` (lumina2.py forward) while diffusers
        negates in the pipeline *after* CFG and renorm
        (pipeline_lumina2.py:761). That is only equivalent because both the CFG
        combination and the renorm are odd in the prediction -- f(-x) = -f(x).

        The equality is exact, not approximate: IEEE negation is exact and
        round-to-nearest is symmetric under it, and ``torch.norm`` is unchanged
        by sign. Asserted bitwise so a future non-odd postprocess turns it red.
        """
        config = Lumina2PipelineConfig()
        torch.manual_seed(0)
        cond = torch.randn(2, 4096, 16)
        uncond = torch.randn(2, 4096, 16)
        scale = 4.0

        # diffusers: combine -> renorm -> negate.
        ref = uncond + scale * (cond - uncond)
        ref = ref * (
            torch.norm(cond, dim=-1, keepdim=True)
            / torch.norm(ref, dim=-1, keepdim=True)
        )
        ref = -ref

        # SGLang: the DiT already negated, so CFG and renorm see -cond/-uncond.
        sgl_combined = (-uncond) + scale * ((-cond) - (-uncond))
        sgl = config.postprocess_cfg_noise(None, sgl_combined, -cond)

        torch.testing.assert_close(sgl, ref, rtol=0, atol=0)

    def test_sigma_grid_matches_the_diffusers_pipeline_linspace(self):
        """diffusers' pipeline builds the grid itself (pipeline_lumina2.py:698)
        and passes it into retrieve_timesteps:

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        The scheduler's *own* default grid -- shift=6.0 applied to an internal
        linspace -- is never used by the pipeline, so comparing against a bare
        ``set_timesteps(steps)`` compares against code nothing calls.
        """
        config = Lumina2PipelineConfig()

        for steps in (1, 2, 8, 30):
            with self.subTest(steps=steps):
                sigmas = config.prepare_sigmas(None, steps)
                # np.linspace, not torch.linspace: the implementation and
                # diffusers both call this one, and the two can disagree in the
                # last bit -- which would make an exact assert lie.
                expected = np.linspace(1.0, 1 / steps, steps)

                self.assertEqual(len(sigmas), steps)
                self.assertEqual(list(sigmas), list(expected))

        # An explicitly supplied grid must pass through untouched.
        given = [1.0, 0.7, 0.3]
        self.assertIs(config.prepare_sigmas(given, 3), given)

    def test_latent_geometry_derives_from_the_vae(self):
        """FluxVAEArchConfig declares spatial_compression_ratio=1 and derives
        the real value in post_init() from block_out_channels, falling back to
        dim_mult. Reading it before post_init() reports 1 and makes the latent
        look 8x too large -- that misread cost a false alarm during validation,
        so both the pre and post values are pinned here."""
        config = Lumina2PipelineConfig()

        self.assertEqual(config.vae_config.arch_config.spatial_compression_ratio, 1)
        config.vae_config.post_init()
        self.assertEqual(config.vae_config.arch_config.spatial_compression_ratio, 8)

        # 16 latent channels, matching the FLUX.1-dev VAE Lumina-2 ships.
        self.assertEqual(config.dit_config.arch_config.num_channels_latents, 16)

        batch = SimpleNamespace(height=1024, width=1024)
        self.assertEqual(config.prepare_latent_shape(batch, 1, 1), (1, 16, 128, 128))

    def test_cfg_normalization_stays_off(self):
        """Lumina2SamplingParams pins this to 0.0 against a change to the base
        default. Turning it on stacks CFGPolicy's global max-norm clip on top of
        the per-position renorm above, which is a different operation."""
        self.assertFalse(Lumina2SamplingParams().cfg_normalization)

    def test_pipeline_config_passes_its_own_validator(self):
        """check_pipeline_config() rejects vae_sp without vae_tiling, and the
        base class defaults both to True. Lumina sets vae_tiling=False, so
        leaving vae_sp inherited fails validation before anything loads --
        i.e. every `sglang generate` / `sglang serve` dies at startup.

        The other Lumina tests build the config and assert on fields; none of
        them ran it through its own validator, which is why that shipped.
        """
        config = Lumina2PipelineConfig()

        self.assertFalse(config.vae_tiling)
        self.assertFalse(config.vae_sp)
        config.check_pipeline_config()

    def test_padding_mask_fires_when_the_tensor_is_wider_than_the_caption(self):
        """Passes before and after the GQA fix; it pins *which path* an
        ordinary request takes.

        tokenize_prompt pads to max_length=256 while the attention mask carries
        the caption's true length (~40 tokens with the system prompt). So
        _padding_mask does not short-circuit, and the caption refiner runs the
        masked attention path on a *single* prompt -- no batching required.
        That is why the GQA defect in USPAttention's masked branch was a launch
        blocker rather than a mixed-length-batch edge case.

        The only case that legitimately skips the mask is a caption that fills
        its tensor exactly.
        """
        device = torch.device("cpu")

        mask, meta = Lumina2Transformer2DModel._padding_mask(
            [37], LUMINA2_MAX_TEXT_LEN, device
        )
        self.assertIsNotNone(mask)
        self.assertIsNotNone(meta)
        self.assertEqual(tuple(mask.shape), (1, LUMINA2_MAX_TEXT_LEN))
        self.assertEqual(int(mask.sum()), 37)
        self.assertTrue(bool(mask[0, 36]))
        self.assertFalse(bool(mask[0, 37]))

        self.assertEqual(
            Lumina2Transformer2DModel._padding_mask(
                [LUMINA2_MAX_TEXT_LEN], LUMINA2_MAX_TEXT_LEN, device
            ),
            (None, None),
        )


if __name__ == "__main__":
    unittest.main()
