# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free contracts against the official Diffusers Ovis implementation."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import AutoencoderKL as ReferenceVAE
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_ovis_image import (
    OvisImageTransformer2DModel as ReferenceDiT,
)
from diffusers.pipelines.ovis_image.pipeline_ovis_image import (
    OvisImagePipeline as ReferencePipeline,
)
from transformers import Qwen3Config, Qwen3Model

from sglang.multimodal_gen.configs.models.dits.ovis_image import OvisImageConfig
from sglang.multimodal_gen.configs.pipeline_configs.ovis_image import (
    OvisImagePipelineConfig,
    ovis_image_text_output,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
    OvisImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3ForCausalLM
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL
from sglang.multimodal_gen.runtime.pipelines.ovis_image import prepare_mu
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.test_utils import CustomTestCase


class TestOvisImageConditioning(CustomTestCase):
    def test_mask_before_crop_keeps_fixed_length_attention(self):
        hidden = torch.arange(2 * 33 * 4).reshape(2, 33, 4).float()
        mask = torch.ones(2, 33, dtype=torch.long)
        mask[0, 30:] = 0
        mask[1, 28:] = 0
        result = ovis_image_text_output(
            SimpleNamespace(last_hidden_state=hidden),
            SimpleNamespace(attention_mask=mask),
        )
        torch.testing.assert_close(
            result.prompt_embeds, (hidden * mask.unsqueeze(-1))[:, 28:]
        )
        self.assertEqual(result.prompt_seq_lens, [5, 5])
        self.assertTrue(result.prompt_embeds_mask.all())

    def test_pack_unpack_matches_official_layout(self):
        config = OvisImagePipelineConfig()
        batch = SimpleNamespace(height=80, width=112)
        latent = torch.arange(4 * 16 * 10 * 14).reshape(4, 16, 10, 14).float()
        packed = config.maybe_pack_latents(latent, 4, batch)
        expected = ReferencePipeline._pack_latents(latent, 4, 16, 10, 14)
        torch.testing.assert_close(packed, expected, atol=0, rtol=0)
        restored = config.post_denoising_loop(packed, batch)
        torch.testing.assert_close(restored, latent, atol=0, rtol=0)

    def test_dynamic_schedule_matches_official_defaults(self):
        config = OvisImagePipelineConfig()
        for size, mu in ((80, 0.4608984375), (512, 0.63), (1024, 1.15)):
            with self.subTest(size=size):
                scheduler = FlowMatchEulerDiscreteScheduler(
                    use_dynamic_shifting=True,
                    base_image_seq_len=256,
                    max_image_seq_len=4096,
                    base_shift=0.5,
                    max_shift=1.15,
                )
                batch = SimpleNamespace(height=size, width=size, scheduler=scheduler)
                _, actual_mu = prepare_mu(
                    batch, SimpleNamespace(pipeline_config=config)
                )
                self.assertAlmostEqual(actual_mu, mu)
                sigmas = config.prepare_sigmas(None, 5)
                np.testing.assert_array_equal(sigmas, np.linspace(1, 1 / 5, 5))
                scheduler.set_timesteps(sigmas=sigmas, mu=actual_mu)
                expected = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
                expected.set_timesteps(sigmas=np.linspace(1, 1 / 5, 5), mu=mu)
                torch.testing.assert_close(scheduler.timesteps, expected.timesteps)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for native attention")
class TestOvisImageNumerics(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.owned_group = not torch.distributed.is_initialized()
        cls.owned_srt_context = False
        cls.kwargs = dict(
            in_channels=64,
            out_channels=64,
            num_attention_heads=2,
            attention_head_dim=8,
            joint_attention_dim=16,
            num_layers=1,
            num_single_layers=1,
            axes_dims_rope=(2, 2, 4),
        )
        cls.config = OvisImageConfig()
        cls.config.update_model_arch(cls.kwargs)
        set_global_server_args(
            ServerArgs(
                model_path="ATH-MaaS/Ovis-Image-7B",
                pipeline_config=OvisImagePipelineConfig(dit_config=cls.config),
                num_gpus=1,
                attention_backend="torch_sdpa",
                performance_mode="manual",
            )
        )
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        from sglang.srt.runtime_context import get_context, publish
        from sglang.srt.server_args import ServerArgs as SrtServerArgs

        cls.owned_srt_context = get_context()._server_args is None
        if cls.owned_srt_context:
            publish(
                SrtServerArgs(model_path="dummy", tp_size=1),
                role="diffusion_gpu_worker",
            )

    @classmethod
    def tearDownClass(cls):
        if cls.owned_group:
            cleanup_dist_env_and_memory()
        if cls.owned_srt_context:
            from sglang.srt.runtime_context import reset_context

            reset_context()

    @torch.no_grad()
    def test_blocks_and_full_transformer(self):
        for dtype in (torch.float32, torch.bfloat16):
            for batch_size, text_len, image_len in ((1, 5, 15), (4, 7, 25)):
                with self.subTest(dtype=dtype, batch=batch_size):
                    torch.manual_seed(42)
                    reference = ReferenceDiT(**self.kwargs).cuda().to(dtype).eval()
                    native = (
                        OvisImageTransformer2DModel(self.config, self.kwargs)
                        .cuda()
                        .to(dtype)
                        .eval()
                    )
                    native.load_weights(reference.state_dict().items())
                    image = torch.randn(
                        batch_size, image_len, 64, device="cuda", dtype=dtype
                    )
                    text = torch.randn(
                        batch_size, text_len, 16, device="cuda", dtype=dtype
                    )
                    # Zero tokenizer padding is still a valid DiT token.
                    text[:, -2:] = 0
                    t = torch.full((batch_size,), 1000.0, device="cuda")
                    txt_ids = torch.zeros(text_len, 3, device="cuda")
                    txt_ids[:, 1:] = torch.arange(text_len, device="cuda")[:, None]
                    img_ids = torch.zeros(image_len, 3, device="cuda")
                    img_ids[:, 1] = torch.arange(image_len, device="cuda") // 5
                    img_ids[:, 2] = torch.arange(image_len, device="cuda") % 5
                    rope = native.rotary_emb(torch.cat([txt_ids, img_ids]))
                    with set_forward_context(current_timestep=0, attn_metadata=None):
                        actual = native(image, text, t, rope)
                    expected = reference(image, text, t / 1000, img_ids, txt_ids).sample
                    atol, rtol = (
                        (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
                    )
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                    block_image = torch.randn(
                        batch_size, image_len, 16, device="cuda", dtype=dtype
                    )
                    block_text = torch.randn(
                        batch_size, text_len, 16, device="cuda", dtype=dtype
                    )
                    temb = torch.randn(batch_size, 16, device="cuda", dtype=dtype)
                    reference_rope = reference.pos_embed(torch.cat([txt_ids, img_ids]))
                    for native_blocks, reference_blocks in (
                        (native.transformer_blocks, reference.transformer_blocks),
                        (
                            native.single_transformer_blocks,
                            reference.single_transformer_blocks,
                        ),
                    ):
                        with set_forward_context(
                            current_timestep=0, attn_metadata=None
                        ):
                            actual_context, actual_image = native_blocks[0](
                                block_image, block_text, temb, rope
                            )
                        expected_context, expected_image = reference_blocks[0](
                            block_image, block_text, temb, reference_rope
                        )
                        torch.testing.assert_close(
                            actual_context, expected_context, atol=atol, rtol=rtol
                        )
                        torch.testing.assert_close(
                            actual_image, expected_image, atol=atol, rtol=rtol
                        )

    @torch.no_grad()
    def test_raw_timestep_matches_reference_precision(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                reference = ReferenceDiT(**self.kwargs).cuda().to(dtype).eval()
                native = (
                    OvisImageTransformer2DModel(self.config, self.kwargs)
                    .cuda()
                    .to(dtype)
                    .eval()
                )
                native.load_weights(reference.state_dict().items())
                image = torch.randn(4, 25, 64, device="cuda", dtype=dtype)
                text = torch.randn(4, 7, 16, device="cuda", dtype=dtype)
                text[:, -2:] = 0
                # These dynamic-shift scheduler timesteps exercise an extra
                # BF16 round at normalization: 502 -> 504 and 510 -> 512.
                timestep = torch.tensor(
                    [1000.0, 502.7401428, 510.5518188, 957.25], device="cuda"
                )
                txt_ids = torch.zeros(7, 3, device="cuda")
                txt_ids[:, 1:] = torch.arange(7, device="cuda")[:, None]
                img_ids = torch.zeros(25, 3, device="cuda")
                img_ids[:, 1] = torch.arange(25, device="cuda") // 5
                img_ids[:, 2] = torch.arange(25, device="cuda") % 5
                rope = native.rotary_emb(torch.cat([txt_ids, img_ids]))
                effective = {}

                def capture(name):
                    def hook(module, inputs):
                        effective[name] = inputs[0].detach().clone()

                    return hook

                hooks = [
                    native.time_proj.register_forward_pre_hook(capture("native")),
                    reference.time_proj.register_forward_pre_hook(capture("reference")),
                ]
                try:
                    with set_forward_context(current_timestep=0, attn_metadata=None):
                        actual = native(image, text, timestep, rope)
                    expected = reference(
                        image, text, timestep.to(dtype) / 1000, img_ids, txt_ids
                    ).sample
                finally:
                    for hook in hooks:
                        hook.remove()
                torch.testing.assert_close(
                    effective["native"], effective["reference"], atol=0, rtol=0
                )
                if dtype == torch.bfloat16:
                    torch.testing.assert_close(
                        effective["native"],
                        torch.tensor(
                            [1000.0, 504.0, 512.0, 956.0],
                            device="cuda",
                            dtype=dtype,
                        ),
                        atol=0,
                        rtol=0,
                    )
                atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    @torch.no_grad()
    def test_blocks_and_transformer_with_full_width_rope(self):
        kwargs = {
            **self.kwargs,
            "attention_head_dim": 128,
            "axes_dims_rope": (16, 56, 56),
        }
        config = OvisImageConfig()
        config.update_model_arch(kwargs)
        for dtype in (torch.float32, torch.bfloat16):
            for batch_size, text_len, image_len in ((1, 5, 15), (4, 7, 25)):
                torch.manual_seed(42)
                reference = ReferenceDiT(**kwargs).cuda().to(dtype).eval()
                native = (
                    OvisImageTransformer2DModel(config, kwargs).cuda().to(dtype).eval()
                )
                native.load_weights(reference.state_dict().items())
                txt_ids = torch.zeros(text_len, 3, device="cuda")
                txt_ids[:, 1:] = torch.arange(text_len, device="cuda")[:, None]
                img_ids = torch.zeros(image_len, 3, device="cuda")
                img_ids[:, 1] = torch.arange(image_len, device="cuda") // 5
                img_ids[:, 2] = torch.arange(image_len, device="cuda") % 5
                rope = native.rotary_emb(torch.cat([txt_ids, img_ids]))
                reference_rope = reference.pos_embed(torch.cat([txt_ids, img_ids]))
                atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
                block_image = torch.randn(
                    batch_size, image_len, 256, device="cuda", dtype=dtype
                )
                block_text = torch.randn(
                    batch_size, text_len, 256, device="cuda", dtype=dtype
                )
                temb = torch.randn(batch_size, 256, device="cuda", dtype=dtype)
                for component in ("transformer_blocks", "single_transformer_blocks"):
                    with self.subTest(
                        dtype=dtype, batch=batch_size, component=component
                    ):
                        with set_forward_context(
                            current_timestep=0, attn_metadata=None
                        ):
                            actual_context, actual_image = getattr(native, component)[
                                0
                            ](block_image, block_text, temb, rope)
                        expected_context, expected_image = getattr(
                            reference, component
                        )[0](block_image, block_text, temb, reference_rope)
                        torch.testing.assert_close(
                            actual_context, expected_context, atol=atol, rtol=rtol
                        )
                        torch.testing.assert_close(
                            actual_image, expected_image, atol=atol, rtol=rtol
                        )
                with self.subTest(dtype=dtype, batch=batch_size, component="full"):
                    image = torch.randn(
                        batch_size, image_len, 64, device="cuda", dtype=dtype
                    )
                    text = torch.randn(
                        batch_size, text_len, 16, device="cuda", dtype=dtype
                    )
                    text[:, -2:] = 0
                    timestep = torch.full((batch_size,), 1000.0, device="cuda")
                    with set_forward_context(current_timestep=0, attn_metadata=None):
                        actual = native(image, text, timestep, rope)
                    expected = reference(
                        image, text, timestep.to(dtype) / 1000, img_ids, txt_ids
                    ).sample
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    @torch.no_grad()
    def test_native_text_encoder_matches_qwen3(self):
        kwargs = dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            pad_token_id=0,
            attention_bias=False,
            rope_theta=1000000.0,
        )
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(17)
                reference_config = Qwen3Config(**kwargs)
                reference_config._attn_implementation = "sdpa"
                reference = Qwen3Model(reference_config).cuda().to(dtype).eval()
                # Learned norm scales expose the difference between casting
                # before or after the scale multiplication in BF16.
                for name, parameter in reference.named_parameters():
                    if name.endswith("norm.weight"):
                        parameter.uniform_(0.5, 1.5)
                config = OvisImagePipelineConfig().text_encoder_configs[0]
                config.update_model_arch(reference_config.to_dict())
                native = Qwen3ForCausalLM(config).cuda().to(dtype).eval()
                native.load_weights(reference.state_dict().items())
                atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
                for case, length in (
                    ("padded", 33),
                    ("all_valid", 65),
                    ("implicit_causal", 65),
                ):
                    with self.subTest(mask=case):
                        ids = torch.randint(1, 128, (2, length), device="cuda")
                        mask = (
                            None if case == "implicit_causal" else torch.ones_like(ids)
                        )
                        if case == "padded":
                            mask[0, 31:] = 0
                            mask[1, 28:] = 0
                            ids[mask == 0] = 0
                        with set_forward_context(
                            current_timestep=0, attn_metadata=None
                        ):
                            actual = native(ids, attention_mask=mask).last_hidden_state
                        expected = reference(
                            ids, attention_mask=mask, use_cache=False
                        ).last_hidden_state
                        if mask is not None:
                            actual = actual * mask.unsqueeze(-1)
                            expected = expected * mask.unsqueeze(-1)
                        torch.testing.assert_close(
                            actual, expected, atol=atol, rtol=rtol
                        )

    @torch.no_grad()
    def test_native_vae_decode_matches_diffusers(self):
        kwargs = dict(
            block_out_channels=(16, 32),
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            layers_per_block=1,
            latent_channels=16,
            norm_num_groups=8,
            sample_size=32,
            use_quant_conv=False,
            use_post_quant_conv=False,
            scaling_factor=0.3611,
            shift_factor=0.1159,
        )
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(19)
                reference = ReferenceVAE(**kwargs).cuda().to(dtype).eval()
                config = OvisImagePipelineConfig()
                config.vae_config.update_model_arch(dict(reference.config))
                native = AutoencoderKL(config.vae_config).cuda().to(dtype).eval()
                native.load_state_dict(reference.state_dict(), strict=True)
                latent = torch.randn(2, 16, 8, 8, device="cuda", dtype=dtype)
                scale, shift = config.get_decode_scale_and_shift("cuda", dtype, native)
                decoded_latent = latent / scale + shift
                actual = native.decode(decoded_latent)
                expected = reference.decode(latent / 0.3611 + 0.1159).sample
                atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class TestOvisImageWeightShards(CustomTestCase):
    @torch.no_grad()
    def test_single_block_tp_gated_projection_matches_full_checkpoint(self):
        """TP must slice both value/gate rows and both attention/MLP columns."""
        from unittest.mock import patch

        import torch.nn.functional as F

        from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
            OvisImageSingleTransformerBlock,
        )

        dim, mlp_dim = 8, 32
        inputs = torch.arange(24).reshape(3, dim).float() / 24
        attention = torch.arange(24).reshape(3, dim).float() / 13 - 1
        mlp_weight = (
            torch.arange(2 * mlp_dim * dim).reshape(2 * mlp_dim, dim).float() / 128
        )
        mlp_bias = torch.arange(2 * mlp_dim).float() / 16 - 2
        output_weight = (
            torch.arange(dim * (dim + mlp_dim)).reshape(dim, dim + mlp_dim).float()
            / 320
            - 0.5
        )
        output_bias = torch.arange(dim).float() / 7
        mlp_full = F.linear(inputs, mlp_weight, mlp_bias)
        value_full, gate_full = mlp_full.chunk(2, dim=-1)
        expected = F.linear(
            torch.cat([attention, value_full * F.silu(gate_full)], dim=-1),
            output_weight,
            output_bias,
        )
        partial_outputs = []
        for rank in range(2):
            group = SimpleNamespace(world_size=2, rank_in_group=rank)
            # No attention is executed: mock the external kernel constructor
            # and process-group boundary, keeping both production loaders.
            with (
                patch(
                    "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                    return_value=group,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.flux.get_tp_world_size",
                    return_value=2,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.flux.USPAttention",
                    return_value=torch.nn.Identity(),
                ),
            ):
                block = OvisImageSingleTransformerBlock(dim, 2, 4, "single")
            for module, weight, bias in (
                (block.proj_mlp, mlp_weight, mlp_bias),
                (block.proj_out, output_weight, output_bias),
            ):
                module.weight.weight_loader(module.weight, weight)
                module.bias.weight_loader(module.bias, bias)
            mlp_local = block.proj_mlp(inputs)[0]
            value, gate = mlp_local.chunk(2, dim=-1)
            attention_local = attention[:, rank * 4 : (rank + 1) * 4]
            local_features = torch.cat([attention_local, value * F.silu(gate)], dim=-1)
            partial_outputs.append(F.linear(local_features, block.proj_out.weight))
        actual = sum(partial_outputs) + output_bias
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
