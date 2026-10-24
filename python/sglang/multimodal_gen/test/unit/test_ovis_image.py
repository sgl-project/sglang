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
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
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

    def test_native_scheduler_steps_match_official(self):
        """Fixed model outputs stay within tolerance across the native Euler path."""
        from diffusers.pipelines.ovis_image.pipeline_ovis_image import calculate_shift

        from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler as NativeEulerScheduler,
        )

        scheduler_kwargs = dict(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=True,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15,
        )
        config = OvisImagePipelineConfig()
        generator = torch.Generator("cpu").manual_seed(42)
        initial = torch.randn(2, 25, 64, generator=generator)
        prediction = torch.randn(2, 25, 64, generator=generator)
        for size in (512, 1024):
            for steps in (20, 50):
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(size=size, steps=steps, dtype=dtype):
                        native = NativeEulerScheduler(**scheduler_kwargs)
                        reference = FlowMatchEulerDiscreteScheduler(**scheduler_kwargs)
                        batch = SimpleNamespace(
                            height=size, width=size, scheduler=native
                        )
                        _, native_mu = prepare_mu(
                            batch, SimpleNamespace(pipeline_config=config)
                        )
                        reference_mu = calculate_shift((size // 16) ** 2)
                        native.set_timesteps(
                            sigmas=config.prepare_sigmas(None, steps), mu=native_mu
                        )
                        reference.set_timesteps(
                            sigmas=np.linspace(1, 1 / steps, steps), mu=reference_mu
                        )
                        native.set_begin_index(0)
                        reference.set_begin_index(0)
                        actual = initial.to(dtype).clone()
                        expected = actual.clone()
                        fixed_prediction = prediction.to(dtype)
                        atol, rtol = (
                            (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
                        )
                        for actual_t, expected_t in zip(
                            native.timesteps, reference.timesteps
                        ):
                            actual = native.step(
                                fixed_prediction, actual_t, actual, return_dict=False
                            )[0]
                            expected = reference.step(
                                fixed_prediction,
                                expected_t,
                                expected,
                                return_dict=False,
                            )[0]
                            torch.testing.assert_close(
                                actual, expected, atol=atol, rtol=rtol
                            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for native attention")
class TestOvisImageNumerics(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from unittest.mock import patch

        from sglang.multimodal_gen.runtime.server_args import (
            server_args as server_args_module,
        )

        # unittest runs class cleanups even if setup or teardown raises.
        args_patch = patch.object(server_args_module, "_global_server_args", None)
        args_patch.start()
        cls.addClassCleanup(args_patch.stop)
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
        cls.server_args = ServerArgs(
            model_path="ATH-MaaS/Ovis-Image-7B",
            pipeline_config=OvisImagePipelineConfig(dit_config=cls.config),
            num_gpus=1,
            attention_backend="torch_sdpa",
            performance_mode="manual",
        )
        set_global_server_args(cls.server_args)
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
    def test_attention_preserves_norm_round_before_rope(self):
        """Weighted QK norm rounds to BF16 before the official FP32 RoPE."""
        from unittest.mock import patch

        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.transformers.transformer_ovis_image import (
            OvisImageAttention as ReferenceAttention,
        )
        from diffusers.models.transformers.transformer_ovis_image import (
            OvisImagePosEmbed,
        )

        import sglang.multimodal_gen.runtime.layers.layernorm as norm_module
        from sglang.multimodal_gen.runtime.loader.utils import (
            set_default_torch_dtype,
        )
        from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
            OvisImageAttention as NativeAttention,
        )

        dtype = torch.bfloat16
        head_dim, heads, text_len, image_len = 128, 2, 5, 7
        dim = head_dim * heads
        # All head rows have exactly representable variance 2.5. Non-unit
        # learned BF16 scales expose the norm-to-RoPE rounding boundary without
        # depending on a particular random reduction near a half-way tie.
        image = torch.tensor([1.0, 2.0], device="cuda", dtype=dtype)
        image = image.repeat(dim // 2).repeat(2, image_len, 1)
        text = torch.tensor([2.0, 1.0], device="cuda", dtype=dtype)
        text = text.repeat(dim // 2).repeat(2, text_len, 1)
        ids = torch.zeros(text_len + image_len, 3, device="cuda", dtype=dtype)
        ids[:text_len, 1:] = torch.arange(text_len, device="cuda")[:, None]
        ids[text_len:, 1] = torch.arange(image_len, device="cuda") // 3 + 1
        ids[text_len:, 2] = torch.arange(image_len, device="cuda") % 3 + 1
        reference_rope = OvisImagePosEmbed(10000, [16, 56, 56])(ids)
        native_rope = tuple(value[:, ::2].contiguous() for value in reference_rope)
        scales = {
            "norm_q": (1.125, 0.875),
            "norm_k": (0.75, 1.25),
            "norm_added_q": (0.625, 1.375),
            "norm_added_k": (0.875, 1.125),
        }
        identity = torch.eye(dim, device="cuda", dtype=dtype)
        for dual_stream in (False, True):
            with self.subTest(dual_stream=dual_stream):
                kwargs = dict(
                    query_dim=dim,
                    dim_head=head_dim,
                    out_dim=dim,
                    bias=True,
                    eps=1e-6,
                    added_kv_proj_dim=dim if dual_stream else None,
                    context_pre_only=False if dual_stream else None,
                    pre_only=not dual_stream,
                )
                reference = ReferenceAttention(heads=heads, **kwargs).cuda().to(dtype)
                with set_default_torch_dtype(dtype):
                    native = NativeAttention(num_heads=heads, **kwargs).cuda()
                projection_names = ["to_q", "to_k", "to_v"]
                if dual_stream:
                    projection_names.extend(["add_q_proj", "add_k_proj", "add_v_proj"])
                for name in projection_names:
                    projection = getattr(reference, name)
                    projection.weight.copy_(identity)
                    projection.bias.zero_()
                for name, pair in scales.items():
                    if hasattr(reference, name):
                        getattr(reference, name).weight.copy_(
                            torch.tensor(pair, device="cuda", dtype=dtype).repeat(
                                head_dim // 2
                            )
                        )
                native.load_state_dict(reference.state_dict(), strict=True)
                captured = {}

                def capture(module, inputs):
                    captured["query"] = inputs[0].detach().clone()
                    captured["key"] = inputs[1].detach().clone()

                hook = native.attn.register_forward_pre_hook(capture)
                try:
                    with (
                        patch.object(
                            norm_module,
                            "fused_inplace_qknorm_rope",
                            wraps=norm_module.fused_inplace_qknorm_rope,
                        ) as fused,
                        set_forward_context(current_timestep=0, attn_metadata=None),
                    ):
                        if dual_stream:
                            native(image, text, native_rope)
                        else:
                            native(
                                torch.cat([text, image], dim=1), freqs_cis=native_rope
                            )
                    if fused.call_count == 0:
                        self.skipTest(
                            "CUDA fused QKNorm+RoPE is unavailable or disabled"
                        )
                    self.assertEqual(fused.call_count, 2 if dual_stream else 1)
                finally:
                    hook.remove()
                if dual_stream:
                    query = torch.cat(
                        [
                            reference.norm_added_q(
                                text.unflatten(-1, (heads, head_dim))
                            ),
                            reference.norm_q(image.unflatten(-1, (heads, head_dim))),
                        ],
                        dim=1,
                    )
                    key = torch.cat(
                        [
                            reference.norm_added_k(
                                text.unflatten(-1, (heads, head_dim))
                            ),
                            reference.norm_k(image.unflatten(-1, (heads, head_dim))),
                        ],
                        dim=1,
                    )
                else:
                    joint = torch.cat([text, image], dim=1).unflatten(
                        -1, (heads, head_dim)
                    )
                    query, key = reference.norm_q(joint), reference.norm_k(joint)
                expected_query = apply_rotary_emb(query, reference_rope, sequence_dim=1)
                expected_key = apply_rotary_emb(key, reference_rope, sequence_dim=1)
                torch.testing.assert_close(
                    captured["query"], expected_query, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    captured["key"], expected_key, atol=0, rtol=0
                )

    @torch.no_grad()
    def test_tp_row_projection_rounds_once_like_dense_linear(self):
        """FP32 partial products reduce before one rounding with the bias."""
        from unittest.mock import patch

        import torch.nn.functional as F

        import sglang.multimodal_gen.runtime.models.dits.ovis_image as ovis_module
        from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
            OvisImageRowParallelLinear,
        )

        dtype = torch.bfloat16
        with set_default_torch_dtype(dtype):
            projection = OvisImageRowParallelLinear(
                3072, 3072, bias=True, input_is_parallel=True
            ).cuda()
        generator = torch.Generator(device="cuda").manual_seed(0)
        projection.weight.copy_(
            torch.randn(3072, 3072, device="cuda", generator=generator) * 0.02
        )
        projection.bias.copy_(torch.randn(3072, device="cuda", generator=generator))
        inputs = torch.randn(
            1, 1029, 3072, device="cuda", dtype=dtype, generator=generator
        )
        # One rank holding the whole contraction must reproduce the dense
        # BF16 linear, which also accumulates in FP32 and rounds once.
        with (
            patch.object(projection, "tp_size", 2),
            patch.object(
                ovis_module,
                "tensor_model_parallel_all_reduce",
                lambda tensor, tp_group: tensor,
            ),
        ):
            actual, bias = projection(inputs)
        self.assertIsNone(bias)
        expected = F.linear(inputs, projection.weight, projection.bias)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_sharded_heads_keep_full_model_flash_kernel(self):
        """Head shards of a short sequence match the full-head flash output."""
        import torch.nn.functional as F

        from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
            OvisImageSDPAImpl,
        )

        heads, head_dim = 24, 128
        for seq_len in (1025, 4224):
            generator = torch.Generator(device="cuda").manual_seed(seq_len)
            query, key, value = (
                torch.randn(
                    1,
                    seq_len,
                    heads,
                    head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                for _ in range(3)
            )
            expected = F.scaled_dot_product_attention(
                *(x.transpose(1, 2) for x in (query, key, value))
            ).transpose(1, 2)
            for local_heads in (24, 12, 6, 3):
                with self.subTest(seq_len=seq_len, local_heads=local_heads):
                    impl = OvisImageSDPAImpl(
                        num_heads=local_heads,
                        head_size=head_dim,
                        causal=False,
                        softmax_scale=head_dim**-0.5,
                    )
                    impl.global_heads = heads
                    actual = torch.cat(
                        [
                            impl.forward(q, k, v, None)
                            for q, k, v in zip(
                                query.split(local_heads, dim=2),
                                key.split(local_heads, dim=2),
                                value.split(local_heads, dim=2),
                            )
                        ],
                        dim=2,
                    )
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_ulysses_tail_pad_attention_matches_unpadded_attention(self):
        """Tail-padded Ulysses SDPA trims the pad instead of masking it."""
        from unittest.mock import patch

        import torch.nn.functional as F

        import sglang.multimodal_gen.runtime.models.dits.ovis_image as ovis_module
        from sglang.multimodal_gen.runtime.layers.attention import USPAttention
        from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
            OvisImageAttention as NativeAttention,
        )
        from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
            OvisImageSDPAImpl,
            OvisImageUSPAttention,
        )
        from sglang.multimodal_gen.runtime.server_args import (
            server_args as server_args_module,
        )

        heads, head_dim = 6, 128
        for dtype in (torch.bfloat16, torch.float32):
            # The unit conftest resets the global args per test; this check
            # targets the explicit torch SDPA backend.
            with (
                set_default_torch_dtype(dtype),
                patch.object(
                    server_args_module, "_global_server_args", self.server_args
                ),
            ):
                native = NativeAttention(
                    query_dim=heads * head_dim,
                    num_heads=heads,
                    dim_head=head_dim,
                    out_dim=heads * head_dim,
                    pre_only=True,
                ).cuda()
            attention = native.attn
            self.assertIsInstance(attention, OvisImageUSPAttention)
            self.assertIsInstance(attention.attn_impl, OvisImageSDPAImpl)
            for batch, valid, pad in ((1, 1025, 1), (4, 20, 2), (1, 16, 0)):
                with self.subTest(dtype=dtype, batch=batch, valid=valid, pad=pad):
                    generator = torch.Generator(device="cuda").manual_seed(valid)
                    q, k, v = (
                        torch.randn(
                            batch,
                            valid + pad,
                            heads,
                            head_dim,
                            device="cuda",
                            dtype=dtype,
                            generator=generator,
                        )
                        for _ in range(3)
                    )
                    meta = {"pad_start": valid, "pad_end": valid + pad} if pad else None
                    expected = F.scaled_dot_product_attention(
                        *(x[:, :valid].transpose(1, 2) for x in (q, k, v))
                    ).transpose(1, 2)
                    with (
                        patch.object(
                            ovis_module, "get_sequence_parallel_world_size", lambda: 2
                        ),
                        patch.object(
                            ovis_module, "get_ulysses_parallel_world_size", lambda: 2
                        ),
                        patch.object(
                            ovis_module, "_ipc_input_a2a_qkv", lambda q, k, v: None
                        ),
                        patch.object(
                            ovis_module,
                            "_usp_input_all_to_all_qkv",
                            lambda q, k, v: (q, k, v),
                        ),
                        patch.object(
                            ovis_module,
                            "_usp_output_all_to_all",
                            lambda x, head_dim: x,
                        ),
                        patch.object(
                            USPAttention, "forward", side_effect=AssertionError
                        ) as parent,
                        set_forward_context(current_timestep=0, attn_metadata=None),
                    ):
                        if not pad:
                            parent.side_effect = None
                            attention(q, k, v, attn_mask_meta=meta)
                            parent.assert_called_once()
                            continue
                        actual = attention(q, k, v, attn_mask_meta=meta)
                        # Explicit masks, replicated text and Ring keep the
                        # shared path.
                        parent.side_effect = None
                        mask = torch.ones(
                            batch, valid + pad, dtype=torch.bool, device="cuda"
                        )
                        attention(q, k, v, attn_mask=mask, attn_mask_meta=meta)
                        attention(q, k, v, attn_mask_meta=meta, num_replicated_prefix=1)
                        with patch.object(
                            ovis_module, "get_ulysses_parallel_world_size", lambda: 1
                        ):
                            attention(q, k, v, attn_mask_meta=meta)
                        self.assertEqual(parent.call_count, 3)
                    torch.testing.assert_close(
                        actual[:, :valid], expected, atol=0, rtol=0
                    )
                    self.assertTrue(torch.all(actual[:, valid:] == 0))

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
    def test_non_aligned_head_dimensions_match_reference(self):
        """A vector RoPE tail must not overwrite the next head or token."""
        for head_dim, axes in ((24, (4, 10, 10)), (18, (2, 8, 8))):
            kwargs = {
                **self.kwargs,
                "attention_head_dim": head_dim,
                "axes_dims_rope": axes,
            }
            config = OvisImageConfig()
            config.update_model_arch(kwargs)
            hidden_size = kwargs["num_attention_heads"] * head_dim
            for dtype in (torch.bfloat16, torch.float16):
                for batch_size, text_len, image_len in ((1, 5, 15), (4, 7, 25)):
                    with self.subTest(head_dim=head_dim, dtype=dtype, batch=batch_size):
                        torch.manual_seed(42)
                        reference = ReferenceDiT(**kwargs).cuda().to(dtype).eval()
                        with set_default_torch_dtype(dtype):
                            native = (
                                OvisImageTransformer2DModel(config, kwargs)
                                .cuda()
                                .eval()
                            )
                        native.load_weights(reference.state_dict().items())
                        txt_ids = torch.zeros(text_len, 3, device="cuda")
                        txt_ids[:, 1:] = torch.arange(text_len, device="cuda")[:, None]
                        img_ids = torch.zeros(image_len, 3, device="cuda")
                        img_ids[:, 1] = torch.arange(image_len, device="cuda") // 5
                        img_ids[:, 2] = torch.arange(image_len, device="cuda") % 5
                        rope = native.rotary_emb(torch.cat([txt_ids, img_ids]))
                        reference_rope = reference.pos_embed(
                            torch.cat([txt_ids, img_ids])
                        )
                        block_image = torch.randn(
                            batch_size,
                            image_len,
                            hidden_size,
                            device="cuda",
                            dtype=dtype,
                        )
                        block_text = torch.randn(
                            batch_size,
                            text_len,
                            hidden_size,
                            device="cuda",
                            dtype=dtype,
                        )
                        temb = torch.randn(
                            batch_size, hidden_size, device="cuda", dtype=dtype
                        )
                        for component in (
                            "transformer_blocks",
                            "single_transformer_blocks",
                        ):
                            with self.subTest(component=component):
                                with set_forward_context(
                                    current_timestep=0, attn_metadata=None
                                ):
                                    actual_context, actual_image = getattr(
                                        native, component
                                    )[0](block_image, block_text, temb, rope)
                                expected_context, expected_image = getattr(
                                    reference, component
                                )[0](block_image, block_text, temb, reference_rope)
                                torch.testing.assert_close(
                                    actual_context,
                                    expected_context,
                                    atol=0.05,
                                    rtol=0.02,
                                )
                                torch.testing.assert_close(
                                    actual_image,
                                    expected_image,
                                    atol=0.05,
                                    rtol=0.02,
                                )
                        image = torch.randn(
                            batch_size, image_len, 64, device="cuda", dtype=dtype
                        )
                        text = torch.randn(
                            batch_size, text_len, 16, device="cuda", dtype=dtype
                        )
                        text[:, -2:] = 0
                        timestep = torch.full((batch_size,), 1000.0, device="cuda")
                        with set_forward_context(
                            current_timestep=0, attn_metadata=None
                        ):
                            actual = native(image, text, timestep, rope)
                        expected = reference(
                            image,
                            text,
                            timestep.to(dtype) / 1000,
                            img_ids,
                            txt_ids,
                        ).sample
                        torch.testing.assert_close(
                            actual, expected, atol=0.05, rtol=0.02
                        )

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


class TestOvisImageQwenPrecision(CustomTestCase):
    @torch.no_grad()
    def test_tp_projection_matches_full_gemm(self):
        """TP shards must reconstruct the HF projection without split-K rounding."""
        from unittest.mock import patch

        import torch.nn.functional as F

        from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
        from sglang.multimodal_gen.runtime.models.encoders.qwen3 import (
            Qwen3HfRowParallelLinear,
        )

        inputs = torch.tensor(
            [[256, 1, -256, 0], [512, 2, -512, 0]], dtype=torch.bfloat16
        )
        weight = torch.tensor([[1, 1, 1, 1], [2, 1, 2, 1]], dtype=torch.bfloat16)
        bias = torch.tensor([0.5, -0.5], dtype=torch.bfloat16)
        for input_is_parallel in (False, True):
            for skip_bias_add in (False, True):
                for rank in range(2):
                    with self.subTest(
                        rank=rank,
                        input_is_parallel=input_is_parallel,
                        skip_bias_add=skip_bias_add,
                    ):
                        group = SimpleNamespace(world_size=2, rank_in_group=rank)
                        with (
                            patch(
                                "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                                return_value=group,
                            ),
                            set_default_torch_dtype(torch.bfloat16),
                        ):
                            layer = Qwen3HfRowParallelLinear(
                                4,
                                2,
                                input_is_parallel=input_is_parallel,
                                skip_bias_add=skip_bias_add,
                            )
                        layer.weight.weight_loader(layer.weight, weight)
                        layer.bias.weight_loader(layer.bias, bias)
                        collective_inputs = []

                        def gather_shards(value, dim=-1, tp_group=None):
                            self.assertIs(tp_group, group)
                            complete = inputs if dim == -1 else weight
                            torch.testing.assert_close(
                                value,
                                complete[:, rank * 2 : (rank + 1) * 2],
                                atol=0,
                                rtol=0,
                            )
                            collective_inputs.append(value.clone())
                            return complete.clone()

                        local_input = (
                            inputs[:, rank * 2 : (rank + 1) * 2]
                            if input_is_parallel
                            else inputs
                        )
                        with (
                            patch(
                                "sglang.multimodal_gen.runtime.models.encoders.qwen3.tensor_model_parallel_all_gather",
                                gather_shards,
                            ),
                            torch.autocast("cpu", dtype=torch.bfloat16),
                        ):
                            actual, output_bias = layer(local_input)
                        expected = F.linear(
                            inputs, weight, None if skip_bias_add else bias
                        )
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        self.assertTrue(
                            all(x.dtype == torch.bfloat16 for x in collective_inputs)
                        )
                        if skip_bias_add:
                            torch.testing.assert_close(
                                output_bias, bias, atol=0, rtol=0
                            )
                        else:
                            self.assertIsNone(output_bias)

    @torch.no_grad()
    def test_tp_projection_can_keep_local_outputs(self):
        """Callers requesting unreduced partials retain the original row contract."""
        from unittest.mock import patch

        import torch.nn.functional as F

        from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
        from sglang.multimodal_gen.runtime.models.encoders.qwen3 import (
            Qwen3HfRowParallelLinear,
        )

        inputs = torch.tensor([[256, 1, -256, 0]], dtype=torch.bfloat16)
        weight = torch.tensor([[1, 1, 1, 1]], dtype=torch.bfloat16)
        for rank in range(2):
            with self.subTest(rank=rank):
                group = SimpleNamespace(world_size=2, rank_in_group=rank)
                with (
                    patch(
                        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                        return_value=group,
                    ),
                    set_default_torch_dtype(torch.bfloat16),
                ):
                    layer = Qwen3HfRowParallelLinear(
                        4, 1, bias=False, reduce_results=False
                    )
                layer.weight.weight_loader(layer.weight, weight)
                local = inputs[:, rank * 2 : (rank + 1) * 2]
                actual, bias = layer(local)
                expected = F.linear(local, weight[:, rank * 2 : (rank + 1) * 2])
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertIsNone(bias)

    @torch.no_grad()
    def test_hf_rope_retains_linear_scaling(self):
        from unittest.mock import patch

        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding,
            apply_rotary_pos_emb,
        )

        from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3Attention

        reference_config = Qwen3Config(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters={
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 2.0,
            },
        )
        config = OvisImagePipelineConfig().text_encoder_configs[0]
        config.update_model_arch(reference_config.to_dict())
        group = SimpleNamespace(world_size=1, rank_in_group=0)
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                return_value=group,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.encoders.qwen3.get_tp_world_size",
                return_value=1,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.encoders.qwen3.LocalAttention",
                return_value=torch.nn.Identity(),
            ),
        ):
            native = Qwen3Attention(
                config,
                hidden_size=32,
                num_heads=4,
                num_kv_heads=2,
                rope_theta=1000000.0,
                rope_scaling=reference_config.rope_parameters,
            )
        reference = Qwen3RotaryEmbedding(reference_config)
        positions = torch.arange(17).expand(2, -1)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(123)
                query = torch.randn(2, 17, 32).to(dtype)
                key = torch.randn(2, 17, 16).to(dtype)
                cos, sin = reference(query, positions)
                expected_query, expected_key = apply_rotary_pos_emb(
                    query.unflatten(-1, (4, 8)).transpose(1, 2),
                    key.unflatten(-1, (2, 8)).transpose(1, 2),
                    cos,
                    sin,
                )
                actual_query, actual_key = native._apply_hf_rope(positions, query, key)
                torch.testing.assert_close(
                    actual_query,
                    expected_query.transpose(1, 2).flatten(2),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    actual_key,
                    expected_key.transpose(1, 2).flatten(2),
                    atol=0,
                    rtol=0,
                )


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
