# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free contracts against the official Diffusers Ovis implementation."""

import unittest
from types import SimpleNamespace

import torch
from diffusers.models.transformers.transformer_ovis_image import (
    OvisImageTransformer2DModel as ReferenceDiT,
)

from sglang.multimodal_gen.configs.models.dits.ovis_image import OvisImageConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ImagePipelineConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
    OvisImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.test_utils import CustomTestCase


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
            pipeline_config=ImagePipelineConfig(dit_config=cls.config),
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
