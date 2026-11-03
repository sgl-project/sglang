import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
)
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * (1.0 + weight.float())).to(dtype)


def neox_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    half = rotary_dim // 2
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotated[..., :half], rotated[..., half:]
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return torch.cat(
        [first * cos - second * sin, second * cos + first * sin, passthrough], dim=-1
    )


class TestFusedQKRMSNormRoPEGate(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)
        self.tokens = 17
        self.num_q_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.rotary_dim = 64
        self.eps = 1e-6
        device, dtype = "cuda", torch.bfloat16
        self.q_gate = torch.randn(
            self.tokens,
            self.num_q_heads * 2 * self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.k = torch.randn(
            self.tokens, self.num_kv_heads * self.head_dim, device=device, dtype=dtype
        )
        self.q_weight = torch.randn(self.head_dim, device=device, dtype=dtype)
        self.k_weight = torch.randn(self.head_dim, device=device, dtype=dtype)
        inv_freq = 10000 ** (
            -torch.arange(self.rotary_dim // 2, device=device).float()
            * 2
            / self.rotary_dim
        )
        angles = torch.arange(512, device=device).float().unsqueeze(1) * inv_freq
        self.cos_sin_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)

    def call(self, positions, cos_sin_cache=None, mrope_axis_map=None, rotary_dim=None):
        if cos_sin_cache is None:
            cos_sin_cache = self.cos_sin_cache
        return fused_qk_gemma_rmsnorm_rope_gate(
            self.q_gate,
            self.k,
            self.q_weight,
            self.k_weight,
            cos_sin_cache,
            positions,
            self.eps,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            rotary_dim or self.rotary_dim,
            has_gate=True,
            mrope_axis_map=mrope_axis_map,
        )

    def build_mrope(self, mrope_section, interleaved):
        return MRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=2 * sum(mrope_section),
            max_position_embeddings=512,
            base=10000,
            is_neox_style=True,
            dtype=torch.bfloat16,
            mrope_section=mrope_section,
            mrope_interleaved=interleaved,
        ).to("cuda")

    def graph_buffer_positions(self):
        buffer = torch.zeros(3, 4 * self.tokens, dtype=torch.int64, device="cuda")
        buffer[:, : self.tokens] = torch.stack(
            [
                torch.arange(self.tokens, device="cuda") % 7,
                torch.arange(self.tokens, device="cuda") % 5 + 3,
                torch.arange(self.tokens, device="cuda") % 3 + 11,
            ]
        )
        return buffer[:, : self.tokens]

    def test_matches_reference_for_1d_positions(self):
        positions = torch.arange(self.tokens, device="cuda", dtype=torch.int64)
        q_out, k_out, gate_out = self.call(positions)

        packed = self.q_gate.view(self.tokens, self.num_q_heads, 2 * self.head_dim)
        cos, sin = self.cos_sin_cache[positions].chunk(2, dim=-1)
        want_q = neox_rope(
            gemma_rmsnorm(packed[..., : self.head_dim], self.q_weight, self.eps),
            cos,
            sin,
            self.rotary_dim,
        )
        want_k = neox_rope(
            gemma_rmsnorm(
                self.k.view(self.tokens, self.num_kv_heads, self.head_dim),
                self.k_weight,
                self.eps,
            ),
            cos,
            sin,
            self.rotary_dim,
        )
        torch.testing.assert_close(
            q_out.view_as(want_q).float(), want_q.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            k_out.view_as(want_k).float(), want_k.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            gate_out.float(), packed[..., self.head_dim :].float(), atol=0, rtol=0
        )

    def test_mrope_matches_the_rotary_module(self):
        """With t == h == w every layout agrees, so only distinct rows catch a wrong
        axis. Interleaved [11, 11, 10] is what Qwen3.6-35B-A3B ships, and [24, 20, 20]
        fills the head dimension, leaving no pass-through tail.
        """
        for section, interleaved in (
            ([11, 11, 10], False),
            ([11, 11, 10], True),
            ([24, 20, 20], True),
        ):
            with self.subTest(section=section, interleaved=interleaved):
                rope = self.build_mrope(section, interleaved)
                positions = self.graph_buffer_positions()

                q_in = self.q_gate.view(
                    self.tokens, self.num_q_heads, 2 * self.head_dim
                )[..., : self.head_dim].reshape(self.tokens, -1)
                want_q, want_k = rope.forward_native(
                    positions,
                    gemma_rmsnorm(
                        q_in.view(self.tokens, self.num_q_heads, self.head_dim),
                        self.q_weight,
                        self.eps,
                    ).reshape(self.tokens, -1),
                    gemma_rmsnorm(
                        self.k.view(self.tokens, self.num_kv_heads, self.head_dim),
                        self.k_weight,
                        self.eps,
                    ).reshape(self.tokens, -1),
                )

                q_out, k_out, _gate = self.call(
                    positions,
                    cos_sin_cache=rope.cos_sin_cache,
                    mrope_axis_map=rope.axis_map,
                    rotary_dim=rope.rotary_dim,
                )
                torch.testing.assert_close(
                    q_out.float(), want_q.float(), atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    k_out.float(), want_k.float(), atol=2e-2, rtol=2e-2
                )

    def test_rejects_positions_and_map_apart(self):
        flat = torch.arange(self.tokens, device="cuda", dtype=torch.int64)
        axis_map = self.build_mrope([11, 11, 10], interleaved=True).axis_map
        cases = ((flat.unsqueeze(0).repeat(3, 1), None), (flat, axis_map))
        for positions, axis_map_passed in cases:
            with self.subTest(mrope_positions=positions.dim() == 2):
                with self.assertRaises(AssertionError):
                    self.call(positions, mrope_axis_map=axis_map_passed)


def reference_qk_rope(
    q_gate,
    k,
    q_weight,
    k_weight,
    cache,
    positions,
    axis_map,
    q_heads,
    kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
    eps,
):
    """Keep the intermediate norm cast used by the separate model operators."""
    tokens = q_gate.shape[0]
    packed_q = q_gate.reshape(tokens, q_heads, (2 if has_gate else 1) * head_dim)
    q = packed_q[..., :head_dim]
    gate = packed_q[..., head_dim:].clone() if has_gate else None

    def norm(x, weight):
        x_fp32 = x.float()
        scaled = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
        return (scaled * (weight.float() + 1)).to(x.dtype)

    q = norm(q, q_weight)
    k = norm(k.reshape(tokens, kv_heads, head_dim), k_weight)
    half = rotary_dim // 2
    lanes = torch.arange(half, device=q.device)
    if positions.ndim == 1:
        selected = positions[:, None].expand(tokens, half)
    else:
        selected = positions[axis_map.long(), :].T
    cos = cache[selected, lanes].float()[:, None, :]
    sin = cache[selected, half + lanes].float()[:, None, :]

    def rotate(x):
        first = x[..., :half].float()
        second = x[..., half:rotary_dim].float()
        output = x.clone()
        output[..., :half] = (first * cos - second * sin).to(x.dtype)
        output[..., half:rotary_dim] = (second * cos + first * sin).to(x.dtype)
        return output.flatten(1)

    return rotate(q), rotate(k), gate


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "CUDA is required"
)
class TestGatedQKNormRoPE(unittest.TestCase):
    def test_startup_warmup_covers_mrope_cache_dtype_and_ragged_stride(self):
        import triton

        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            warmup_fused_qk_rmsnorm_rope_gate,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.bfloat16
        axes = tuple(i % 3 for i in range(32))
        axis_map = torch.tensor(axes, device=device, dtype=torch.int64)
        q_weight = torch.randn(256, device=device, dtype=dtype)
        k_weight = torch.randn_like(q_weight)
        for cache_dtype in (torch.float32, torch.bfloat16):
            self.assertTrue(
                warmup_fused_qk_rmsnorm_rope_gate(
                    device=device,
                    dtype=dtype,
                    weight_dtype=dtype,
                    cache_dtype=cache_dtype,
                    num_heads=32,
                    num_kv_heads=2,
                    head_size=256,
                    rotary_dim=64,
                    eps=1e-6,
                    has_gate=True,
                    mrope_axes=axes,
                )
            )
            angles = torch.randn(128, 32, device=device)
            cache = torch.cat((angles.cos(), angles.sin()), -1).to(cache_dtype)
            for index_dtype in (torch.int32, torch.int64):
                for rows in (1, 17, 64, 129):
                    with self.subTest(
                        cache_dtype=cache_dtype, index_dtype=index_dtype, rows=rows
                    ):
                        projection = torch.randn(
                            rows, 17408, device=device, dtype=dtype
                        )
                        q_gate, k, _ = projection.split((16384, 512, 512), -1)
                        positions = torch.randint(
                            128, (3, rows), device=device, dtype=index_dtype
                        )
                        expected = reference_qk_rope(
                            q_gate,
                            k,
                            q_weight,
                            k_weight,
                            cache,
                            positions,
                            axis_map,
                            32,
                            2,
                            256,
                            64,
                            True,
                            1e-6,
                        )
                        with patch.object(
                            triton.runtime.driver.active.utils,
                            "load_binary",
                            side_effect=AssertionError(
                                "MRoPE loaded a new CUDA module"
                            ),
                        ):
                            actual = fused_qk_gemma_rmsnorm_rope_gate(
                                q_gate,
                                k,
                                q_weight,
                                k_weight,
                                cache,
                                positions,
                                1e-6,
                                32,
                                2,
                                256,
                                64,
                                mrope_axis_map=axis_map,
                            )
                        for output, reference in zip(actual[:2], expected[:2]):
                            torch.testing.assert_close(
                                output, reference, rtol=1e-2, atol=1.5625e-2
                            )
                        torch.testing.assert_close(
                            actual[2], expected[2], rtol=0, atol=0
                        )

    def test_cp_output_gate_reuses_attention_storage(self):
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(397)
        for dtype in (torch.bfloat16, torch.float16):
            for tokens in (0, 1, 17, 128, 4096):
                with self.subTest(dtype=dtype, tokens=tokens):
                    output = torch.randn(
                        (tokens, 8192), device=device, dtype=dtype, generator=generator
                    )
                    gate = torch.randn(
                        output.shape, device=device, dtype=dtype, generator=generator
                    )
                    original_gate = gate.clone()
                    reference = fused_sigmoid_mul(output, gate, inplace=False)
                    actual = fused_sigmoid_mul(output, gate, inplace=True)
                    torch.cuda.synchronize(device)
                    self.assertIs(actual, output)
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    torch.testing.assert_close(gate, original_gate, rtol=0, atol=0)

    def test_actual_normalization_and_rotary_operators(self):
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
            mrope_axis_map_values,
        )
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
        from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(399)
        q_heads, kv_heads, head_dim = 32, 2, 256
        for dtype in (torch.bfloat16, torch.float16):
            for tokens in (17, 4096):
                for layout in ("1d", "sections", "interleaved", "glm"):
                    with self.subTest(dtype=dtype, tokens=tokens, layout=layout):
                        # The separate GLM CUDA kernel indexes its axis map up
                        # to head_size / 2. Use full RoPE for that comparator;
                        # partial GLM is checked against the independent oracle.
                        rotary_dim = head_dim if layout == "glm" else 64
                        rope_args = dict(
                            head_size=head_dim,
                            rotary_dim=rotary_dim,
                            max_position_embeddings=8192,
                            base=1000000,
                            is_neox_style=True,
                            dtype=dtype,
                        )
                        if layout == "1d":
                            rope = RotaryEmbedding(**rope_args).to(device=device)
                            positions = torch.randint(
                                0, 8192, (tokens,), device=device, generator=generator
                            )
                            axis_map = None
                        else:
                            sections = [32, 48, 48] if layout == "glm" else [11, 11, 10]
                            rope = MRotaryEmbedding(
                                **rope_args,
                                mrope_section=sections,
                                mrope_interleaved=layout != "sections",
                                mrope_interleaved_glm=layout == "glm",
                            ).to(device=device)
                            positions = torch.randint(
                                0, 8192, (3, tokens), device=device, generator=generator
                            )
                            axis_map = torch.tensor(
                                mrope_axis_map_values(
                                    rotary_dim,
                                    sections,
                                    interleaved=layout != "sections",
                                    interleaved_glm=layout == "glm",
                                ),
                                dtype=torch.int64,
                                device=device,
                            )
                        q_norm = GemmaRMSNorm(head_dim).to(device=device, dtype=dtype)
                        k_norm = GemmaRMSNorm(head_dim).to(device=device, dtype=dtype)
                        with torch.no_grad():
                            q_norm.weight.copy_(
                                torch.randn(
                                    head_dim,
                                    dtype=dtype,
                                    device=device,
                                    generator=generator,
                                )
                            )
                            k_norm.weight.copy_(
                                torch.randn(
                                    head_dim,
                                    dtype=dtype,
                                    device=device,
                                    generator=generator,
                                )
                            )
                        q_width, kv_width = q_heads * head_dim * 2, kv_heads * head_dim
                        packed = torch.randn(
                            tokens,
                            q_width + 2 * kv_width,
                            dtype=dtype,
                            device=device,
                            generator=generator,
                        )
                        q_gate, k, _ = packed.split((q_width, kv_width, kv_width), -1)
                        q, gate = q_gate.reshape(tokens, q_heads, head_dim * 2).chunk(
                            2, -1
                        )
                        expected_q = q_norm.forward_cuda(q).flatten(1)
                        expected_k = k_norm.forward_cuda(
                            k.reshape(tokens, kv_heads, head_dim)
                        ).flatten(1)
                        # These are the production separate CUDA operators, not
                        # only a mathematical oracle. MRoPE also rounds its cache.
                        expected_q, expected_k = rope.forward_cuda(
                            positions, expected_q, expected_k
                        )
                        actual_q, actual_k, actual_gate = (
                            fused_qk_gemma_rmsnorm_rope_gate(
                                q_gate,
                                k,
                                q_norm.weight,
                                k_norm.weight,
                                rope.cos_sin_cache,
                                positions,
                                1e-6,
                                q_heads,
                                kv_heads,
                                head_dim,
                                rotary_dim,
                                mrope_axis_map=axis_map,
                            )
                        )
                        tolerance = (
                            dict(rtol=1e-2, atol=1.5625e-2)
                            if dtype == torch.bfloat16
                            else dict(rtol=2e-3, atol=2e-3)
                        )
                        torch.testing.assert_close(actual_q, expected_q, **tolerance)
                        torch.testing.assert_close(actual_k, expected_k, **tolerance)
                        torch.testing.assert_close(actual_gate, gate, rtol=0, atol=0)

    def test_fp32_weight_cache_and_int32_positions(self):
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
            mrope_axis_map_values,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(398)
        for dtype in (torch.bfloat16, torch.float16):
            for layout in ("1d", "sections", "glm_partial"):
                mrope = layout != "1d"
                with self.subTest(dtype=dtype, layout=layout):
                    tokens, q_heads, kv_heads, head_dim, rotary_dim = 37, 4, 1, 128, 64
                    q_gate = torch.randn(
                        tokens,
                        q_heads * head_dim * 2,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                    k = torch.randn(
                        tokens,
                        kv_heads * head_dim,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                    q_weight = torch.randn(head_dim, device=device, generator=generator)
                    k_weight = torch.randn(head_dim, device=device, generator=generator)
                    angles = torch.randn(
                        128, rotary_dim // 2, device=device, generator=generator
                    )
                    cache = torch.cat((angles.cos(), angles.sin()), -1)
                    positions = torch.randint(
                        0,
                        128,
                        (3, tokens) if mrope else (tokens,),
                        dtype=torch.int32,
                        device=device,
                        generator=generator,
                    )
                    axis_map = (
                        (
                            torch.arange(
                                rotary_dim // 2, dtype=torch.int32, device=device
                            )
                            % 3
                        )
                        if mrope
                        else None
                    )
                    if layout == "glm_partial":
                        axis_map = torch.tensor(
                            mrope_axis_map_values(
                                rotary_dim,
                                [8, 12, 12],
                                interleaved=True,
                                interleaved_glm=True,
                            ),
                            dtype=torch.int32,
                            device=device,
                        )
                    expected = reference_qk_rope(
                        q_gate,
                        k,
                        q_weight,
                        k_weight,
                        cache,
                        positions,
                        axis_map,
                        q_heads,
                        kv_heads,
                        head_dim,
                        rotary_dim,
                        True,
                        1e-6,
                    )
                    actual = fused_qk_gemma_rmsnorm_rope_gate(
                        q_gate,
                        k,
                        q_weight,
                        k_weight,
                        cache,
                        positions,
                        1e-6,
                        q_heads,
                        kv_heads,
                        head_dim,
                        rotary_dim,
                        mrope_axis_map=axis_map,
                    )
                    tolerance = (
                        dict(rtol=1e-2, atol=1.5625e-2)
                        if dtype == torch.bfloat16
                        else dict(rtol=2e-3, atol=2e-3)
                    )
                    torch.testing.assert_close(actual[0], expected[0], **tolerance)
                    torch.testing.assert_close(actual[1], expected[1], **tolerance)
                    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_unsupported_metadata_keeps_fallback(self):
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            can_use_fused_qk_gemma_rmsnorm_rope_gate,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        arguments = dict(
            q_gate=torch.zeros(7, 512, dtype=torch.bfloat16, device=device),
            k=torch.zeros(7, 128, dtype=torch.bfloat16, device=device),
            q_weight=torch.zeros(128, dtype=torch.bfloat16, device=device),
            k_weight=torch.zeros(128, dtype=torch.bfloat16, device=device),
            cos_sin_cache=torch.zeros(8, 64, device=device),
            positions=torch.zeros(7, dtype=torch.int64, device=device),
            eps=1e-6,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=128,
            rotary_dim=64,
        )
        self.assertTrue(can_use_fused_qk_gemma_rmsnorm_rope_gate(**arguments))
        variants = (
            {"q_gate": arguments["q_gate"].float()},
            {"k": arguments["k"].half()},
            {
                "q_gate": torch.zeros(7, 1024, dtype=torch.bfloat16, device=device)[
                    :, ::2
                ]
            },
            {"q_weight": torch.zeros(256, dtype=torch.bfloat16, device=device)[::2]},
            {"cos_sin_cache": torch.zeros(8, 32, device=device)},
            {"positions": torch.zeros(14, dtype=torch.int64, device=device)[::2]},
            {"positions": torch.zeros(3, 7, dtype=torch.int64, device=device)},
            {"is_neox_style": False},
            {"rotary_dim": 63},
            {"eps": 0},
        )
        for variant in variants:
            with self.subTest(changed=tuple(variant)):
                self.assertFalse(
                    can_use_fused_qk_gemma_rmsnorm_rope_gate(**(arguments | variant))
                )

    def test_strided_projection_parity(self):
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(397)
        case_count = 0
        for dtype in (torch.bfloat16, torch.float16):
            for q_heads, kv_heads, head_dim, rotary_dim in (
                (32, 2, 256, 64),
                (4, 1, 128, 128),
            ):
                for tokens in (0, 1, 17, 4096):
                    for has_gate in (False, True):
                        for mrope in (False, True):
                            with self.subTest(
                                dtype=dtype,
                                tokens=tokens,
                                head_dim=head_dim,
                                rotary_dim=rotary_dim,
                                has_gate=has_gate,
                                mrope=mrope,
                            ):
                                q_width = q_heads * head_dim * (2 if has_gate else 1)
                                kv_width = kv_heads * head_dim
                                # Slices of a joint QKV projection have row padding.
                                packed = torch.randn(
                                    tokens,
                                    q_width + 2 * kv_width,
                                    dtype=dtype,
                                    device=device,
                                    generator=generator,
                                )
                                before = packed.clone()
                                q_gate = packed[:, :q_width]
                                k = packed[:, q_width : q_width + kv_width]
                                q_weight = torch.randn(
                                    head_dim,
                                    device=device,
                                    dtype=dtype,
                                    generator=generator,
                                )
                                k_weight = torch.randn(
                                    head_dim,
                                    device=device,
                                    dtype=dtype,
                                    generator=generator,
                                )
                                angles = torch.randn(
                                    8192,
                                    rotary_dim // 2,
                                    device=device,
                                    generator=generator,
                                )
                                cache = torch.cat((angles.cos(), angles.sin()), -1).to(
                                    dtype
                                )
                                if mrope:
                                    # Distinct temporal/spatial positions detect lane errors.
                                    positions = torch.randint(
                                        0,
                                        cache.shape[0],
                                        (3, tokens),
                                        device=device,
                                        generator=generator,
                                    )
                                    axis_map = (
                                        torch.arange(rotary_dim // 2, device=device) % 3
                                    )
                                    axis_map = axis_map.to(torch.int32)
                                else:
                                    positions = torch.randint(
                                        0,
                                        cache.shape[0],
                                        (tokens,),
                                        device=device,
                                        generator=generator,
                                    )
                                    axis_map = None
                                expected = reference_qk_rope(
                                    q_gate,
                                    k,
                                    q_weight,
                                    k_weight,
                                    cache,
                                    positions,
                                    axis_map,
                                    q_heads,
                                    kv_heads,
                                    head_dim,
                                    rotary_dim,
                                    has_gate,
                                    1e-6,
                                )
                                actual = fused_qk_gemma_rmsnorm_rope_gate(
                                    q_gate,
                                    k,
                                    q_weight,
                                    k_weight,
                                    cache,
                                    positions,
                                    1e-6,
                                    q_heads,
                                    kv_heads,
                                    head_dim,
                                    rotary_dim,
                                    has_gate=has_gate,
                                    mrope_axis_map=axis_map,
                                )
                                tolerance = (
                                    dict(rtol=1e-2, atol=1.5625e-2)
                                    if dtype == torch.bfloat16
                                    else dict(rtol=2e-3, atol=2e-3)
                                )
                                torch.testing.assert_close(
                                    actual[0], expected[0], **tolerance
                                )
                                torch.testing.assert_close(
                                    actual[1], expected[1], **tolerance
                                )
                                if has_gate:
                                    torch.testing.assert_close(
                                        actual[2], expected[2], rtol=0, atol=0
                                    )
                                else:
                                    self.assertIsNone(actual[2])
                                # The attention V slice shares the input allocation.
                                torch.testing.assert_close(
                                    packed, before, rtol=0, atol=0
                                )
                                case_count += 1
        torch.cuda.synchronize(device)
        self.assertEqual(case_count, 64)
        print(
            f"Qwen3.5 fused QK/RoPE: {case_count} GPU parity cases passed", flush=True
        )


if __name__ == "__main__":
    unittest.main()
