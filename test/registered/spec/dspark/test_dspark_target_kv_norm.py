"""Narrow rounding boundaries in KV-input DSpark normalization and pool writes."""

import unittest
from itertools import product
from types import MethodType, SimpleNamespace

import torch
from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
    fused_kv_norm_rope_write,
)
from sglang.srt.models.dspark_target_kv import (
    TargetKVAttention,
    TargetKVRMSNorm,
    TargetKVSiluAndMul,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dflash_utils import table_qk_norm_rope_
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def norm_reference(x, weight, *, cast_before_weight):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    if cast_before_weight:
        value = value.to(x.dtype).float()
    return (value * weight.float()).to(x.dtype)


def patterned(shape):
    size = torch.Size(shape).numel()
    return ((torch.arange(size, device="cuda") % 4 + 1).to(torch.bfloat16)).reshape(
        shape
    )


def rope_inputs(dim, rows, nonzero):
    frequency = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.arange(4096).float()[:, None] * frequency
    table = torch.cat((angles.cos(), angles.sin()), -1).cuda()
    positions = torch.tensor([0, 1, 159, 162, 2048, 4095][:rows], device="cuda")
    if not nonzero:
        positions.zero_()
    return table, positions


def rope_reference(value, table, positions):
    cosine, sine = table[positions].to(value.dtype).chunk(2, -1)
    cosine, sine = cosine[:, None], sine[:, None]
    first, second = value.chunk(2, -1)
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine), -1
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTargetKVNorm(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    @torch.no_grad()
    def test_residual_and_weight_rounding_match_training(self):
        for hidden in (128, 1024):
            with self.subTest(hidden=hidden):
                norm = TargetKVRMSNorm(hidden, 1e-6).cuda().bfloat16()
                norm.weight.fill_(1.703125)
                x = patterned((3, hidden))
                residual = x * 0.0078125 + 0.00390625
                summed = x + residual
                expected = norm_reference(summed, norm.weight, cast_before_weight=True)
                actual, saved = norm(x, residual)
                torch.testing.assert_close(saved, summed, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(norm(summed), expected, rtol=0, atol=0)
                legacy = norm_reference(summed, norm.weight, cast_before_weight=False)
                self.assertFalse(torch.equal(expected, legacy))

    @torch.no_grad()
    def test_qk_fusion_preserves_v_and_cast_order(self):
        for dim in (64, 128):
            for hf, rounded in product((False, True), repeat=2):
                with self.subTest(dim=dim, hf=hf, rounded=rounded):
                    q_heads, k_heads, rows = 4, 2, 5
                    # A padded row stride exercises the in-place fused QKV view.
                    storage = patterned((rows, (q_heads + 2 * k_heads) * dim + 16))
                    qkv = storage[:, :-16]
                    initial = qkv.clone()
                    weight = torch.full(
                        (dim,), 1.703125, device="cuda", dtype=qkv.dtype
                    )
                    table, positions = rope_inputs(dim, rows, rounded)
                    table_qk_norm_rope_(
                        qkv,
                        positions,
                        weight,
                        weight,
                        table,
                        q_heads,
                        k_heads,
                        dim,
                        1e-6,
                        cast_x_before_out_mul=hf,
                        round_rope_intermediates=rounded,
                    )
                    qk_size = (q_heads + k_heads) * dim
                    expected = norm_reference(
                        initial[:, :qk_size].reshape(rows, -1, dim),
                        weight,
                        cast_before_weight=hf,
                    )
                    expected = rope_reference(expected, table, positions).reshape(
                        rows, -1
                    )
                    torch.testing.assert_close(
                        qkv[:, :qk_size], expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        qkv[:, qk_size:], initial[:, qk_size:], rtol=0, atol=0
                    )

    @torch.no_grad()
    def test_context_fusion_honors_norm_mode_and_commit_ownership(self):
        dim, heads, layers, rows = 128, 2, 2, 6
        kv_size = dim * heads
        kv = patterned((rows, layers * 2 * kv_size))
        locs = torch.tensor([7, 5, 9, 3, 1, 11], device="cuda")
        commits = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
        weights = torch.full((layers, dim), 1.703125, device="cuda", dtype=kv.dtype)
        for hf, rounded in product((False, True), repeat=2):
            with self.subTest(hf=hf, rounded=rounded):
                table, positions = rope_inputs(dim, rows, rounded)
                buffers = [
                    [
                        torch.full((16, heads, dim), -42, device="cuda", dtype=kv.dtype)
                        for _ in range(2)
                    ]
                    for _ in range(layers)
                ]
                meta = torch.tensor(
                    [
                        [k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0)]
                        for k, v in buffers
                    ],
                    device="cuda",
                    dtype=torch.int64,
                )
                fused_kv_norm_rope_write(
                    kv,
                    meta,
                    weights,
                    table,
                    positions,
                    locs,
                    layers,
                    kv_size,
                    dim,
                    1e-6,
                    commits,
                    3,
                    cast_x_before_out_mul=hf,
                    round_rope_intermediates=rounded,
                )
                valid_rows = [0, 3, 4]
                for index, (keys, values) in enumerate(buffers):
                    expected_keys = torch.full_like(keys, -42)
                    expected_values = torch.full_like(values, -42)
                    source = kv.view(rows, layers, 2, heads, dim)[:, index]
                    normalized = norm_reference(
                        source[valid_rows, 0], weights[index], cast_before_weight=hf
                    )
                    expected_keys[locs[valid_rows]] = rope_reference(
                        normalized, table, positions[valid_rows]
                    )
                    expected_values[locs[valid_rows]] = source[valid_rows, 1]
                    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
                    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)

    @torch.no_grad()
    def test_rope_fallback_and_graph_replay_preserve_narrow_products(self):
        rows, heads, dim = 6, 2, 128
        table, positions = rope_inputs(dim, rows, True)
        attn = SimpleNamespace(
            head_dim=dim, rotary_emb=SimpleNamespace(cos_sin_cache=table)
        )
        attn._rotate = MethodType(TargetKVAttention._rotate, attn)
        source = patterned((rows, 3 * heads * dim))
        qkv = torch.empty_like(source)
        weight = torch.full((dim,), 1.703125, device="cuda", dtype=source.dtype)

        def run():
            qkv.copy_(source)
            table_qk_norm_rope_(
                qkv,
                positions,
                weight,
                weight,
                table,
                heads,
                heads,
                dim,
                1e-6,
                cast_x_before_out_mul=True,
                round_rope_intermediates=True,
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for offset in (0, 7):
            source.add_(0.125)
            positions.add_(offset).remainder_(len(table))
            graph.replay()
            expected = norm_reference(
                source[:, : 2 * heads * dim].reshape(rows, -1, dim),
                weight,
                cast_before_weight=True,
            )
            rotated = rope_reference(expected, table, positions)
            torch.testing.assert_close(
                qkv[:, : 2 * heads * dim].reshape_as(rotated), rotated, rtol=0, atol=0
            )
            q, k = expected.flatten(1).chunk(2, -1)
            fallback_q, fallback_k = TargetKVAttention.apply_qk_rope(
                attn, positions, q, k
            )
            torch.testing.assert_close(
                torch.cat((fallback_q, fallback_k), -1),
                rotated.flatten(1),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                TargetKVAttention.apply_k_rope(attn, positions, k),
                fallback_k,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                qkv[:, 2 * heads * dim :], source[:, 2 * heads * dim :], rtol=0, atol=0
            )

    @torch.no_grad()
    def test_silu_rounds_before_multiplication(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                gate = torch.linspace(-5, 5, 1024, device="cuda", dtype=dtype)
                up = torch.linspace(0.3, 2.5, 1024, device="cuda", dtype=dtype)
                x = torch.cat((gate, up))[None]
                expected = torch.nn.functional.silu(gate) * up
                actual = TargetKVSiluAndMul()(x)
                torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
                unrounded = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                    dtype
                )
                self.assertFalse(torch.equal(expected, unrounded))


if __name__ == "__main__":
    unittest.main()
