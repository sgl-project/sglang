"""Narrow rounding boundaries in KV-input DSpark normalization and pool writes."""

import unittest

import torch
from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
    fused_kv_norm_rope_write,
)
from sglang.srt.models.dspark_target_kv import TargetKVRMSNorm
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
            for hf in (False, True):
                with self.subTest(dim=dim, hf=hf):
                    q_heads, k_heads, rows = 4, 2, 5
                    # A padded row stride exercises the in-place fused QKV view.
                    storage = patterned((rows, (q_heads + 2 * k_heads) * dim + 16))
                    qkv = storage[:, :-16]
                    initial = qkv.clone()
                    weight = torch.full(
                        (dim,), 1.703125, device="cuda", dtype=qkv.dtype
                    )
                    table = torch.cat(
                        (torch.ones(1, dim // 2), torch.zeros(1, dim // 2)), -1
                    ).cuda()
                    table_qk_norm_rope_(
                        qkv,
                        torch.zeros(rows, device="cuda", dtype=torch.long),
                        weight,
                        weight,
                        table,
                        q_heads,
                        k_heads,
                        dim,
                        1e-6,
                        cast_x_before_out_mul=hf,
                    )
                    qk_size = (q_heads + k_heads) * dim
                    expected = norm_reference(
                        initial[:, :qk_size].reshape(rows, -1, dim),
                        weight,
                        cast_before_weight=hf,
                    ).reshape(rows, -1)
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
        positions = torch.zeros(rows, device="cuda", dtype=torch.long)
        locs = torch.tensor([7, 5, 9, 3, 1, 11], device="cuda")
        commits = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
        table = torch.cat(
            (torch.ones(1, dim // 2), torch.zeros(1, dim // 2)), -1
        ).cuda()
        weights = torch.full((layers, dim), 1.703125, device="cuda", dtype=kv.dtype)
        for hf in (False, True):
            with self.subTest(hf=hf):
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
                )
                valid_rows = [0, 3, 4]
                for index, (keys, values) in enumerate(buffers):
                    expected_keys = torch.full_like(keys, -42)
                    expected_values = torch.full_like(values, -42)
                    source = kv.view(rows, layers, 2, heads, dim)[:, index]
                    expected_keys[locs[valid_rows]] = norm_reference(
                        source[valid_rows, 0], weights[index], cast_before_weight=hf
                    )
                    expected_values[locs[valid_rows]] = source[valid_rows, 1]
                    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
                    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
