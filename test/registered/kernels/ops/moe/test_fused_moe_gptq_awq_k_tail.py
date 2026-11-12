"""The GPTQ/AWQ MoE kernel must keep its weight loads inside K on a partial K tile."""

import itertools
import unittest

import torch
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _align(topk_ids, num_experts, block_m):
    """Sort routes by expert and pad each expert to whole BLOCK_SIZE_M tiles."""
    flat = topk_ids.flatten()
    sorted_ids, expert_ids = [], []
    for expert in range(num_experts):
        routes = (flat == expert).nonzero().flatten().tolist()
        routes += [flat.numel()] * (-len(routes) % block_m)
        sorted_ids += routes
        expert_ids += [expert] * (len(routes) // block_m)
    return (
        torch.tensor(sorted_ids, device="cuda", dtype=torch.int32),
        torch.tensor(expert_ids, device="cuda", dtype=torch.int32),
        torch.tensor([len(sorted_ids)], device="cuda", dtype=torch.int32),
    )


def _run(a, qweight, scale, qzeros, topk_ids, config, bits, group_size):
    m, top_k = topk_ids.shape
    sorted_ids, expert_ids, num_post_padded = _align(
        topk_ids=topk_ids,
        num_experts=qweight.shape[0],
        block_m=config["BLOCK_SIZE_M"],
    )
    output = torch.empty(
        m, top_k, qweight.shape[1], device="cuda", dtype=torch.bfloat16
    )
    invoke_fused_moe_kernel(
        A=a,
        B=qweight,
        bias=None,
        C=output,
        A_scale=None,
        B_scale=scale,
        B_zp=qzeros,
        topk_weights=torch.ones(m, top_k, device="cuda"),
        topk_ids=topk_ids,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=bits == 8,
        use_int4_w4a16=bits == 4,
        per_channel_quant=False,
        block_shape=[0, group_size],
    )
    torch.cuda.synchronize()
    return output


def _reference(a, q, zp, scale, group_size):
    """a: [m, k]; q, zp, scale: per route, [m, top_k, n, ...] unpacked."""
    weight = (q.float() - zp.float().repeat_interleave(group_size, -1)) * (
        scale.float().repeat_interleave(group_size, -1)
    )
    # The kernel rounds the dequantized weight to bf16 before the dot.
    weight = weight.to(torch.bfloat16).double()
    return torch.einsum("mk,mtnk->mtn", a.double(), weight)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestFusedMoeGptqAwqKTail(CustomTestCase):
    def test_partial_k_tile(self):
        torch.manual_seed(0)
        m, e, n, top_k, group_size = 7, 4, 128, 2, 32
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
        for k, bits, has_zp in itertools.product((96, 160), (4, 8), (False, True)):
            with self.subTest(k=k, bits=bits, has_zp=has_zp):
                a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 4
                q = torch.randint(
                    0, 2**bits, (e, n, k), device="cuda", dtype=torch.uint8
                )
                scale = torch.rand(e, n, k // group_size, device="cuda") / 16 + 1 / 64
                scale = scale.to(torch.bfloat16)
                if has_zp:
                    zp = torch.randint(
                        0, 2**bits, (e, n, k // group_size), device="cuda"
                    ).to(torch.uint8)
                else:
                    zp = torch.full_like(scale, 2 ** (bits - 1), dtype=torch.uint8)
                # int4 packs two K values per byte (even K in the low nibble) and
                # two N values per zero-point byte.
                if bits == 4:
                    qweight = q[..., 0::2] | (q[..., 1::2] << 4)
                    qzeros = zp[:, 0::2] | (zp[:, 1::2] << 4)
                else:
                    qweight, qzeros = q, zp
                topk_ids = torch.stack(
                    [torch.randperm(e, device="cuda")[:top_k] for _ in range(m)]
                ).to(torch.int32)
                output = _run(
                    a=a,
                    qweight=qweight,
                    scale=scale,
                    qzeros=qzeros if has_zp else None,
                    topk_ids=topk_ids,
                    config=config,
                    bits=bits,
                    group_size=group_size,
                )
                ids = topk_ids.long()
                reference = _reference(
                    a=a, q=q[ids], zp=zp[ids], scale=scale[ids], group_size=group_size
                )
                torch.testing.assert_close(
                    output.double(), reference, rtol=1e-2, atol=1e-2
                )

    def test_weight_tail_at_allocation_end(self):
        """Qwen3.8-Flash-Next's int4 down projection at TP 4, at decode size.

        Its K is 160 (moe_wna16 lowers the group size to 32) and the default
        decode config uses BLOCK_SIZE_K 64, so the last K tile is partial. The
        weights are allocated last: without the K mask, the last expert's last
        row reads past the end of the allocation (illegal memory access on L40S).
        """
        torch.manual_seed(0)
        m, e, n, k, top_k, group_size = 32, 512, 2560, 160, 10, 32
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 4
        scale = torch.full(
            (e, n, k // group_size), 1 / 64, device="cuda", dtype=torch.bfloat16
        )
        topk_ids = torch.stack(
            [torch.randperm(e, device="cuda")[:top_k] for _ in range(m)]
        ).to(torch.int32)
        topk_ids[0, 0] = e - 1
        qweight = torch.randint(
            0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8
        )
        output = _run(
            a=a,
            qweight=qweight,
            scale=scale,
            qzeros=None,
            topk_ids=topk_ids,
            config=config,
            bits=4,
            group_size=group_size,
        )
        # Check the first token, which routes to the last expert.
        packed = qweight[topk_ids[:1].long()]
        q = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)
        zp = torch.full_like(scale[: q.shape[1]], 8, dtype=torch.uint8)
        reference = _reference(
            a=a[:1],
            q=q,
            zp=zp.unsqueeze(0),
            scale=scale[: q.shape[1]].unsqueeze(0),
            group_size=group_size,
        )
        torch.testing.assert_close(output[:1].double(), reference, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
