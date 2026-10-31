"""Block-FP8 scales must broadcast over the right axes after swapping A and B."""

import itertools
import unittest

import torch
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import fused_moe_kernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "Requires SM90 for the FP8 swap_ab path",
)
class TestFusedMoeBlockFp8Scales(CustomTestCase):
    def test_scale_broadcast(self):
        torch.manual_seed(0)
        m, n, k = 19, 256, 256
        block_m, group_n, group_k = 32, 128, 128
        # Small integers keep the FP8 dot exact under Hopper's fast accumulation.
        a = torch.randint(-2, 3, (m, k), device="cuda").float()
        b = torch.randint(-2, 3, (1, n, k), device="cuda").float()
        a, b = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
        a_scale = torch.rand(m, k // group_k, device="cuda") + 0.1
        b_scale = torch.rand(1, n // group_n, k // group_k, device="cuda") + 0.1
        weights = torch.rand(m, 1, device="cuda")
        # Permute valid tokens and pad the last tile to exercise output indexing.
        sorted_ids = torch.cat(
            [
                torch.arange(m - 1, -1, -1, device="cuda", dtype=torch.int32),
                torch.full((block_m - m,), m, device="cuda", dtype=torch.int32),
            ]
        )
        expert_ids = torch.zeros(1, device="cuda", dtype=torch.int32)
        padded = torch.tensor([block_m], device="cuda", dtype=torch.int32)
        # Dequantize before matmul, independently of the kernel's tile layout.
        a_dequant = a.float() * a_scale.repeat_interleave(group_k, dim=1)
        b_dequant = b[0].float() * b_scale[0].repeat_interleave(
            group_n, dim=0
        ).repeat_interleave(group_k, dim=1)
        reference = (a_dequant.double() @ b_dequant.double().T).float() * weights

        # Scalar weight scales (< and == group_n), plus the vector-scale branch.
        for swap_ab, block_n in itertools.product((False, True), (64, 128, 256)):
            with self.subTest(swap_ab=swap_ab, block_n=block_n):
                output = torch.empty(m, n, device="cuda", dtype=torch.float32)
                fused_moe_kernel[(n // block_n,)](
                    a_ptr=a,
                    a_desc=None,
                    b_ptr=b,
                    b_desc=None,
                    bias_ptr=None,
                    c_ptr=output,
                    a_scale_ptr=a_scale,
                    b_scale_ptr=b_scale,
                    topk_weights_ptr=weights,
                    sorted_token_ids_ptr=sorted_ids,
                    expert_ids_ptr=expert_ids,
                    num_tokens_post_padded_ptr=padded,
                    add_mask_ptr=None,
                    N=n,
                    K=k,
                    EM=block_m,
                    num_valid_tokens=m,
                    stride_am=a.stride(0),
                    stride_ak=a.stride(1),
                    stride_be=b.stride(0),
                    stride_bk=b.stride(2),
                    stride_bn=b.stride(1),
                    stride_bias_e=0,
                    stride_bias_n=0,
                    stride_cm=output.stride(0),
                    stride_cn=output.stride(1),
                    stride_asm=a_scale.stride(0),
                    stride_ask=a_scale.stride(1),
                    stride_bse=b_scale.stride(0),
                    stride_bsk=b_scale.stride(2),
                    stride_bsn=b_scale.stride(1),
                    group_n=group_n,
                    group_k=group_k,
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=group_k,
                    GROUP_SIZE_M=1,
                    MUL_ROUTED_WEIGHT=True,
                    top_k=1,
                    compute_type=tl.float32,
                    use_fp8_w8a8=True,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    per_channel_quant=False,
                    even_Ks=True,
                    c_sorted=False,
                    swap_ab=swap_ab,
                    FUSE_ADD_TO_OUTPUT=False,
                    MASK_OUTPUT=False,
                    FUSE_SUM_ALL_REDUCE=False,
                    ROUTER_TOPK=1,
                    num_warps=4,
                    num_stages=2,
                )
                torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
