import dataclasses
from typing import Optional

import pytest
import torch
from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

from sglang.srt.layers.moe.fused_moe_ibm.persistent_gg_bf16 import (
    grouped_gemm_persistent as ibm_gg_bf16,
)


def _fp8_perm(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    A permutation routine that works on fp8 types.
    """
    if torch.is_floating_point(m) and m.dtype.itemsize == 1:
        return m.view(dtype=torch.uint8)[idx, ...].view(dtype=m.dtype)
    else:
        return m[idx, ...]


def _moe_permute(
    curr_hidden_states: torch.Tensor,
    a1q_scale: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    num_experts: int,
    expert_map: Optional[torch.Tensor],
    block_size: int,
) -> tuple[
    torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Determine the sorted_token_ids, expert_ids for the given problem size.
    Permute the hidden states and scales according to `sorted_token_ids`.
    """
    top_k_num = topk_ids.size(1)

    tokens_in_chunk = curr_hidden_states.size(0)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids_cuda = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    sgl_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        cumsum_buffer,
    )

    inv_perm: Optional[torch.Tensor] = None

    num_tokens = top_k_num * tokens_in_chunk
    expert_ids_cuda = torch.repeat_interleave(expert_ids_cuda, block_size, dim=0)
    inv_perm = torch.argsort(sorted_ids_cuda)[:num_tokens]

    # Permute according to sorted token ids.
    sorted_ids_cuda = sorted_ids_cuda.clamp(max=num_tokens - 1)

    curr_hidden_states = _fp8_perm(curr_hidden_states, sorted_ids_cuda // top_k_num)

    if a1q_scale is not None:
        a1q_scale = a1q_scale[sorted_ids_cuda // top_k_num]

    return (curr_hidden_states, a1q_scale, sorted_ids_cuda, expert_ids_cuda, inv_perm)


@dataclasses.dataclass
class TestConfig:
    m: int
    n: int
    k: int
    e: int
    num_topk: int
    dtype: torch.dtype
    group_size: int


@dataclasses.dataclass
class TestTensors:
    a: torch.Tensor
    w: torch.Tensor
    topk_ids: torch.Tensor

    def __repr__(self):

        def describe_t(t: torch.Tensor, name: str) -> str:
            return f"  - {name} : {t.shape} {t.dtype} {t.device}\n"

        s = ""
        s += "Test Tensors :\n"
        s += describe_t(self.a, "a")
        s += describe_t(self.w, "w")
        s += describe_t(self.topk_ids, "topk_ids")
        return s

    @staticmethod
    def make(cfg: TestConfig) -> "TestTensors":
        m, n, k, e, num_topk, dtype = (
            cfg.m,
            cfg.n,
            cfg.k,
            cfg.e,
            cfg.num_topk,
            cfg.dtype,
        )
        device = "cuda"
        a = torch.randn((m, k), device=device, dtype=dtype) / 10
        w = torch.randn((e, n, k), device=device, dtype=dtype) / 10
        topk_ids = torch.randint(low=0, high=e, size=(m, num_topk), device=device)
        return TestTensors(a=a, w=w, topk_ids=topk_ids)


@dataclasses.dataclass
class GroupedTestTensors:
    a_grouped: torch.Tensor
    w: torch.Tensor
    expert_ids_grouped: torch.Tensor
    inv_perm: torch.Tensor

    group_size: int

    @staticmethod
    def make(tt: TestTensors, tc: TestConfig) -> "GroupedTestTensors":
        a_grouped, _, _, expert_ids, inv_perm = _moe_permute(
            curr_hidden_states=tt.a,
            a1q_scale=None,
            topk_ids=tt.topk_ids,
            num_experts=tc.e,
            expert_map=None,
            block_size=tc.group_size,
        )

        return GroupedTestTensors(
            a_grouped, tt.w, expert_ids, inv_perm, group_size=tc.group_size
        )


def torch_gg(gtt: GroupedTestTensors) -> torch.Tensor:
    group_size = gtt.group_size
    m = gtt.a_grouped.size(0)
    n = gtt.w.size(1)
    torch_out = torch.empty((m, n), dtype=gtt.a_grouped.dtype, device="cuda")
    expert_ids_cpu = gtt.expert_ids_grouped.to(device="cpu")
    num_groups = m // group_size

    for g in range(num_groups):
        s = g * group_size
        e = s + group_size
        ei = expert_ids_cpu[s]
        o = torch_out[s:e]
        a = gtt.a_grouped[s:e]
        torch.mm(a, gtt.w[ei].t(), out=o)

    return torch_out


Ms = [128, 256]
Ns = [512, 1024, 2048]
Ks = [1024]
Es = [128]
TOPKs = [4]
IMPLs = [ibm_gg_bf16]


@pytest.mark.parametrize("M", Ms)
@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("K", Ks)
@pytest.mark.parametrize("E", Es)
@pytest.mark.parametrize("TOPK", TOPKs)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("impl", IMPLs)
def test_ibm_bf16(M, N, K, E, TOPK, dtype, impl):

    config: TestConfig = TestConfig(
        m=M, n=N, k=K, e=E, num_topk=TOPK, dtype=dtype, group_size=128
    )
    tt: TestTensors = TestTensors.make(config)
    gtt: GroupedTestTensors = GroupedTestTensors.make(tt, config)

    print(
        f"M {M} N {N} K {K} E {E} TOPK {TOPK} | A {gtt.a_grouped.shape}  ...",
        flush=True,
    )

    ref_output = torch_gg(gtt)
    impl_output = impl(gtt.a_grouped, gtt.w, gtt.expert_ids_grouped)

    torch.testing.assert_close(ref_output, impl_output)
