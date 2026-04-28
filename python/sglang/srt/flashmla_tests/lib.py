import dataclasses
import enum
import os
import random
from typing import List, Optional

from sglang.srt.utils import is_hip

if is_hip():
    pass
else:
    import flash_mla

import torch

from . import kernelkit as kk
from . import quant


class TestTarget(enum.Enum):
    FWD = 0
    DECODE = 1


@dataclasses.dataclass
class ExtraTestParamForDecode:
    b: int
    is_varlen: bool
    have_zero_seqlen_k: bool
    extra_s_k: Optional[int] = None
    extra_topk: Optional[int] = None
    block_size: int = 64
    extra_block_size: Optional[int] = None
    have_extra_topk_length: bool = False


@dataclasses.dataclass
class TestParam:
    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    seed: int = -1  # -1: to be filled automatically
    check_correctness: bool = True
    is_all_indices_invalid: bool = (
        False  # All indices are invalid, i.e., all indices are set to a large number (e.g., 2147483647)
    )
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    decode: Optional[ExtraTestParamForDecode] = None


@dataclasses.dataclass
class RawTestParamForDecode:
    """
    "Flattened" test parameters for decoding test

    In our test script, to maintain compatibility with TestParam, we embed decode-only parameters into TestParam.decode, which is not very convenient when construct testcases. So here we have a "flattened" version of test parameters for decoding test.
    """

    b: int
    h_q: int
    s_q: int
    h_kv: int
    s_kv: int
    is_varlen: bool
    topk: int
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    have_topk_length: bool = False
    enable_attn_sink: bool = True
    extra_s_k: Optional[int] = None
    extra_topk: Optional[int] = None
    block_size: int = 64
    extra_block_size: Optional[int] = None
    have_extra_topk_length: bool = False
    d_qk: int = 576  # Q/K head dim (= dv + RoPE dim)
    d_v: int = 512  # V head dim
    check_correctness: bool = True
    num_runs: int = 10
    seed: int = -1

    def to_test_param(self) -> TestParam:
        return TestParam(
            self.s_q,
            self.s_kv,
            self.topk,
            self.h_q,
            self.h_kv,
            self.d_qk,
            self.d_v,
            self.seed,
            self.check_correctness,
            self.is_all_indices_invalid,
            self.num_runs,
            self.enable_attn_sink,
            self.have_topk_length,
            decode=ExtraTestParamForDecode(
                self.b,
                self.is_varlen,
                self.have_zero_seqlen_k,
                self.extra_s_k,
                self.extra_topk,
                self.block_size,
                self.extra_block_size,
                self.have_extra_topk_length,
            ),
        )


@dataclasses.dataclass
class Testcase:
    p: TestParam
    doubt: torch.Tensor  # [s_q, h_q, d_v]
    q: torch.Tensor  # [s_q, h_q, d_qk]
    kv: torch.Tensor  # [s_kv, h_kv, d_qk]
    indices: torch.Tensor  # [s_q, h_kv, topk]
    sm_scale: float
    attn_sink: Optional[torch.Tensor]  # [h_q]
    topk_length: Optional[torch.Tensor]  # [s_q]


def _randperm_batch(
    batch_size: int, perm_range: torch.Tensor, perm_size: int, paddings: List[int]
) -> torch.Tensor:
    """
    Generate random permutations in batch
    The return tensor, denoted as `res`, has a shape of [batch_size, perm_size]. `0 <= res[i, :] < perm_range[i]` holds.
    Values within each row are unique.
    If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains values in `[0, perm_range[i])` as many as possible, and the rest are filled with `padding`.
    """
    assert not torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
    rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32)
    rand[
        torch.arange(0, perm_range_max).broadcast_to(batch_size, perm_range_max)
        >= perm_range.view(batch_size, 1)
    ] = float(
        "-inf"
    )  # Fill invalid positions, so that the following `topk` operators will select positions within `perm_range` first
    res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
    if len(paddings) == 1:
        res[res >= perm_range.view(batch_size, 1)] = paddings[0]
    else:
        fillers = torch.tensor(paddings, dtype=torch.int32).index_select(
            0, torch.randint(0, len(paddings), (res.numel(),), dtype=torch.int32)
        )
        res.masked_scatter_(res >= perm_range.view(batch_size, 1), fillers)
    torch.use_deterministic_algorithms(False)
    return res


def generate_testcase(t: TestParam) -> Testcase:
    kk.set_random_seed(t.seed)
    q = (
        torch.randn((t.s_q, t.h_q, t.d_qk), dtype=torch.bfloat16) / 10
        + (random.random() - 0.5) / 10
    )
    kv = (
        torch.randn((t.s_kv, t.h_kv, t.d_qk), dtype=torch.bfloat16) / 10
        + (random.random() - 0.5) / 10
    )
    do = (
        torch.randn((t.s_q, t.h_q, t.d_v), dtype=torch.bfloat16) / 10
        + (random.random() - 0.5) / 10
    )

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    do.clamp_(-10, 10)

    invalid_indices_candidate = [
        -2147483648,
        -123456,
        -1,
        t.s_kv,
        114514,
        1919810,
        2147480000,
        2147483647,
    ]
    indices = _randperm_batch(
        t.s_q,
        torch.full((t.s_q,), t.s_kv, dtype=torch.int32),
        t.topk,
        invalid_indices_candidate,
    ).view(t.s_q, t.h_kv, t.topk)

    if t.is_all_indices_invalid:
        all_indices_invalid_mask = torch.randn(t.s_q, device="cpu") < -2
        indices[all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)] = (
            random.choice(invalid_indices_candidate)
        )
    indices = indices.to(q.device)

    attn_sink = None
    if t.have_attn_sink:
        attn_sink = torch.randn((t.h_q,), dtype=torch.float32)
        mask = torch.randn((t.h_q,), dtype=torch.float32)
        attn_sink[mask < -0.5] = float("-inf")
        attn_sink[mask > +0.5] = float("+inf")

    topk_length = None
    if t.have_topk_length:
        topk_length = torch.randint(
            0, max(t.topk + 1, 64), (t.s_q,), dtype=torch.int32, device=q.device
        ).clamp_max(t.topk)

    q = kk.non_contiguousify(q)
    kv = kk.non_contiguousify(kv)
    do = kk.non_contiguousify(do)
    indices = kk.non_contiguousify(indices)

    return Testcase(
        p=t,
        doubt=do,
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.5,  # Otherwise dK is too small compared to dV
        attn_sink=attn_sink,
        topk_length=topk_length,
    )


@dataclasses.dataclass
class KVScope:
    t: TestParam
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor
    blocked_k: torch.Tensor
    abs_indices: torch.Tensor
    indices_in_kvcache: torch.Tensor
    topk_length: Optional[torch.Tensor]
    blocked_k_quantized: Optional[torch.Tensor] = None

    def quant_and_dequant_(self):
        """
        For FP8 cases, we need to quantize the KV cache for Flash MLA.
        Besides, the quantization error may be too large to be distinguished from wrong kernels, so we de-quantize kvcache here to mitigate quantization error
        """
        fp8_kvcache_layout = None
        if self.t.d_qk == 576:
            fp8_kvcache_layout = quant.FP8KVCacheLayout.V32_FP8Sparse
        elif self.t.d_qk == 512:
            assert self.abs_indices is not None
            fp8_kvcache_layout = quant.FP8KVCacheLayout.MODEL1_FP8Sparse
        else:
            assert False
        self.blocked_k_quantized = quant.quantize_k_cache(
            self.blocked_k, fp8_kvcache_layout
        )
        blocked_k_dequantized = quant.dequantize_k_cache(
            self.blocked_k_quantized, fp8_kvcache_layout
        )
        self.blocked_k = blocked_k_dequantized

    def get_kvcache_for_flash_mla(self) -> torch.Tensor:
        """
        Return the quantized blocked_k for Flash MLA
        """
        assert (
            self.blocked_k_quantized is not None
        ), "Please call `quant_and_dequant_` first before calling `get_kvcache_for_flash_mla`"
        return self.blocked_k_quantized

    def apply_perm(self, perm: torch.Tensor) -> "KVScope":
        """
        Apply a batch permutation to this KVScope. Used for batch-invariance test
        """
        new_kvscope = KVScope(
            self.t,
            self.cache_seqlens[perm],
            self.block_table[perm],
            self.blocked_k,
            self.abs_indices[perm],
            self.indices_in_kvcache[perm],
            self.topk_length[perm] if self.topk_length is not None else None,
            self.blocked_k_quantized,
        )
        return new_kvscope


@dataclasses.dataclass
class TestcaseForDecode:
    p: TestParam
    q: torch.Tensor  # [b, s_q, h_q, d_qk]
    attn_sink: Optional[torch.Tensor]  # [h_q]
    sm_scale: float
    kv_scope: KVScope
    extra_kv_scope: Optional[KVScope]


def generate_testcase_for_decode(t: TestParam) -> TestcaseForDecode:
    kk.set_random_seed(t.seed)
    assert t.h_q % t.h_kv == 0
    assert t.decode is not None

    q = torch.randn((t.decode.b, t.s_q, t.h_q, t.d_qk))
    q.clamp_(min=-1.0, max=1.0)

    attn_sink = None
    if t.have_attn_sink:
        attn_sink = torch.randn((t.h_q,), dtype=torch.float32)
        inf_mask = torch.randn((t.h_q,), dtype=torch.float32)
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = float("-inf")

    def generate_one_k_scope(
        s_k: int,
        block_size: int,
        topk: int,
        is_varlen: bool,
        have_zero_seqlen: bool,
        is_all_indices_invalid: bool,
        have_topk_length: bool,
    ) -> KVScope:
        b = t.decode.b  # type: ignore
        cache_seqlens_cpu = torch.full((b,), s_k, dtype=torch.int32, device="cpu")
        if is_varlen:
            for i in range(b):
                cache_seqlens_cpu[i] = max(random.normalvariate(s_k, s_k / 2), t.s_q)

        if have_zero_seqlen:
            zeros_mask = torch.randn(b, dtype=torch.float32, device="cpu") > 0
            cache_seqlens_cpu[zeros_mask] = 0

        max_seqlen_alignment = 4 * block_size
        max_seqlen_pad = (
            max(kk.cdiv(int(cache_seqlens_cpu.max().item()), max_seqlen_alignment), 1)
            * max_seqlen_alignment
        )
        cache_seqlens = cache_seqlens_cpu.cuda()

        assert max_seqlen_pad % block_size == 0
        block_table = torch.arange(
            b * max_seqlen_pad // block_size, dtype=torch.int32
        ).view(b, max_seqlen_pad // block_size)
        block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(
            b, -1
        )

        blocked_k = (
            kk.gen_non_contiguous_randn_tensor(
                (block_table.numel(), block_size, t.h_kv, t.d_qk)
            )
            / 10
        )
        blocked_k.clamp_(min=-1.0, max=1.0)

        abs_indices = torch.empty((b, t.s_q, topk), dtype=torch.int32)
        if is_all_indices_invalid:
            abs_indices.fill_(-1)
        else:
            abs_indices[:] = _randperm_batch(
                b * t.s_q, cache_seqlens.repeat_interleave(t.s_q), topk, [-1]
            ).view(b, t.s_q, topk)
        indices_in_kvcache = quant.abs_indices2indices_in_kvcache(
            abs_indices, block_table, block_size
        )

        topk_length = (
            torch.randint(0, topk + 1, (b,), dtype=torch.int32, device=q.device)
            if have_topk_length
            else None
        )

        # Mask nonused KV as NaN
        if have_topk_length:
            indices_in_kvcache_masked = indices_in_kvcache.clone()
            indices_in_kvcache_masked[
                torch.arange(0, topk).view(1, 1, topk).broadcast_to(b, t.s_q, topk)
                >= (topk_length.view(b, 1, 1) if have_topk_length else topk)
            ] = -1
        else:
            indices_in_kvcache_masked = indices_in_kvcache

        blocked_k = blocked_k.view(-1, t.h_kv, t.d_qk)
        nonused_indices_mask = torch.ones(
            blocked_k.size(0) * blocked_k.size(1), dtype=torch.bool, device="cpu"
        )
        nonused_indices_mask[indices_in_kvcache_masked] = False
        blocked_k[nonused_indices_mask, :, :] = float("nan")
        blocked_k = blocked_k.view(-1, block_size, t.h_kv, t.d_qk)

        block_table = kk.non_contiguousify(block_table)
        abs_indices = kk.non_contiguousify(abs_indices)
        indices_in_kvcache = kk.non_contiguousify(indices_in_kvcache)
        return KVScope(
            t,
            cache_seqlens,
            block_table,
            blocked_k,
            abs_indices,
            indices_in_kvcache,
            topk_length,
        )

    kv_scope0 = generate_one_k_scope(
        t.s_kv,
        t.decode.block_size,
        t.topk,
        t.decode.is_varlen,
        t.decode.have_zero_seqlen_k,
        t.is_all_indices_invalid,
        t.have_topk_length,
    )
    kv_scope0.quant_and_dequant_()
    if t.decode.extra_topk is not None:
        if t.decode.extra_s_k is None:
            t.decode.extra_s_k = t.decode.extra_topk * 2
        if t.decode.extra_block_size is None:
            t.decode.extra_block_size = t.decode.block_size
        kv_scope1 = generate_one_k_scope(
            t.decode.extra_s_k,
            t.decode.extra_block_size,
            t.decode.extra_topk,
            t.decode.is_varlen,
            t.decode.have_zero_seqlen_k,
            t.is_all_indices_invalid,
            t.decode.have_extra_topk_length,
        )
        kv_scope1.quant_and_dequant_()
    else:
        assert (
            t.decode.extra_block_size is None
            and t.decode.extra_s_k is None
            and not t.decode.have_extra_topk_length
        )
        kv_scope1 = None

    sm_scale = t.d_qk**-0.55

    q = kk.non_contiguousify(q)
    return TestcaseForDecode(t, q, attn_sink, sm_scale, kv_scope0, kv_scope1)


def run_flash_mla_sparse_fwd(p: TestParam, t: Testcase, return_p_sum: bool):
    assert not return_p_sum
    return flash_mla.flash_mla_sparse_fwd(
        t.q,
        t.kv,
        t.indices,
        sm_scale=t.sm_scale,
        attn_sink=t.attn_sink,
        topk_length=t.topk_length,
    )


def run_flash_mla_decode(
    p: TestParam, t: TestcaseForDecode, tile_scheduler_metadata, num_splits
):
    assert p.decode is not None
    return flash_mla.flash_mla_with_kvcache(
        t.q,
        t.kv_scope.get_kvcache_for_flash_mla(),
        None,
        None,
        p.d_v,
        tile_scheduler_metadata,
        num_splits,
        t.sm_scale,
        False,
        True,
        t.kv_scope.indices_in_kvcache,
        t.attn_sink,
        (
            t.extra_kv_scope.get_kvcache_for_flash_mla()
            if t.extra_kv_scope is not None
            else None
        ),
        t.extra_kv_scope.indices_in_kvcache if t.extra_kv_scope is not None else None,
        t.kv_scope.topk_length,
        (
            t.extra_kv_scope.topk_length
            if t.extra_kv_scope is not None and t.extra_kv_scope.topk_length is not None
            else None
        ),
    )


@dataclasses.dataclass
class FlopsAndMemVolStatistics:
    """
    FLOPs and memory volume statistics for prefilling
    """

    fwd_flop: float
    fwd_mem_vol: float


def count_flop_and_mem_vol(p: TestParam, t: Testcase) -> FlopsAndMemVolStatistics:
    total_topk = (
        (p.s_q * p.topk) if t.topk_length is None else t.topk_length.sum().item()
    )
    indices_valid_mask = (t.indices >= 0) & (t.indices < p.s_kv)
    if t.topk_length is not None:
        indices_valid_mask &= (
            torch.arange(p.topk)[None, None, :].broadcast_to(p.s_q, p.h_kv, p.topk)
        ) < t.topk_length[:, None, None]
    num_valid_indices = indices_valid_mask.sum().item()

    fwd_flop = 2 * total_topk * p.h_q * (p.d_qk + p.d_v)
    fwd_mem_vol = num_valid_indices * p.d_qk * 2 + p.s_q * p.h_q * (p.d_qk + p.d_v) * 2
    return FlopsAndMemVolStatistics(
        fwd_flop,
        fwd_mem_vol,
    )


@dataclasses.dataclass
class FlopsAndMemVolStatisticsForDecode:
    """
    FLOPs and memory volume statistics for decoding
    """

    flop: float
    mem_vol: float


def count_flop_and_mem_vol_for_decode(
    p: TestParam, t: TestcaseForDecode
) -> FlopsAndMemVolStatisticsForDecode:
    assert p.decode
    b = p.decode.b

    def get_num_attended_tokens(kv_scope: KVScope) -> int:
        topk = kv_scope.indices_in_kvcache.shape[-1]
        if kv_scope.topk_length is None:
            return b * p.s_q * topk
        else:
            return int(kv_scope.topk_length.sum().item()) * p.s_q

    def get_num_retrieved_tokens(kv_scope: KVScope) -> int:
        if kv_scope.topk_length is None:
            indices = kv_scope.indices_in_kvcache
        else:
            indices = kv_scope.indices_in_kvcache.clone()
            batch, s_q, topk = indices.shape
            mask = torch.arange(0, topk, device=indices.device).view(
                1, 1, topk
            ).broadcast_to(batch, s_q, topk) >= kv_scope.topk_length.view(batch, 1, 1)
            indices[mask] = -1
        num_unique_tokens = indices.unique().numel()  # type: ignore
        return num_unique_tokens

    num_attended_tokens = get_num_attended_tokens(t.kv_scope) + (
        get_num_attended_tokens(t.extra_kv_scope) if t.extra_kv_scope is not None else 0
    )
    num_retrieved_tokens = get_num_retrieved_tokens(t.kv_scope) + (
        get_num_retrieved_tokens(t.extra_kv_scope)
        if t.extra_kv_scope is not None
        else 0
    )

    compute_flop = 2 * p.h_q * num_attended_tokens * (p.d_qk + p.d_v)
    kv_token_size = 656 if p.d_qk == 576 else 576  # Assume FP8 KV Cache
    mem_vol = sum(
        [
            2 * b * p.s_q * p.h_q * p.d_qk,  # Q
            num_retrieved_tokens * kv_token_size,  # K
            2 * b * p.s_q * p.h_q * p.d_v,  # O
        ]
    )
    return FlopsAndMemVolStatisticsForDecode(compute_flop, mem_vol)


def is_no_cooldown() -> bool:
    return os.environ.get("NO_COOLDOWN", "").lower() in ["1", "yes", "y"]
