"""Gemma-4 fused all-reduce sites: glue between ``models/gemma4_causal.py`` and
``kernels.ops.communication.gemma4_ar_fusion``.

Opt-in with ``SGLANG_GEMMA4_AR_FUSION=1``. Extend, mixed and decode batches
of up to ``SGLANG_GEMMA4_AR_FUSION_MAX_TOKENS`` tokens run every
TP all-reduce + RMSNorm -> residual add (x layer_scalar) -> RMSNorm -> per-128
FP8 quant of the dense layer stack as one NVLS kernel each; everything else
takes the regular path. A decode batch inside a CUDA graph has the graph's
batch size, the same on every rank, so a captured graph is fused or stock as
a whole. Needs SM90 with NVLink multicast (the TP group's symmetric workspace,
whose pull semaphores the barriers use), TP > 1 on one node, and block-FP8
weights whose
linear takes pre-quantized ``(fp8, scales)`` input.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

GROUP = 128


def _tma_aligned_rows(num_tokens: int) -> int:
    # 16-byte aligned fp32 columns, as create_per_token_group_quant_fp8_output_scale
    return (num_tokens + 3) // 4 * 4


def scales_view(storage: torch.Tensor, groups: int, num_tokens: int) -> torch.Tensor:
    """[num_tokens, groups] column-major view of the fp32 scale storage, columns
    padded to 16 bytes, starting at its base (the address its multicast VA maps):
    the layout DeepGEMM reads without reformatting it."""
    ld = _tma_aligned_rows(num_tokens)
    return storage.view(-1)[: groups * ld].view(groups, ld).t()[:num_tokens]


def requested() -> bool:
    return bool(envs.SGLANG_GEMMA4_AR_FUSION.get())


class _State(NamedTuple):
    group: torch.distributed.ProcessGroup  # the TP CPU group
    world_size: int
    rank: int
    group_name: str


@cache_once
def _get_state() -> Optional[_State]:
    """The TP group's symmetric workspace when it can carry the fused sites."""
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        WorkspaceRequest,
        request_group_workspace,
    )
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils.device import get_device_sm

    if get_device_sm() != 90:
        logger.warning(
            "SGLANG_GEMMA4_AR_FUSION=1 ignored: the FP8 scale layout the kernel "
            "writes is the SM90 one"
        )
        return None
    if get_parallel().tp_size <= 1:
        logger.warning("SGLANG_GEMMA4_AR_FUSION=1 ignored: no tensor parallel")
        return None
    group = get_parallel().tp_group
    # barriers only; any request gets at least one semaphore per SM
    lease = request_group_workspace(
        group, WorkspaceRequest(owner="gemma4_ar_fusion", semaphores=1)
    )
    if lease is None or not lease.has_multicast:
        logger.warning(
            "SGLANG_GEMMA4_AR_FUSION=1 ignored: a multicast symmetric workspace "
            "is unavailable"
        )
        return None
    from sglang.kernels.ops.communication import gemma4_ar_fusion as mod
    from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
        get_all_reduce_config,
    )

    # the fused sites' grid: the custom all-reduce's pull width, which they were tuned with
    pull_blocks = get_all_reduce_config(lease.world_size).num_pull_blocks
    mod.register_comm(
        lambda: lease.get_communicator(pull_blocks=pull_blocks),
        world_size=lease.world_size,
    )
    return _State(lease.group, lease.world_size, lease.rank, group.cpu_group.group_name)


class _Symm(NamedTuple):
    """One persistent symmetric buffer: this rank's view plus its multicast VA."""

    tensor: torch.Tensor
    mc: int


def _symm_buffer(shape: Tuple[int, int], dtype: torch.dtype, group_name: str) -> _Symm:
    from torch._C._distributed_c10d import _SymmetricMemory

    # outside inference mode on purpose (see k3_ar_fusion._create_buffer)
    with torch.inference_mode(False):
        t = _SymmetricMemory.empty_strided_p2p(
            shape,
            [shape[1], 1],
            dtype,
            torch.device("cuda", torch.cuda.current_device()),
            group_name,
        )
    handle = _SymmetricMemory.rendezvous(t)
    if handle is None or handle.multicast_ptr == 0:
        raise RuntimeError("no NVLink multicast on this TP group")
    t.zero_()
    return _Symm(t, int(handle.multicast_ptr))


class Gemma4ArFusion:
    """Buffers and launches for the fused sites of one Gemma-4 model on one TP rank.

    Create through :func:`create`, which returns None (with a warning) when the
    fusion was requested but cannot run here.
    """

    def __init__(self, state: _State, hidden_size: int, eps: float):
        from sglang.kernels.ops.communication import gemma4_ar_fusion as mod

        self._mod = mod
        self.world_size, self.rank = state.world_size, state.rank
        self.eps = float(eps)
        self.hidden = hidden_size
        self.max_tokens = int(envs.SGLANG_GEMMA4_AR_FUSION_MAX_TOKENS.get())
        T, H = self.max_tokens, hidden_size
        # partial sums in (bf16); the last layer multicasts the bf16 residual into it
        self.partial = _symm_buffer((T, H), torch.bfloat16, state.group_name)
        self.out = _symm_buffer((T, H), torch.float8_e4m3fn, state.group_name)
        # stored [H / 128, align4(T)]: each site hands the block-FP8 GEMM the
        # MN-major, TMA-aligned [T, H / 128] view DeepGEMM consumes as is
        self.scales = _symm_buffer(
            (H // GROUP, _tma_aligned_rows(T)), torch.float32, state.group_name
        )
        dev = self.partial.tensor.device
        self.one = torch.ones(1, dtype=torch.float32, device=dev)
        self.used = set()  # "extend" / "decode" once their first fused forward ran
        torch.cuda.synchronize()
        torch.distributed.barrier(state.group)
        logger.info(
            "gemma4 all-reduce fusion: tp=%d, up to %d tokens, %.0f MB symmetric memory",
            self.world_size,
            self.max_tokens,
            sum(b.tensor.nbytes for b in (self.partial, self.out, self.scales)) / 2**20,
        )

    @classmethod
    def create(cls, hidden_size: int, eps: float) -> Optional[Gemma4ArFusion]:
        state = _get_state()
        if state is None:
            return None
        try:
            return cls(state, hidden_size, eps)
        except Exception as e:  # buffers or the JIT build
            logger.warning("SGLANG_GEMMA4_AR_FUSION=1 ignored: %s", e)
            return None

    def applies(self, num_tokens: int) -> bool:
        # Every size: on H200 the fused site beats the stock chain from 2 tokens
        # at TP4 and outside 160..768 tokens at TP2; where it does not, it is
        # within ~1.5 us per site, under 2 % of a decode or prefill step.
        return 0 < num_tokens <= self.max_tokens

    def rows(self, num_tokens: int) -> Tuple[int, int]:
        """(first row, row count) this rank owns out of ``num_tokens``."""
        per_rank = (num_tokens + self.world_size - 1) // self.world_size
        # trailing ranks of a small batch own no rows; keep row0 inside the batch
        row0 = min(self.rank * per_rank, num_tokens)
        return row0, min(num_tokens, row0 + per_rank) - row0

    def partial_out(self, num_tokens: int) -> torch.Tensor:
        """The symmetric storage a site reads its partial sums from. Pass it to
        the producing linear (``RowParallelLinear(output_tensor=...)``) so the
        GEMM writes it directly; a partial that lives elsewhere is copied in."""
        return self.partial.tensor[:num_tokens]

    def _stage(self, partial: torch.Tensor) -> torch.Tensor:
        staged = self.partial_out(partial.shape[0])
        if partial.data_ptr() != staged.data_ptr():
            staged.copy_(partial)
        return staged

    def site(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        w_a: torch.Tensor,
        w_b: torch.Tensor,
        scalar: Optional[torch.Tensor],
        round_a: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """allreduce(partial) -> norm(w_a) -> + residual (x scalar) -> norm(w_b) -> FP8.

        Updates ``residual`` (this rank's rows) in place; returns ``(fp8 [T, H],
        fp32 scales [T, H / 128])`` for every row, the pre-quantized input of the
        block-FP8 linear. The scales are column-major with TMA-aligned rows, the
        layout DeepGEMM reads without reformatting them.
        """
        T = partial.shape[0]
        row0, rows = self.rows(T)
        scales = scales_view(self.scales.tensor, self.scales.tensor.shape[0], T)
        self._mod.site(
            self._stage(partial),
            self.partial.mc,
            self.out.tensor[:T],
            self.out.mc,
            scales,
            self.scales.mc,
            residual,
            w_a,
            w_b,
            self.one if scalar is None else scalar,
            self.eps,
            row0,
            rows,
            round_a,
            world_size=self.world_size,
        )
        return self.out.tensor[:T], scales

    def final(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        w_a: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        """allreduce(partial) -> norm(w_a) -> (+ residual) x scalar, every row as
        bf16 (in the partial buffer)."""
        T = partial.shape[0]
        row0, rows = self.rows(T)
        staged = self._stage(partial)
        self._mod.final(
            staged,
            self.partial.mc,
            staged,
            self.partial.mc,
            residual,
            w_a,
            scalar,
            self.eps,
            row0,
            rows,
            world_size=self.world_size,
        )
        return staged
