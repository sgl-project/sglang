# SPDX-License-Identifier: Apache-2.0
"""SubBlock block-sparse attention backend.

Routes the same 64-token SubBlock plan to SGLang's CuTe-DSL block-sparse
FlashAttention kernel on SM90 or FlashInfer's architecture-specific kernels on
SM100 and SM120. A log-sum-exp over query/key sub-block pairs selects the blocks
(see ``backends/subblock_sparse/``).
Everything is training-free: the router runs before attention and produces
the ``q2k_block_index`` the selected kernel consumes.

Sparsity is not applied everywhere. The early denoise steps settle the layout
of the sample and tolerate approximation badly, so the backend falls back to
dense attention for them. Depth turns out not to matter the same way, which is
why the layer cutoff defaults to zero -- see the defaults below. The schedule
is configured through ``--attention-backend-config``, which overrides
individual keys of the defaults below::

    --attention-backend subblock_sparse_attn \
    --attention-backend-config '{"sparsity": 0.85}'

Requirements inherited from the kernels: compute capability 9.0 (Hopper) or
10.0/12.0 (Blackwell), bf16, head_dim 128. Hopper uses SGLang's CuTe-DSL SM90
block-sparse FlashAttention kernel; B200 and SM120 devices use FlashInfer's
architecture-specific blk64 kernels. Inside the DiT, any call the kernels cannot
serve -- cross/refiner attention, short sequences, non-bf16 -- runs dense instead.
On any other GPU the resolver refuses the backend at startup rather than falling back.

``--attention-backend`` reaches every component, and the text encoder admits
only fa / torch_sdpa / sage_attn_3. Pair it with
``--component-attention-backends text_encoder=fa`` on SM90/SM100, or
``text_encoder=torch_sdpa`` on SM120; see the README.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
    SubBlockRouter,
    load_bsa_attn_blk64_fwd,
    load_bsa_attn_sm120_blk64_fwd,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# The kernel is fixed at 64-token blocks and 128-wide heads.
SUBBLOCK_SPARSE_BLOCK_SIZE = 64
SUBBLOCK_SPARSE_HEAD_DIM = 128

# Defaults for the schedule; override through --attention-backend-config.
# Sparsity is the speed lever, and it saturates: on MiniMax-H3 t2va at 37.7k
# tokens, 0.75 gives 1.14x, 0.80 gives 1.18x and 0.85 gives 1.21x -- cutting the
# block budget by 40% past 0.75 buys 6%, because attention is no longer the bulk
# of the step. 0.85 was the worst arm on cosine-vs-dense on both clips rendered
# across all three grades, and 0.80 costs 0.017 / 0.006 cos_c against 0.75 on
# those same two clips for 3.5% of the time, so the default takes the quality.
DEFAULT_SPARSITY = 0.75
# The two cutoffs were swept independently on MiniMax-H3 t2va (1344x768, 5 s,
# 50 steps, n_k=4, sparsity 0.75) and behave nothing alike. Lowering the step
# cutoff from 10 to 5 halves cosine-vs-dense (0.558 -> 0.310 on two clips) and
# visibly re-frames the shot; going to 0 leaves the sample essentially
# uncorrelated with dense for 1.20x -> 1.30x. Lowering the layer cutoff from 2
# to 0 costs 0.0013 of that cosine -- inside the 0.02 run-to-run noise floor --
# and is worth ~1%, so the first DiT blocks get no special treatment.
DEFAULT_SKIP_FIRST_STEPS = 10
DEFAULT_SKIP_FIRST_LAYERS = 0
DEFAULT_N_K = 4
# Query-side splitting. Splitting Q *alone* is worse than not splitting -- with
# one key vector to score against, the query detail averages out -- which is
# where the "n_q is worthless" reading came from. Splitting both sides together
# is a different estimator: the log-sum-exp then runs over query-key sub-block
# pairs. It is the only estimator change in this family that has reproduced end
# to end, and it costs 0.5% of the denoise time. Measured against n_q=1 on
# fifteen t2va prompts, every arm rendered in one session against that session's
# own dense render, as cosine of the decoded video:
#     sparsity 0.90   +0.062   paired t = +2.6   better on 13/15
#     sparsity 0.75   +0.008   paired t = +2.1   better on 10/15
# The margin shrinks as the budget loosens, which is the pattern every estimator
# comparison here has followed: at the shipped 148 of 590 blocks the rules mostly
# agree on what to keep.
DEFAULT_N_Q = 4
# Below this many keys the router costs more than the blocks it saves, and the
# top-k budget collapses to a handful of blocks.
DEFAULT_MIN_SEQ_LEN = 4096

# ``blocks.<idx>.attn`` is a DiT layer; ``token_refiner.blocks.<idx>.attn`` and
# anything else is not and stays dense.
_DIT_LAYER_PREFIX = re.compile(r"^blocks\.(\d+)\.")


def _dit_layer_index(prefix: str) -> int | None:
    match = _DIT_LAYER_PREFIX.match(prefix)
    return int(match.group(1)) if match else None


@functools.lru_cache(maxsize=8)
def _cached_block_sizes(seq_len: int, device: torch.device) -> torch.Tensor:
    """Per-block real token counts; identical for every layer and step.

    Rebuilding it per call costs an arange plus a clamp launch on the critical
    path for a tensor that only depends on the sequence length.
    """
    return SubBlockRouter.block_sizes(seq_len, device)


@functools.lru_cache(maxsize=1)
def _load_sm90_block_sparse_attention():
    """Load the CuTe-DSL Hopper path only when an SM90 device selects it.

    Keeping these imports lazy avoids pulling the sizeable CuTe dependency tree
    into the existing SM100 path, whose FlashInfer blk64 kernel is plain CUDA.
    """
    from sglang.kernels.ops.attention.flash_attn.cute.block_sparsity import (
        BlockSparseTensorsTorch,
    )
    from sglang.kernels.ops.attention.flash_attn.cute.interface import flash_attn_func

    return BlockSparseTensorsTorch, flash_attn_func


def _sm90_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    topk: int,
    softmax_scale: float,
    block_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a SubBlock routing plan through the existing SM90 CuTe kernel."""
    BlockSparseTensorsTorch, flash_attn_func = _load_sm90_block_sparse_attention()

    # The caller sorts each active sparse prefix for SM90. Dense rows are the
    # already-sorted complete range; entries beyond each row's count are ignored.
    ordered_index = q2k_block_index
    if block_counts is None:
        block_counts = torch.full(
            ordered_index.shape[:-1],
            topk,
            dtype=torch.int32,
            device=ordered_index.device,
        )
    sparse_tensors = BlockSparseTensorsTorch(
        mask_block_cnt=block_counts,
        mask_block_idx=ordered_index,
        # There are no always-dense blocks in a SubBlock routing plan. The
        # block-sparse broadcast pattern records both absent tensors as None
        # and participates in the compile key, so mask-only and mask+full calls
        # cannot share a compiled kernel.
        full_block_cnt=None,
        full_block_idx=None,
        block_size=(SUBBLOCK_SPARSE_BLOCK_SIZE, SUBBLOCK_SPARSE_BLOCK_SIZE),
    )
    out, _ = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        num_splits=1,
        block_sparse_tensors=sparse_tensors,
    )
    return out


def _sm100_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    topk: int,
    softmax_scale: float,
    block_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a SubBlock routing plan through FlashInfer's SM100 kernel."""
    out = load_bsa_attn_blk64_fwd()(
        q,
        k,
        v,
        q2k_block_index,
        topk,
        block_sizes=_cached_block_sizes(k.shape[1], k.device),
        q2k_block_nums=block_counts,
        softmax_scale=softmax_scale,
    )
    return out[0] if isinstance(out, tuple) else out


def _sm120_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    topk: int,
    softmax_scale: float,
    block_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a SubBlock routing plan through FlashInfer's SM120 kernel."""
    logger.info_once("SubBlock sparse attention kernel active: FlashInfer SM120 blk64")
    out = load_bsa_attn_sm120_blk64_fwd()(
        q,
        k,
        v,
        q2k_block_index,
        topk,
        block_sizes=_cached_block_sizes(k.shape[1], k.device),
        q2k_block_nums=block_counts,
        softmax_scale=softmax_scale,
    )
    return out[0] if isinstance(out, tuple) else out


@functools.lru_cache(maxsize=None)
def _get_subblock_sparse_attention_runner(device: torch.device):
    """Resolve the architecture-specific kernel once per CUDA device."""
    capability = torch.cuda.get_device_capability(device)
    if capability == (9, 0):
        return _sm90_sparse_attention
    if capability == (10, 0):
        return _sm100_sparse_attention
    if capability == (12, 0):
        return _sm120_sparse_attention
    raise RuntimeError(
        "SubBlock sparse attention supports compute capability 9.0, 10.0, or 12.0; "
        f"this tensor is on a {capability[0]}.{capability[1]} device."
    )


def _run_subblock_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    topk: int,
    softmax_scale: float,
    block_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch a prepared 64x64 routing plan to Hopper or Blackwell.

    SM90 requires every active index prefix to be sorted in ascending order;
    SM100 and SM120 accept the router's original order. Heterogeneous callers
    must sort compact sparse prefixes before expanding them to full-width dense
    rows.
    """
    runner = _get_subblock_sparse_attention_runner(q.device)
    return runner(
        q,
        k,
        v,
        q2k_block_index,
        topk,
        softmax_scale,
        block_counts,
    )


class SubBlockSparseAttentionBackend(AttentionBackend):
    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [SUBBLOCK_SPARSE_HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN

    @staticmethod
    def get_impl_cls() -> type[SubBlockSparseAttentionImpl]:
        return SubBlockSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[SubBlockSparseAttentionMetadata]:
        return SubBlockSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[SubBlockSparseAttentionMetadataBuilder]:
        return SubBlockSparseAttentionMetadataBuilder


@dataclass
class SubBlockSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int


class SubBlockSparseAttentionMetadataBuilder(AttentionMetadataBuilder):
    # The base class declares __init__ abstract, so a builder that does not
    # override it cannot be instantiated at all.
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self, current_timestep: int, **kwargs: dict[str, Any]
    ) -> SubBlockSparseAttentionMetadata:
        return SubBlockSparseAttentionMetadata(current_timestep=current_timestep)


class SubBlockSparseSchedule(msgspec.Struct, frozen=True):
    """When sparsity is allowed to apply, and how much of it."""

    sparsity: float
    skip_first_steps: int
    skip_first_layers: int
    n_k: int
    n_q: int
    min_seq_len: int

    @classmethod
    def from_server_args(cls) -> SubBlockSparseSchedule:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        config = get_global_server_args().attention_backend_config or {}
        schedule = SubBlockSparseSchedule(
            sparsity=float(config.get("sparsity", DEFAULT_SPARSITY)),
            skip_first_steps=int(
                config.get("skip_first_steps", DEFAULT_SKIP_FIRST_STEPS)
            ),
            skip_first_layers=int(
                config.get("skip_first_layers", DEFAULT_SKIP_FIRST_LAYERS)
            ),
            n_k=int(config.get("n_k", DEFAULT_N_K)),
            n_q=int(config.get("n_q", DEFAULT_N_Q)),
            min_seq_len=int(config.get("min_seq_len", DEFAULT_MIN_SEQ_LEN)),
        )
        if not 0.0 <= schedule.sparsity < 1.0:
            raise ValueError(
                f"subblock sparsity must be in [0, 1), got {schedule.sparsity}"
            )
        for name, value in (("n_k", schedule.n_k), ("n_q", schedule.n_q)):
            if value not in (1, 2, 4, 8):
                raise ValueError(f"subblock {name} must be 1, 2, 4 or 8, got {value}")
        if schedule.skip_first_steps < 0 or schedule.skip_first_layers < 0:
            raise ValueError("subblock skip_first_* must be non-negative")
        return schedule


class SubBlockSparseAttentionImpl(AttentionImpl):
    """Block-sparse attention with a dense fallback for the excluded region.

    One impl instance is built per attention module, so ``prefix`` fixes the
    layer for the lifetime of the object; only the denoise step varies per
    call and it comes from the forward context.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else head_size**-0.5
        )
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        self.schedule = SubBlockSparseSchedule.from_server_args()
        self.layer_idx = _dit_layer_index(prefix)
        # A layer outside the DiT stack (token refiner, cross attention) never
        # runs sparse: its sequences are short and its budget meaningless.
        self.layer_enabled = (
            self.layer_idx is not None
            and self.layer_idx >= self.schedule.skip_first_layers
            and head_size == SUBBLOCK_SPARSE_HEAD_DIM
            and self.schedule.sparsity > 0.0
        )
        self.router = (
            SubBlockRouter(n_k=self.schedule.n_k, n_q=self.schedule.n_q)
            if self.layer_enabled
            else None
        )
        self.dense_impl = self._build_dense_impl(causal=causal)
        if self.layer_enabled:
            logger.info_once(
                f"SubBlock sparse attention: sparsity={self.schedule.sparsity:.3f} "
                f"n_k={self.schedule.n_k} n_q={self.schedule.n_q}, dense for the first "
                f"{self.schedule.skip_first_steps} denoise steps and the first "
                f"{self.schedule.skip_first_layers} DiT layers"
            )

    def _build_dense_impl(self, *, causal: bool) -> AttentionImpl:
        """Flash attention, used wherever the schedule excludes sparsity."""
        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            get_attn_backend,
        )

        backend = get_attn_backend(
            self.head_size,
            torch.bfloat16,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
            selected_attention_backend=AttentionBackendEnum.FA,
        )
        return backend.get_impl_cls()(
            num_heads=self.num_heads,
            head_size=self.head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{self.prefix}.dense",
        )

    def _step_enabled(self) -> bool:
        return get_forward_context().current_timestep >= self.schedule.skip_first_steps

    def _sparse_ready(self, q: torch.Tensor, k: torch.Tensor) -> bool:
        return (
            self.layer_enabled
            and self._step_enabled()
            and q.dtype == torch.bfloat16
            and k.shape[-3] >= self.schedule.min_seq_len
            and not self.causal
        )

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sparse_query_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Q ``[1, Sq, H, 128]`` against K/V ``[1, Sk, H, 128]``."""
        plan = self.router.route(
            q,
            k,
            sparsity=self.schedule.sparsity,
            softmax_scale=self.softmax_scale,
        )
        expected_q_blocks = -(-q.shape[1] // SUBBLOCK_SPARSE_BLOCK_SIZE)
        if plan.index.shape[2] != expected_q_blocks:
            raise ValueError(
                "SubBlock routing/kernel query-block mismatch: "
                f"plan has {plan.index.shape[2]}, kernel needs {expected_q_blocks}"
            )
        # Proof that the sparse path actually ran -- the construction-time log
        # above only says the layer was eligible.
        if sparse_query_block_mask is None:
            logger.info_once(
                f"SubBlock sparse attention active: Sq={q.shape[1]} "
                f"Sk={k.shape[1]} heads={q.shape[2]} "
                f"keeping {plan.topk}/{plan.num_blocks} key blocks per query "
                f"block (sparsity {1 - plan.density:.4f})"
            )
        else:
            logger.info_once(
                f"SubBlock heterogeneous BSA active: Sq={q.shape[1]} "
                f"Sk={k.shape[1]} heads={q.shape[2]}; selected query blocks "
                f"keep {plan.topk}/{plan.num_blocks} key blocks and unselected "
                "query blocks are dense"
            )
        block_counts = None
        runner = _get_subblock_sparse_attention_runner(q.device)
        block_index = (
            plan.index.sort(dim=-1).values
            if runner is _sm90_sparse_attention
            else plan.index
        )
        kernel_topk = plan.topk
        if sparse_query_block_mask is not None:
            sparse_query_block_mask = sparse_query_block_mask.to(
                device=q.device, dtype=torch.bool
            ).view(-1)
            if sparse_query_block_mask.numel() != expected_q_blocks:
                raise ValueError(
                    "SubBlock sparse query-block mask length does not match Q"
                )
            num_k_blocks = plan.num_blocks
            full_index = torch.arange(
                num_k_blocks, device=q.device, dtype=block_index.dtype
            ).view(1, 1, 1, num_k_blocks)
            heterogeneous_index = full_index.expand(
                *block_index.shape[:-1], num_k_blocks
            ).clone()
            sparse_rows = sparse_query_block_mask.view(1, 1, -1, 1)
            heterogeneous_index[..., : plan.topk] = torch.where(
                sparse_rows,
                block_index,
                heterogeneous_index[..., : plan.topk],
            )
            block_counts = (
                torch.where(
                    sparse_query_block_mask.view(1, 1, -1),
                    plan.topk,
                    num_k_blocks,
                )
                .expand(*block_index.shape[:-1])
                .to(torch.int32)
            )
            block_index = heterogeneous_index
            kernel_topk = num_k_blocks
        return _run_subblock_sparse_attention(
            q,
            k,
            v,
            block_index,
            kernel_topk,
            self.softmax_scale,
            block_counts,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SubBlockSparseAttentionMetadata | None = None,
    ) -> torch.Tensor:
        """query/key/value: ``[B, S, H, D]``."""
        if not self._sparse_ready(query, key):
            return self.dense_impl.forward(query, key, value, attn_metadata)
        return self._sparse_attention(query, key, value)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
        first_segment_sparse_query_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Packed ``[T, H, D]`` rows split into documents by ``cu_seqlens``.

        Each packed document keeps its own full K/V context. The optional
        first-segment mask selects sparse Q blocks; unselected blocks stay
        dense within the same heterogeneous BSA call.
        Documents shorter than ``min_seq_len`` -- in H3, the padding tail --
        stay on the existing dense segment path.
        """

        def all_dense() -> torch.Tensor:
            return self.dense_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        if cu_seqlens_host is None or not self._sparse_ready(query, key):
            return all_dense()

        segments = [
            (start, stop)
            for start, stop in zip(cu_seqlens_host[:-1], cu_seqlens_host[1:])
            if stop > start
        ]
        sparse_segments = {
            (start, stop)
            for start, stop in segments
            if stop - start >= self.schedule.min_seq_len
        }
        if not sparse_segments:
            return all_dense()

        out = torch.empty_like(query)
        # cu_seqlens covers every packed row in practice; a caller that leaves
        # a tail outside the last document would otherwise read uninitialized
        # memory back out.
        if segments[-1][1] < query.shape[0]:
            out[segments[-1][1] :].zero_()
        for start, stop in segments:
            # Deliberately not `.contiguous()`. After the Ulysses all-to-all,
            # q/k/v are last-dim slices of one packed buffer, so they are
            # strided; the attention kernels handle those views directly, and
            # forcing contiguity here measured as a wasted
            # full-tensor copy (0.46 ms per call at S=37.7k on B200).
            q_seg = query[start:stop].unsqueeze(0)
            k_seg = key[start:stop].unsqueeze(0)
            v_seg = value[start:stop].unsqueeze(0)
            if (start, stop) in sparse_segments:
                sparse_query_block_mask = (
                    first_segment_sparse_query_block_mask
                    if (start, stop) == segments[0]
                    else None
                )
                seg_out = self._sparse_attention(
                    q_seg,
                    k_seg,
                    v_seg,
                    sparse_query_block_mask=sparse_query_block_mask,
                )
            else:
                seg_out = self._dense_segment(q_seg, k_seg, v_seg)
            out[start:stop] = seg_out[0]
        return out

    def _dense_segment(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Dense attention over one packed document, ``[1, S, H, D]``."""
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=self.causal,
            scale=self.softmax_scale,
        ).transpose(1, 2)


__all__ = [
    "SubBlockSparseAttentionBackend",
    "SubBlockSparseAttentionImpl",
    "SubBlockSparseAttentionMetadata",
    "SubBlockSparseAttentionMetadataBuilder",
    "SubBlockSparseSchedule",
]
