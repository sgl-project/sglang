"""HALOps dispatch for ragged DSA MQA logits and top-k."""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_prefill_cp_interleave,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import get_int_env_var

logger = logging.getLogger(__name__)

try:
    from halops.indexer.kernels.mqa_logits_fused_topk.config import (
        DEFAULT_KV_SAMPLE,
        SamplingConfig,
        base_candidate_capacity,
        bounded_workspace_bytes,
        unsupported_reason,
    )
    from halops.indexer.mqa_logits_fused_topk import (
        mqa_logits_fused_topk as _halops_mqa_logits_fused_topk,
    )
    from halops.indexer.mqa_logits_fused_topk import warmup as _halops_warmup
except (ImportError, OSError) as exc:  # HALOps is an optional runtime wheel.
    _halops_mqa_logits_fused_topk = None
    _halops_warmup = None
    DEFAULT_KV_SAMPLE = (64, 4)
    SamplingConfig = None
    base_candidate_capacity = None
    bounded_workspace_bytes = None
    unsupported_reason = None
    logger.debug("HALOps MQA logits top-k is unavailable: %s", exc)


_failed_warmups: set[tuple] = set()
# Configs already warmed up: every layer of a model resolves the same
# gate, so only the first one pays for the kernel warmup.
_warmed: set[tuple] = set()

_MQA_LOGITS_FUSED_TOPK_MIN_KV = get_int_env_var(
    "SGLANG_DSA_MQA_LOGITS_FUSED_TOPK_MIN_KV", -1
)
_MQA_LOGITS_FUSED_TOPK_MIN_Q = get_int_env_var(
    "SGLANG_DSA_MQA_LOGITS_FUSED_TOPK_MIN_Q",
    -1,
)


def resolve_kv_sample() -> tuple[int, int]:
    """Read and validate the pilot sampling pair used by HALOps."""
    value = os.environ.get(
        "SGLANG_DSA_MQA_LOGITS_FUSED_TOPK_KV_SAMPLE",
        ",".join(map(str, DEFAULT_KV_SAMPLE)),
    )
    if SamplingConfig is None:
        return DEFAULT_KV_SAMPLE
    try:
        sampling = SamplingConfig.from_string(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid SGLANG_DSA_MQA_LOGITS_FUSED_TOPK_KV_SAMPLE: {value!r}. {exc}"
        ) from exc
    return sampling.group, sampling.take


def _get_indexer_num_heads(indexer) -> Optional[int]:
    """Resolve the actual kernel width, including FlashIndexer selection."""
    gate = getattr(indexer, "linear_gate", None)
    if gate is None:
        return indexer.n_heads
    from sglang.srt.layers.attention.dsa.flashindexer import FLASHINDEXER_TOPK_HEADS

    return FLASHINDEXER_TOPK_HEADS


def resolve_indexer_gate(
    indexer,
) -> bool:
    """Warm HALOps and return whether this indexer may use it."""
    enabled = envs.SGLANG_DSA_MQA_LOGITS_FUSED_TOPK.get()
    if (
        not enabled
        or _halops_mqa_logits_fused_topk is None
        or _halops_warmup is None
        or unsupported_reason is None
        or not torch.cuda.is_available()
        or not envs.SGLANG_DSA_FUSE_TOPK.get()
        or getattr(indexer, "num_init_tokens", 0)
        or getattr(indexer, "num_local_tokens", 0)
    ):
        if enabled:
            logger.warning(
                "HALOps MQA logits top-k requested but unavailable; "
                "using the existing DSA logits path."
            )
        return False

    qkv_dtype = "mxfp4" if indexer._HEAD_DIM_WITH_SF == 68 else "fp8"

    kv_sample = resolve_kv_sample()
    indexer.mqa_logits_fused_topk_kv_sample = kv_sample
    num_sms = max(1, int(getattr(indexer, "sm_count", 0)))
    num_head = _get_indexer_num_heads(indexer)
    reason = unsupported_reason(
        num_head,
        indexer.head_dim,
        indexer.index_topk,
        qkv_dtype=qkv_dtype,
    )
    if reason is not None:
        logger.info(
            "HALOps MQA logits top-k disabled on layer %s: %s",
            getattr(indexer, "layer_id", "?"),
            reason,
        )
        return False

    # The head gates reach the top-k kernel as fp32 even with a bf16 gate, so
    # warm the same fp32-score config the runtime dispatch will select.
    scores_bf16 = False
    config = (
        torch.cuda.current_device(),
        qkv_dtype,
        scores_bf16,
        num_head,
        indexer.index_topk,
        num_sms,
        kv_sample,
    )
    if config in _failed_warmups:
        return False
    if config not in _warmed:
        try:
            _halops_warmup(
                qkv_dtype=qkv_dtype,
                scores_bf16=scores_bf16,
                num_heads=num_head,
                top_k=indexer.index_topk,
                num_sms=num_sms,
                kv_sample=kv_sample,
                ordered_bounds=True,
            )
        except Exception:
            _failed_warmups.add(config)
            logger.warning(
                "HALOps MQA logits top-k warmup failed for %s; "
                "keeping the existing DSA logits path.",
                config,
                exc_info=True,
            )
            return False
        _warmed.add(config)
    logger.info(
        "DSA mqa_logits_fused_topk enabled on layer %d: index-K %s, scores %s, "
        "H=%d, D=%d, topk=%d; kv_sample=%s.",
        indexer.layer_id,
        qkv_dtype,
        "BF16" if scores_bf16 else "FP32",
        num_head,
        indexer.head_dim,
        indexer.index_topk,
        indexer.mqa_logits_fused_topk_kv_sample,
    )
    return True


def _estimate_ragged_work(
    seq_lens_cpu: Optional[torch.Tensor],
    extend_lens_cpu: Optional[Sequence[int]],
    num_rows: int,
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Optional[Tuple[int, int]]:
    """Return visible score work and request-weighted KV work from host metadata."""
    if not isinstance(seq_lens_cpu, torch.Tensor):
        return None
    if (
        seq_lens_cpu.device.type != "cpu"
        or seq_lens_cpu.ndim != 1
        or not isinstance(extend_lens_cpu, (list, tuple))
        or not isinstance(cp_size, int)
        or not isinstance(cp_rank, int)
        or cp_size <= 0
        or not 0 <= cp_rank < cp_size
    ):
        return None

    seq_lens = seq_lens_cpu.tolist()
    if len(seq_lens) != len(extend_lens_cpu):
        return None

    total_rows = 0
    q_offset = 0
    work = 0
    request_kv_work = 0
    for seq_len_value, extend_len_value in zip(seq_lens, extend_lens_cpu, strict=True):
        if not isinstance(seq_len_value, int) or not isinstance(extend_len_value, int):
            return None
        seq_len, extend_len = seq_len_value, extend_len_value
        if not 0 <= extend_len <= seq_len:
            return None

        first = (cp_rank - q_offset) % cp_size
        q_offset += extend_len
        local_rows = max(0, (extend_len - first + cp_size - 1) // cp_size)
        total_rows += local_rows
        request_kv_work += local_rows * seq_len
        if local_rows == 0:
            continue

        last = seq_len - extend_len + 1 + first + (local_rows - 1) * cp_size
        work += local_rows * (2 * last - (local_rows - 1) * cp_size) // 2

    if total_rows != num_rows:
        return None
    return work, request_kv_work


def should_use_fused(
    num_rows: int,
    num_kv_rows: int,
    topk: int,
    *,
    qkv_dtype: str,
    scores_bf16: bool,
    effective_work: Optional[int],
    request_kv_work: Optional[int],
    kv_sample: tuple[int, int],
    num_heads: Optional[int],
) -> bool:
    """Apply shape, pilot, work and workspace guards before launching HALOps."""
    if (
        SamplingConfig is None
        or base_candidate_capacity is None
        or bounded_workspace_bytes is None
        or num_rows <= 0
        or num_kv_rows <= 0
        or effective_work is None
        or request_kv_work is None
        or not isinstance(effective_work, int)
        or not isinstance(request_kv_work, int)
    ):
        return False
    try:
        sampling = SamplingConfig(*kv_sample)
    except ValueError:
        return False
    if not sampling.has_pilot_sample(num_kv_rows):
        return False

    min_work = _MQA_LOGITS_FUSED_TOPK_MIN_KV
    min_rows = _MQA_LOGITS_FUSED_TOPK_MIN_Q
    min_request_kv = 0
    if (
        min_work < 0
        and min_rows < 0
        and num_heads == 8
        and topk == 2048
        and not scores_bf16
        and kv_sample == DEFAULT_KV_SAMPLE
    ):
        # SM100 real-data replay: FP8 needs large queries and long local KV;
        # MXFP4 also benefits on small queries that the generic gate excludes.
        if qkv_dtype == "fp8":
            min_rows = 3072
            min_request_kv = 98304
        elif qkv_dtype == "mxfp4":
            min_rows = 256
            min_work = 1 << 23
    if min_work < 0:
        min_work = 1 << 25
    if min_rows < 0:
        min_rows = 0 if min_work <= 0 else 512
    min_work *= 3 if scores_bf16 else 1
    if num_rows < min_rows or effective_work < min_work:
        return False
    if num_heads is None or request_kv_work < num_rows * max(
        min_request_kv, 8 * base_candidate_capacity(topk)
    ):
        return False

    score_bytes = num_rows * num_kv_rows * (2 if scores_bf16 else 4)
    workspace_bytes = bounded_workspace_bytes(
        num_rows,
        num_kv_rows,
        topk,
        qkv_dtype=qkv_dtype,
        kv_sample=kv_sample,
    )
    return workspace_bytes < score_bytes


def mqa_logits_fused_topk(
    indexer,
    metadata,
    *,
    q: torch.Tensor,
    q_sf: Optional[torch.Tensor],
    kv: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    forward_batch=None,
) -> Optional[torch.Tensor]:
    """Return HALOps top-k output or ``None`` for the baseline path."""
    if _halops_mqa_logits_fused_topk is None:
        return None
    if getattr(
        metadata, "topk_transform_method", None
    ) != TopkTransformMethod.RAGGED or getattr(metadata, "force_unfused_topk", False):
        return None
    qkv_dtype = "fp8" if q.dtype == torch.float8_e4m3fn else "mxfp4"
    num_heads = q.shape[1]
    ks, ke = metadata.get_indexer_kvcache_range()

    seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
    extend_lens_cpu = metadata.get_dsa_extend_len_cpu()
    cp_size, cp_rank = 1, 0
    if getattr(indexer, "dsa_enable_prefill_cp", False):
        if forward_batch is None or not can_dsa_prefill_cp_interleave(forward_batch):
            return None
        parallel = get_parallel()
        cp_size, cp_rank = parallel.attn_cp_size, parallel.attn_cp_rank
        seq_lens_cpu = forward_batch.seq_lens_cpu
        extend_lens_cpu = forward_batch.extend_seq_lens_cpu

    ragged_work = _estimate_ragged_work(
        seq_lens_cpu,
        extend_lens_cpu,
        ks.shape[0],
        cp_size=cp_size,
        cp_rank=cp_rank,
    )
    if ragged_work is None:
        return None
    kv_sample = getattr(indexer, "mqa_logits_fused_topk_kv_sample", DEFAULT_KV_SAMPLE)
    if not should_use_fused(
        ks.shape[0],
        kv[0].shape[0],
        indexer.index_topk,
        qkv_dtype=qkv_dtype,
        scores_bf16=weights.dtype == torch.bfloat16,
        effective_work=ragged_work[0],
        request_kv_work=ragged_work[1],
        kv_sample=kv_sample,
        num_heads=num_heads,
    ):
        return None
    if qkv_dtype == "mxfp4":
        if q_sf is None:
            q, q_sf = indexer._quantize_fp4(q)
        # Byte views of the packed E2M1 values and UE8M0 exponents.
        kv = (kv[0].view(torch.uint8), kv[1].view(torch.int32))

    return _halops_mqa_logits_fused_topk(
        q=q,
        q_sf=q_sf,
        kv=kv,
        weights=weights,
        ks=ks,
        ke=ke,
        topk_indices_offset=metadata.attn_metadata.topk_indices_offset,
        top_k=indexer.index_topk,
        scores_bf16=weights.dtype == torch.bfloat16,
        num_sms=max(1, int(getattr(indexer, "sm_count", 0))),
        kv_sample=kv_sample,
        ordered_bounds=True,
        topk_indices=out,
    )
