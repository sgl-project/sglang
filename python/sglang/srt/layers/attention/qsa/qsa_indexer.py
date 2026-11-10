"""QSA indexer for Qwen4-Exp checkpoints."""

from __future__ import annotations

import os
import functools
import logging
from typing import Optional, Tuple

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    qsa_fast_topk,
)
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
)
from sglang.srt.layers.attention.qsa import verify_topk
from sglang.srt.layers.attention.qsa.mqa import (
    HAS_TILELANG,
    _positive_env_int,
    qsa_mqa_decode,
    qsa_mqa_prefill,
    qsa_mqa_prefill_relative,
)
from sglang.srt.layers.cp.utils import (
    ContextParallelStrategyKind,
    cp_materialize_global_token_order,
    cp_shard_hidden_states,
    cp_shard_position_ids,
    get_cp_strategy,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_sm100_supported

logger = logging.getLogger(__name__)

# SGLANG_QSA_PREFILL_TOPK_BRIDGE_DIRECT (default off): the breakable prefill graph's top-k bridge
# (models/qwen4_exp.py _breakable_qsa_indexer) lends its rows to the next prefill selection, which
# writes them in place of a new tensor, so the bridge's full fill and its [rows, 2051] int32 D2D copy
# are skipped. One shot: the bridge is taken by the first select_prefill_tokens call inside
# prefill_topk_output() or by none. Same kernels, same values: bitwise.
QSA_PREFILL_TOPK_BRIDGE_DIRECT = os.environ.get(
    "SGLANG_QSA_PREFILL_TOPK_BRIDGE_DIRECT", "0"
).strip().lower() in ("1", "true", "yes", "on")
_PREFILL_TOPK_OUT: list = [None]


class prefill_topk_output:
    """Offer ``buffer`` (int32 [>= rows, token_topk + compress_ratio - 1], unit last stride) to the
    next select_prefill_tokens call; withdrawn on exit whether taken or not."""

    __slots__ = ("_buffer",)

    def __init__(self, buffer: torch.Tensor):
        self._buffer = buffer

    def __enter__(self):
        _PREFILL_TOPK_OUT[0] = self._buffer
        return self

    def __exit__(self, *exc):
        _PREFILL_TOPK_OUT[0] = None
        return False


def _take_prefill_topk_output(rows: int, width: int, device) -> Optional[torch.Tensor]:
    buffer = _PREFILL_TOPK_OUT[0]
    if buffer is None:
        return None
    _PREFILL_TOPK_OUT[0] = None
    if (
        buffer.dtype != torch.int32
        or buffer.dim() != 2
        or buffer.shape[0] < rows
        or buffer.shape[1] != width
        or buffer.stride(1) != 1
        or buffer.device != device
    ):
        return None
    view = buffer[:rows]
    return view if view.is_contiguous() else None

# Decode/verify indexer fast paths (both exact; set to 0 to run the stock kernels):
# SGLANG_QSA_GROUPED_SCORE scores a request's target-verify rows from one load of
# each compressed key tile (bit-identical logits) for batches of 1-16 requests;
# SGLANG_QSA_SPLIT_TOPK runs the cluster-split exact top-k for up to 64 rows.
# Both were measured at 4-32 rows; larger batches keep the stock kernels.
# SGLANG_QSA_GROUPED_SCORE_MAX_REQUESTS raises the grouped scorer's request cap;
# above 16 requests the rows need SGLANG_ENABLE_QSA_GROUPED4_VERIFY's kernel.
_QSA_GROUPED_SCORE_MIN_REQUESTS = 1
_QSA_GROUPED_SCORE_MAX_REQUESTS = _positive_env_int(
    envs.SGLANG_QSA_GROUPED_SCORE_MAX_REQUESTS
)
_QSA_GROUPED4_DRAFT_EXTEND = envs.SGLANG_ENABLE_QSA_GROUPED4_DRAFT_EXTEND.get()
_QSA_DRAFT_EXTEND_UNION = envs.SGLANG_ENABLE_QSA_DRAFT_EXTEND_UNION.get()
_QSA_SPLIT_TOPK_MAX_ROWS = 64


@functools.lru_cache(maxsize=None)
def _qsa_env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _qpf_enabled() -> bool:
    from sglang.srt.layers.attention.qsa.prologue_fold import QPF_ENABLED

    return QPF_ENABLED


@functools.lru_cache(maxsize=None)
def _log_grouped_cap_once(requests: int) -> None:
    logger.debug(
        "QSA grouped verify scorer: requests=%d max_requests=%d",
        requests,
        _QSA_GROUPED_SCORE_MAX_REQUESTS,
    )


# Decode/verify selection paths ("mk_score", "split_topk") that have logged their first use
# (at graph capture); log only, so a run can confirm its small-batch path stayed on.
_DECODE_PATH_LOGGED: set = set()


def _log_decode_path_once(path: str, rows: int) -> None:
    if path in _DECODE_PATH_LOGGED:
        return
    _DECODE_PATH_LOGGED.add(path)
    name = {"mk_score": "SGLANG_Q38C_MK_QSA (score megakernel)",
            "split_topk": "SGLANG_QSA_SPLIT_TOPK (split top-k)"}[path]
    logger.info("QSA decode path engaged: %s, rows=%d", name, rows)


_QSA_SHADOW_STATS = {"calls": 0, "rows": 0, "grouped_calls": 0, "logit_rows_diff": 0,
                     "set_equal": 0, "tie_only": 0, "value_mismatch": 0,
                     "mismatch_new_exact": 0, "mismatch_stock_exact": 0,
                     "mismatch_stock_rerun_differs": 0, "mismatch_rows_over_32k": 0}


def _qsa_shadow_check(q, cache, table, lengths, max_model_len, k, logits, indices, rows_per_group):
    """Debug only (SGLANG_QSA_SHADOW_CHECK=1, eager decode): rerun the stock per-row
    scorer and fast_topk on the same inputs and count rows whose logits or
    selected sets differ; ties at the k-th value are reported separately."""
    if torch.cuda.is_current_stream_capturing():
        return
    import logging

    from sglang.kernels.ops.attention.fast_topk import fast_topk

    ref_logits = qsa_mqa_decode(q, cache, table, lengths, max_model_len, min_scored_len=k)
    lens = lengths.to(torch.int32)
    ref = fast_topk(ref_logits, lens, topk=k, row_starts=None)
    st = _QSA_SHADOW_STATS
    st["calls"] += 1
    st["grouped_calls"] += int(rows_per_group > 1)
    for r, length in enumerate(lens.tolist()):
        st["rows"] += 1
        if length > k and not torch.equal(logits[r, :length], ref_logits[r, :length]):
            st["logit_rows_diff"] += 1
        a, b = indices[r], ref[r]
        if set(a.tolist()) == set(b.tolist()):
            st["set_equal"] += 1
            continue
        row = ref_logits[r, :length]
        va = torch.sort(row[a[a >= 0].long()]).values
        vb = torch.sort(row[b[b >= 0].long()]).values
        if torch.equal(va, vb):
            st["tie_only"] += 1
        else:
            # Classify against the exact fp32 top-k and a second stock run
            # (fast_topk drops threshold-bin entries past 4096 in atomic order).
            st["value_mismatch"] += 1
            truth = torch.sort(torch.topk(row, k).values).values
            st["mismatch_new_exact"] += int(torch.equal(va, truth))
            st["mismatch_stock_exact"] += int(torch.equal(vb, truth))
            rerun = fast_topk(ref_logits[r : r + 1], lens[r : r + 1], topk=k, row_starts=None)[0]
            vr = torch.sort(row[rerun[rerun >= 0].long()]).values
            st["mismatch_stock_rerun_differs"] += int(not torch.equal(vr, vb))
            st["mismatch_rows_over_32k"] += int(length > 32768)
    if st["calls"] % 200 == 0:
        logging.getLogger(__name__).info("QSA shadow check: %s", st)


# Cap on the fp32 [query_rows, compressed_keys] prefill logits workspace;
# top-k is per row, so tiling rows does not change the selection.
_QSA_PREFILL_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024
_Q38C_SCORE_Q4 = os.environ.get("SGLANG_Q38C_SCORE_Q4", "0") == "1"
_Q38C_MK_QSA = os.environ.get("SGLANG_Q38C_MK_QSA", "0") == "1"
# qsa/prefill_select.py: the same prefill selection with fewer, fuller launches.
_QSA_PREFILL_FAST_SELECT = _qsa_env_flag("SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT", "0")


def _qsa_prefill_row_chunk_size(rows: int, keys: int, heads: int) -> int:
    if rows <= 0 or keys <= 0:
        return max(rows, 1)
    block_q = max(1, 128 // heads)
    bytes_per_row = keys * torch.float32.itemsize
    max_padded_rows = max(block_q, _QSA_PREFILL_LOGITS_BUDGET_BYTES // bytes_per_row)
    max_padded_rows = max(block_q, max_padded_rows // block_q * block_q)
    return min(rows, max_padded_rows)


class QSAIndexer(MultiPlatformOp):
    """Config-driven fused-QK, weight-free sparse-attention indexer."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        rotary_emb=None,
    ) -> None:
        self._validate_config(config)
        super().__init__()
        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.block_topk = self.token_topk // self.compress_ratio
        if rotary_emb is None:
            raise ValueError("QSAIndexer must reuse its Qwen4-Exp attention RoPE")
        self.rotary_emb = rotary_emb
        if not 0 < self.rotary_emb.rotary_dim <= self.index_head_dim:
            raise ValueError(
                "Qwen4-Exp attention RoPE rotary_dim must fit the QSA index head: "
                f"{self.rotary_emb.rotary_dim=} {self.index_head_dim=}"
            )
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self._rope_axis_map_cache = None
        self._qpf_shadow_q_ref = None

    @staticmethod
    def _validate_config(config) -> None:
        names = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        missing = [name for name in names if getattr(config, name, None) is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")
        values = {name: int(getattr(config, name)) for name in names}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}")
        if values["indexer_compress_ratio"] < 2:
            # DP token-padding rows carry logical length 1, which must never
            # reach a compression boundary; ratio >= 2 guarantees that.
            raise ValueError(
                "QSA requires indexer_compress_ratio >= 2, got "
                f"{values['indexer_compress_ratio']}"
            )
        if values["indexer_kv_heads"] != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if values["indexer_budget"] % values["indexer_compress_ratio"] != 0:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        block_topk = values["indexer_budget"] // values["indexer_compress_ratio"]
        if block_topk not in (512, 2048):
            raise ValueError(
                "fast_topk_v2 requires indexer_budget / indexer_compress_ratio "
                f"to be 512 or 2048, got {block_topk}"
            )

    def _use_fused_prep(self, tensor: torch.Tensor) -> bool:
        """Whether the fused indexer-prep kernels support this configuration."""
        return (
            tensor.is_cuda
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and self.index_head_dim in (64, 128, 256)
            and self.rotary_emb.rotary_dim % 2 == 0
            and not getattr(self.rotary_emb, "mrope_interleaved_glm", False)
            and len(getattr(self.rotary_emb, "mrope_section", None) or ()) in (0, 3)
            and getattr(self.rotary_emb, "cos_sin_cache", None) is not None
            and self.rotary_emb.cos_sin_cache.is_cuda
            and self.rotary_emb.cos_sin_cache.dtype == torch.float32
        )

    def _rope_axis_map(self, device) -> torch.Tensor:
        """axis_map[i] is the MRoPE position axis whose cos/sin rotary pair i reads."""
        cache = self._rope_axis_map_cache
        if cache is not None and cache.device == device:
            return cache
        half = self.rotary_emb.rotary_dim // 2
        section = getattr(self.rotary_emb, "mrope_section", None) or None
        axis_map = torch.zeros(half, dtype=torch.int32)
        if section is not None:
            s0, s1, s2 = (int(v) for v in section)
            if getattr(self.rotary_emb, "mrope_interleaved", False):
                pair = torch.arange(half, dtype=torch.int32)
                axis_map[((pair % 3) == 1) & (pair < s1 * 3)] = 1
                axis_map[((pair % 3) == 2) & (pair < s2 * 3)] = 2
            else:
                axis_map[s0 : s0 + s1] = 1
                axis_map[s0 + s1 :] = 2
        self._rope_axis_map_cache = axis_map.to(device)
        return self._rope_axis_map_cache

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        pool=None,
        cache_loc: torch.Tensor | None = None,
        q_heads_padded: int | None = None,
        ensure_cos_sin_cache: bool = True,
        projected_qk: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        if projected_qk is None:
            qk, _ = self.index_qk_proj(hidden_states)
        else:
            qk = projected_qk
        token_k = qk[:, self.index_n_heads * self.index_head_dim :].reshape(
            -1, self.index_kv_heads, self.index_head_dim
        )
        if (
            pool is not None
            and cache_loc is not None
            and qk.shape[0] > 0
            and self._use_fused_prep(qk)
        ):
            from sglang.kernels.ops.attention.qsa_indexer import (
                qsa_index_q_norm_rope_store,
            )

            if (
                ensure_cos_sin_cache
                and not get_is_capture_mode()
                and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length")
            ):
                self.rotary_emb._ensure_cos_sin_cache_length(
                    int(positions.max().item())
                )
            key_state_buffer = pool.get_qsa_key_state_buffer(self.layer_id)
            q = qsa_index_q_norm_rope_store(
                qk,
                positions.long(),
                self.rotary_emb.cos_sin_cache,
                self._rope_axis_map(qk.device),
                self.q_layernorm.weight.data,
                cache_loc[: qk.shape[0]].long(),
                key_state_buffer.view(key_state_buffer.shape[0], -1),
                pool.qsa_rope_position_buffer,
                self.index_n_heads,
                self.rotary_emb.rotary_dim,
                self.q_layernorm.variance_epsilon,
                self.rotary_emb.is_neox_style,
                q_heads_padded=q_heads_padded,
                out_dtype=pool.qsa_compressed_dtype,
            )
            return q, token_k, True
        q_raw = qk[:, : self.index_n_heads * self.index_head_dim]
        q = self.q_layernorm(q_raw.reshape(-1, self.index_head_dim)).reshape(
            -1, self.index_n_heads, self.index_head_dim
        )
        q = self.apply_rope(positions, q, ensure_cos_sin_cache=ensure_cos_sin_cache)
        if pool is not None:
            q = q.to(pool.qsa_compressed_dtype)
        return q, token_k, False

    def _qpf_observe(self, forward_batch, metadata, rows, result, reason):
        if forward_batch is None:
            return
        from sglang.srt.layers.attention.qsa.observability import observe

        observe(
            "QPF", self.layer_id, forward_batch, rows, result=result, reason=reason,
            graph=metadata.is_cuda_graph,
            capturing=torch.cuda.is_current_stream_capturing(),
        )

    def _qpf_metadata_reason(self, hidden_states, pool, state_slots, group_locs, write_locs):
        # Called only after the unchanged original gate rejected the invocation.
        if pool is None:
            return "missing_pool"
        if state_slots is None:
            return "missing_state_slots"
        if group_locs is None:
            return "missing_graph_ring_group_locs"
        if write_locs is None:
            return "missing_graph_write_locs"
        if hidden_states.shape[0] == 0:
            return "empty_hidden_rows"
        return "fused_compress_ineligible"

    def _qpf_shape_reason(self, qk, pool, ring, compressed, out_dtype, group_locs, write_locs):
        # Tensor metadata only: no device values, reductions or synchronization.
        if qk.shape[0] == 0:
            return "empty_projected_rows"
        if not self._use_fused_prep(qk):
            return "fused_prep_ineligible"
        if not qk.is_contiguous():
            return "qk_not_contiguous"
        if ring.dtype != qk.dtype or pool.index_state_dtype != qk.dtype:
            return "ring_dtype_mismatch"
        if not ring.is_contiguous():
            return "ring_not_contiguous"
        if compressed.dtype != out_dtype:
            return "compressed_dtype_mismatch"
        if not 0 < self.compress_ratio <= 16:
            return "compress_ratio_out_of_range"
        if group_locs.dtype != torch.int32 or write_locs.dtype != torch.int32:
            return "group_or_write_dtype_mismatch"
        if group_locs.ndim != 2 or group_locs.shape[1] != self.compress_ratio:
            return "group_shape_mismatch"
        if group_locs.shape[0] == 0:
            return "empty_group_locs"
        if write_locs.shape != group_locs.shape[:1]:
            return "write_shape_mismatch"
        return "group_locs_not_contiguous"

    def _try_prologue_fold(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata,
        state_slots: torch.Tensor | None,
        projected_qk: torch.Tensor | None,
        *,
        forward_batch=None,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """h12: rev4's q_prep (project_qk's fused branch) and the graph compression
        (_compress_decode_cuda_graph's fused branch) as one launch. Returns (q, qk); q is None
        when the rows do not qualify, and qk (if projected here) is handed to the fallback."""
        from sglang.srt.layers.attention.qsa import prologue_fold as qpf

        pool = metadata.token_to_kv_pool
        group_locs = metadata.graph_ring_group_locs
        write_locs = metadata.graph_write_locs
        if (
            pool is None
            or state_slots is None
            or group_locs is None
            or write_locs is None
            or hidden_states.shape[0] == 0
            or not self._use_fused_compress(pool)
        ):
            self._qpf_observe(
                forward_batch, metadata, hidden_states.shape[0], "FALLBACK",
                self._qpf_metadata_reason(hidden_states, pool, state_slots, group_locs, write_locs),
            )
            return None, projected_qk
        qk = projected_qk
        if qk is None:
            qk, _ = self.index_qk_proj(hidden_states)
        key_state_buffer = pool.get_qsa_key_state_buffer(self.layer_id)
        ring = key_state_buffer.view(key_state_buffer.shape[0], -1)
        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)
        compressed = compressed_buffer.view(compressed_buffer.shape[0], -1)
        out_dtype = pool.qsa_compressed_dtype
        if (
            qk.shape[0] == 0
            or not self._use_fused_prep(qk)
            or not qk.is_contiguous()
            or ring.dtype != qk.dtype
            or pool.index_state_dtype != qk.dtype
            or not ring.is_contiguous()
            or compressed.dtype != out_dtype
            or not 0 < self.compress_ratio <= 16
            or group_locs.dtype != torch.int32
            or write_locs.dtype != torch.int32
            or group_locs.ndim != 2
            or group_locs.shape[1] != self.compress_ratio
            or group_locs.shape[0] == 0
            or write_locs.shape != group_locs.shape[:1]
            or not group_locs.is_contiguous()
        ):
            self._qpf_observe(
                forward_batch, metadata, hidden_states.shape[0], "FALLBACK",
                self._qpf_shape_reason(qk, pool, ring, compressed, out_dtype, group_locs, write_locs),
            )
            return None, qk
        if not get_is_capture_mode() and hasattr(
            self.rotary_emb, "_ensure_cos_sin_cache_length"
        ):
            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))
        rope_buffer = pool.qsa_rope_position_buffer
        kwargs = dict(
            qk=qk,
            positions=positions.long(),
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            axis_map=self._rope_axis_map(qk.device),
            q_weight=self.q_layernorm.weight.data,
            cache_loc=state_slots[: qk.shape[0]].long(),
            num_q_heads=self.index_n_heads,
            rotary_dim=self.rotary_emb.rotary_dim,
            q_eps=self.q_layernorm.variance_epsilon,
            is_neox_style=self.rotary_emb.is_neox_style,
            q_heads_padded=((self.index_n_heads + 7) // 8) * 8,
            out_dtype=out_dtype,
            group_locs=group_locs,
            k_weight=self.k_layernorm.weight.data,
            write_locs=write_locs,
            compress_ratio=self.compress_ratio,
            k_eps=self.k_layernorm.variance_epsilon,
        )
        snap = qpf.shadow_snapshot(ring, rope_buffer) if qpf.QPF_SHADOW else None
        q = qpf.qsa_index_prologue_fold(
            key_state_buffer=ring,
            rope_position_buffer=rope_buffer,
            compressed_k_buffer=compressed,
            **kwargs,
        )
        self._qpf_observe(forward_batch, metadata, hidden_states.shape[0], "ACTIVE", "eligible")
        if not getattr(self, "_qpf_engaged_logged", False):
            qpf.logger.info("Q38_QPF_ACTIVE layer=%s rows=%s groups=%s dtype=%s", self.layer_id, qk.shape[0], group_locs.shape[0], qk.dtype)
            self._qpf_engaged_logged = True
        if snap is not None:
            ref = qpf.shadow_reference(snap, **kwargs)
            qpf.shadow_compare(
                ref,
                q=q,
                key_state_buffer=ring,
                rope_position_buffer=rope_buffer,
                compressed_k_buffer=compressed,
                write_locs=write_locs,
                compress_ratio=self.compress_ratio,
            )
            self._qpf_shadow_q_ref = ref[0]
        return q, qk

    def normalize_compressed_keys(
        self, compressed_keys: torch.Tensor, block_positions: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.k_layernorm(
            compressed_keys.reshape(-1, self.index_head_dim)
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)
        return self.apply_rope(block_positions, normalized)

    def _use_fused_compress(self, pool) -> bool:
        return getattr(
            pool, "qsa_rope_position_buffer", None
        ) is not None and self._use_fused_prep(
            pool.get_qsa_key_state_buffer(self.layer_id)
        )

    def _fused_compress_store(
        self,
        pool,
        group_locs: torch.Tensor,
        write_locs: torch.Tensor,
        source_keys: torch.Tensor | None = None,
        source_rope: torch.Tensor | None = None,
    ) -> None:
        """Fused mean -> gemma norm -> MRoPE -> compressed-cache store;
        a None source_keys/source_rope reads the members from the pending ring."""
        from sglang.kernels.ops.attention.qsa_indexer import (
            qsa_index_k_compress_store,
        )

        if source_keys is None:
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
        if source_rope is None:
            source_rope = pool.qsa_rope_position_buffer
        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)
        qsa_index_k_compress_store(
            source_keys.reshape(source_keys.shape[0], -1)
            .contiguous()
            .to(pool.index_state_dtype),
            group_locs.to(torch.int32),
            source_rope,
            self.rotary_emb.cos_sin_cache,
            self._rope_axis_map(source_keys.device),
            self.k_layernorm.weight.data,
            write_locs.to(torch.int32),
            compressed_buffer.view(compressed_buffer.shape[0], -1),
            self.compress_ratio,
            self.rotary_emb.rotary_dim,
            self.k_layernorm.variance_epsilon,
            self.rotary_emb.is_neox_style,
        )

    def _pending_ring_slots(
        self, metadata, logical_positions: torch.Tensor, is_extend: bool
    ) -> torch.Tensor:
        return build_pending_ring_slots(
            token_to_batch_idx=metadata.token_to_batch_idx,
            req_pool_indices=metadata.req_pool_indices,
            sequence_lengths=metadata.sequence_lengths,
            logical_positions=logical_positions,
            compress_ratio=self.compress_ratio,
            is_extend=is_extend,
        )

    def _group_ring_slots(
        self, metadata, group_end_positions: torch.Tensor, sequence_ids: torch.Tensor
    ) -> torch.Tensor:
        return build_group_ring_slots(
            req_pool_indices=metadata.req_pool_indices,
            group_end_positions=group_end_positions,
            sequence_ids=sequence_ids,
            compress_ratio=self.compress_ratio,
        )

    def update_key_state_and_compress(
        self,
        token_k: torch.Tensor,
        logical_positions: torch.Tensor,
        rope_positions: torch.Tensor,
        metadata,
        state_slots: torch.Tensor | None = None,
        state_stored: bool = False,
    ) -> None:
        """Store the pending-group ring and compress each completed group."""

        pool = metadata.token_to_kv_pool
        is_extend = metadata.compress_member_rows is not None
        if not state_stored:
            if state_slots is None:
                state_slots = self._pending_ring_slots(
                    metadata, logical_positions, is_extend
                )
            pool.set_qsa_key_state_buffer(
                self.layer_id, state_slots[: token_k.shape[0]], token_k
            )
            pool.set_qsa_rope_position_buffer(
                state_slots[: token_k.shape[0]], rope_positions
            )

        if metadata.is_cuda_graph:
            self._compress_decode_cuda_graph(metadata)
            return

        if metadata.write_locs is None:
            raise RuntimeError(
                "QSA metadata is missing the precomputed write plan; the "
                "sparse-attention backend derives it from the batch lengths"
            )
        if metadata.write_locs.numel() == 0:
            return
        group_end_positions = metadata.compress_group_positions.long()
        compressed_locs = metadata.write_locs
        if is_extend:
            # Extend chunks are group-aligned; each planned group lies in this forward,
            # so read its members from the packed chunk tensors.
            member_rows = metadata.compress_member_rows.long()
            group_locs = member_rows[:, None] + torch.arange(
                self.compress_ratio, device=member_rows.device, dtype=torch.long
            )
            source_keys = token_k
            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)
            source_rope = metadata.extend_rope_matrix
            if source_rope is None:
                source_rope = build_rope_position_matrix(
                    rope_positions, token_k.shape[0]
                )
        else:
            # Paged eager rows (speculative fallback) complete at most one
            # group each; its members are exactly the pending ring window.
            group_locs = metadata.compress_group_ring_locs
            if group_locs is None:
                group_locs = self._group_ring_slots(
                    metadata,
                    group_end_positions,
                    metadata.compress_sequence_ids.long(),
                )
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
            source_rope = pool.qsa_rope_position_buffer
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                compressed_locs,
                source_keys=source_keys,
                source_rope=source_rope,
            )
            return
        key_groups = source_keys[group_locs]
        pooled = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            source_rope[group_locs[:, 0]]
        )
        normalized = self.normalize_compressed_keys(pooled, compressed_rope_positions)
        pool.set_qsa_compressed_k_buffer(self.layer_id, compressed_locs, normalized)

    def _compress_decode_cuda_graph(self, metadata) -> None:
        """Fixed-shape graph-replay compression; non-boundary rows write slot 0."""

        if metadata.graph_write_locs is None or metadata.graph_ring_group_locs is None:
            raise RuntimeError("QSA CUDA graph compression metadata is incomplete")
        pool = metadata.token_to_kv_pool
        group_locs = metadata.graph_ring_group_locs
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                metadata.graph_write_locs,
            )
            return
        key_groups = pool.get_qsa_key_state_buffer(self.layer_id)[group_locs]
        compressed = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            pool.qsa_rope_position_buffer[group_locs[:, 0]]
        )
        compressed = self.normalize_compressed_keys(
            compressed, compressed_rope_positions
        )
        pool.set_qsa_compressed_k_buffer(
            self.layer_id, metadata.graph_write_locs, compressed.contiguous()
        )

    def _rope_from_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """[n, 3] slot coordinates -> the layout apply_rope expects."""
        positions = positions.transpose(0, 1)
        if not getattr(self.rotary_emb, "mrope_section", None):
            return positions[0]
        return positions

    def apply_rope(
        self,
        positions: torch.Tensor,
        tensor: torch.Tensor,
        ensure_cos_sin_cache: bool = True,
    ) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        positions = positions.long()
        num_positions = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if num_positions != tensor.shape[0]:
            raise ValueError("QSA RoPE positions must match the token dimension")
        if (
            ensure_cos_sin_cache
            and not get_is_capture_mode()
            and hasattr(self.rotary_emb, "_ensure_cos_sin_cache_length")
        ):
            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))

        # position_cos/position_sin repeat cos/sin to the full rotary width;
        # apply_rotary_emb consumes one half.
        self.rotary_emb.get_cos_sin_with_position(positions)
        rotary_dim = self.rotary_emb.rotary_dim
        half_rotary_dim = rotary_dim // 2
        cos = self.rotary_emb.position_cos.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        sin = self.rotary_emb.position_sin.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        rotated = apply_rotary_emb(
            tensor[..., :rotary_dim],
            cos,
            sin,
            self.rotary_emb.is_neox_style,
        )
        return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)

    def select_prefill_tokens(
        self,
        q: torch.Tensor,
        compressed_keys: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths_for_rows: torch.Tensor,
        max_row_keys: Optional[int] = None,
    ) -> torch.Tensor:
        """Top-k token selection for packed prefill rows.

        ``max_row_keys`` (host int, the longest request's compressed-block count)
        switches scoring to request-relative logits: [rows, max_row_keys] instead
        of [rows, all compressed keys of the batch]. The latter caps a chunk at
        a few dozen rows once a batch holds many long prefixes, so the scoring
        kernel ran as thousands of one-CTA launches. Selection is unchanged: the
        top-k reads only each row's own section and returns indices relative to
        the row's start either way."""
        rows = q.shape[0]
        output = (
            _take_prefill_topk_output(
                rows, self.token_topk + self.compress_ratio - 1, q.device
            )
            if _PREFILL_TOPK_OUT[0] is not None
            else None
        )
        if output is None:
            output = torch.empty(
                (rows, self.token_topk + self.compress_ratio - 1),
                dtype=torch.int32,
                device=q.device,
            )
        if rows == 0:
            return output
        if _QSA_PREFILL_FAST_SELECT:
            from sglang.srt.layers.attention.qsa.prefill_select import (
                select_prefill_fast,
            )

            # Bit-identical to the code below; False when its shape is not covered.
            if select_prefill_fast(
                self,
                q,
                compressed_keys,
                row_starts,
                row_ends,
                query_positions,
                sequence_lengths_for_rows,
                max_row_keys,
                output,
            ):
                return output

        relative = (
            max_row_keys is not None and max_row_keys > 0 and q.is_cuda and HAS_TILELANG
        )
        row_chunk_size = _qsa_prefill_row_chunk_size(
            rows,
            max_row_keys if relative else compressed_keys.shape[0],
            q.shape[1],
        )
        for row_start in range(0, rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, rows)
            chunk_slice = slice(row_start, row_end)
            if compressed_keys.shape[0] == 0:
                block_indices = torch.full(
                    (row_end - row_start, self.block_topk),
                    -1,
                    dtype=torch.int32,
                    device=q.device,
                )
                logits = None
            elif relative:
                chunk_starts = row_starts[chunk_slice]
                chunk_lengths = row_ends[chunk_slice] - chunk_starts
                logits = qsa_mqa_prefill_relative(
                    q[chunk_slice],
                    compressed_keys,
                    chunk_starts,
                    row_ends[chunk_slice],
                    max_row_keys,
                )
                block_indices = qsa_fast_topk(
                    logits,
                    torch.zeros_like(chunk_starts),
                    chunk_lengths,
                    topk=self.block_topk,
                )
            else:
                logits = qsa_mqa_prefill(
                    q[chunk_slice],
                    compressed_keys,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                )
                block_indices = qsa_fast_topk(
                    logits,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                    topk=self.block_topk,
                )
            selected = expand_qsa_block_indices(
                block_indices,
                query_positions[chunk_slice],
                sequence_lengths_for_rows[chunk_slice],
                compress_ratio=self.compress_ratio,
                token_topk=self.token_topk,
            )
            output[chunk_slice].copy_(selected)
            del logits, block_indices, selected
        return output

    def select_decode_tokens(
        self,
        q: torch.Tensor,
        compressed_cache: torch.Tensor,
        compressed_page_table: torch.Tensor,
        compressed_lengths: torch.Tensor,
        max_model_len: int,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        defer_expansion: bool = False,
        score_q4_requests: torch.Tensor | None = None,
        rows_per_group: int = 1,
        return_logits: bool = False,
        grouped4_only: bool = False,
        sampled_verify_topk: bool = False,
    ) -> torch.Tensor:
        # fast_topk (k = 512) writes 0..length-1 for rows that fit k without reading
        # their scores, and never reads past a row's length.
        use_fast_topk = (
            q.is_cuda
            and self.block_topk == 512
            and not is_batch_invariant_mode_enabled()
        )
        if not (
            get_parallel().tp_size == 1
            and is_sm100_supported()
            and rows_per_group > 1
            and q.shape[0] % rows_per_group == 0
            and _QSA_GROUPED_SCORE_MIN_REQUESTS
            <= q.shape[0] // rows_per_group
            <= _QSA_GROUPED_SCORE_MAX_REQUESTS
            and _qsa_env_flag("SGLANG_QSA_GROUPED_SCORE")
        ):
            rows_per_group = 1
        elif q.shape[0] // rows_per_group > 16:
            _log_grouped_cap_once(q.shape[0] // rows_per_group)
        use_mk = _Q38C_MK_QSA and score_q4_requests is not None and not is_batch_invariant_mode_enabled()
        if score_q4_requests is not None and not is_batch_invariant_mode_enabled():
            if use_mk:
                from sglang.kernels.ops.attention.q38_mk_score import score_mk as score_q4

                _log_decode_path_once("mk_score", q.shape[0])
            else:
                from sglang.kernels.ops.attention.q38_score_q4 import score_q4

            logits = score_q4(
                q, compressed_cache, compressed_page_table, compressed_lengths,
                max_model_len, score_q4_requests, min_scored_len=self.block_topk,
            )
        else:
            logits = qsa_mqa_decode(
                q,
                compressed_cache,
                compressed_page_table,
                compressed_lengths,
                max_model_len,
                min_scored_len=self.block_topk if use_fast_topk else 0,
                rows_per_group=rows_per_group,
                num_query_heads=self.index_n_heads,
                grouped4_only=grouped4_only and rows_per_group > 1,
            )
        if use_fast_topk:
            # Decode rows start at zero, so compressed lengths double as row lengths;
            # skip the generic zero-fill + subtract.
            from sglang.srt.layers.attention.qsa.split_topk import (
                split_fast_topk,
                split_topk_supported,
            )

            if (
                get_parallel().tp_size == 1
                and is_sm100_supported()
                and logits.shape[0] <= _QSA_SPLIT_TOPK_MAX_ROWS
                and _qsa_env_flag("SGLANG_QSA_SPLIT_TOPK")
                and split_topk_supported(logits, self.block_topk)
            ):
                _log_decode_path_once("split_topk", logits.shape[0])
                block_indices = split_fast_topk(
                    logits,
                    compressed_lengths.to(torch.int32),
                    topk=self.block_topk,
                )
            elif sampled_verify_topk and not use_mk:
                block_indices = verify_topk.select(logits, compressed_lengths.to(torch.int32))
            else:
                if use_mk:
                    from sglang.kernels.ops.attention.q38_mk_topk import topk_mk as fast_topk
                else:
                    from sglang.kernels.ops.attention.fast_topk import fast_topk

                block_indices = fast_topk(
                    logits,
                    compressed_lengths.to(torch.int32),
                    topk=self.block_topk,
                    row_starts=None,
                )
            if _qsa_env_flag("SGLANG_QSA_SHADOW_CHECK", "0"):
                _qsa_shadow_check(
                    q,
                    compressed_cache,
                    compressed_page_table,
                    compressed_lengths,
                    max_model_len,
                    self.block_topk,
                    logits,
                    block_indices,
                    rows_per_group,
                )
        else:
            row_starts = torch.zeros_like(compressed_lengths, dtype=torch.int32)
            block_indices = qsa_fast_topk(
                logits, row_starts, compressed_lengths, topk=self.block_topk
            )
        if defer_expansion:
            return (block_indices, logits) if return_logits else block_indices
        expanded = expand_qsa_block_indices(
            block_indices,
            query_positions,
            sequence_lengths,
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
        )
        return (expanded, None) if return_logits else expanded

    def forward_cuda_cp(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        global_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill CP: score only this rank's zigzag query rows.

        ``hidden_states``/``positions`` are the local (padded) rows and their
        RoPE positions. Index q/k are projected for the local rows only; the raw
        keys are all-gathered into global token order so the pending-group ring
        store and the block compression run on the batch's global write plan
        exactly as without CP (every rank ends up with the complete compressed-K
        cache). Selection then scores the local rows against all compressed
        blocks of their request, position-driven, so the output rows line up
        with the local query rows the sparse attention consumes.
        """
        strategy = get_cp_strategy()
        if strategy is None or strategy.kind != ContextParallelStrategyKind.ZIGZAG:
            strategy_name = strategy.name if strategy is not None else "none"
            raise NotImplementedError(
                "QSA prefill CP only supports the zigzag strategy; "
                f"got {strategy_name}."
            )
        meta = forward_batch.attn_cp_metadata
        num_local = int(meta.total_q_prev_tokens + meta.total_q_next_tokens)
        global_logical = getattr(forward_batch, "positions", None)
        global_logical = (
            global_logical[0] if global_logical.ndim == 2 else global_logical
        ).flatten()
        num_global = indexer_metadata.get_token_to_batch_idx().numel()
        global_logical = global_logical[:num_global]
        local_logical = cp_shard_position_ids(global_logical, forward_batch)[:num_local]
        local_sequence_ids = cp_shard_hidden_states(
            indexer_metadata.get_token_to_batch_idx(), forward_batch
        )[:num_local]
        hidden_states = hidden_states[:num_local]
        positions = (
            positions[:, :num_local] if positions.ndim == 2 else positions[:num_local]
        )
        # Use the compressed-cache dtype for Q. Leaving cache_loc unset defers
        # the ring store until raw local K has been gathered into global order.
        # The load-time RoPE reservation covers these positions, as this layer's
        # attention already assumed; re-checking costs a device sync per layer.
        q, token_k, _ = self.project_qk(
            hidden_states,
            positions,
            pool=indexer_metadata.token_to_kv_pool,
            ensure_cos_sin_cache=False,
        )
        # Raw keys of all new tokens in global packed order: the ring store
        # and compression use the global write plan unchanged. Cached prefix
        # groups already reside in the compressed-K pool on every rank.
        token_k_local = token_k.reshape(num_local, -1).contiguous()
        token_k_full = cp_materialize_global_token_order(
            token_k_local, forward_batch
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)
        global_rope = (
            global_rope_positions[:, :num_global]
            if global_rope_positions.ndim == 2
            else global_rope_positions[:num_global]
        )
        self.update_key_state_and_compress(
            token_k_full,
            global_logical,
            global_rope,
            indexer_metadata,
            state_slots=indexer_metadata.pending_ring_slots,
            state_stored=False,
        )
        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(
                self.layer_id, local_logical, query_sequence_ids=local_sequence_ids
            )
        )
        row_sequence_lengths = sequence_lengths.index_select(
            0, local_sequence_ids.long()
        )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            local_logical,
            row_sequence_lengths,
            max_row_keys=indexer_metadata.prefill_max_row_keys(
                getattr(forward_batch, "seq_lens_cpu", None)
            ),
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        cp_global_rope_positions: torch.Tensor | None = None,
        projected_qk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Portable orchestration shared by different platforms.
        Fast paths are gated per platforms inside kernel calls.
        """
        forward_mode = forward_batch.forward_mode
        is_target_verify = getattr(forward_mode, "is_target_verify", lambda: False)()
        is_draft_extend = getattr(forward_mode, "is_draft_extend_v2", lambda: False)()
        if cp_global_rope_positions is not None and not (
            forward_mode.is_decode() or is_target_verify or is_draft_extend
        ):
            return self.forward_cuda_cp(
                hidden_states,
                positions,
                forward_batch,
                indexer_metadata,
                cp_global_rope_positions,
            )
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            # EAGLE/MTP may advance the model's RoPE coordinate independently
            # from the physical paged-KV position.  Compression and sparse
            # selection must use the latter; otherwise a draft step can index
            # token_slot_table[:, seq_len] one past the valid range.
            logical_positions = indexer_metadata.decode_logical_positions
            if logical_positions is None:
                logical_positions = indexer_metadata.get_seqlens_expanded() - 1
        else:
            logical_positions = getattr(forward_batch, "positions", None)
            if logical_positions is None:
                logical_positions = positions[0] if positions.ndim == 2 else positions
            logical_positions = logical_positions.flatten()
        # DP MAX_LEN padding adds token rows without assigning them to a
        # request. token_to_batch_idx is the source of truth for semantic rows.
        num_valid_tokens = indexer_metadata.get_token_to_batch_idx().numel()
        if logical_positions.numel() < num_valid_tokens:
            raise ValueError(
                "QSA logical positions are shorter than the request mapping: "
                f"positions={logical_positions.numel()}, mapping={num_valid_tokens}"
            )
        if hidden_states.shape[0] < num_valid_tokens:
            raise ValueError(
                "QSA hidden states are shorter than the request mapping: "
                f"hidden={hidden_states.shape[0]}, mapping={num_valid_tokens}"
            )
        position_tokens = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if position_tokens < num_valid_tokens:
            raise ValueError(
                "QSA RoPE positions are shorter than the request mapping: "
                f"positions={position_tokens}, mapping={num_valid_tokens}"
            )

        logical_positions = logical_positions[:num_valid_tokens]
        hidden_states = hidden_states[:num_valid_tokens]
        positions = (
            positions[:, :num_valid_tokens]
            if positions.ndim == 2
            else positions[:num_valid_tokens]
        )
        state_slots = indexer_metadata.pending_ring_slots
        if state_slots is None:
            state_slots = self._pending_ring_slots(
                indexer_metadata,
                logical_positions,
                indexer_metadata.compress_member_rows is not None,
            )
        is_decode_like = forward_mode.is_decode() or is_target_verify or is_draft_extend
        q = None
        if is_decode_like and indexer_metadata.is_cuda_graph and _qpf_enabled():
            q, projected_qk = self._try_prologue_fold(
                hidden_states, positions, indexer_metadata, state_slots, projected_qk,
                forward_batch=forward_batch,
            )
        if is_decode_like and not indexer_metadata.is_cuda_graph and _qpf_enabled():
            self._qpf_observe(forward_batch, indexer_metadata, hidden_states.shape[0], "FALLBACK", "non_graph_path")
        if is_decode_like and indexer_metadata.is_cuda_graph and not _qpf_enabled():
            self._qpf_observe(forward_batch, indexer_metadata, hidden_states.shape[0], "FALLBACK", "qpf_disabled")
        if q is None:
            q, token_k, state_stored = self.project_qk(
                hidden_states,
                positions,
                pool=indexer_metadata.token_to_kv_pool,
                cache_loc=state_slots,
                q_heads_padded=(
                    # The tilelang decode MQA kernel needs query heads in multiples of 8.
                    ((self.index_n_heads + 7) // 8) * 8
                    if (forward_mode.is_decode() or is_target_verify or is_draft_extend)
                    else None
                ),
                projected_qk=projected_qk,
            )
            self.update_key_state_and_compress(
                token_k,
                logical_positions,
                positions,
                indexer_metadata,
                state_slots=state_slots,
                state_stored=state_stored,
            )
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            compressed_cache, page_table, compressed_lengths, max_model_len = (
                indexer_metadata.get_decode_mqa_inputs(self.layer_id)
            )
            rows_per_group = 1
            if is_target_verify and get_parallel().tp_size == 1:
                width = int(getattr(forward_batch.spec_info, "draft_token_num", 1) or 1)
                if width > 1 and num_valid_tokens % width == 0:
                    rows_per_group = width
            grouped4_only = False
            if (
                _QSA_GROUPED4_DRAFT_EXTEND
                and is_draft_extend
                and get_parallel().tp_size == 1
                and forward_batch.req_pool_indices.numel() > 0
                and num_valid_tokens == 4 * forward_batch.req_pool_indices.numel()
            ):
                # Spec-v2 draft extend packs one equal window per request, rows in
                # position order (qsa/graph_metadata.py mode 2), so a window of four
                # groups like target verify.
                rows_per_group = 4
                grouped4_only = True
            score_q4_requests = None
            if (
                (_Q38C_SCORE_Q4 or _Q38C_MK_QSA)
                and (is_target_verify or is_draft_extend)
                and (is_draft_extend or get_parallel().tp_size != 1
                     or not _qsa_env_flag("SGLANG_QSA_GROUPED_SCORE"))
                and indexer_metadata.is_cuda_graph
                and q.is_cuda
                and is_sm100_supported()
                and q.shape == (4, 8, 128)
                and q.dtype == torch.bfloat16
                and compressed_cache.dtype == torch.bfloat16
                and compressed_cache.shape[1:3] == (16, 1)
                and self.index_n_heads == 4
                and self.block_topk == 512
                and self.compress_ratio == 4
                and forward_batch.req_pool_indices.numel() == 1
                and indexer_metadata.req_pool_indices is not None
                and indexer_metadata.req_pool_indices.numel() == 4
            ):
                score_q4_requests = indexer_metadata.req_pool_indices
            selected = self.select_decode_tokens(
                q,
                compressed_cache,
                page_table,
                compressed_lengths,
                max_model_len,
                logical_positions,
                indexer_metadata.get_seqlens_int32(),
                defer_expansion=q.is_cuda and indexer_metadata.defer_block_expansion,
                score_q4_requests=score_q4_requests,
                rows_per_group=rows_per_group,
                grouped4_only=grouped4_only,
                # SGLANG_QSA_VERIFY_TOPK=sampled: fast_topk's exact output, faster at verify rows;
                # SGLANG_ENABLE_QSA_DRAFT_EXTEND_UNION takes it for the grouped4 draft extend too.
                sampled_verify_topk=(
                    (is_target_verify and rows_per_group > 1 and verify_topk.MODE == "sampled")
                    or (_QSA_DRAFT_EXTEND_UNION and grouped4_only)
                ),
            )

            q_ref = self._qpf_shadow_q_ref
            if q_ref is not None:
                # SGLANG_QSA_QPF_SHADOW: the same selection from rev4's Q.
                from sglang.srt.layers.attention.qsa import prologue_fold as qpf

                self._qpf_shadow_q_ref = None
                select_args = (
                    compressed_cache,
                    page_table,
                    compressed_lengths,
                    max_model_len,
                    logical_positions,
                    indexer_metadata.get_seqlens_int32(),
                )
                select_kwargs = dict(
                    defer_expansion=q.is_cuda and indexer_metadata.defer_block_expansion,
                    score_q4_requests=score_q4_requests,
                    rows_per_group=rows_per_group,
                    grouped4_only=grouped4_only,
                    return_logits=True,
                )
                # rev4's selection from rev4's Q, and rev4's selection rerun on the fused Q (A/A:
                # the top-k's own choice among tied scores), each with its scorer logits.
                ref_sel, ref_logits = self.select_decode_tokens(q_ref, *select_args, **select_kwargs)
                aa_sel, aa_logits = self.select_decode_tokens(q, *select_args, **select_kwargs)
                qpf.shadow_compare_topk(
                    selected, ref_sel, aa_sel, aa_logits, ref_logits,
                    lengths=compressed_lengths, min_scored=self.block_topk,
                )
            return selected

        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(self.layer_id, logical_positions)
        )
        query_sequence_ids = indexer_metadata.get_token_to_batch_idx()
        row_sequence_lengths = sequence_lengths.index_select(
            0, query_sequence_ids.long()
        )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            logical_positions,
            row_sequence_lengths,
            max_row_keys=indexer_metadata.prefill_max_row_keys(
                getattr(forward_batch, "seq_lens_cpu", None)
            ),
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        cp_global_rope_positions: torch.Tensor | None = None,
        projected_qk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states,
            positions,
            forward_batch,
            indexer_metadata,
            cp_global_rope_positions,
            projected_qk=projected_qk,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states, positions, forward_batch, indexer_metadata
        )


__all__ = [
    "QSAIndexer",
]
