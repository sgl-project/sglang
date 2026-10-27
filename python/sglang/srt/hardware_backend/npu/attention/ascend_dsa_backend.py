from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_dsa_index_kpool,
    get_dsa_index_topk,
)
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    AscendAttnMultiStepDraftBackend,
    ForwardMetadata,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.dsa.kpool_plan import (
    KPoolExtendPlan,
    KPoolWritePlan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class AscendDSAForwardMetadata(ForwardMetadata):
    """Ascend attention metadata extended with the DSA/KPool contract.

    Only fields consumed by NPU runtime paths are retained.  All tensors
    live on device; no host-side mirrors are stored.
    """

    # ── Core attention ──
    page_size: int = 0
    cache_seqlens_int32: Optional[torch.Tensor] = None
    real_page_table: Optional[torch.Tensor] = None

    # ── DSA indexer (consumed by IndexerKPool.forward_npu) ──
    dsa_seqlens_expanded: Optional[torch.Tensor] = None
    token_to_batch_idx: Optional[torch.Tensor] = None

    # ── KPool plans ──
    kpool_extend_plan: Optional[KPoolExtendPlan] = None
    kpool_write_plan: Optional[KPoolWritePlan] = None


@dataclass(frozen=True)
class AscendDSAIndexerMetadata(BaseIndexerMetadata):
    """Read-only adapter from Ascend attention metadata to the indexer API."""

    attn_metadata: AscendDSAForwardMetadata

    def get_seqlens_int32(self) -> torch.Tensor:
        assert self.attn_metadata.cache_seqlens_int32 is not None
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        assert self.attn_metadata.real_page_table is not None
        return self.attn_metadata.real_page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        assert self.attn_metadata.dsa_seqlens_expanded is not None
        return self.attn_metadata.dsa_seqlens_expanded

    def get_page_table_1(self) -> torch.Tensor:
        raise NotImplementedError(
            "NPU kpool path uses get_page_table_64, not get_page_table_1"
        )

    def get_token_to_batch_idx(self) -> torch.Tensor:
        assert self.attn_metadata.token_to_batch_idx is not None
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(self, logits: torch.Tensor, topk: int, **_) -> torch.Tensor:
        raise NotImplementedError(
            "NPU KPool uses the vendor lightning indexer instead of topk_transform"
        )


class AscendDSAAttnBackend(AscendAttnBackend):
    """Add the KPool metadata required by Ascend attention for DSA models."""

    def __init__(self, model_runner: ModelRunner, speculative_step_id: int = 0):
        super().__init__(model_runner, speculative_step_id=speculative_step_id)
        # KPool planning still consumes host sequence lengths. The base DSA
        # backend can run mirrorless, but this subclass does not support it yet.
        self.needs_cpu_seq_lens = True
        hf_config = model_runner.model_config.hf_config
        self.dsa_index_topk = get_dsa_index_topk(hf_config)
        self.dsa_index_kpool = get_dsa_index_kpool(hf_config)
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        if self.dsa_index_kpool > 1:
            assert self.page_size % self.dsa_index_kpool == 0, (
                "Ascend DSA KPool requires a pool size that "
                f"divides page_size; got page_size={self.page_size}, "
                f"index_kpool={self.dsa_index_kpool}."
            )

    def _create_forward_metadata(self) -> AscendDSAForwardMetadata:
        return AscendDSAForwardMetadata(page_size=self.page_size)

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> AscendDSAForwardMetadata:
        metadata = super()._init_cuda_graph_metadata(
            bs, forward_mode, seq_lens, out_cache_loc
        )
        # DSA derives effective lengths from the raw graph inputs after the base
        # update. Keep a separate buffer for each backend and capture bucket.
        metadata.seq_lens = seq_lens.clone()
        return metadata

    def _query_lens_device(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Return per-request query lengths without a device-host round trip."""
        mode = forward_batch.forward_mode
        batch_size = forward_batch.batch_size

        # Target verify and draft extend v2 both use a fixed batch_size * N query layout.
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            extend_seq_lens = torch.full(
                (batch_size,),
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            forward_batch.extend_seq_lens = extend_seq_lens
            return extend_seq_lens
        if mode.is_extend():
            assert forward_batch.extend_seq_lens is not None
            if forward_batch.extend_seq_lens.shape[0] < batch_size:
                raise ValueError(
                    "DSA metadata requires at least one query length per request; "
                    f"got {forward_batch.extend_seq_lens.shape[0]} lengths for "
                    f"batch_size={batch_size}."
                )
            return forward_batch.extend_seq_lens[:batch_size].to(torch.int32)
        return torch.ones(
            (batch_size,),
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )

    def _query_layout_sizes(self, forward_batch: ForwardBatch) -> tuple[int, int]:
        """Return packed-token count and maximum query length from host metadata."""
        mode = forward_batch.forward_mode
        batch_size = forward_batch.batch_size

        if mode.is_target_verify() or mode.is_draft_extend_v2():
            query_len = int(self.speculative_num_draft_tokens)
            return batch_size * query_len, query_len

        if mode.is_extend():
            query_lens_cpu = forward_batch.extend_seq_lens_cpu
            if query_lens_cpu is not None:
                if len(query_lens_cpu) < batch_size:
                    raise ValueError(
                        "DSA metadata requires at least one host query length per "
                        f"request; got {len(query_lens_cpu)} lengths for "
                        f"batch_size={batch_size}."
                    )
                query_lens = [int(value) for value in query_lens_cpu[:batch_size]]
                return sum(query_lens), max(query_lens, default=0)

            # Normally extend has a CPU mirror. The packed input shape still lets
            # custom ForwardBatch producers size the output without a device read.
            if forward_batch.input_ids is None:
                raise ValueError(
                    "DSA extend metadata requires extend_seq_lens_cpu or input_ids"
                )
            total_q = int(forward_batch.input_ids.shape[0])
            return total_q, total_q

        return batch_size, 1

    def _cache_seqlens(self, forward_batch: ForwardBatch) -> torch.Tensor:
        cache_seqlens = forward_batch.seq_lens[: forward_batch.batch_size].to(
            torch.int32
        )
        if forward_batch.forward_mode.is_target_verify():
            cache_seqlens = cache_seqlens + self.speculative_num_draft_tokens
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            cache_seqlens = cache_seqlens + self.speculative_step_id + 1
        return cache_seqlens

    def _cache_seqlens_cpu_max(self, forward_batch: ForwardBatch) -> int:
        """Return the effective maximum KV length from the existing CPU mirror."""
        assert forward_batch.seq_lens_cpu is not None
        seq_lens_cpu = forward_batch.seq_lens_cpu[: forward_batch.batch_size]
        max_seq_len = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() > 0 else 0
        if forward_batch.forward_mode.is_target_verify():
            max_seq_len += self.speculative_num_draft_tokens
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            max_seq_len += self.speculative_step_id + 1
        return max_seq_len

    def _populate_dsa_metadata(
        self, forward_batch: ForwardBatch, _graph_capture: bool = False
    ) -> None:
        metadata = self.forward_metadata
        assert isinstance(metadata, AscendDSAForwardMetadata)
        device = forward_batch.seq_lens.device

        # ── Device tensors ──
        cache_seqlens = self._cache_seqlens(forward_batch)
        query_lens = self._query_lens_device(forward_batch)

        # The KPool write plan indexes the page table with one row per request.
        # Expand it per query at the indexer's paged entry point for MTP reads;
        # do not change the row semantics here.
        max_seq_len_k = self._cache_seqlens_cpu_max(forward_batch)
        page_table_1 = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[: forward_batch.batch_size],
            :max_seq_len_k,
        ]
        real_page_table_computed = (
            (page_table_1[:, :: self.page_size] // self.page_size)
            .to(torch.int32)
            .contiguous()
        )

        if _graph_capture:
            max_rows = real_page_table_computed.shape[0]
            max_cols = self.req_to_token_pool.req_to_token.shape[1] // self.page_size
            real_page_table = torch.zeros(
                (max_rows, max(max_cols, real_page_table_computed.shape[1])),
                dtype=torch.int32,
                device=device,
            )
            real_page_table[:, : real_page_table_computed.shape[1]] = (
                real_page_table_computed
            )
        else:
            real_page_table = real_page_table_computed

        mode = forward_batch.forward_mode
        needs_causal_expansion = (
            mode.is_extend() or mode.is_target_verify() or mode.is_draft_extend_v2()
        )
        if needs_causal_expansion:
            from sgl_kernel_npu.operators.indexer import expand_causal_seqlens

            total_q, max_query_len = self._query_layout_sizes(forward_batch)
            seq_lens_expanded, token_request_ids = expand_causal_seqlens(
                cache_seqlens,
                query_lens,
                device,
                total_q,
                max_query_len,
            )
        else:
            seq_lens_expanded = None
            token_request_ids = torch.arange(
                forward_batch.batch_size,
                device=page_table_1.device,
                dtype=torch.int64,
            )
        # ── Populate only consumed fields ──
        metadata.page_size = self.page_size
        metadata.cache_seqlens_int32 = cache_seqlens
        metadata.real_page_table = real_page_table
        metadata.dsa_seqlens_expanded = seq_lens_expanded
        metadata.token_to_batch_idx = token_request_ids

        # Allocate the plan once during capture; _replay_dsa_metadata refreshes
        # it in place during replay.
        metadata.kpool_extend_plan = None
        metadata.kpool_write_plan = None
        if self.dsa_index_kpool > 1:
            self._init_kpool_metadata(
                metadata,
                forward_batch,
                metadata.real_page_table,
                metadata.dsa_seqlens_expanded,
            )

    def _init_kpool_metadata(
        self,
        metadata: AscendDSAForwardMetadata,
        forward_batch: ForwardBatch,
        real_page_table: torch.Tensor,
        seq_lens_expanded: torch.Tensor,
    ) -> None:
        """Compute KPoolExtendPlan / KPoolWritePlan (NPU PyTorch variants).

        Mirrors dsa_backend.py _init_kpool_metadata but uses NPU-safe
        PyTorch implementations instead of CUDA Triton kernels.
        """
        from sglang.srt.layers.attention.dsa.kpool_plan import (
            _alloc_kpool_write_plan_buffers,
            init_kpool_extend_metadata_npu,
            update_kpool_write_plan_npu,
        )

        pool_size = self.dsa_index_kpool
        slots_per_page = self.page_size
        forward_mode = forward_batch.forward_mode

        # ── Extend: build KPoolExtendPlan (writes + tails) ──
        if forward_mode.is_extend_without_speculative():
            init_kpool_extend_metadata_npu(
                metadata,
                forward_batch,
                pool_size=pool_size,
                real_page_size=self.page_size,
                slots_per_page=slots_per_page,
                full_real_page_table=real_page_table,
                local_seqlens_expanded=seq_lens_expanded,
            )
            return

        # ── Verify / draft-extend-v2: build KPoolWritePlan ──
        # Plain NPU decode updates the KPool cache directly through
        # kpool_decode_update_index_cache and does not consume this plan.
        if not (forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()):
            return

        is_v2 = forward_mode.is_draft_extend_v2()
        num_draft_tokens = self.speculative_num_draft_tokens

        if num_draft_tokens == 0:
            return

        # Allocate the plan here for capture/eager execution; graph replay only
        # refreshes these buffers in place.
        plan = _alloc_kpool_write_plan_buffers(
            max_bs=forward_batch.batch_size,
            num_draft_tokens=num_draft_tokens,
            kpool=pool_size,
            device=forward_batch.seq_lens.device,
            is_verify=True,
            is_v2=is_v2,
        )
        metadata.kpool_write_plan = plan

        # The two MTP stages use different seq_lens semantics, so recover the
        # ring-write start for this iteration.
        seq_lens = forward_batch.seq_lens[: forward_batch.batch_size]
        if is_v2:
            write_start = (seq_lens - self.speculative_num_draft_tokens).to(torch.int32)
        else:
            write_start = seq_lens.to(torch.int32)

        # Draft extend v2 runs with fixed N rows but may commit only the accepted
        # tokens for each request.
        effective_n_per_batch = None
        if is_v2 and forward_batch.spec_info is not None:
            effective_n_per_batch = getattr(
                forward_batch.spec_info, "num_accept_tokens", None
            )
            if effective_n_per_batch is not None:
                effective_n_per_batch = effective_n_per_batch[
                    : forward_batch.batch_size
                ]

        # Refresh the plan in place and let the NPU plan entry point select the
        # implementation, preserving capture-time addresses.
        update_kpool_write_plan_npu(
            metadata,
            write_start=write_start,
            req_pool_indices=forward_batch.req_pool_indices[: forward_batch.batch_size],
            real_page_table=real_page_table,
            pool_size=pool_size,
            real_page_size=self.page_size,
            num_draft_tokens=num_draft_tokens,
            forward_mode=forward_mode,
            slots_per_page=slots_per_page,
            effective_n_per_batch=effective_n_per_batch,
        )

    def _replay_dsa_metadata(self, forward_batch: ForwardBatch) -> None:
        """Update DSA metadata tensors in-place during graph replay."""
        metadata = self.forward_metadata
        assert isinstance(metadata, AscendDSAForwardMetadata)
        forward_mode = forward_batch.forward_mode
        bs = forward_batch.batch_size

        # 1. cache_seqlens_int32 — in-place copy
        cache_seqlens = self._cache_seqlens(forward_batch)
        if metadata.cache_seqlens_int32 is None or (
            metadata.cache_seqlens_int32.shape != cache_seqlens.shape
        ):
            raise RuntimeError(
                "DSA graph replay cache_seqlens shape differs from capture: "
                f"captured={getattr(metadata.cache_seqlens_int32, 'shape', None)}, "
                f"replay={cache_seqlens.shape}, batch_size={bs}."
            )
        metadata.cache_seqlens_int32.copy_(cache_seqlens)

        # Query lengths are mode-dependent. In particular, regular draft decode
        # intentionally leaves ForwardBatch.extend_seq_lens unset and has one
        # query token per request, so consumers must use this normalized tensor.
        query_lens = self._query_lens_device(forward_batch)

        # 2. real_page_table — recompute and in-place copy
        max_seq_len_k = self._cache_seqlens_cpu_max(forward_batch)
        page_table_1 = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[:bs], :max_seq_len_k
        ]
        new_real_page_table = (
            (page_table_1[:, :: self.page_size] // self.page_size)
            .to(torch.int32)
            .contiguous()
        )

        n_rows, n_cols = new_real_page_table.shape
        if (
            metadata.real_page_table is None
            or metadata.real_page_table.shape[0] != n_rows
            or metadata.real_page_table.shape[1] < n_cols
        ):
            raise RuntimeError(
                "DSA graph replay page-table shape exceeds capture buffer: "
                f"captured={getattr(metadata.real_page_table, 'shape', None)}, "
                f"replay={new_real_page_table.shape}, batch_size={bs}."
            )
        metadata.real_page_table[:n_rows, :n_cols].copy_(new_real_page_table)

        # 3. dsa_seqlens_expanded — update causal lengths for each query token
        needs_causal_expansion = (
            forward_mode.is_extend()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        )
        total_q, max_query_len = self._query_layout_sizes(forward_batch)
        if needs_causal_expansion:
            from sgl_kernel_npu.operators.indexer import expand_causal_seqlens

            expected_shape = (total_q,)
            if (
                metadata.dsa_seqlens_expanded is None
                or metadata.dsa_seqlens_expanded.shape != expected_shape
                or metadata.token_to_batch_idx is None
                or metadata.token_to_batch_idx.shape != expected_shape
            ):
                raise RuntimeError(
                    "DSA graph replay packed-query shape differs from capture: "
                    "captured_seqlens="
                    f"{getattr(metadata.dsa_seqlens_expanded, 'shape', None)}, "
                    "captured_token_map="
                    f"{getattr(metadata.token_to_batch_idx, 'shape', None)}, "
                    f"replay={expected_shape}, batch_size={bs}."
                )
            expand_causal_seqlens(
                cache_seqlens,
                query_lens,
                cache_seqlens.device,
                total_q,
                max_query_len,
                seqlens_out=metadata.dsa_seqlens_expanded,
                token_to_batch_out=metadata.token_to_batch_idx,
            )
        else:
            expected_shape = (bs,)
            if metadata.token_to_batch_idx is None or (
                metadata.token_to_batch_idx.shape != expected_shape
            ):
                raise RuntimeError(
                    "DSA graph replay token-map shape differs from capture: "
                    f"captured={getattr(metadata.token_to_batch_idx, 'shape', None)}, "
                    f"replay={expected_shape}, batch_size={bs}."
                )
            torch.arange(bs, out=metadata.token_to_batch_idx)
        # 4. kpool_write_plan — update for verify/draft-extend-v2.
        # Plain decode has no plan because its index-cache update consumes the
        # ForwardBatch and indexer metadata tensors directly.
        if (
            self.dsa_index_kpool > 1
            and metadata.kpool_write_plan is not None
            and (forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2())
        ):
            from sglang.srt.layers.attention.dsa.kpool_plan import (
                update_kpool_write_plan_npu,
            )

            is_v2 = forward_mode.is_draft_extend_v2()
            num_draft_tokens = self.speculative_num_draft_tokens
            if num_draft_tokens > 0:
                seq_lens = forward_batch.seq_lens[:bs]
                if is_v2:
                    write_start = (seq_lens - self.speculative_num_draft_tokens).to(
                        torch.int32
                    )
                else:
                    write_start = seq_lens.to(torch.int32)

                # Draft v2 keeps a fixed-N graph shape but commits only the
                # tokens accepted in this iteration.
                effective_n_per_batch = None
                if is_v2 and forward_batch.spec_info is not None:
                    effective_n_per_batch = getattr(
                        forward_batch.spec_info, "num_accept_tokens", None
                    )
                    if effective_n_per_batch is not None:
                        effective_n_per_batch = effective_n_per_batch[
                            : forward_batch.batch_size
                        ]
                update_kpool_write_plan_npu(
                    metadata,
                    write_start=write_start,
                    req_pool_indices=forward_batch.req_pool_indices[:bs],
                    real_page_table=metadata.real_page_table,
                    pool_size=self.dsa_index_kpool,
                    real_page_size=self.page_size,
                    num_draft_tokens=num_draft_tokens,
                    forward_mode=forward_mode,
                    slots_per_page=self.page_size,
                    effective_n_per_batch=effective_n_per_batch,
                )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        super().init_forward_metadata(forward_batch)
        self._populate_dsa_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        # Capture allocates stable addresses; replay updates only the contents
        # in place so graph references remain valid.
        if in_capture:
            self._populate_dsa_metadata(forward_batch, _graph_capture=True)
        else:
            self._replay_dsa_metadata(forward_batch)

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> AscendDSAIndexerMetadata:
        metadata = self.forward_metadata
        assert isinstance(metadata, AscendDSAForwardMetadata)
        return AscendDSAIndexerMetadata(attn_metadata=metadata)


class AscendDSAAttnMultiStepDraftBackend(AscendAttnMultiStepDraftBackend):
    """Create an Ascend backend that generates DSA metadata for each draft decode step."""

    backend_cls = AscendDSAAttnBackend
