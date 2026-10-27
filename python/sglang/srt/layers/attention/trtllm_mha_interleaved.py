"""Address-stable TRTLLM MHA metadata for interleaved decode graphs."""

from dataclasses import fields

import torch


class TRTLLMInterleavedGraphMetadata:
    """One bucket/stage owns metadata; buckets reuse scratch by stage index.

    Distinct stages can execute concurrently. Different buckets of one stage
    execute serially, so sharing their workspace avoids retaining 512 MiB per
    captured bucket. Native graph caches are never rebound or overwritten.
    """

    def __init__(self, backend, forward_batch, *, stage_index, metadata_type):
        backend.validate_interleaved_graph_support()
        self._backend = backend
        self._batch_size = forward_batch.batch_size
        self._validate_batch(forward_batch)
        # Construct the native metadata schema without borrowing its shared
        # graph buffers or temporarily mutating its batch-size cache.
        bs = self._batch_size
        device = forward_batch.seq_lens.device
        self.metadata = metadata_type(
            cache_seqlens_int32=torch.zeros(bs, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.zeros(bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros(
                (bs, backend.max_num_pages), dtype=torch.int32, device=device
            ),
        )
        backend.bind_interleaved_stage_workspace(self.metadata, stage_index=stage_index)
        self._cache = backend.decode_cuda_graph_metadata
        self._identity = self._fingerprint()
        self.refresh(forward_batch)

    def _validate_batch(self, batch):
        if (
            batch.batch_size != self._batch_size
            or not batch.forward_mode.is_decode_or_idle()
            or batch.spec_info is not None
        ):
            raise RuntimeError("TRTLLM_INTERLEAVED_REQUIRES_NON_SPEC_DECODE")

    def _fingerprint(self):
        return tuple(
            (field.name, value.data_ptr(), tuple(value.shape), value.stride())
            for field in fields(self.metadata)
            if torch.is_tensor(value := getattr(self.metadata, field.name))
        )

    def is_valid(self):
        return (
            self._backend is not None
            and self._backend.decode_cuda_graph_metadata is self._cache
            and self._identity == self._fingerprint()
        )

    def refresh(self, forward_batch):
        if not self.is_valid():
            raise RuntimeError("TRTLLM_INTERLEAVED_METADATA_DRIFT")
        self._validate_batch(forward_batch)
        backend = self._backend
        if backend.kv_index_translator.is_translating:
            backend.kv_index_translator.copy_page_table(
                forward_batch.kv_loc_plan, out=self.metadata.page_table
            )
        # Static-pool page tables and sequence lengths are refreshed by the
        # metadata kernel recorded at bind(), on every graph replay.
        backend.forward_metadata = self.metadata

    def bind(self, forward_batch):
        self._validate_batch(forward_batch)
        self._backend._apply_cuda_graph_metadata(
            bs=self._batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            forward_mode=forward_batch.forward_mode,
            spec_info=None,
            out_cache_loc=forward_batch.out_cache_loc,
            metadata_override=self.metadata,
        )

    def close(self):
        self.metadata = self._cache = self._backend = None
