from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.minicpm_sala import get_block_table
from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
)
from sglang.srt.layers.attention.minicpm.attention_adapter import (
    MiniCPMFlashAttentionAdapter,
    MiniCPMFlashInferAdapter,
)
from sglang.srt.layers.attention.minicpm.cache import attach_compressed_cache
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    CompressionLevelMetadata,
    MiniCPMSparseMetadata,
    RepeatedSegmentLayout,
    _assign_row_metadata,
    _build_dense_verify_overwrite,
    _build_k1_k2_compression_metadata,
    _build_sparse_decode_metadata,
    _build_sparse_verify_rows,
    _plan_repeated_segments,
    _plan_sparse_decode,
    _plan_sparse_prefill,
    _plan_sparse_verify,
    allocate_and_compress_keys,
    batched_gather,
    compressed_attention,
    compressed_attention_tilelang,
    get_compress_k_v2,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_platform,
    get_schedule,
    get_spec,
)
from sglang.srt.utils import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


def _transpose_head_group_layout(
    tensor: torch.Tensor,
    spans: list[tuple[int, int]],
    *,
    head_group_num: int,
    heads_per_group: int,
    to_group_major: bool,
) -> None:
    for start, seq_len in spans:
        end = start + head_group_num * seq_len
        leading_shape = (
            (seq_len, head_group_num) if to_group_major else (head_group_num, seq_len)
        )
        tensor[start:end] = (
            tensor[start:end]
            .clone()
            .view(*leading_shape, heads_per_group, tensor.shape[-1])
            .transpose(0, 1)
            .reshape(-1, heads_per_group, tensor.shape[-1])
        )


def _copy_dense_page_table(
    destination: torch.Tensor,
    destination_row: int,
    source: torch.Tensor,
    source_row: int,
    kv_len: int,
    head_group_num: int,
) -> None:
    for group in range(head_group_num):
        destination[destination_row + group, :kv_len] = (
            source[source_row, :kv_len] * head_group_num + group
        )


def _gather_compressed_keys(
    full_compressed_k: torch.Tensor,
    level: CompressionLevelMetadata,
    batches: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [
        level.cu_seqlens_cpu[batch + 1] - level.cu_seqlens_cpu[batch]
        for batch in batches
    ]
    compact_k = torch.cat(
        [
            full_compressed_k[
                level.cu_seqlens_cpu[batch] : level.cu_seqlens_cpu[batch + 1]
            ]
            for batch in batches
        ]
    )
    compact_cu_seqlens = torch.tensor(
        [0, *lengths], dtype=torch.int32, device=full_compressed_k.device
    ).cumsum(0, dtype=torch.int32)
    return compact_k, compact_cu_seqlens


def _copy_dense_page_tables(
    sparse_page_table: torch.Tensor,
    page_table: torch.Tensor,
    *,
    dense_rows: list[tuple[int, int, int]],
    head_group_num: int,
) -> None:
    for row_start, batch_idx, kv_len in dense_rows:
        _copy_dense_page_table(
            destination=sparse_page_table,
            destination_row=row_start,
            source=page_table,
            source_row=batch_idx,
            kv_len=kv_len,
            head_group_num=head_group_num,
        )


class MiniCPMSparseBackend(AttentionBackend):
    """MiniCPM sparse dispatch layered on the standard FlashAttention backend."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        fa_impl_ver=3,
        *,
        use_flashinfer: bool,
    ):
        super().__init__()
        use_blackwell = get_platform().is_blackwell
        if use_blackwell:
            fa_impl_ver = 4
        self.flash_attn_backend = FlashAttentionBackend(
            model_runner,
            skip_prefill=skip_prefill,
            fa_impl_ver=fa_impl_ver,
        )
        self.forward_metadata: Optional[MiniCPMSparseMetadata] = None
        self.max_context_len = self.flash_attn_backend.max_context_len
        self.device = self.flash_attn_backend.device
        self.model_dtype = model_runner.dtype
        self._use_cuda_graph_buffers = False
        self.decode_cuda_graph_metadata = (
            self.flash_attn_backend.decode_cuda_graph_metadata
        )
        self.req_to_token_pool = self.flash_attn_backend.req_to_token_pool
        self.token_to_kv_pool = self.flash_attn_backend.token_to_kv_pool
        self.page_size = self.flash_attn_backend.page_size
        tp_size = get_parallel().attn_tp_size
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(tp_size)
        self.num_q_heads = model_runner.model_config.num_attention_heads // tp_size

        # Sparse attention configuration (required for MiniCPM)
        hf_config = model_runner.model_config.hf_config

        if not isinstance(hf_config, MiniCPMHybridConfig) or not (
            hf_config.has_minicpm_sparse_attention
        ):
            raise ValueError(
                "MiniCPM model must have sparse attention enabled. "
                "Please ensure the model config has MiniCPM sparse attention enabled."
            )
        sparse_config = hf_config.sparse_config
        self.kernel_size = sparse_config["kernel_size"]
        self.kernel_stride = sparse_config["kernel_stride"]
        self.init_blocks = sparse_config["init_blocks"]
        self.block_size = sparse_config["block_size"]
        self.window_size = sparse_config["window_size"]
        if (
            self.kernel_stride <= 0
            or self.kernel_size <= 0
            or self.block_size <= 0
            or self.window_size < 0
            or self.kernel_size % self.kernel_stride
            or self.block_size % self.kernel_stride
            or self.window_size % self.block_size
        ):
            raise ValueError(
                "MiniCPM sparse kernel_stride must divide kernel_size and "
                "block_size, and block_size must divide window_size."
            )
        attach_compressed_cache(
            self.req_to_token_pool,
            model_runner.token_to_kv_pool_allocator,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
            enable_memory_saver=get_exec().features.enable_memory_saver,
        )
        self.req_to_sparse_k1_token = self.req_to_token_pool.req_to_sparse_k1_token
        self.req_to_sparse_k2_token = self.req_to_token_pool.req_to_sparse_k2_token
        self.minicpm_dense_as_sparse = envs.SGLANG_MINICPM_DENSE_AS_SPARSE.get()
        self.dense_len = (
            0 if self.minicpm_dense_as_sparse else sparse_config["dense_len"]
        )
        # Seeds graph captures' history length and bounds the startup context check;
        # unlike dense_len, the dense-as-sparse env does not zero it.
        self.config_dense_len = sparse_config["dense_len"]
        topk = sparse_config["topk"]
        self.local_blocks = self.window_size // self.block_size
        self.sparse_topk = topk + (self.window_size // self.block_size)
        self.num_sparse_topk_tokens = self.block_size * self.sparse_topk
        required_context_len = max(self.config_dense_len, self.num_sparse_topk_tokens)
        if self.max_context_len < required_context_len:
            raise ValueError(
                "MiniCPM sparse attention requires context_length >= "
                f"{required_context_len}, got {self.max_context_len}."
            )

        # Head group number derived from model configuration
        self.head_dim = model_runner.model_config.head_dim
        self.head_group_num = self.num_kv_heads
        self.heads_per_group = self.num_q_heads // self.head_group_num
        if self.heads_per_group != 16:
            raise ValueError(
                "MiniCPM sparse attention requires 16 query heads per KV head, "
                f"got {self.heads_per_group}."
            )
        self.k1_kernel_size = self.kernel_size
        self.k1_kernel_stride = self.kernel_stride
        self.k2_kernel_size = self.kernel_size * 4
        self.k2_kernel_stride = self.kernel_stride * 4

        self._init_spec_decode_config()

        self.minicpm_fuse_topk = (
            use_blackwell and use_flashinfer
        ) or envs.SGLANG_MINICPM_FUSE_TOPK.get()
        # The fused top-k scores on the k1 level alone,
        # so the k2 repeat layout has no readers under minicpm_fuse_topk.
        self.verify_repeat_levels = frozenset(
            name for name in ("k1", "k2") if name != "k2" or not self.minicpm_fuse_topk
        )
        dtype_str = str(self.model_dtype).removeprefix("torch.")
        if self.minicpm_fuse_topk and dtype_str not in ("bfloat16", "float16"):
            raise ValueError(
                "MiniCPM fused top-k only supports bfloat16 and float16, "
                f"got {self.model_dtype}."
            )

        max_cache_len = self.max_context_len
        pooled_k_len = (max_cache_len + self.block_size - 1) // self.block_size

        output_topk = min(self.sparse_topk, pooled_k_len)

        # For the kernel, we need power of 2 topk
        topk_power2 = next_power_of_2(output_topk)
        kernel_topk = min(topk_power2, pooled_k_len)
        # Make sure it's still power of 2
        if kernel_topk != next_power_of_2(kernel_topk):
            kernel_topk = next_power_of_2(kernel_topk) // 2
        kernel_topk = max(8, kernel_topk)
        self.kernel_topk = kernel_topk
        self.decode_fused_kernels = {}
        self.prefill_fused_kernels = {}
        bucketed_pooled_k_len = next_power_of_2(pooled_k_len)

        pooling_block_stride = self.block_size // self.kernel_stride
        pooling_pad_len = self.kernel_size // self.kernel_stride - 1
        pooling_num_offs = (
            self.kernel_size // self.kernel_stride
            + self.block_size // self.kernel_stride
            - 1
        )
        self.fused_kernel_kwargs = {
            "groups": self.heads_per_group,
            "heads": self.num_q_heads,
            "dim": self.head_dim,
            "topk": self.kernel_topk,
            "pooled_k_len": bucketed_pooled_k_len,
            "m_block_dim": self.heads_per_group,
            "block_M": self.heads_per_group,
            "block_stride": pooling_block_stride,
            "pad_len": pooling_pad_len,
            "num_offs": pooling_num_offs,
            "kernel_stride": self.kernel_stride,
            "block_size": self.block_size,
            "dense_len": self.dense_len,
            "init_blocks": self.init_blocks,
            "local_blocks": self.local_blocks,
            "dtype_str": dtype_str,
        }
        chunked_prefill_size = get_schedule().chunked_prefill_size
        if self.minicpm_fuse_topk and chunked_prefill_size <= 0:
            raise ValueError(
                "MiniCPM fused top-k requires a positive --chunked-prefill-size."
            )
        self.prefill_kernel_max_seqlen_q_grid = chunked_prefill_size
        if self.minicpm_fuse_topk:
            for batch_size in range(1, model_runner.max_running_requests + 1):
                self._get_fused_topk_kernel(batch_size, is_prefill=True)

        self.attention_adapter = (
            MiniCPMFlashInferAdapter(
                model_runner,
                head_group_num=self.head_group_num,
                heads_per_group=self.heads_per_group,
                head_dim=self.head_dim,
                page_size=self.page_size,
                max_kv_tokens_per_row=max(
                    self.dense_len,
                    self.num_sparse_topk_tokens,
                ),
                rows_per_req=self.head_group_num
                * (self.speculative_num_draft_tokens or 1),
            )
            if use_flashinfer
            else MiniCPMFlashAttentionAdapter(self.flash_attn_backend)
        )

    def _init_spec_decode_config(self):
        spec = get_spec()
        self.speculative_num_draft_tokens = spec.speculative_num_draft_tokens
        # A target-only server carries no speculative args (eagle_topk is
        # None); treat that as chain drafts.
        if (spec.speculative_eagle_topk or 1) > 1:
            raise NotImplementedError(
                "MiniCPM backend target verify supports chain drafts only "
                "(speculative_eagle_topk == 1)"
            )

    def _get_fused_topk_kernel(self, batch_size: int, *, is_prefill: bool):
        if not self.minicpm_fuse_topk:
            return None

        from sglang.srt.layers.attention.minicpm.fuse_kernel import (
            fused_attn_pooling_online_topk_decode,
            fused_attn_pooling_online_topk_prefill,
        )

        cache = self.prefill_fused_kernels if is_prefill else self.decode_fused_kernels
        if batch_size not in cache:
            kwargs = dict(self.fused_kernel_kwargs, batch_size=batch_size)
            if is_prefill:
                kwargs["max_seqlen_q_grid"] = self.prefill_kernel_max_seqlen_q_grid
                cache[batch_size] = fused_attn_pooling_online_topk_prefill(**kwargs)
            else:
                cache[batch_size] = fused_attn_pooling_online_topk_decode(**kwargs)
        return cache[batch_size]

    def update_batch_for_sparse(
        self, forward_batch: ForwardBatch, metadata: MiniCPMSparseMetadata
    ):
        if forward_batch.forward_mode.is_target_verify():
            self._update_batch_for_verify(forward_batch, metadata)
            return

        metadata.k1, metadata.k2 = _build_k1_k2_compression_metadata(
            req_pool_indices=forward_batch.req_pool_indices,
            base_metadata=metadata.base,
            req_to_sparse_k1_token=self.req_to_sparse_k1_token,
            req_to_sparse_k2_token=self.req_to_sparse_k2_token,
            k1_kernel_size=self.k1_kernel_size,
            k1_kernel_stride=self.k1_kernel_stride,
            k2_kernel_size=self.k2_kernel_size,
            k2_kernel_stride=self.k2_kernel_stride,
            cu_seqlens_q=metadata.base.cu_seqlens_q,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

        if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            _plan_sparse_prefill(
                forward_batch,
                metadata,
                head_group_num=self.head_group_num,
                heads_per_group=self.heads_per_group,
                dense_len=self.dense_len,
                sparse_topk=self.sparse_topk,
                block_size=self.block_size,
            )
            _copy_dense_page_tables(
                metadata.sparse_page_table,
                metadata.base.page_table,
                dense_rows=metadata.dense_rows,
                head_group_num=self.head_group_num,
            )
        else:
            _plan_sparse_decode(
                forward_batch,
                metadata,
                head_group_num=self.head_group_num,
                heads_per_group=self.heads_per_group,
                dense_len=self.dense_len,
                sparse_topk=self.sparse_topk,
                block_size=self.block_size,
            )
            _copy_dense_page_tables(
                metadata.sparse_page_table,
                metadata.base.page_table,
                dense_rows=metadata.dense_rows,
                head_group_num=self.head_group_num,
            )

    def _update_batch_for_verify(
        self,
        forward_batch: ForwardBatch,
        metadata: MiniCPMSparseMetadata,
    ):
        # seq_lens + num_draft_tokens: compression must cover the draft keys
        # too, or block scoring never sees the proposed tokens.
        metadata.k1, metadata.k2 = _build_k1_k2_compression_metadata(
            req_pool_indices=forward_batch.req_pool_indices,
            base_metadata=metadata.base,
            req_to_sparse_k1_token=self.req_to_sparse_k1_token,
            req_to_sparse_k2_token=self.req_to_sparse_k2_token,
            k1_kernel_size=self.k1_kernel_size,
            k1_kernel_stride=self.k1_kernel_stride,
            k2_kernel_size=self.k2_kernel_size,
            k2_kernel_stride=self.k2_kernel_stride,
            cu_seqlens_q=metadata.base.cu_seqlens_q,
            seq_lens_cpu=(
                forward_batch.seq_lens_cpu + self.speculative_num_draft_tokens
            ),
        )
        _plan_sparse_verify(
            forward_batch,
            metadata,
            head_group_num=self.head_group_num,
            heads_per_group=self.heads_per_group,
            dense_len=self.dense_len,
            sparse_topk=self.sparse_topk,
            block_size=self.block_size,
            num_draft_tokens=self.speculative_num_draft_tokens,
        )
        # Varlen segments cannot be shared across verify rows.
        metadata.k1_repeat = _plan_repeated_segments(
            metadata.k1.cu_seqlens,
            total_tokens=metadata.k1.cu_seqlens_cpu[-1],
            repeats=self.speculative_num_draft_tokens,
        )
        if "k2" in self.verify_repeat_levels:
            metadata.k2_repeat = _plan_repeated_segments(
                metadata.k2.cu_seqlens,
                total_tokens=metadata.k2.cu_seqlens_cpu[-1],
                repeats=self.speculative_num_draft_tokens,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            raise NotImplementedError(
                "MiniCPM backend does not support draft extend mode"
            )

        self._use_cuda_graph_buffers = False
        self.flash_attn_backend.init_forward_metadata(forward_batch)
        metadata = MiniCPMSparseMetadata(base=self.flash_attn_backend.forward_metadata)
        if forward_batch.forward_mode.is_idle():
            self.forward_metadata = metadata
            return
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        self.update_batch_for_sparse(forward_batch, metadata)
        self.attention_adapter.prepare_forward(
            metadata,
            # Verify rows are single-query paged-decode rows,
            # so they plan on the decode wrapper; verify dispatches as an extend.
            is_prefill=not (
                forward_batch.forward_mode.is_decode_or_idle() or is_target_verify
            ),
            graph=False,
        )
        self.forward_metadata = metadata

    def _compress_decode_keys(
        self,
        query_states: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self.forward_metadata
        compressed = []
        for name, level in (("k1", metadata.k1), ("k2", metadata.k2)):
            total = level.cu_seqlens_cpu[-1]
            if self._use_cuda_graph_buffers:
                buffer = self.decode_cuda_graph_metadata[f"compress_{name}"][:total]
            else:
                buffer = torch.empty(
                    (total, layer.tp_k_head_num, layer.head_dim),
                    dtype=query_states.dtype,
                    device=self.device,
                )
            compressed.append(buffer)
        compressed_k, compressed_k2 = compressed

        get_compress_k_v2(
            layer=layer,
            forward_batch=forward_batch,
            metadata=metadata,
            full_compressed_k1=compressed_k,
            full_compressed_k2=compressed_k2,
            max_context_length=self.max_context_len,
            k1_kernel_size=self.k1_kernel_size,
            k1_kernel_stride=self.k1_kernel_stride,
            k2_kernel_size=self.k2_kernel_size,
            k2_kernel_stride=self.k2_kernel_stride,
        )
        return compressed_k, compressed_k2

    def get_topk_for_sparse(
        self,
        query_states,
        key_states,
        layer,
        forward_batch,
        is_prefill=True,
    ):
        """Select this layer's top-k blocks for every planned row
        (prefill: the sparse sub-batch), with layout [head_group, token, topk]."""
        if is_prefill:
            metadata = self.forward_metadata
            sparse_bs = metadata.sparse_bs_list
            full_compressed_k1, full_compressed_k2 = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
                k1_token_nums=metadata.k1.cu_seqlens_cpu[-1],
                k2_token_nums=metadata.k2.cu_seqlens_cpu[-1],
                k1_kernel_size=self.k1_kernel_size,
                k1_kernel_stride=self.k1_kernel_stride,
                k2_kernel_size=self.k2_kernel_size,
                k2_kernel_stride=self.k2_kernel_stride,
                dtype=key_states.dtype,
                device=key_states.device,
                max_context_length=self.max_context_len,
            )

            compressed = []
            if len(sparse_bs) == forward_batch.batch_size:
                compressed = [
                    (full_compressed_k1, metadata.k1.cu_seqlens),
                    (full_compressed_k2, metadata.k2.cu_seqlens),
                ]
            else:
                query_states = batched_gather(
                    query_states.reshape(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.extend_seq_lens_cpu,
                    sparse_bs,
                )
                for full_compressed_k, level in (
                    (full_compressed_k1, metadata.k1),
                    (full_compressed_k2, metadata.k2),
                ):
                    compressed.append(
                        _gather_compressed_keys(full_compressed_k, level, sparse_bs)
                    )

            (
                (compressed_k, compressed_cu_seqlens),
                (
                    compressed_k2,
                    compressed_cu_seqlens2,
                ),
            ) = compressed

            ret = self.sparse_get_topk_impl(
                query_states,
                metadata.topk_cu_seqlens_q,
                metadata.topk_cu_seqlens_k,
                metadata.topk_max_seqlen_q,
                metadata.topk_max_seqlen_k,
                compressed_k=compressed_k,
                compressed_cu_seqlens=compressed_cu_seqlens,
                compressed_k2=compressed_k2,
                compressed_cu_seqlens2=compressed_cu_seqlens2,
                fused_kernel=self._get_fused_topk_kernel(
                    len(sparse_bs),
                    is_prefill=True,
                ),
            )
            return ret
        else:
            metadata = self.forward_metadata
            compressed_k, compressed_k2 = self._compress_decode_keys(
                query_states,
                layer,
                forward_batch,
            )
            compressed_cu_seqlens = metadata.k1.cu_seqlens
            compressed_cu_seqlens2 = metadata.k2.cu_seqlens
            if metadata.k1_repeat is not None:
                compressed_k = compressed_k.index_select(0, metadata.k1_repeat.index)
                compressed_cu_seqlens = metadata.k1_repeat.cu_seqlens
                if metadata.k2_repeat is not None:
                    compressed_k2 = compressed_k2.index_select(
                        0, metadata.k2_repeat.index
                    )
                    compressed_cu_seqlens2 = metadata.k2_repeat.cu_seqlens

            ret = self.sparse_get_topk_impl(
                query_states,
                metadata.topk_cu_seqlens_q,
                metadata.topk_cu_seqlens_k,
                metadata.topk_max_seqlen_q,
                metadata.topk_max_seqlen_k,
                compressed_k=compressed_k,
                compressed_cu_seqlens=compressed_cu_seqlens,
                compressed_k2=compressed_k2,
                compressed_cu_seqlens2=compressed_cu_seqlens2,
                fused_kernel=self._get_fused_topk_kernel(
                    metadata.topk_cu_seqlens_q.shape[0] - 1,
                    is_prefill=False,
                ),
            )

        return ret

    def sparse_get_topk_impl(
        self,
        query_layer,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        compressed_k=None,
        compressed_cu_seqlens=None,
        compressed_k2=None,
        compressed_cu_seqlens2=None,
        fused_kernel=None,
    ):
        cache_lens = None
        if max_seqlen_in_batch_k > max_seqlen_in_batch_q:
            if max_seqlen_in_batch_q == 1:
                # Single-query rows: decode stages kv_len - 1 (seq_lens_k - seq_lens_q),
                # verify stages each draft row's causal prefix (token position - 1).
                cache_lens = self.forward_metadata.cache_seqlens_int32_stage1
            else:
                seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                cache_lens = seq_lens_k - seq_lens_q
        else:
            batch_size = cu_seqlens_q.shape[0] - 1
            cache_lens = torch.zeros(
                batch_size, dtype=torch.int32, device=cu_seqlens_q.device
            )

        if not self.minicpm_fuse_topk:
            topk_idx = compressed_attention(
                query_layer,
                compressed_k,
                compressed_k2,
                self.kernel_stride,
                self.block_size,
                self.sparse_topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                compressed_cu_seqlens2,
                max_seqlen_in_batch_q,
                self.max_context_len,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
                cu_seqlens_q_adjusted=self.forward_metadata.cu_seqlens_q_adjusted,
                max_seqlen_q_adjusted=self.forward_metadata.max_seqlen_q_adjusted,
            )
        else:
            topk_idx = compressed_attention_tilelang(
                query_layer,
                compressed_k,
                self.block_size,
                self.sparse_topk,
                self.kernel_topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                cache_lens=cache_lens,
                fused_kernel=fused_kernel,
                max_cache_len=self.max_context_len,
            )

        return topk_idx

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if layer.is_cross_attention:
            raise NotImplementedError(
                "MiniCPM backend does not support cross attention"
            )
        if layer.sliding_window_size not in (None, -1):
            raise NotImplementedError(
                "MiniCPM backend does not support sliding-window attention"
            )
        if forward_batch.forward_mode.is_draft_extend_v2():
            raise NotImplementedError(
                "MiniCPM backend does not support draft extend mode"
            )
        if forward_batch.forward_mode.is_target_verify():
            return self._forward_paged_rows(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        metadata = self.forward_metadata
        q, q_rope, k_rope, k_descale, v_descale = (
            self.flash_attn_backend.prepare_paged_mha_query(
                q,
                q_rope,
                k_rope,
                layer,
                logical_batch_size=forward_batch.batch_size,
                kv_head_num=layer.tp_k_head_num,
                is_prefill=True,
            )
        )
        page_table = metadata.base.page_table

        if metadata.sparse_bs_list:
            q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            topk_idx = self.get_topk_for_sparse(
                query_states=q_reshaped,
                key_states=k,
                layer=layer,
                forward_batch=forward_batch,
            )

            sparse_page_table_sparse_bs = get_block_table(
                topk_idx,
                page_table[metadata.sparse_bs_list],
                metadata.token_to_bs,
                metadata.token_pos_in_bs,
                metadata.seqlen_k_sparse_bs_tensor,
                head_group_num=self.head_group_num,
                block_size=self.block_size,
                elementwise=False,
            ).reshape(-1, self.num_sparse_topk_tokens)

            # copy page table for sparse bs
            metadata.sparse_page_table[
                metadata.sparse_idx, : self.num_sparse_topk_tokens
            ] = sparse_page_table_sparse_bs
        else:
            total_k1 = self.forward_metadata.k1.cu_seqlens_cpu[-1]
            total_k2 = self.forward_metadata.k2.cu_seqlens_cpu[-1]

            allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                k1_token_nums=total_k1,
                k2_token_nums=total_k2,
                k1_kernel_size=self.k1_kernel_size,
                k1_kernel_stride=self.k1_kernel_stride,
                k2_kernel_size=self.k2_kernel_size,
                k2_kernel_stride=self.k2_kernel_stride,
                dtype=k.dtype,
                device=k.device,
                max_context_length=self.max_context_len,
            )

        return self._forward_grouped_heads(
            q,
            layer,
            metadata,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=sinks,
        )

    def _forward_grouped_heads(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        metadata: MiniCPMSparseMetadata,
        *,
        k_descale: Optional[torch.Tensor],
        v_descale: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        dense_layout_spans = [
            (query_start, query_len)
            for _, _, query_start, query_len in metadata.dense_layout
        ]

        q_by_head_group = q.contiguous().view(-1, self.heads_per_group, layer.head_dim)
        _transpose_head_group_layout(
            q_by_head_group,
            dense_layout_spans,
            head_group_num=self.head_group_num,
            heads_per_group=self.heads_per_group,
            to_group_major=True,
        )

        key_cache, value_cache = self.flash_attn_backend.get_paged_mha_kv_cache(
            layer,
            head_group_num=self.head_group_num,
        )

        result = self.attention_adapter.forward(
            q_by_head_group,
            key_cache,
            value_cache,
            metadata,
            layer,
            is_prefill=True,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=sinks,
        )

        _transpose_head_group_layout(
            result,
            dense_layout_spans,
            head_group_num=self.head_group_num,
            heads_per_group=self.heads_per_group,
            to_group_major=False,
        )

        return result.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _forward_paged_rows(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        q_rope: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bs = forward_batch.batch_size
        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        metadata = self.forward_metadata
        q, q_rope, k_rope, k_descale, v_descale = (
            self.flash_attn_backend.prepare_paged_mha_query(
                q,
                q_rope,
                k_rope,
                layer,
                logical_batch_size=bs,
                kv_head_num=layer.tp_k_head_num,
                is_prefill=False,
            )
        )
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        if self._rows_need_block_selection(bs=bs, metadata=metadata):
            topk_idx = self.get_topk_for_sparse(
                query_states=q_reshaped,
                key_states=k,
                layer=layer,
                forward_batch=forward_batch,
                is_prefill=False,
            )
            self._plan_paged_sparse_page_table(topk_idx=topk_idx, metadata=metadata)
        else:
            # Dense rows still extend the persistent chunk table,
            # which the first sparse round past dense_len reads as history.
            self._compress_decode_keys(
                query_states=q_reshaped, layer=layer, forward_batch=forward_batch
            )

        key_cache, value_cache = self.flash_attn_backend.get_paged_mha_kv_cache(
            layer,
            head_group_num=self.head_group_num,
        )
        result = self.attention_adapter.forward(
            # One row per (token, head group).
            q_reshaped.reshape(-1, self.heads_per_group, layer.head_dim),
            key_cache,
            value_cache,
            metadata,
            layer,
            is_prefill=False,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=sinks,
        )

        return result.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _rows_need_block_selection(
        self, *, bs: int, metadata: MiniCPMSparseMetadata
    ) -> bool:
        # Selection output is fully masked out when every decode row is dense.
        # The graph binds leave dense_rows unset, so captured shapes stay static.
        return metadata.dense_rows is None or len(metadata.dense_rows) < bs

    def _plan_paged_sparse_page_table(
        self,
        *,
        topk_idx: torch.Tensor,
        metadata: MiniCPMSparseMetadata,
    ) -> None:
        topk_rows = get_block_table(
            topk_idx=topk_idx,
            block_table=metadata.base.page_table,
            token_to_bs=metadata.token_to_bs,
            token_pos_in_bs=metadata.token_pos_in_bs,
            seqlen_q=metadata.seqlen_k_sparse_bs_tensor,
            head_group_num=self.head_group_num,
            block_size=self.block_size,
            elementwise=True,
        ).reshape(-1, self.num_sparse_topk_tokens)
        destination = metadata.sparse_page_table[:, : self.num_sparse_topk_tokens]
        torch.where(metadata.sparse_row_mask, topk_rows, destination, out=destination)
        if metadata.verify_dense_rows is not None:
            self._overwrite_dense_verify_rows(metadata)

    def _overwrite_dense_verify_rows(self, metadata: MiniCPMSparseMetadata):
        # Rows read the prefix pages exactly as _copy_dense_page_table writes them.
        src = metadata.verify_dense_rows
        rows = metadata.sparse_page_table.view(src.shape[0], self.head_group_num, -1)
        rows[:, :, : src.shape[2]] = torch.where(
            metadata.verify_dense_mask, src, rows[:, :, : src.shape[2]]
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if layer.is_cross_attention:
            raise NotImplementedError(
                "MiniCPM backend does not support cross attention"
            )
        if layer.sliding_window_size not in (None, -1):
            raise NotImplementedError(
                "MiniCPM backend does not support sliding-window attention"
            )

        return self._forward_paged_rows(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            q_rope=q_rope,
            k_rope=k_rope,
            sinks=sinks,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.flash_attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        # max_num_tokens (= max_bs * draft width) already covers the verify rows.
        # The flashinfer adapter sizes its cuda_graph_custom_mask from this row
        # count; the sparse path never reads it, so the over-allocation is accepted.
        self.attention_adapter.init_cuda_graph_state(
            max_num_tokens * self.head_group_num,
        )
        self.decode_cuda_graph_metadata = (
            self.flash_attn_backend.decode_cuda_graph_metadata
        )
        self._init_sparse_row_graph_buffers(max_bs, max_num_tokens)
        self._init_verify_graph_buffers(max_bs, max_num_tokens)
        self._init_compression_graph_buffers(max_bs)

    def _init_sparse_row_graph_buffers(self, max_bs: int, max_num_tokens: int):
        buffers = self.decode_cuda_graph_metadata
        sparse_max_num_pages = (
            max(self.dense_len, self.num_sparse_topk_tokens) + self.page_size - 1
        ) // self.page_size
        max_rows = max_num_tokens * self.head_group_num
        buffers.update(
            {
                "sparse_cache_seqlens": torch.full(
                    (max_rows,),
                    self.num_sparse_topk_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "sparse_cu_seqlens_q": torch.arange(
                    0,
                    max_rows + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "sparse_cu_seqlens_k": torch.arange(
                    0,
                    (max_rows + 1) * self.num_sparse_topk_tokens,
                    self.num_sparse_topk_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "token_to_bs": torch.arange(
                    0, max_bs, dtype=torch.int32, device=self.device
                ),
                "sparse_page_table": torch.zeros(
                    max_rows,
                    sparse_max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "sparse_row_mask": torch.zeros(
                    max_rows, 1, dtype=torch.bool, device=self.device
                ),
                "cu_seqlens_q_adjusted": torch.arange(
                    0, max_num_tokens + 1, dtype=torch.int32, device=self.device
                )
                * self.heads_per_group,
                "cache_seqlens_int32_stage1": torch.zeros(
                    max_num_tokens, dtype=torch.int32, device=self.device
                ),
            }
        )

    def _init_verify_graph_buffers(self, max_bs: int, max_num_tokens: int):
        if self.speculative_num_draft_tokens is None:
            return
        buffers = self.decode_cuda_graph_metadata
        draft_width = self.speculative_num_draft_tokens
        assert max_bs * draft_width <= max_num_tokens, (
            f"verify tokens {max_bs * draft_width} exceed the graph runner's "
            f"token budget {max_num_tokens}"
        )
        buffers["verify_token_to_bs"] = torch.arange(
            0, max_bs, dtype=torch.int32, device=self.device
        ).repeat_interleave(draft_width)
        buffers["token_pos_in_bs"] = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        # Verify rows never share sparse_row_mask with decode replays, which
        # rewrite that buffer; every verify row takes the selection output.
        buffers["verify_row_mask"] = torch.ones(
            max_num_tokens * self.head_group_num,
            1,
            dtype=torch.bool,
            device=self.device,
        )
        if self.dense_len > 0:
            buffers["verify_dense_rows"] = torch.zeros(
                max_num_tokens,
                self.head_group_num,
                self.dense_len,
                dtype=torch.int32,
                device=self.device,
            )
            buffers["verify_dense_mask"] = torch.zeros(
                max_num_tokens,
                1,
                self.dense_len,
                dtype=torch.bool,
                device=self.device,
            )

    def _init_compression_graph_buffers(self, max_bs: int):
        buffers = self.decode_cuda_graph_metadata
        for name, kernel_size, kernel_stride in (
            ("k1", self.k1_kernel_size, self.k1_kernel_stride),
            ("k2", self.k2_kernel_size, self.k2_kernel_stride),
        ):
            max_num_pages = (
                max(
                    0,
                    (self.max_context_len - kernel_size) // kernel_stride + 1,
                )
                + self.page_size
                - 1
            ) // self.page_size
            buffers[f"compress_{name}"] = torch.zeros(
                (
                    max_bs * self.max_context_len // kernel_stride,
                    self.head_group_num,
                    self.head_dim,
                ),
                dtype=self.model_dtype,
                device=self.device,
            )
            buffers[f"{name}.table"] = torch.zeros(
                max_bs, max_num_pages, dtype=torch.int32, device=self.device
            )
            buffers[f"{name}.history_compress_token_nums"] = torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            )
            if self.speculative_num_draft_tokens is not None and (
                name in self.verify_repeat_levels
            ):
                # The captured gather records this index's address; it must
                # outlive the graphs, and every round rewrites its packed layout.
                max_verify_tokens = max_bs * self.speculative_num_draft_tokens
                buffers[f"{name}_repeat"] = RepeatedSegmentLayout(
                    index=torch.zeros(
                        max_verify_tokens * (self.max_context_len // kernel_stride),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    cu_seqlens=torch.zeros(
                        max_verify_tokens + 1, dtype=torch.int32, device=self.device
                    ),
                )
            for field in (
                "cu_seqlens",
                "cu_new_token_nums",
                "cu_total_compress_token_nums",
            ):
                buffers[f"{name}.{field}"] = torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        if not (forward_batch.forward_mode.is_decode_or_idle() or is_target_verify):
            raise NotImplementedError(
                "MiniCPM backend CUDA graph only supports decode/idle and "
                f"target verify modes, got {forward_batch.forward_mode}"
            )

        self._use_cuda_graph_buffers = True
        num_selection_rows = forward_batch.batch_size
        if is_target_verify:
            num_selection_rows *= self.speculative_num_draft_tokens
        self._get_fused_topk_kernel(
            num_selection_rows,
            is_prefill=False,
        )
        self.flash_attn_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture
        )
        metadata = MiniCPMSparseMetadata(base=self.flash_attn_backend.forward_metadata)
        if is_target_verify:
            self._bind_sparse_verify_graph_metadata(
                forward_batch,
                metadata,
                in_capture=in_capture,
            )
            if not in_capture:
                self._replay_sparse_verify_graph_metadata(forward_batch, metadata)
        else:
            self._bind_sparse_graph_metadata(
                forward_batch,
                metadata,
                in_capture=in_capture,
            )
            if not in_capture:
                self._replay_sparse_graph_metadata(forward_batch, metadata)
        self.attention_adapter.prepare_forward(
            metadata,
            is_prefill=False,
            graph=True,
        )
        self.forward_metadata = metadata

    def _build_sparse_decode_replay_metadata(
        self,
        metadata: MiniCPMSparseMetadata,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ):
        decode_metadata = _build_sparse_decode_metadata(
            seq_lens_cpu=seq_lens_cpu,
            base_metadata=metadata.base,
            head_group_num=self.head_group_num,
            dense_len=self.dense_len,
            sparse_topk=self.sparse_topk,
            block_size=self.block_size,
        )
        compression_metadata = _build_k1_k2_compression_metadata(
            req_pool_indices=req_pool_indices,
            base_metadata=metadata.base,
            req_to_sparse_k1_token=self.req_to_sparse_k1_token,
            req_to_sparse_k2_token=self.req_to_sparse_k2_token,
            k1_kernel_size=self.k1_kernel_size,
            k1_kernel_stride=self.k1_kernel_stride,
            k2_kernel_size=self.k2_kernel_size,
            k2_kernel_stride=self.k2_kernel_stride,
            cu_seqlens_q=metadata.base.cu_seqlens_q,
            seq_lens_cpu=seq_lens_cpu,
        )
        return decode_metadata, compression_metadata

    def _bind_sparse_graph_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: MiniCPMSparseMetadata,
        *,
        in_capture: bool,
    ):
        bs = forward_batch.batch_size
        buffers = self.decode_cuda_graph_metadata
        sparse_rows = self.head_group_num * bs

        assume_kv_len = self.config_dense_len
        if in_capture:
            metadata.base.cu_seqlens_k.copy_(
                torch.arange(bs + 1, device=self.device, dtype=torch.int32)
                * assume_kv_len
            )
            metadata.base.max_seq_len_k = assume_kv_len

        self._bind_compression_graph_metadata(
            bs=bs, metadata=metadata, in_capture=in_capture, assume_kv_len=assume_kv_len
        )

        metadata.sparse_row_mask = buffers["sparse_row_mask"][:sparse_rows]
        _assign_row_metadata(
            metadata,
            sparse_cache_seqlens_int32=buffers["sparse_cache_seqlens"][:sparse_rows],
            sparse_cu_seqlens_q=buffers["sparse_cu_seqlens_q"][: sparse_rows + 1],
            sparse_cu_seqlens_k=buffers["sparse_cu_seqlens_k"][: sparse_rows + 1],
            sparse_page_table=buffers["sparse_page_table"][:sparse_rows],
            token_to_bs=buffers["token_to_bs"][:bs],
            token_pos_in_bs=metadata.base.cache_seqlens_int32,
            cache_seqlens_int32_stage1=buffers["cache_seqlens_int32_stage1"][:bs],
            cu_seqlens_q_adjusted=buffers["cu_seqlens_q_adjusted"][: bs + 1],
            max_seqlen_q_adjusted=metadata.base.max_seq_len_q * self.heads_per_group,
            topk_cu_seqlens_q=metadata.base.cu_seqlens_q,
            topk_cu_seqlens_k=metadata.base.cu_seqlens_k,
            topk_max_seqlen_q=1,
            topk_max_seqlen_k=metadata.base.max_seq_len_k,
        )

    def _bind_compression_graph_metadata(
        self,
        bs: int,
        metadata: MiniCPMSparseMetadata,
        *,
        in_capture: bool,
        assume_kv_len: int,
    ):
        metadata.k1 = self._bind_compression_level(
            "k1",
            kernel_size=self.k1_kernel_size,
            kernel_stride=self.k1_kernel_stride,
            bs=bs,
            in_capture=in_capture,
            assume_kv_len=assume_kv_len,
        )
        metadata.k2 = self._bind_compression_level(
            "k2",
            kernel_size=self.k2_kernel_size,
            kernel_stride=self.k2_kernel_stride,
            bs=bs,
            in_capture=in_capture,
            assume_kv_len=assume_kv_len,
        )

    def _bind_compression_level(
        self,
        name: str,
        *,
        kernel_size: int,
        kernel_stride: int,
        bs: int,
        in_capture: bool,
        assume_kv_len: int,
    ) -> CompressionLevelMetadata:
        buffers = self.decode_cuda_graph_metadata
        level = CompressionLevelMetadata()
        level_len = max(0, (assume_kv_len - kernel_size) // kernel_stride + 1)
        level.cu_seqlens_cpu = [index * level_len for index in range(bs + 1)]
        level.cu_seqlens = buffers[f"{name}.cu_seqlens"][: bs + 1]
        if in_capture:
            level.cu_seqlens.copy_(
                torch.arange(bs + 1, device=self.device, dtype=torch.int32) * level_len
            )
        level.table = buffers[f"{name}.table"][:bs]
        level.history_compress_token_nums = buffers[
            f"{name}.history_compress_token_nums"
        ][:bs]
        level.cu_new_token_nums = buffers[f"{name}.cu_new_token_nums"][: bs + 1]
        level.cu_total_compress_token_nums = buffers[
            f"{name}.cu_total_compress_token_nums"
        ][: bs + 1]
        return level

    def _replay_sparse_graph_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: MiniCPMSparseMetadata,
    ):
        bs = forward_batch.batch_size
        real_bs = bs - forward_batch.num_padding
        if real_bs == 0:
            # All-padding replay: stale row tables feed only discarded padding rows,
            # and the zeroed lengths keep the captured gathers in bounds.
            metadata.sparse_cache_seqlens_int32.zero_()
            metadata.sparse_cu_seqlens_k.zero_()
            metadata.cache_seqlens_int32_stage1.zero_()
            metadata.sparse_row_mask.zero_()
            for level in (metadata.k1, metadata.k2):
                level.history_compress_token_nums.zero_()
                level.cu_seqlens.zero_()
                level.cu_new_token_nums.zero_()
                level.cu_total_compress_token_nums.zero_()
            return

        decode_metadata, compression_metadata = (
            self._build_sparse_decode_replay_metadata(
                metadata,
                req_pool_indices=forward_batch.req_pool_indices[:real_bs],
                seq_lens_cpu=forward_batch.seq_lens_cpu[:real_bs],
            )
        )
        real_sparse_rows = self.head_group_num * real_bs
        metadata.sparse_cache_seqlens_int32[:real_sparse_rows].copy_(
            decode_metadata.sparse_cache_seqlens_int32
        )
        metadata.sparse_cu_seqlens_k[: real_sparse_rows + 1].copy_(
            decode_metadata.sparse_cu_seqlens_k
        )
        metadata.cache_seqlens_int32_stage1[:real_bs].copy_(
            metadata.base.cache_seqlens_int32[:real_bs] - 1
        )
        metadata.sparse_row_mask[:real_sparse_rows].copy_(
            decode_metadata.sparse_row_mask
        )
        _copy_dense_page_tables(
            metadata.sparse_page_table,
            metadata.base.page_table,
            dense_rows=decode_metadata.dense_rows,
            head_group_num=self.head_group_num,
        )

        self._replay_compression_graph_metadata(
            forward_batch=forward_batch,
            metadata=metadata,
            compression_metadata=compression_metadata,
            real_bs=real_bs,
            cu_fields=(
                "cu_seqlens",
                "cu_new_token_nums",
                "cu_total_compress_token_nums",
            ),
        )

        if real_bs < bs:
            metadata.sparse_cache_seqlens_int32[real_sparse_rows:].zero_()
            metadata.sparse_cu_seqlens_k[real_sparse_rows + 1 :].fill_(
                decode_metadata.sparse_cu_seqlens_k[-1]
            )
            metadata.cache_seqlens_int32_stage1[real_bs:].zero_()
            metadata.sparse_row_mask[real_sparse_rows:].zero_()

    def _replay_compression_graph_metadata(
        self,
        *,
        forward_batch: ForwardBatch,
        metadata: MiniCPMSparseMetadata,
        compression_metadata,
        real_bs: int,
        cu_fields: tuple[str, ...],
    ):
        bs = forward_batch.batch_size
        for (name, req_to_sparse), src in zip(
            (
                ("k1", self.req_to_sparse_k1_token),
                ("k2", self.req_to_sparse_k2_token),
            ),
            compression_metadata,
        ):
            dst = getattr(metadata, name)
            dst.history_compress_token_nums[:real_bs].copy_(
                src.history_compress_token_nums
            )
            if real_bs < bs:
                dst.history_compress_token_nums[real_bs:].zero_()
            for field in cu_fields:
                dst_field = getattr(dst, field)
                src_field = getattr(src, field)
                dst_field[: real_bs + 1].copy_(src_field)
                if real_bs < bs:
                    # Padded rows get zero-length tails
                    # so the captured gathers stay in bounds.
                    dst_field[real_bs + 1 :].fill_(src_field[-1])
            dst.table.copy_(req_to_sparse[forward_batch.req_pool_indices])

    def _bind_sparse_verify_graph_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: MiniCPMSparseMetadata,
        *,
        in_capture: bool,
    ):
        bs = forward_batch.batch_size
        num_draft_tokens = self.speculative_num_draft_tokens
        num_verify_tokens = bs * num_draft_tokens
        sparse_rows = num_verify_tokens * self.head_group_num
        buffers = self.decode_cuda_graph_metadata
        metadata.sparse_row_mask = buffers["verify_row_mask"][:sparse_rows]
        if self.dense_len > 0:
            num_cols = min(self.dense_len, metadata.base.page_table.shape[1])
            metadata.verify_dense_rows = buffers["verify_dense_rows"][
                :num_verify_tokens, :, :num_cols
            ]
            metadata.verify_dense_mask = buffers["verify_dense_mask"][
                :num_verify_tokens, :, :num_cols
            ]
        _assign_row_metadata(
            metadata,
            sparse_cache_seqlens_int32=buffers["sparse_cache_seqlens"][:sparse_rows],
            sparse_cu_seqlens_q=buffers["sparse_cu_seqlens_q"][: sparse_rows + 1],
            sparse_cu_seqlens_k=buffers["sparse_cu_seqlens_k"][: sparse_rows + 1],
            sparse_page_table=buffers["sparse_page_table"][:sparse_rows],
            token_to_bs=buffers["verify_token_to_bs"][:num_verify_tokens],
            token_pos_in_bs=buffers["token_pos_in_bs"][:num_verify_tokens],
            # These buffers are prefix-stable (arange / zeros),
            # so the decode ones serve verify's row count too.
            cache_seqlens_int32_stage1=buffers["cache_seqlens_int32_stage1"][
                :num_verify_tokens
            ],
            cu_seqlens_q_adjusted=buffers["cu_seqlens_q_adjusted"][
                : num_verify_tokens + 1
            ],
            max_seqlen_q_adjusted=self.heads_per_group,
            topk_cu_seqlens_q=buffers["sparse_cu_seqlens_q"][: num_verify_tokens + 1],
            topk_cu_seqlens_k=metadata.base.cu_seqlens_k,
            topk_max_seqlen_q=1,
            # Static across capture and replay: the captured selection froze its
            # cache_lens branch here, and replays do not refresh base.max_seq_len_k.
            topk_max_seqlen_k=self.max_context_len,
        )

        assume_kv_len = min(
            self.config_dense_len + num_draft_tokens, self.max_context_len
        )
        if in_capture:
            metadata.base.cu_seqlens_k.copy_(
                torch.arange(bs + 1, device=self.device, dtype=torch.int32)
                * assume_kv_len
            )
            metadata.base.cache_seqlens_int32.fill_(assume_kv_len)
            seed_seq_lens = torch.full(
                (bs,),
                assume_kv_len - num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            self._write_verify_graph_rows(
                metadata, seed_seq_lens, real_rows=sparse_rows
            )
        self._bind_compression_graph_metadata(
            bs=bs, metadata=metadata, in_capture=in_capture, assume_kv_len=assume_kv_len
        )
        if in_capture:
            self._bind_verify_repeat_layouts(
                metadata, bs=bs, num_verify_tokens=num_verify_tokens
            )

    def _bind_verify_repeat_layouts(
        self,
        metadata: MiniCPMSparseMetadata,
        *,
        bs: int,
        num_verify_tokens: int,
    ):
        metadata.k1_repeat = self._bind_verify_repeat_level(
            metadata.k1,
            name="k1",
            kernel_stride=self.k1_kernel_stride,
            bs=bs,
            num_verify_tokens=num_verify_tokens,
        )
        metadata.k2_repeat = self._bind_verify_repeat_level(
            metadata.k2,
            name="k2",
            kernel_stride=self.k2_kernel_stride,
            bs=bs,
            num_verify_tokens=num_verify_tokens,
        )

    def _bind_verify_repeat_level(
        self,
        level_metadata: CompressionLevelMetadata,
        *,
        name: str,
        kernel_stride: int,
        bs: int,
        num_verify_tokens: int,
    ) -> Optional[RepeatedSegmentLayout]:
        segment_rows = self.max_context_len // kernel_stride
        packed_total = level_metadata.cu_seqlens_cpu[-1]
        level_metadata.cu_total_compress_token_nums.copy_(
            torch.arange(bs + 1, device=self.device, dtype=torch.int32) * segment_rows
        )
        # _compress_decode_keys sizes the compression view from cu_seqlens_cpu,
        # so seed it to the slab extent the compression kernel writes into.
        level_metadata.cu_seqlens_cpu = [row * segment_rows for row in range(bs + 1)]
        # Mirror the eager planner: under minicpm_fuse_topk the k2 repeat is unset,
        # and the captured gather records the plain level cu_seqlens.
        if name not in self.verify_repeat_levels:
            return None
        repeat = self._verify_repeat_layout_views(
            name, bs=bs, num_verify_tokens=num_verify_tokens, segment_rows=segment_rows
        )
        self._write_verify_repeat_layout(
            level_metadata, repeat, packed_total=packed_total
        )
        return repeat

    def _verify_repeat_layout_views(
        self, name: str, *, bs: int, num_verify_tokens: int, segment_rows: int
    ) -> RepeatedSegmentLayout:
        # The captured gather recorded the retained buffers' addresses,
        # so capture and replay both work through prefix views of them.
        retained = self.decode_cuda_graph_metadata[f"{name}_repeat"]
        return RepeatedSegmentLayout(
            index=retained.index[: num_verify_tokens * segment_rows],
            cu_seqlens=retained.cu_seqlens[: num_verify_tokens + 1],
        )

    def _write_verify_repeat_layout(
        self,
        level_metadata: CompressionLevelMetadata,
        repeat: RepeatedSegmentLayout,
        *,
        packed_total: int,
    ) -> None:
        # The compression kernel writes request b's chunks at slab offset
        # cu_total[b], so the packed rows gather from there.
        layout = _plan_repeated_segments(
            level_metadata.cu_seqlens,
            total_tokens=packed_total,
            repeats=self.speculative_num_draft_tokens,
            segment_starts=level_metadata.cu_total_compress_token_nums[:-1],
        )
        packed = layout.index.numel()
        repeat.index[:packed].copy_(layout.index)
        # The captured gather reads the whole view, so the unreferenced tail
        # must stay inside the smallest compression view: point it at row 0.
        repeat.index[packed:].zero_()
        repeat.cu_seqlens.copy_(layout.cu_seqlens)

    def _write_verify_graph_rows(
        self,
        metadata: MiniCPMSparseMetadata,
        seq_lens: torch.Tensor,
        *,
        real_rows: int,
    ):
        """Refresh the static verify row buffers; the row-length outputs past
        real_rows (CUDA-graph padding) are zeroed."""
        num_draft_tokens = self.speculative_num_draft_tokens
        token_pos, row_lens = _build_sparse_verify_rows(
            seq_lens,
            head_group_num=self.head_group_num,
            dense_len=self.dense_len,
            sparse_topk=self.sparse_topk,
            block_size=self.block_size,
            num_draft_tokens=num_draft_tokens,
        )
        row_lens[real_rows:] = 0
        metadata.token_pos_in_bs.copy_(token_pos)
        metadata.sparse_cache_seqlens_int32.copy_(row_lens)
        torch.cumsum(
            row_lens, dim=0, dtype=torch.int32, out=metadata.sparse_cu_seqlens_k[1:]
        )
        # Per-row decode-form cache length for the block selection.
        metadata.cache_seqlens_int32_stage1.copy_(token_pos - 1)
        if self.dense_len > 0:
            _build_dense_verify_overwrite(
                token_pos=token_pos,
                token_to_bs=metadata.token_to_bs,
                page_table=metadata.base.page_table,
                dense_len=self.dense_len,
                head_group_num=self.head_group_num,
                out_rows=metadata.verify_dense_rows,
                out_mask=metadata.verify_dense_mask,
            )

    def _replay_sparse_verify_graph_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: MiniCPMSparseMetadata,
    ):
        bs = forward_batch.batch_size
        num_draft_tokens = self.speculative_num_draft_tokens
        # num_padding comes only from the graph runner's synthetic replay batch;
        # it is not a ForwardBatch contract member.
        real_bs = bs - forward_batch.num_padding
        if real_bs == 0:
            # All-padding replay: zero this round's lengths so the captured
            # compression and gathers see empty segments, not the previous round's.
            metadata.sparse_cache_seqlens_int32.zero_()
            metadata.sparse_cu_seqlens_k.zero_()
            metadata.cache_seqlens_int32_stage1.zero_()
            for level in (metadata.k1, metadata.k2):
                level.history_compress_token_nums.zero_()
                level.cu_seqlens.zero_()
                level.cu_new_token_nums.zero_()
            for name in self.verify_repeat_levels:
                self.decode_cuda_graph_metadata[f"{name}_repeat"].cu_seqlens[
                    : bs * num_draft_tokens + 1
                ].zero_()
            return

        # Padding slots keep the runner's seq-len fill value,
        # so their token positions stay in range.
        self._write_verify_graph_rows(
            metadata,
            forward_batch.seq_lens,
            real_rows=real_bs * num_draft_tokens * self.head_group_num,
        )

        compression_metadata = _build_k1_k2_compression_metadata(
            req_pool_indices=forward_batch.req_pool_indices[:real_bs],
            base_metadata=metadata.base,
            req_to_sparse_k1_token=self.req_to_sparse_k1_token,
            req_to_sparse_k2_token=self.req_to_sparse_k2_token,
            k1_kernel_size=self.k1_kernel_size,
            k1_kernel_stride=self.k1_kernel_stride,
            k2_kernel_size=self.k2_kernel_size,
            k2_kernel_stride=self.k2_kernel_stride,
            cu_seqlens_q=metadata.base.cu_seqlens_q,
            seq_lens_cpu=forward_batch.seq_lens_cpu[:real_bs] + num_draft_tokens,
        )
        # cu_total_compress_token_nums keeps the capture-time slab offsets the
        # compression kernel writes at; only the packed lengths are refreshed.
        self._replay_compression_graph_metadata(
            forward_batch=forward_batch,
            metadata=metadata,
            compression_metadata=compression_metadata,
            real_bs=real_bs,
            cu_fields=("cu_seqlens", "cu_new_token_nums"),
        )
        for name, kernel_stride, level_metadata, src in (
            ("k1", self.k1_kernel_stride, metadata.k1, compression_metadata[0]),
            ("k2", self.k2_kernel_stride, metadata.k2, compression_metadata[1]),
        ):
            if name in self.verify_repeat_levels:
                self._write_verify_repeat_layout(
                    level_metadata,
                    self._verify_repeat_layout_views(
                        name,
                        bs=bs,
                        num_verify_tokens=bs * num_draft_tokens,
                        segment_rows=self.max_context_len // kernel_stride,
                    ),
                    packed_total=src.cu_seqlens_cpu[-1],
                )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.flash_attn_backend.get_cuda_graph_seq_len_fill_value()
