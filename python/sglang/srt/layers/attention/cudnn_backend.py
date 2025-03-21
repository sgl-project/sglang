from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import cudnn
import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


# Helper functions
def convert_to_cudnn_type(torch_type):
    """Convert PyTorch data type to CuDNN data type."""
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


# Convert sglang's kv cache to the container format required by cudnn
def create_cudnn_container(cache: torch.Tensor, page_size: int):
    # [max_token_number, head_num, head_dim]
    # [page_number, (page_size, head_num, head_dim)]

    page_num = cache.size(0) // page_size
    cache = cache[: page_num * page_size]
    cache = cache.view(page_num, page_size, cache.size(1), cache.size(2))
    return cache.permute(0, 2, 1, 3)


# Convert sglang's KV indices to the page_table format required by CuDNN
# TODO: Accelerate this function
def create_cudnn_pagetable(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    max_page_per_req: int,
    page_size: int,
    device: torch.device,
):

    page_tables = torch.empty_strided(
        (seq_lens.size(0), 1, max_page_per_req, 1),
        (max_page_per_req, 1, 1, 1),
        device=device,
        dtype=torch.int32,
    )

    for seq_idx in range(seq_lens.shape[0]):
        idx = req_pool_indices[seq_idx]
        seq_len = seq_lens[seq_idx].item()

        index = req_to_token[idx, :seq_len]
        page_starts = torch.arange(0, seq_len, page_size, device=index.device)
        page_table = index[page_starts] // page_size

        num_pages = page_table.size(0)
        page_tables[seq_idx, 0, :num_pages, 0] = page_table

    return page_tables


# Since the actual batch_size in sglang may exceed the predefined batch_size during graph building,
# we need to process the query in chunks.
# Thanks to the ragged layout, we can maintain the query shape (S, Hq, Dqk)
def chunk_attention_data(
    batch_size, query, page_table_k, page_table_v, seq_lens, extend_seq_lens
):
    """
    Efficiently batch attention inputs for processing in smaller chunks.

    Args:
        batch_size: Maximum number of sequences to process in each batch
        query: Query tensor to be batched
        page_table_k: Key page table tensor to be batched
        page_table_v: Value page table tensor to be batched
        seq_lens: Sequence lengths tensor to be batched
        extend_seq_lens: Extended sequence lengths tensor to be batched

    Yields:
        Tuples of batched tensors (query, key_page_table, value_page_table, seq_lengths, extended_seq_lengths)
    """
    total_seqs = seq_lens.shape[0]
    query_start_idx = torch.zeros(1, device=query.device, dtype=torch.int32).item()
    for start_idx in range(0, total_seqs, batch_size):
        end_idx = min(start_idx + batch_size, total_seqs)
        query_end_idx = query_start_idx + extend_seq_lens[start_idx:end_idx].sum()
        if end_idx - start_idx == batch_size:
            yield (
                query[query_start_idx:query_end_idx],
                page_table_k[start_idx:end_idx],
                page_table_v[start_idx:end_idx],
                seq_lens[start_idx:end_idx],
                extend_seq_lens[start_idx:end_idx],
                query_start_idx,
                query_end_idx,
            )
        else:
            seq = torch.full(
                (batch_size, 1, 1, 1), 0, device=query.device, dtype=torch.int32
            )
            extend_seq = torch.full(
                (batch_size, 1, 1, 1), 0, device=query.device, dtype=torch.int32
            )
            seq[: end_idx - start_idx] = seq_lens[start_idx:end_idx]
            extend_seq[: end_idx - start_idx] = extend_seq_lens[start_idx:end_idx]

            page_table_k_item = torch.empty_strided(
                (batch_size, 1, page_table_k.shape[2], 1),
                page_table_k.stride(),
                device=query.device,
                dtype=torch.int32,
            )
            page_table_v_item = torch.empty_strided(
                (batch_size, 1, page_table_k.shape[2], 1),
                page_table_k.stride(),
                device=query.device,
                dtype=torch.int32,
            )
            page_table_k_item[: end_idx - start_idx] = page_table_k[start_idx:end_idx]
            page_table_v_item[: end_idx - start_idx] = page_table_v[start_idx:end_idx]

            yield (
                query[query_start_idx:query_end_idx],
                page_table_k_item,
                page_table_v_item,
                seq,
                extend_seq,
                query_start_idx,
                query_end_idx,
            )
        query_start_idx = query_end_idx


# Compute the exclusive prefix sum of a tensor
def compute_exclusive_prefix_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute exclusive prefix sum of a tensor.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with exclusive prefix sum
    """
    assert tensor.size(1) == tensor.size(2) == tensor.size(3) == 1
    return torch.cat(
        (
            torch.zeros(1, 1, 1, 1, dtype=torch.int64, device=tensor.device),
            torch.cumsum(tensor, dim=0),
        )
    )


# TODO: Support non-ragged Query
# TODO: Support cuda graph
class CuDNNAttnBackend(AttentionBackend):

    class UIDs(Enum):
        """Unique identifiers for tensors in CuDNN graph."""

        Q_UID = 0
        K_UID = 1
        V_UID = 2
        O_UID = 3
        STATS_UID = 4
        PAGE_TABLE_K_UID = 5
        PAGE_TABLE_V_UID = 6
        SEQ_LEN_Q_UID = 7
        SEQ_LEN_KV_UID = 8
        ATTN_SCALE_UID = 9
        Q_RAGGED_OFFSET_UID = 10
        O_RAGGED_OFFSET_UID = 11

    def __init__(self, model_runner: ModelRunner):
        """
        Initialize the CuDNN attention backend.

        Args:
            model_runner: The model runner instance
        """
        super().__init__()
        self.forward_metadata = None
        self.dtype = model_runner.model_config.dtype
        self.device = model_runner.device

        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.max_seq_q = (
            model_runner.server_args.chunked_prefill_size
        )  # TODO: TO check if this is correct
        self.batch_size = 32
        self.page_size = model_runner.server_args.page_size

        # Parameters for building the CuDNN graph
        self.max_context_len = min(
            model_runner.model_config.context_len, model_runner.max_total_num_tokens
        )
        self.max_num_pages = (
            model_runner.max_total_num_tokens + self.page_size - 1
        ) // self.page_size
        self.max_page_per_req = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size

        # Initialize CuDNN handle
        self.cudnn_handle = cudnn.create_handle()
        stream = torch.cuda.current_stream().cuda_stream
        cudnn.set_stream(handle=self.cudnn_handle, stream=stream)
        self.cudnn_dtype = convert_to_cudnn_type(model_runner.model_config.dtype)
        self.cudnn_version = cudnn.backend_version_string()

        # Create CuDNN graphs for extend and decode operations
        self.extend_graph = self._create_cudnn_graph(
            self.batch_size,
            self.max_seq_q,
            self.max_num_pages,
            self.max_page_per_req,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            self.cudnn_handle,
            self.cudnn_version,
            True,  # TODO: This is a design flaw, the is_causal attribute should be managed by the layer itself. This is a simplified approach that needs to be fixed.
        )
        self.extend_workspace = torch.empty(
            self.extend_graph.get_workspace_size(),
            device=self.device,
            dtype=torch.uint8,
        )

        self.decode_graph = self._create_cudnn_graph(
            self.batch_size,
            2,  # Should be 1, but causes an unidentified bug. Setting to 2 works correctly
            self.max_num_pages,
            self.max_page_per_req,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            self.cudnn_handle,
            self.cudnn_version,
            True,
        )
        self.decode_workspace = torch.empty(
            self.decode_graph.get_workspace_size(),
            device=self.device,
            dtype=torch.uint8,
        )

        self.container_k_gpu = None
        self.container_v_gpu = None
        self.page_table_k_gpu = None
        self.page_table_v_gpu = None
        self.seq_len_q_gpu = None
        self.seq_len_kv_gpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        self.page_table_k_gpu = create_cudnn_pagetable(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.req_to_token_pool.req_to_token,
            self.max_page_per_req,
            self.page_size,
            self.device,
        )

        self.page_table_v_gpu = create_cudnn_pagetable(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.req_to_token_pool.req_to_token,
            self.max_page_per_req,
            self.page_size,
            self.device,
        )

        self.seq_len_kv_gpu = forward_batch.seq_lens.reshape(-1, 1, 1, 1).to(
            dtype=torch.int32
        )  # Required by CuDNN documentation
        if forward_batch.forward_mode.is_decode_or_idle():
            self.seq_len_q_gpu = torch.ones(
                (forward_batch.seq_lens.size(0), 1, 1, 1),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.seq_len_q_gpu = forward_batch.extend_seq_lens.reshape(-1, 1, 1, 1).to(
                dtype=torch.int32
            )  # Required by CuDNN documentation

    def _create_cudnn_graph(
        self,
        bs: int,
        Sq: int,
        max_num_pages: int,
        max_num_pages_per_req: int,
        qo_heads: int,
        kv_heads: int,
        head_dim: int,
        page_size: int,
        handle: cudnn.handle,
        cudnn_version: str,
        is_causal: bool,
    ):
        """Create a CuDNN graph for the SDPA operation.

        Args:
            bs: Batch size
            Sq: Max Query Length
            max_num_pages: Max Key-Value Length
            max_num_pages_per_req: Max Pages per Request
            qo_heads: Number of Query-Output Heads
            kv_heads: Number of Key-Value Heads
            head_dim: Head Dimension
            page_size: Page Size
            handle: CuDNN Handle
            cudnn_version: CuDNN Version
            is_causal: Whether to use causal attention

        Returns:
            output: cudnn.pygraph
        """

        graph = cudnn.pygraph(
            io_data_type=self.cudnn_dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=handle,
            version=cudnn_version,
        )

        Hqo, Hkv = qo_heads, kv_heads
        D = head_dim

        qo_shape = (bs, Hqo, Sq, D)
        qo_stride = (Sq * Hqo * D, D, Hqo * D, 1)

        kv_shape = (max_num_pages + 1, Hkv, page_size, D)
        kv_stride = (page_size * Hkv * D, D, Hkv * D, 1)

        page_table_shape = (bs, 1, max_num_pages_per_req, 1)
        page_table_stride = (max_num_pages_per_req, 1, 1, 1)

        attn_scale_shape = (1, 1, 1, 1)
        seq_len_shape = (bs, 1, 1, 1)
        qo_ragged_offset_shape = (bs + 1, 1, 1, 1)

        q = graph.tensor(dim=qo_shape, stride=qo_stride, data_type=self.cudnn_dtype)
        container_k = graph.tensor(
            dim=kv_shape, stride=kv_stride, data_type=self.cudnn_dtype
        )
        container_v = graph.tensor(
            dim=kv_shape, stride=kv_stride, data_type=self.cudnn_dtype
        )
        page_table_k = graph.tensor(
            dim=page_table_shape,
            stride=page_table_stride,
            data_type=cudnn.data_type.INT32,
        )
        page_table_v = graph.tensor(
            dim=page_table_shape,
            stride=page_table_stride,
            data_type=cudnn.data_type.INT32,
        )

        q_ragged_offset = graph.tensor(
            dim=qo_ragged_offset_shape,
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        o_ragged_offset = graph.tensor(
            dim=qo_ragged_offset_shape,
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        attn_scale = graph.tensor(
            dim=attn_scale_shape, stride=(1, 1, 1, 1), data_type=self.cudnn_dtype
        )
        seq_len_q = graph.tensor(
            dim=seq_len_shape, stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32
        )
        seq_len_kv = graph.tensor(
            dim=seq_len_shape, stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32
        )
        q.set_ragged_offset(q_ragged_offset)

        o, _ = graph.sdpa(
            name="sdpa_decode" if Sq == 1 else "sdpa_extend",
            q=q,
            k=container_k,
            v=container_v,
            is_inference=True,
            use_padding_mask=True,
            attn_scale=attn_scale,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            diagonal_band_right_bound=0 if is_causal else None,
            diagonal_alignment=(
                cudnn.diagonal_alignment.BOTTOM_RIGHT
                if is_causal
                else cudnn.diagonal_alignment.TOP_LEFT
            ),
            paged_attention_k_table=page_table_k,
            paged_attention_v_table=page_table_v,
        )

        o.set_output(True).set_dim(qo_shape).set_stride(qo_stride)
        o.set_ragged_offset(o_ragged_offset)

        q.set_uid(self.UIDs.Q_UID.value)
        container_k.set_uid(self.UIDs.K_UID.value)
        container_v.set_uid(self.UIDs.V_UID.value)
        o.set_uid(self.UIDs.O_UID.value)
        page_table_k.set_uid(self.UIDs.PAGE_TABLE_K_UID.value)
        page_table_v.set_uid(self.UIDs.PAGE_TABLE_V_UID.value)
        attn_scale.set_uid(self.UIDs.ATTN_SCALE_UID.value)
        seq_len_q.set_uid(self.UIDs.SEQ_LEN_Q_UID.value)
        seq_len_kv.set_uid(self.UIDs.SEQ_LEN_KV_UID.value)
        q_ragged_offset.set_uid(self.UIDs.Q_RAGGED_OFFSET_UID.value)
        o_ragged_offset.set_uid(self.UIDs.O_RAGGED_OFFSET_UID.value)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        return graph

    def _run_cudnn_sdpa_forward(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table_k_gpu: torch.Tensor,
        page_table_v_gpu: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling: torch.Tensor,
        causal: bool,
        graph: cudnn.pygraph,
        workspace: torch.Tensor,
    ):
        """
        Run the CuDNN SDPA forward pass.

        Args:
            query: Query tensor
            output: Output tensor (pre-allocated)
            k_cache: Key cache
            v_cache: Value cache
            page_table_k_gpu: Key page table
            page_table_v_gpu: Value page table
            seq_lens: Sequence lengths
            extend_seq_lens: Extended sequence lengths
            scaling: Attention scaling factor
            graph: CuDNN graph to execute
            workspace: Workspace tensor for CuDNN
            causal: Whether to use causal attention (Currently not actually used)
        Returns:
            Output tensor after attention
        """

        S, Hqo, Dqo = query.shape

        qo_ragged_offset = compute_exclusive_prefix_sum(extend_seq_lens) * Hqo * Dqo

        query = query.reshape(
            -1, query.size(0), query.size(1), query.size(2)
        ).contiguous()
        output = output.reshape(
            -1, output.size(0), output.size(1), output.size(2)
        ).contiguous()
        # To ensure stride requirements are met

        query = query.movedim(1, query.dim() - 2)
        output = output.movedim(1, output.dim() - 2)

        variant_pack = {
            self.UIDs.Q_UID.value: query,
            self.UIDs.K_UID.value: k_cache,
            self.UIDs.V_UID.value: v_cache,
            self.UIDs.O_UID.value: output,
            self.UIDs.PAGE_TABLE_K_UID.value: page_table_k_gpu,
            self.UIDs.PAGE_TABLE_V_UID.value: page_table_v_gpu,
            self.UIDs.SEQ_LEN_Q_UID.value: extend_seq_lens,
            self.UIDs.SEQ_LEN_KV_UID.value: seq_lens,
            self.UIDs.ATTN_SCALE_UID.value: scaling,
            self.UIDs.Q_RAGGED_OFFSET_UID.value: qo_ragged_offset,
            self.UIDs.O_RAGGED_OFFSET_UID.value: qo_ragged_offset,
        }

        graph.execute(variant_pack, workspace, handle=self.cudnn_handle)
        torch.cuda.synchronize()
        # TODO: We should avoid synchronization to improve performance.

        output = output.squeeze(0)
        output = output.movedim(output.dim() - 2, 0)
        return output

    def _common_forward(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        graph,
        workspace,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        container_k_gpu = create_cudnn_container(k_cache, self.page_size)
        container_v_gpu = create_cudnn_container(v_cache, self.page_size)

        page_table_k_gpu = self.page_table_k_gpu
        page_table_v_gpu = self.page_table_v_gpu

        seq_len_q_gpu = self.seq_len_q_gpu
        seq_len_kv_gpu = self.seq_len_kv_gpu
        scaling = torch.tensor(
            layer.scaling, dtype=self.dtype, device=self.device
        ).reshape(1, 1, 1, 1)

        for (
            query,
            page_table_k,
            page_table_v,
            seq_lens,
            extend_seq_lens,
            o_start_idx,
            o_end_idx,
        ) in chunk_attention_data(
            batch_size=self.batch_size,
            query=q_,
            page_table_k=page_table_k_gpu,
            page_table_v=page_table_v_gpu,
            seq_lens=seq_len_kv_gpu,
            extend_seq_lens=seq_len_q_gpu,
        ):
            o1 = self._run_cudnn_sdpa_forward(
                query,
                o_[o_start_idx:o_end_idx],
                container_k_gpu,
                container_v_gpu,
                page_table_k,
                page_table_v,
                seq_lens,
                extend_seq_lens,
                scaling=scaling,
                causal=not layer.is_cross_attention,
                graph=graph,
                workspace=workspace,
            )
            o_[o_start_idx:o_end_idx] = o1

        return o_.reshape(o.shape)

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):

        return self._common_forward(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache,
            graph=self.extend_graph,
            workspace=self.extend_workspace,
        )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):

        return self._common_forward(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache,
            graph=self.decode_graph,
            workspace=self.decode_workspace,
        )
