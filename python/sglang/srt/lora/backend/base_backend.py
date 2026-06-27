from typing import Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.lora.backend.lmhead_mixing import LoRABackendLmHeadMixing
from sglang.srt.lora.utils import LoRABatchInfo, MoELoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseLoRABackend(LoRABackendLmHeadMixing):
    """Base class for different Lora backends.
       Each backend has its own implementation of Lora kernels.

    Args:
        max_loras_per_batch: maximum number of different lora weights
                             that can be applied in a single forward batch.
        device: the device where the backend runs.
    """

    def __init__(self, max_loras_per_batch: int, device: torch.device):
        self.max_loras_per_batch = max_loras_per_batch
        self.device = device
        self.init_lm_head_config()
        self._is_moe_lora = False

    def run_lora_a_embedding(
        self,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
        vocab_size: int,
        extra_embeddings: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run LoRA A embedding lookup with CUDA graph support.

        Args:
            input_ids: token IDs with shape (s,), where s is the sum of all sequence lengths
            weights: LoRA A embedding weights with shape (num_loras, rank, vocab_size)
            vocab_size: base vocabulary size (tokens >= vocab_size are extra tokens)
            extra_embeddings: extra token embeddings with shape (num_loras, num_extra_tokens, rank)
            Only needed if there are added tokens beyond base vocabulary.

        Returns:
            result with shape (s, rank)
        """
        pass

    def run_extra_token_embedding(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor,
        extra_embeddings: torch.Tensor,
        vocab_size: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply extra token embeddings to output in-place.

        Args:
            input_ids: (s,) token IDs
            output: (s, embed_dim) output tensor to be modified
            extra_embeddings: (num_loras, num_extra_tokens, embed_dim) extra embeddings
            vocab_size: base vocabulary size

        Returns:
            output: modified output tensor
        """
        raise NotImplementedError

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora a modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             weights: a set of lora weights with shape (num_lora, c * r, input_dim),
                      here r is lora rank, c is a multiplier for stacked modules (e.g., c=3 for qkv_proj, c=2 for gate_up_proj)
                      usually input_dim is much larger than r
        Returns:
             result with shape (s, c * r)
        """
        pass

    def run_lora_b_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora b modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.

        Args:
             x: input matrix with shape (s, r), here s is the sum of all sequence lengths, r is lora rank
             weights: a set of lora weights with shape (num_lora, output_dim, r)
                      usually output_dim is much larger than r
        Returns:
             result with shape (s, output_dim)
        """
        pass

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for QKV Layer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            qkv_lora_a: lora_a module for qkv, with shape (num_lora, 3 * r, input_dim)
            qkv_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora,output_dim_q + 2 * output_dim_kv, r)
                        If passed in as a tuple of two tensors, it should contain:
                           a lora_b module for q, with shape (1, num_lora, output_dim_q, r)
                           and a combined lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)
        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
        """
        pass

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for gate_up_proj, usually attached to MergedColumnParallelLayer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            gate_up_lora_a: lora_a module for gate_up_proj, with shape (num_lora, 2 * r, input_dim)
            gate_up_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora, 2 * output_dim, r)
                        If passed in as a tuple, it should contain two tensors with shape (num_lora, output_dim, r)
        Returns:
            result with shape (s, 2 * output_dim)
        """
        pass

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        """Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.

        Called during CudaGraphRunner.__init__(), after init_memory_pool().

        Args:
            max_bs_in_cuda_graph: maximum batch size for CUDA Graph mode
            num_tokens_per_bs: number of tokens per sequence (1 for decoding, >1 for target_verify)
        """
        pass

    @property
    def is_moe_lora(self) -> bool:
        return self._is_moe_lora

    @is_moe_lora.setter
    def is_moe_lora(self, value: bool):
        self._is_moe_lora = value

    def init_cuda_graph_moe_buffers(
        self,
        max_bs: int,
        max_loras: int,
        compute_dtype: torch.dtype,
        moe_layer,
    ):
        """Phase 1 of LoRA CUDA graph init: MoE intermediate buffers.

        Called once before init_memory_pool() with a representative MoE layer
        to extract dimensions.  All FusedMoEWithLoRA layers share the same
        buffers since they execute sequentially during forward.

        This is backend-agnostic because MoE LoRA always uses the same
        fused Triton kernel (TritonRunnerCoreWithLoRA) regardless of which
        dense LoRA backend is selected.
        """
        base = moe_layer.base_layer
        top_k = base.top_k
        qinfo = moe_layer._quant_info
        E, N, _ = qinfo.w13_weight.shape
        hidden_dim = qinfo.w2_weight.shape[1]
        device = qinfo.w13_weight.device
        dtype = compute_dtype
        num_experts = base.num_experts

        block_size_m = 64
        max_num_tokens_padded = max_bs * top_k + num_experts * (block_size_m - 1)
        max_num_tokens_padded = (
            (max_num_tokens_padded + block_size_m - 1) // block_size_m
        ) * block_size_m
        max_num_m_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m

        self.moe_cg_buffers = {
            "intermediate_cache1": torch.empty(
                (max_bs, top_k, N), device=device, dtype=dtype
            ),
            "intermediate_cache2": torch.empty(
                (max_bs * top_k, N // 2), device=device, dtype=dtype
            ),
            "intermediate_cache3": torch.empty(
                (max_bs, top_k, hidden_dim), device=device, dtype=dtype
            ),
            "out_hidden_states": torch.empty(
                (max_bs, hidden_dim), device=device, dtype=dtype
            ),
            "sorted_token_ids_lora": torch.empty(
                (max_loras * max_num_tokens_padded,),
                device=device,
                dtype=torch.int32,
            ),
            "expert_ids_lora": torch.empty(
                (max_loras * max_num_m_blocks,),
                device=device,
                dtype=torch.int32,
            ),
            "num_tokens_post_padded_lora": torch.empty(
                (max_loras,), device=device, dtype=torch.int32
            ),
            "adapter_enabled": torch.zeros(max_loras, dtype=torch.int32, device=device),
            # int64 copy of weight_indices for index_fill_(), which requires
            # LongTensor.  weight_indices itself must stay int32 because the
            # CUDA moe_lora_align kernel casts it to int32_t*.
            "weight_indices_long": torch.zeros(
                max_bs, dtype=torch.int64, device=device
            ),
            "lora_ids": torch.arange(max_loras, dtype=torch.int32, device=device),
            "cumsum_buffer": torch.zeros(
                max_loras * (num_experts + 1),
                dtype=torch.int32,
                device=device,
            ),
            "token_mask": torch.empty(
                (max_loras * max_bs * top_k,),
                dtype=torch.int32,
                device=device,
            ),
            "max_num_tokens_padded": max_num_tokens_padded,
            "max_num_m_blocks": max_num_m_blocks,
            "token_lora_mapping": torch.full(
                (max_bs,), -1, dtype=torch.int32, device=device
            ),
        }

    def _add_moe_lora_info(
        self, forward_batch: ForwardBatch, batch_info: LoRABatchInfo
    ) -> LoRABatchInfo:
        if not self.is_moe_lora:
            return batch_info

        if batch_info.use_cuda_graph:
            adapter_enabled = self.moe_cg_buffers["adapter_enabled"]
            token_lora_mapping = self.moe_cg_buffers["token_lora_mapping"]
        else:
            adapter_enabled = None
            token_lora_mapping = None

        num_tokens = (
            sum(forward_batch.extend_seq_lens_cpu)
            if forward_batch.forward_mode.is_extend()
            else forward_batch.batch_size
        )
        max_len = (
            max(forward_batch.extend_seq_lens_cpu)
            if forward_batch.forward_mode.is_extend()
            else 1
        )

        if (
            batch_info.req_seg_indptr is not None
            or batch_info.req_weight_indices is not None
        ):
            assert batch_info.req_seg_indptr is not None
            assert batch_info.req_weight_indices is not None
            num_moe_segments = batch_info.bs
            seg_indptr = batch_info.req_seg_indptr[: num_moe_segments + 1]
            req_to_lora = batch_info.req_weight_indices[:num_moe_segments]
        else:
            num_moe_segments = batch_info.num_segments
            seg_indptr = batch_info.seg_indptr[: num_moe_segments + 1]
            req_to_lora = batch_info.weight_indices[:num_moe_segments]

        adapter_enabled, token_lora_mapping = _compute_moe_lora_info(
            num_tokens,
            seg_indptr,
            batch_info.lora_ranks,
            req_to_lora,
            adapter_enabled,
            token_lora_mapping,
            max_len=max_len,
        )

        batch_info.moe_lora_info = MoELoRABatchInfo(
            seg_indptr=seg_indptr,
            req_to_lora=req_to_lora,
            adapter_enabled=adapter_enabled,
            token_lora_mapping=token_lora_mapping,
        )

        return batch_info

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
    ):
        """Prepare the lora weights and batch info for current forward batch.

        This method provides a hook for each backend to conduct its own preparation
        logic for each forward batch.

        Args:
            forward_batch: the ForwardBatch object for current forward pass
            weight_indices: list of indices of lora weights to be applied for current batch
            lora_ranks: list of lora ranks corresponding to weight_indices
            scalings: list of scaling factors corresponding to weight_indices
            use_cuda_graph: whether to use CUDA Graph for this batch
        """
        pass


@triton.jit
def _compute_moe_lora_info_kernel(
    seg_indptr_ptr,
    lora_ranks_ptr,
    weight_indices_ptr,
    adapter_enabled_ptr,
    token_lora_mapping_ptr,
    num_segments,
    max_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(max_len, BLOCK_SIZE)

    pid_seg = pid // num_pid_m
    pid_m = pid % num_pid_m
    seg_start = tl.load(seg_indptr_ptr + pid_seg)
    seg_end = tl.load(seg_indptr_ptr + pid_seg + 1)
    seg_len = seg_end - seg_start

    offs = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < seg_len
    lora_id = tl.load(weight_indices_ptr + pid_seg)
    valid_lora = lora_id >= 0
    safe_lora_id = tl.maximum(lora_id, 0)
    lora_rank = tl.load(lora_ranks_ptr + safe_lora_id, mask=valid_lora, other=0)
    active_lora = valid_lora & (lora_rank > 0)
    tl.store(
        adapter_enabled_ptr + safe_lora_id,
        active_lora.to(tl.int32),
        mask=(pid_m == 0) & valid_lora,
    )
    token_lora_id = tl.where(active_lora, lora_id, -1)
    tl.store(token_lora_mapping_ptr + seg_start + offs, token_lora_id, mask=valid)


def _compute_moe_lora_info(
    num_tokens: int,
    seg_indptr: torch.Tensor,
    lora_ranks: torch.Tensor,
    weight_indices: torch.Tensor,
    adapter_enabled: torch.Tensor | None,
    token_lora_mapping: torch.Tensor | None,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_lora_mapping is not None:
        assert (
            num_tokens <= token_lora_mapping.shape[0]
        ), "num_tokens must be less than or equal to the shape of token_lora_mapping"
        # CUDA graph captures kernels against the full preallocated mapping
        # buffer.  Runtime batches may be padded to a larger graph bucket than
        # their real token count, so reset the whole buffer and only rewrite the
        # active prefix below.
        token_lora_mapping.fill_(-1)
        token_lora_mapping = token_lora_mapping[:num_tokens]
    else:
        token_lora_mapping = torch.empty(
            (num_tokens,), dtype=torch.int32, device=seg_indptr.device
        )

    if adapter_enabled is not None:
        assert (
            len(lora_ranks) <= adapter_enabled.shape[0]
        ), "lora_ranks must be less than or equal to the shape of adapter_enabled"
    else:
        adapter_enabled = torch.empty(
            len(lora_ranks), dtype=torch.int32, device=lora_ranks.device
        )

    adapter_enabled.zero_()

    has_segments = weight_indices.numel() != 0
    use_cuda_kernel = (
        num_tokens != 0 and has_segments and seg_indptr.device.type == "cuda"
    )
    if use_cuda_kernel:
        block_size = 256
        tiles_per_segment = triton.cdiv(max_len, block_size)
        grid_size = tiles_per_segment * weight_indices.numel()
        assert grid_size * block_size >= num_tokens, (
            f"MoE LoRA token-mapping launch under-covers tokens: "
            f"{grid_size=} {block_size=} {num_tokens=}"
        )
        _compute_moe_lora_info_kernel[(grid_size,)](
            seg_indptr,
            lora_ranks,
            weight_indices,
            adapter_enabled,
            token_lora_mapping,
            weight_indices.numel(),
            max_len,
            BLOCK_SIZE=block_size,
        )
        return adapter_enabled, token_lora_mapping

    if has_segments:
        valid_weight_indices = weight_indices >= 0
        safe_weight_indices = torch.where(valid_weight_indices, weight_indices, 0)
        active_ranks = torch.where(
            valid_weight_indices,
            lora_ranks[safe_weight_indices.long()],
            0,
        )
        valid_indices = weight_indices[valid_weight_indices]
        if valid_indices.numel() != 0:
            adapter_enabled.scatter_(
                0,
                valid_indices.long(),
                (active_ranks[valid_weight_indices] > 0).to(torch.int32),
            )
    if num_tokens == 0:
        return adapter_enabled, token_lora_mapping
    if not has_segments:
        token_lora_mapping.fill_(-1)
        return adapter_enabled, token_lora_mapping

    token_positions = torch.arange(
        num_tokens, device=seg_indptr.device, dtype=torch.int32
    )
    # There is a torch.compile bug so we can't use seg_indptr[1:] here.
    # Instead we pass seg_indptr and then subtract 1 from the result.
    # This works because seg_indptr[0] == 0.
    req_indices = (
        torch.searchsorted(seg_indptr.to(torch.int32), token_positions, right=True) - 1
    )

    active_weight_indices = torch.where(
        valid_weight_indices & (active_ranks > 0),
        weight_indices.to(torch.int32),
        -1,
    )
    token_lora_mapping = torch.index_select(
        active_weight_indices, 0, req_indices, out=token_lora_mapping
    )

    return adapter_enabled, token_lora_mapping
