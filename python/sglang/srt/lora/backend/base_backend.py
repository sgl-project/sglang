from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.lora.backend.lmhead_mixing import LoRABackendLmHeadMixing
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    MoELoRABatchInfo,
    get_batch_token_counts,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseLoRABackend(LoRABackendLmHeadMixing):
    """Base class for different Lora backends.
       Each backend has its own implementation of Lora kernels.

    Args:
        max_loras_per_batch: maximum number of different lora weights
                             that can be applied in a single forward batch.
        device: the device where the backend runs.
    """

    supports_lora_a_overlap = False
    skip_inactive_lora_batches = False
    # Requires a separate "nolora" decode graph for batches without adapters.
    skip_inactive_dense_lora = False

    # Supporting backends implement init_prefill_cuda_graph_batch_info() and
    # honor use_prefill_cuda_graph in prepare_lora_batch().
    supports_prefill_cuda_graph: bool = False

    def __init__(self, max_loras_per_batch: int, device: torch.device):
        self.max_loras_per_batch = max_loras_per_batch
        self.device = device
        # Set by prepare_lora_batch() before each forward; cleared by
        # reset_batch_state() on DP-attention idle forwards. None means "no
        # batch prepared" — the LoRA layers read it to skip LoRA application.
        self.batch_info: Optional[LoRABatchInfo] = None
        self.init_lm_head_config()
        self._is_moe_lora = False
        # Static metadata read by prefill-CUDA-graph kernels, refreshed in
        # place every prefill batch.
        self.prefill_cuda_graph_batch_info: LoRABatchInfo | None = None
        # Request/token caps for serving a batch from the static metadata.
        self.prefill_cuda_graph_max_bs: int | None = None
        self.prefill_cuda_graph_max_tokens: int | None = None
        # Separate scratch sized for the largest prefill token bucket.
        self.prefill_moe_cg_buffers: dict | None = None
        # Sequential MoE layers share one workspace, including graph scratch.
        self.lora_workspace = None

    def reset_batch_state(self):
        """Idle-forward counterpart of prepare_lora_batch(): clears all
        per-batch metadata. batch_info=None is the master "no batch
        prepared" signal that the layer guards (lora_active) read."""
        self.batch_info = None
        self.lm_head_batch_info = None
        self.lm_head_pass_batch_infos = None
        self._lm_head_pass_idx = None

    def reset_routing_cache(self) -> None:
        """Clear cached routes without changing batch metadata or storage."""
        if self.lora_workspace is not None:
            self.lora_workspace.routes.clear()

    def validate_lora_targets(
        self,
        base_model: torch.nn.Module,
        target_modules: set[str],
    ) -> None:
        """Raise before wrapping when this backend cannot execute its targets."""
        pass

    def forward_with_base(
        self,
        layer,
        x: torch.Tensor,
        base_fn,
        lora_a,
        lora_b,
        output_offset,
        offsets,
        *,
        all_reduce=None,
        kind="linear",
        pruned_batch_info=None,
    ) -> torch.Tensor:
        """Run base + LoRA, retaining separate base/A reductions for legacy backends."""
        if all_reduce is None:
            return layer.apply_lora(base_fn(), x)
        lora_a_output = self.run_lora_a_sgemm(x, lora_a)
        output = all_reduce(base_fn())
        lora_a_output = all_reduce(lora_a_output)
        return self.run_lora_b_sgemm(
            x=lora_a_output,
            weights=lora_b,
            output_offset=output_offset,
            output_offset_cpu=layer.output_offset_cpu,
            base_output=output,
        )

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

    def init_decode_cuda_graph_batch_info(
        self, max_bs_in_cuda_graph: int, num_tokens_per_req: int
    ):
        """Allocate decode-runner metadata, including target verification."""
        pass

    def init_prefill_cuda_graph_batch_info(
        self,
        max_num_tokens: int,
        max_num_requests: Optional[int] = None,
    ):
        """Allocate static LoRA batch metadata for the prefill CUDA graph,
        sized for the largest captured token bucket. Called before capture."""
        raise NotImplementedError(
            f"LoRA backend {type(self).__name__} does not support the prefill CUDA graph."
        )

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
        *,
        prefill: bool = False,
    ):
        """Allocate shared MoE routing buffers for decode or prefill captures.

        max_bs counts tokens; layers reuse buffers sequentially. The LoRA MoE
        runner owns its scratch and needs only metadata on the base device.
        """
        include_legacy_kernel_buffers = not moe_layer._lora_runner_backend.is_lora()
        if include_legacy_kernel_buffers:
            quant_info = moe_layer._quant_info
            # Marlin quant info exposes packed weights as w13_qweight.
            weight = getattr(quant_info, "w13_weight", None)
            if weight is None:
                weight = quant_info.w13_qweight
            device = weight.device
        else:
            device = moe_layer.base_layer.w13_weight.device
        buffers = {
            "adapter_enabled": torch.zeros(max_loras, dtype=torch.int32, device=device),
            "token_lora_mapping": torch.full(
                (max_bs,), -1, dtype=torch.int32, device=device
            ),
        }
        if include_legacy_kernel_buffers:
            base = moe_layer.base_layer
            top_k = base.top_k
            num_experts = base.num_experts

            block_size_m = 64
            max_num_tokens_padded = max_bs * top_k + num_experts * (block_size_m - 1)
            max_num_tokens_padded = (
                (max_num_tokens_padded + block_size_m - 1) // block_size_m
            ) * block_size_m
            max_num_m_blocks = (
                max_num_tokens_padded + block_size_m - 1
            ) // block_size_m
            buffers.update(
                {
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
                    "lora_ids": torch.arange(
                        max_loras, dtype=torch.int32, device=device
                    ),
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
                }
            )

        if prefill:
            self.prefill_moe_cg_buffers = buffers
        else:
            self.moe_cg_buffers = buffers

    def _add_moe_lora_info(
        self, forward_batch: ForwardBatch, batch_info: LoRABatchInfo
    ) -> LoRABatchInfo:
        if not self.is_moe_lora:
            return batch_info

        prefill = batch_info is self.prefill_cuda_graph_batch_info
        if batch_info.use_cuda_graph:
            buffers = self.prefill_moe_cg_buffers if prefill else self.moe_cg_buffers
            if prefill and buffers is None:
                raise RuntimeError(
                    "prefill MoE-LoRA CUDA graph buffers were not initialized"
                )
            adapter_enabled = buffers["adapter_enabled"]
            token_lora_mapping = buffers["token_lora_mapping"]
        else:
            adapter_enabled = None
            token_lora_mapping = None

        num_tokens, max_len = get_batch_token_counts(forward_batch)

        # Capture fixes the segment count; include every prefill request slot.
        # Unused slots contain empty segments.
        if (
            batch_info.req_seg_indptr is not None
            or batch_info.req_weight_indices is not None
        ):
            assert batch_info.req_seg_indptr is not None
            assert batch_info.req_weight_indices is not None
            num_moe_segments = (
                batch_info.req_weight_indices.shape[0] if prefill else batch_info.bs
            )
            seg_indptr = batch_info.req_seg_indptr[: num_moe_segments + 1]
            req_to_lora = batch_info.req_weight_indices[:num_moe_segments]
        else:
            num_moe_segments = (
                batch_info.weight_indices.shape[0]
                if prefill
                else batch_info.num_segments
            )
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
        use_decode_cuda_graph: bool,
        use_prefill_cuda_graph: bool = False,
    ):
        """Bind eager metadata or update the selected graph family's static metadata."""
        pass

    def prepare_lora_token_segments(
        self,
        *,
        segment_lens: list[int],
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
    ) -> None:
        """Prepare explicit eager token-row LoRA segments."""
        raise NotImplementedError(
            f"LoRA backend {type(self).__name__} does not support explicit "
            "token segments."
        )


@triton.jit
def _compute_moe_lora_info_kernel(
    seg_indptr_ptr,
    lora_ranks_ptr,
    weight_indices_ptr,
    adapter_enabled_ptr,
    token_lora_mapping_ptr,
    max_len,
    num_tokens,
    bucket_len,
    live_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= live_programs:
        # A tile of the static bucket past the live tokens: no adapter.
        offs = (
            num_tokens + (pid - live_programs) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        tl.store(
            token_lora_mapping_ptr + offs,
            tl.full((BLOCK_SIZE,), -1, tl.int32),
            mask=offs < bucket_len,
        )
        return
    num_pid_m = tl.cdiv(max_len, BLOCK_SIZE)

    pid_seg = pid // num_pid_m
    pid_m = pid % num_pid_m
    seg_start = tl.load(seg_indptr_ptr + pid_seg)
    seg_end = tl.load(seg_indptr_ptr + pid_seg + 1)
    seg_len = seg_end - seg_start

    offs = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < seg_len
    lora_id = tl.load(weight_indices_ptr + pid_seg)
    lora_rank = tl.load(lora_ranks_ptr + lora_id)
    adapter_is_enabled = lora_rank > 0
    if adapter_enabled_ptr is not None:
        tl.store(
            adapter_enabled_ptr + lora_id,
            adapter_is_enabled.to(tl.int32),
            mask=pid_m == 0,
        )
    tl.store(
        token_lora_mapping_ptr + seg_start + offs,
        tl.where(adapter_is_enabled, lora_id, -1),
        mask=valid,
    )


def _compute_moe_lora_info(
    num_tokens: int,
    seg_indptr: torch.Tensor,
    lora_ranks: torch.Tensor,
    weight_indices: torch.Tensor,
    adapter_enabled: torch.Tensor | None,
    token_lora_mapping: torch.Tensor | None,
    max_len: int,
    bucket_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build token slots and the active-adapter mask for legacy backends."""
    if adapter_enabled is None:
        adapter_enabled = torch.empty(
            len(lora_ranks), dtype=torch.int32, device=lora_ranks.device
        )
    else:
        assert len(lora_ranks) <= adapter_enabled.shape[0], (
            "lora_ranks must be less than or equal to the shape of adapter_enabled"
        )
    adapter_enabled.zero_()
    token_lora_mapping = _compute_token_lora_mapping(
        num_tokens,
        seg_indptr,
        lora_ranks,
        weight_indices,
        token_lora_mapping,
        max_len,
        bucket_len=bucket_len,
        adapter_enabled=adapter_enabled,
    )
    return adapter_enabled, token_lora_mapping


def _compute_token_lora_mapping(
    num_tokens: int,
    seg_indptr: torch.Tensor,
    lora_ranks: torch.Tensor,
    weight_indices: torch.Tensor,
    token_lora_mapping: torch.Tensor | None,
    max_len: int,
    bucket_len: int | None = None,
    adapter_enabled: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand request slots, clearing graph padding in the same CUDA launch."""
    tail_tiles = 0
    full_mapping = token_lora_mapping
    if token_lora_mapping is not None:
        assert num_tokens <= token_lora_mapping.shape[0], (
            "num_tokens must be less than or equal to the shape of token_lora_mapping"
        )
        # Clear graph padding here unless the CUDA kernel will clear it.
        if num_tokens < token_lora_mapping.shape[0]:
            if bucket_len is not None and token_lora_mapping.device.type == "cuda":
                tail_tiles = triton.cdiv(bucket_len - num_tokens, 256)
            else:
                token_lora_mapping[num_tokens:].fill_(-1)
        token_lora_mapping = token_lora_mapping[:num_tokens]
    else:
        token_lora_mapping = torch.empty(
            (num_tokens,), dtype=torch.int32, device=seg_indptr.device
        )

    has_segments = weight_indices.numel() != 0
    needs_launch = num_tokens != 0 and has_segments
    if needs_launch:
        block_size = 256
        tiles_per_segment = triton.cdiv(max_len, block_size)
        grid_size = tiles_per_segment * weight_indices.numel()
        assert grid_size * block_size >= num_tokens, (
            f"MoE LoRA token-mapping launch under-covers tokens: "
            f"{grid_size=} {block_size=} {num_tokens=}"
        )

    # Triton kernel on CUDA only; every other device (e.g. XPU) falls through to
    # the native torch path below, which yields the same mapping.
    if needs_launch and seg_indptr.device.type == "cuda":
        _compute_moe_lora_info_kernel[(grid_size + tail_tiles,)](
            seg_indptr,
            lora_ranks,
            weight_indices,
            adapter_enabled,
            token_lora_mapping,
            max_len,
            num_tokens,
            bucket_len if tail_tiles else 0,
            grid_size,
            BLOCK_SIZE=block_size,
        )
        return token_lora_mapping
    if tail_tiles:
        full_mapping[num_tokens:bucket_len].fill_(-1)

    if has_segments and adapter_enabled is not None:
        active_ranks = lora_ranks[weight_indices.long()]
        adapter_enabled.scatter_(
            0, weight_indices.long(), (active_ranks > 0).to(torch.int32)
        )
    if num_tokens == 0:
        return token_lora_mapping
    if not has_segments:
        token_lora_mapping.fill_(-1)
        return token_lora_mapping

    token_positions = torch.arange(
        num_tokens, device=seg_indptr.device, dtype=torch.int32
    )
    # There is a torch.compile bug so we can't use seg_indptr[1:] here.
    # Instead we pass seg_indptr and then subtract 1 from the result.
    # This works because seg_indptr[0] == 0.
    req_indices = (
        torch.searchsorted(seg_indptr.to(torch.int32), token_positions, right=True) - 1
    )

    torch.index_select(
        weight_indices.to(torch.int32), 0, req_indices, out=token_lora_mapping
    )
    token_lora_ranks = torch.index_select(
        lora_ranks, 0, token_lora_mapping.to(torch.int64)
    )
    token_lora_mapping.masked_fill_(token_lora_ranks <= 0, -1)

    return token_lora_mapping
