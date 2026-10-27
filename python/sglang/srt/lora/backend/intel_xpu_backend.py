from typing import List, Optional

import torch

# Native Intel XPU LoRA kernels (from sgl-kernel-xpu). Aliased with an `xpu_`
# prefix to keep the kernel family explicit at each call site.
from sgl_kernel import (
    chunked_sgmv_lora_expand_fwd as xpu_chunked_sgmv_lora_expand_forward,
)
from sgl_kernel import (
    chunked_sgmv_lora_shrink_fwd as xpu_chunked_sgmv_lora_shrink_forward,
)
from sgl_kernel import embedding_lora_a_fwd as xpu_embedding_lora_a_fwd
from sgl_kernel import gate_up_lora_b_fwd as xpu_gate_up_lora_b_fwd
from sgl_kernel import qkv_lora_b_fwd as xpu_qkv_lora_b_fwd
from sgl_kernel import sgemm_lora_a_fwd as xpu_sgemm_lora_a_fwd
from sgl_kernel import sgemm_lora_b_fwd as xpu_sgemm_lora_b_fwd

from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class IntelXpuLoRABackend(TritonLoRABackend):
    """LoRA on native Intel XPU kernels: per-shape sgemm kernels for contiguous prefill,
    chunked-SGMV (gather/GEMM/scatter by permutation) for adapter-sorted decode."""

    name = "intel_xpu"
    # Triton's DP-attention global routing has no XPU kernel path.
    supports_dp_attention = False
    supports_prefill_cuda_graph = True
    # sgl-kernel-xpu LoRA kernels ignore lora_ranks for the GEMM extent.
    uses_max_rank_spaced_lora_layout = True

    @staticmethod
    def _is_permuted(batch_info: LoRABatchInfo) -> bool:
        """True when the batch metadata is adapter-sorted (decode), in which
        case the contiguous-only XPU kernels cannot be used."""
        return batch_info.permutation is not None

    def _lora_a_shrink(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        sgemm_info: LoRABatchInfo,
        stack_num: int,
    ) -> torch.Tensor:
        """LoRA A shrink: sgemm_lora_a_fwd on prefill, chunked-SGMV on decode (no
        max_chunk_size needed). Output is always in physical token order."""
        if self._is_permuted(sgemm_info):
            return xpu_chunked_sgmv_lora_shrink_forward(
                input_x=x,
                weights=weights,
                stack_num=stack_num,
                num_segments=sgemm_info.num_segments,
                seg_indptr=sgemm_info.seg_indptr,
                weight_indices=sgemm_info.weight_indices,
                lora_ranks=sgemm_info.lora_ranks,
                permutation=sgemm_info.permutation,
            )
        return xpu_sgemm_lora_a_fwd(
            input_x=x,
            weights=weights,
            stack_num=stack_num,
            seg_indptr=sgemm_info.seg_indptr,
            weight_indices=sgemm_info.weight_indices,
            lora_ranks=sgemm_info.lora_ranks,
            seg_lens=sgemm_info.seg_lens,
        )

    def _lora_b_expand(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        sgemm_info: LoRABatchInfo,
        slice_offsets: torch.Tensor,
        max_slice_size: int,
        num_slices: int,
        base_output: torch.Tensor = None,
    ) -> torch.Tensor:
        """Chunked-SGMV B expand for any slice count; permutation=None on prefill.
        Takes the shrink's (num_tokens, num_slices * max_rank) output as-is."""
        return xpu_chunked_sgmv_lora_expand_forward(
            input_x=x,
            weights=weights,
            slice_offsets=slice_offsets,
            max_slice_size=max_slice_size,
            num_slices=num_slices,
            num_segments=sgemm_info.num_segments,
            seg_indptr=sgemm_info.seg_indptr,
            weight_indices=sgemm_info.weight_indices,
            lora_ranks=sgemm_info.lora_ranks,
            scalings=sgemm_info.scalings,
            permutation=sgemm_info.permutation,
            base_output=base_output,
        )

    # ---- LoRA kernel entry points ----

    def run_lora_a_embedding(
        self,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
        vocab_size: int,
        extra_embeddings: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # self.batch_info is always contiguous (permutation=None) for
        # embeddings, so the XPU kernel always applies.
        bi = self.batch_info
        assert bi is not None, (
            "LoRA kernel invoked with no prepared batch (DP-attention idle "
            "forward?). Gate the caller on lora_active."
        )
        return xpu_embedding_lora_a_fwd(
            input_ids=input_ids,
            weights=weights,
            vocab_size=vocab_size,
            seg_indptr=bi.seg_indptr,
            weight_indices=bi.weight_indices,
            lora_ranks=bi.lora_ranks,
            extra_embeddings=extra_embeddings,
            seg_lens=bi.seg_lens,
            max_len=bi.max_len
        )

    def run_lora_a_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        pruned_batch_info: LoRABatchInfo = None,
        stack_num: int = 1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        sgemm_info = self._sgemm_info(pruned_batch_info)
        return self._lora_a_shrink(x, weights, sgemm_info, stack_num)

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        output_offset: torch.Tensor,
        base_output: torch.Tensor = None,
        pruned_batch_info: LoRABatchInfo = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        sgemm_info = self._sgemm_info(pruned_batch_info)
        if self._is_permuted(sgemm_info):
            # Decode (permuted): single-slice chunked-SGMV expand on XPU.
            return self._lora_b_expand(
                x=x,
                weights=weights,
                sgemm_info=sgemm_info,
                slice_offsets=output_offset,
                max_slice_size=weights.shape[-2],
                num_slices=1,
                base_output=base_output,
            )
        return xpu_sgemm_lora_b_fwd(
            input_x=x,
            weights=weights,
            seg_indptr=sgemm_info.seg_indptr,
            weight_indices=sgemm_info.weight_indices,
            lora_ranks=sgemm_info.lora_ranks,
            scalings=sgemm_info.scalings,
            seg_lens=sgemm_info.seg_lens,
            base_output=base_output,
        )

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        n_slices: int = 3,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        # x: (s, input_dim), qkv_lora_a: (num_lora, n_slices * r, input_dim),
        # qkv_lora_b: (num_lora, total_output_dim, r)
        assert isinstance(qkv_lora_b, torch.Tensor)

        sgemm_info = self._sgemm_info()
        # A-shrink runs on XPU on both paths (chunked-SGMV on the permuted
        # decode); its output is in physical token order.
        lora_a_output = self._lora_a_shrink(x, qkv_lora_a, sgemm_info, n_slices)
        # qkv_lora_b_fwd is fixed at 3 slices; MLA fused_qkv_a_proj_with_mqa has 2.
        if self._is_permuted(sgemm_info) or n_slices != 3:
            return self._lora_b_expand(
                x=lora_a_output,
                weights=qkv_lora_b,
                sgemm_info=sgemm_info,
                slice_offsets=output_offset,
                max_slice_size=max_qkv_out_dim,
                num_slices=n_slices,
                base_output=base_output,
            )

        # Prefill (contiguous): XPU B-expand.
        return xpu_qkv_lora_b_fwd(
            input_x=lora_a_output,
            qkv_lora_b=qkv_lora_b,
            output_offset=output_offset,
            max_qkv_out_dim=max_qkv_out_dim,
            seg_indptr=sgemm_info.seg_indptr,
            weight_indices=sgemm_info.weight_indices,
            lora_ranks=sgemm_info.lora_ranks,
            scalings=sgemm_info.scalings,
            seg_lens=sgemm_info.seg_lens,
            base_output=base_output,
        )

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        # x: (s, input_dim), gate_up_lora_a: (num_lora, 2 * r, input_dim),
        # gate_up_lora_b: (num_lora, 2 * output_dim, r)
        assert isinstance(gate_up_lora_b, torch.Tensor)
        output_dim = gate_up_lora_b.shape[-2] // 2

        sgemm_info = self._sgemm_info()
        # A-shrink runs on XPU on both paths (chunked-SGMV on the permuted
        # decode); its output is in physical token order.
        lora_a_output = self._lora_a_shrink(x, gate_up_lora_a, sgemm_info, 2)
        if self._is_permuted(sgemm_info):
            # Decode (permuted): fused 2-way (gate, up) chunked-SGMV expand on XPU.
            return self._lora_b_expand(
                x=lora_a_output,
                weights=gate_up_lora_b,
                sgemm_info=sgemm_info,
                slice_offsets=output_offset,
                max_slice_size=output_dim,
                num_slices=2,
                base_output=base_output,
            )

        # Prefill (contiguous): XPU B-expand.
        return xpu_gate_up_lora_b_fwd(
            input_x=lora_a_output,
            gate_up_lora_b=gate_up_lora_b,
            output_dim=output_dim,
            seg_indptr=sgemm_info.seg_indptr,
            weight_indices=sgemm_info.weight_indices,
            lora_ranks=sgemm_info.lora_ranks,
            scalings=sgemm_info.scalings,
            seg_lens=sgemm_info.seg_lens,
            base_output=base_output,
        )

    def _prepare_prefill_cuda_graph_sgemm_batch_info(
        self, forward_batch: ForwardBatch, weight_indices: List[int]
    ) -> Optional[LoRABatchInfo]:
        # XPU kernels tile M per segment; fixed-size chunks only add partial tiles.
        return None
