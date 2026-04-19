from typing import List, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.lora.utils import LoRABatchInfo, build_lm_head_pass_segments
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class LoRABackendLmHeadMixing:
    def init_lm_head_config(self):
        self.lm_head_batch_info = None
        # Precomputed per-pass lm_head batch_infos.  When the logits processor
        # calls lm_head in multiple passes (chunked logprobs), each pass gets
        # its own batch_info from this list.
        self.lm_head_pass_batch_infos = None
        # Current pass index.  When set, apply_lora uses
        # lm_head_pass_batch_infos[idx] instead of lm_head_batch_info.
        self._lm_head_pass_idx = None

    def _get_lm_head_pass_segments(
        self,
        weight_indices: list[int],
        pruned_lens: List[int],
    ) -> Optional[List[Tuple[List[int], List[int]]]]:
        """Compute per-pass segment info for lm_head LoRA logprobs chunking.

        When LogitsProcessor splits pruned states into fixed-size passes,
        each pass needs its own segmentation so that lm_head LoRA operates
        on the correct adapter assignments.  This method returns the generic
        per-pass (seg_weight_indices, seg_lens) tuples; each backend is
        responsible for converting them into backend-specific LoRABatchInfo.

        Returns None if logprobs chunking is disabled or the pruned token
        count does not exceed the logprobs chunk size.
        """
        logprobs_chunk_size = envs.SGLANG_LOGITS_PROCESSER_CHUNK_SIZE.get()
        enable_logprobs_chunk = envs.SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK.get()
        pruned_total = sum(pruned_lens)

        if not enable_logprobs_chunk or pruned_total <= logprobs_chunk_size:
            return None

        return build_lm_head_pass_segments(
            weight_indices, pruned_lens, logprobs_chunk_size
        )

    def _prepare_lm_head_batch_info(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        batch_info: LoRABatchInfo,
    ) -> Tuple[Optional[LoRABatchInfo], Optional[List[LoRABatchInfo]]]:
        """Prepare the lm_head batch info for the current forward batch."""
        """It returns a tuple of (lm_head_batch_info, lm_head_pass_batch_infos)."""
        pass

    def _build_lm_head_batch_info(
        self,
        lm_head_segments: Tuple[List[int], List[int]],
        batch_info: LoRABatchInfo,
        chunk_size: int,
        expected_tokens: int,
    ) -> LoRABatchInfo:
        """Build a LoRABatchInfo for pruned lm_head input."""
        pass
