import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.srt.speculative.uno_info import UnoForwardInput
from sglang.srt.speculative.uno_lora import UnoCudaGraphLoRAState


class UnoDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Decode graph runner for linear UNO and both tree forward roles.

    Linear UNO uses two-variant F-wide capture. Tree UNO uses
    separate runner instances for its F-wide LoRA draft
    and native Q/K EAGLE target verification.
    """

    def __init__(
        self,
        model_runner,
        *,
        tree_draft_attn_backend=None,
        tree_draft_width=None,
        **kwargs,
    ):
        candidate_top_k = get_spec().speculative_eagle_topk
        self._tree_mode = candidate_top_k > 1
        self._tree_draft_mode = tree_draft_width is not None

        if self._tree_draft_mode:
            self.record_nolora_graph = False
            self._capture_spec_input_type = SpecInputType.UNO_DRAFT
            self._lora_state = UnoCudaGraphLoRAState(
                model_runner.lora_manager,
                model_runner.uno_lora_id,
                tree_draft_width,
            )
            model_runner.lora_manager.reset_lora_batch()
            kwargs.update(
                attn_backend=tree_draft_attn_backend,
                speculative_num_steps=1,
                speculative_num_draft_tokens=tree_draft_width,
            )
            super().__init__(model_runner, **kwargs)
            model_runner.lora_manager.reset_lora_batch()
            return

        if self._tree_mode:
            # Capture exactly one base-model target graph.  The internal UNO
            # adapter is active only in the rejected F-wide draft phase.
            self.record_nolora_graph = False
            model_runner.lora_manager.reset_lora_batch()
            kwargs.update(
                attn_backend=model_runner.attn_backend,
                speculative_num_steps=get_spec().speculative_num_steps,
                speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            )
            super().__init__(model_runner, **kwargs)
            model_runner.lora_manager.reset_lora_batch()
            return

        forward_width = model_runner.decode_num_tokens_per_req()
        self.record_nolora_graph = forward_width > 1
        self._capture_spec_input_type = SpecInputType.UNO_VERIFY
        self._lora_state = UnoCudaGraphLoRAState(
            model_runner.lora_manager,
            model_runner.uno_lora_id,
            forward_width,
        )
        model_runner.lora_manager.reset_lora_batch()
        super().__init__(model_runner, **kwargs)

    def capture_prepare(self, size, stream_idx=None, num_tokens=None):
        forward_batch, attn_backend, pp_proxy_tensors = super().capture_prepare(
            size, stream_idx=stream_idx, num_tokens=num_tokens
        )
        # UNO owns token-row routing directly. K2's generic graph runner now
        # keys LoRA setup off lora_manager presence, so suppress its synthetic
        # request-level base-adapter routing during UNO capture.
        forward_batch.lora_ids = None
        return forward_batch, attn_backend, pp_proxy_tensors

    def can_run_graph(self, forward_batch):
        spec_info = forward_batch.spec_info
        if self._tree_draft_mode:
            if not isinstance(spec_info, UnoForwardInput):
                return False
            if spec_info.spec_input_type != SpecInputType.UNO_DRAFT:
                return False
            return super().can_run_graph(forward_batch)

        if self._tree_mode:
            if not isinstance(spec_info, EagleVerifyInput):
                return False
            return super().can_run_graph(forward_batch)

        if not isinstance(spec_info, UnoForwardInput):
            return False
        # At F=1 both phases are base-only and share variant_label=None.
        if spec_info.spec_input_type not in {
            SpecInputType.UNO_DRAFT,
            SpecInputType.UNO_VERIFY,
        }:
            return False
        return super().can_run_graph(forward_batch)

    def _resolve_lora_variant(self, forward_batch):
        """borrowed technique from multi-LoRA serving
        to capture separate graph for each step."""
        if self._tree_mode:
            # Tree verification is always the clean target model.  Keeping the
            # graph keys unlabeled is sufficient because draft and verify own
            # separate runners.
            return None
        if not self.record_nolora_graph:
            return None
        if forward_batch.spec_info.spec_input_type == SpecInputType.UNO_DRAFT:
            return "lora"
        return "nolora"

    def capture_one_shape(
        self,
        size,
        forward,
        stream_idx=None,
        variant_label=None,
        dsa_variant=None,
    ):
        """capture one CUDA graph with/out UNO LoRA."""
        if self._tree_draft_mode:
            self._lora_state.capture_draft(size)
            try:
                return super().capture_one_shape(
                    size,
                    forward,
                    stream_idx,
                    None,
                    dsa_variant,
                )
            finally:
                self._lora_state.reset()

        if self._tree_mode:
            self.model_runner.lora_manager.reset_lora_batch()
            try:
                return super().capture_one_shape(
                    size,
                    forward,
                    stream_idx,
                    None,
                    dsa_variant,
                )
            finally:
                self.model_runner.lora_manager.reset_lora_batch()

        if variant_label == "lora":
            self._capture_spec_input_type = SpecInputType.UNO_DRAFT
            self._lora_state.capture_draft(size)
        else:
            self._capture_spec_input_type = SpecInputType.UNO_VERIFY
            self._lora_state.reset()

        super().capture_one_shape(
            size,
            forward,
            stream_idx,
            variant_label,
            dsa_variant,
        )
        self._lora_state.reset()

    def get_spec_info(self, num_tokens: int):
        if self._tree_draft_mode:
            return UnoForwardInput(
                spec_input_type=SpecInputType.UNO_DRAFT,
                positions=self.buffers.positions[:num_tokens],
                draft_token_num=self.captured_req_width,
            )

        if self._tree_mode:
            # This deliberately mirrors DecodeCudaGraphRunner's EAGLE capture
            # input.  Current eagle_prepare_for_verify requests FULL hidden
            # capture for every non-STANDALONE algorithm, including UNO.
            spec_info = EagleVerifyInput(
                draft_token=None,
                custom_mask=self.buffers.custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_cum_len=None,
                spec_steps=self.speculative_num_steps,
                topk=get_spec().speculative_eagle_topk,
                draft_token_num=self.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )
            spec_info.hidden_states = torch.zeros(
                (num_tokens, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
                device=self.model_runner.device,
            )
            return spec_info

        return UnoForwardInput(
            spec_input_type=self._capture_spec_input_type,
            positions=self.buffers.positions[:num_tokens],
            draft_token_num=self.captured_req_width,
        )
