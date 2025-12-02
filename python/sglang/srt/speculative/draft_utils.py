import logging

from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils.common import is_blackwell

logger = logging.getLogger(__name__)


class DraftBackendFactory:
    def __init__(
        self,
        server_args: ServerArgs,
        draft_model_runner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.server_args = server_args
        self.draft_model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

    def _create_backend(
        self, backend_name: str, backend_map: dict, error_template: str
    ):
        backend_type = getattr(self.server_args, backend_name)
        if backend_type is None:
            backend_type = self.server_args.attention_backend

        if backend_type not in backend_map:
            raise ValueError(error_template.format(backend_type=backend_type))

        return backend_map[backend_type]()

    def create_decode_backend(self):
        if self.speculative_num_steps == 1:
            return None

        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            "aiter": self._create_aiter_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            "hybrid_linear_attn": (
                self._create_fa3_decode_backend
                if not is_blackwell()
                else self._create_triton_decode_backend
            ),
            "flashmla": self._create_flashmla_decode_backend,
            "trtllm_mha": self._create_trtllm_mha_decode_backend,
            "trtllm_mla": self._create_trtllm_mla_decode_backend,
            "nsa": self._create_nsa_decode_backend,
            "ascend": self._create_ascend_decode_backend,
        }

        return self._create_backend(
            "decode_attention_backend",
            backend_map,
            "EAGLE is not supported in decode attention backend {backend_type}",
        )

    def create_draft_extend_backend(self):
        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "aiter": self._create_aiter_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            "hybrid_linear_attn": (
                self._create_fa3_prefill_backend
                if not is_blackwell()
                else self._create_triton_prefill_backend
            ),
            "flashmla": self._create_flashmla_prefill_backend,
            "trtllm_mha": self._create_trtllm_mha_prefill_backend,
            "trtllm_mla": self._create_trtllm_mla_prefill_backend,
            "nsa": self._create_nsa_prefill_backend,
            "ascend": self._create_ascend_prefill_backend,
        }
        backend_name = (
            "decode_attention_backend"
            if self.server_args.speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        return self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )

    def _create_nsa_decode_backend(self):
        from sglang.srt.layers.attention.nsa_backend import (
            NativeSparseAttnMultiStepBackend,
        )

        return NativeSparseAttnMultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_nsa_prefill_backend(self):
        from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend

        return NativeSparseAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_flashinfer_decode_backend(self):
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            return FlashInferMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            return FlashInferMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )

    def _create_triton_decode_backend(self):
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        return TritonMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_aiter_decode_backend(self):
        from sglang.srt.layers.attention.aiter_backend import AiterMultiStepDraftBackend

        return AiterMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_fa3_decode_backend(self):
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionMultiStepBackend,
        )

        return FlashAttentionMultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_flashmla_decode_backend(self):
        from sglang.srt.layers.attention.flashmla_backend import (
            FlashMLAMultiStepDraftBackend,
        )

        return FlashMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mha_decode_backend(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        return TRTLLMHAAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mla_decode_backend(self):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )

        return TRTLLMMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_ascend_decode_backend(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnMultiStepDraftBackend,
        )

        return AscendAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_flashinfer_prefill_backend(self):
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            return FlashInferAttnBackend(self.draft_model_runner, skip_prefill=False)
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return FlashInferMLAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_triton_prefill_backend(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return TritonAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_aiter_prefill_backend(self):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        return AiterAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_fa3_prefill_backend(self):
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        return FlashAttentionBackend(self.draft_model_runner, skip_prefill=False)

    def _create_trtllm_mha_prefill_backend(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        return TRTLLMHAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_trtllm_mla_prefill_backend(self):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        return TRTLLMMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_ascend_prefill_backend(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        return AscendAttnBackend(self.draft_model_runner)

    def _create_flashmla_prefill_backend(self):
        logger.warning(
            "flashmla prefill backend is not yet supported for draft extend."
        )
        return None
