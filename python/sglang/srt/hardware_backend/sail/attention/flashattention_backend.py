from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.hardware_backend.sail.attention.flash_attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMultiStepBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class SailFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention backend running on the SAIL FA3 build."""

    def _init_flash_attention_functions(self, device_capability):
        if self.fa_impl_ver != 3:
            raise NotImplementedError(
                f"FA{self.fa_impl_ver} is not supported on SAIL; use fa3."
            )
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self._get_scheduler_metadata = get_scheduler_metadata
        self._get_fa_runtime_policy = None


class SailFlashAttentionMultiStepBackend(FlashAttentionMultiStepBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        fa_impl_ver: int = 3,
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.kv_index_translator = model_runner.kv_index_translator
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                SailFlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    fa_impl_ver=fa_impl_ver,
                )
            )
        self.attn_backend_list = self.attn_backends
        self.forward_metadata = None
