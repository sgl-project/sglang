from __future__ import annotations


class PrefillAdderDllmMixin:
    def add_dllm_prompt_cache_req(self, req):
        from sglang.srt.managers.schedule_policy import (
            CLIP_MAX_NEW_TOKENS,
            AddReqResult,
        )

        total_tokens = req.extend_range.length + min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )

        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        input_tokens = self.ceil_paged_tokens(req.extend_range.length)

        if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        self.can_run_list.append(req)
        self._update_prefill_budget(
            len(req.prefix_indices),
            input_tokens,
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
            req.retracted_stain,
        )
        return AddReqResult.CONTINUE
