"""prefill admission for retained Unlimited-OCR v2 document windows"""

from sglang.srt.managers.schedule_policy import PrefillAdder


class UocrPrefillAdder(PrefillAdder):
    @staticmethod
    def release_request_slot(tree_cache, waiting_queue) -> bool:
        return tree_cache.uocr_doc_reuse.free_for_admission(waiting_queue)

    def should_defer(self, req) -> bool:
        return self.tree_cache.uocr_doc_reuse.defer_new_document(
            req=req,
            adder=self,
            nothing_running=self.running_batch.is_empty() and not self.can_run_list,
        )

    def batch_is_full_after_rejection(self, req) -> bool:
        full = super().batch_is_full_after_rejection(req)
        if self.tree_cache.uocr_doc_reuse.on_no_token(
            req=req, adder=self, nothing_running=self.running_batch.is_empty()
        ):
            return False
        return full
