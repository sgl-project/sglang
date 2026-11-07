"""prefill admission for retained Unlimited-OCR v2 document windows"""

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder


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

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        reuse = self.tree_cache.uocr_doc_reuse
        if not reuse.is_held(req):
            return super().add_one_req(req, has_chunked_req, truncation_align_size)
        # A window held for its decoding predecessor starts only as the one
        # chunked request of an otherwise idle queue: it keeps that slot until
        # the predecessor ends. Otherwise it waits, and the queue moves on.
        if (
            not has_chunked_req
            and not self.can_run_list
            and self.waiting_queue_len <= 1
            and reuse.match_early(req, self.tree_cache)
        ):
            saved = self.rem_chunk_tokens
            cap = reuse.early_first_chunk(req)
            self.rem_chunk_tokens = cap if saved is None else min(saved, cap)
            res = super().add_one_req(req, has_chunked_req, truncation_align_size)
            if self.new_chunked_req is req:
                self.rem_chunk_tokens = 0
                return res
            self.rem_chunk_tokens = saved
            reuse.unmatch_early(req)
        return AddReqResult.CONTINUE

    def add_chunked_req(self, req):
        cap = self.tree_cache.uocr_doc_reuse.early_chunk_cap(req)
        if cap is None:
            return super().add_chunked_req(req)
        if cap > 0:
            if self.chunked_req_limit is not None:
                cap = min(cap, self.chunked_req_limit)
            self.chunked_req_limit = cap
            req = super().add_chunked_req(req)
        # A held window owns the chunked-prefill slot, whether it ran or not.
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens = 0
        return req
