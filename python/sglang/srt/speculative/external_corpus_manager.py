"""Manages external SAM corpora for ngram speculative decoding.

Handles add/remove/list operations and async background loading.
Used by the Scheduler — not a mixin, a standalone manager object.
"""

import logging
import threading
from typing import Callable, Optional, Tuple

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
)

logger = logging.getLogger(__name__)


class ExternalCorpusManager:
    """Manages external SAM corpus lifecycle for a single scheduler.

    Args:
        draft_worker: the NGRAMWorker instance (must have add_external_corpus,
            remove_external_corpus, list_external_corpora methods).
        send_response: callable(output, recv_req) to send deferred responses
            back to the tokenizer manager.
    """

    def __init__(self, draft_worker, send_response: Callable):
        self._worker = draft_worker
        self._send_response = send_response
        self._pending_load: Optional[
            Tuple[AddExternalCorpusReqInput, threading.Thread]
        ] = None
        self._load_result: Optional[AddExternalCorpusReqOutput] = None

    def check_pending_load(self):
        """Poll from the scheduler event loop. Sends response when done."""
        if self._pending_load is None:
            return
        recv_req, thread = self._pending_load
        if thread.is_alive():
            return
        self._pending_load = None
        thread.join()  # formal happens-before for _load_result visibility
        result = self._load_result
        self._load_result = None
        self._send_response(result, recv_req)

    def add(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        if self._pending_load is not None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Another corpus load is already in progress.",
            )

        def _build():
            try:
                loaded = self._worker.add_external_corpus(
                    recv_req.corpus_id, recv_req.token_chunks
                )
                self._load_result = AddExternalCorpusReqOutput(
                    success=True,
                    corpus_id=recv_req.corpus_id,
                    message=f"Loaded corpus '{recv_req.corpus_id}' with {loaded} tokens.",
                    loaded_token_count=loaded,
                )
            except Exception as e:
                self._load_result = AddExternalCorpusReqOutput(
                    success=False, message=str(e)
                )

        thread = threading.Thread(target=_build, daemon=True)
        self._pending_load = (recv_req, thread)
        thread.start()
        return None  # response sent later by check_pending_load

    def remove(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        try:
            self._worker.remove_external_corpus(recv_req.corpus_id)
            return RemoveExternalCorpusReqOutput(
                success=True,
                message=f"Removed corpus '{recv_req.corpus_id}'.",
            )
        except Exception as e:
            return RemoveExternalCorpusReqOutput(success=False, message=str(e))

    def list(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        try:
            ids = self._worker.list_external_corpora()
            return ListExternalCorporaReqOutput(success=True, corpus_ids=ids)
        except Exception as e:
            return ListExternalCorporaReqOutput(success=False, message=str(e))
