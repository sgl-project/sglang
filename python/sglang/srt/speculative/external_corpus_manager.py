"""Manages external SAM corpora for ngram speculative decoding.

Handles add/remove/list operations and async background loading.
Used by the Scheduler — not a mixin, a standalone manager object.
"""

import logging
import threading
from typing import Callable, Optional, Tuple

import torch

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
)
from sglang.srt.runtime_context import get_parallel

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

    @staticmethod
    def _all_tp_ranks_agree(local_flag: bool) -> bool:
        """Return True only if ``local_flag`` is True on every TP rank."""
        tp_group = get_parallel().tp_group
        if tp_group.world_size <= 1:
            return local_flag
        flag = torch.tensor([int(local_flag)], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=tp_group.cpu_group
        )
        return bool(flag.item())

    def check_pending_load(self):
        """Poll from the scheduler event loop. Sends response when done.

        Control requests are broadcast to every TP rank, so ``_pending_load``
        is set in lockstep on all ranks, but the background build threads
        finish at different times. The load is only committed (and
        ``_pending_load`` cleared) once the thread has finished on every TP
        rank, so the pending state stays identical across ranks and ``remove``
        is accepted or rejected consistently.
        """
        if self._pending_load is None:
            return
        recv_req, thread = self._pending_load
        load_finished = not thread.is_alive()
        if not self._all_tp_ranks_agree(load_finished):
            return
        self._pending_load = None
        thread.join()  # formal happens-before for _load_result visibility
        result = self._load_result
        self._load_result = None
        if result.success:
            self._worker.commit_corpus_load(result.corpus_id, result.loaded_token_count)
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
        # A finished thread may still have uncommitted corpus bookkeeping.
        # _pending_load is cleared on all TP ranks in the same scheduler
        # iteration (see check_pending_load), so this check agrees across ranks.
        if self._pending_load is not None:
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Cannot remove a corpus while a corpus load is pending.",
            )
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
            token_counts = self._worker.list_external_corpora()
            return ListExternalCorporaReqOutput(
                success=True,
                corpus_token_counts=token_counts,
            )
        except Exception as e:
            return ListExternalCorporaReqOutput(success=False, message=str(e))
