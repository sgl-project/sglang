"""Manages external SAM corpora for ngram speculative decoding.

Handles add/remove/list operations and async background loading.
Used by the Scheduler — not a mixin, a standalone manager object.
"""

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


class ExternalCorpusManager:
    """Manages external SAM corpus lifecycle for a single scheduler.

    Args:
        draft_worker: the NGRAMWorker instance (must have add_external_corpus,
            remove_external_corpus, list_external_corpora methods).
        send_response: callable(output, recv_req) to send deferred responses
            back to the tokenizer manager.
        synchronize_load_result: callable that aggregates a completed
            local load across worker ranks before it is committed and returned.
        cancel_load: callable that discards the worker's unpublished staging
            corpus after a distributed build or commit failure.
    """

    def __init__(
        self,
        draft_worker,
        send_response: Callable,
        synchronize_load_result: Callable,
        cancel_load: Callable,
    ):
        self._worker = draft_worker
        self._send_response = send_response
        self._synchronize_load_result = synchronize_load_result
        self._cancel_load = cancel_load
        self._pending_load: Optional[
            Tuple[AddExternalCorpusReqInput, threading.Thread]
        ] = None
        self._load_result: Optional[AddExternalCorpusReqOutput] = None

    def check_pending_load(self):
        """Coordinate, publish, and reply after a background build finishes."""
        if self._pending_load is None:
            return
        recv_req, thread = self._pending_load
        corpus_id = recv_req.corpus_id or ""
        local_result = None
        if not thread.is_alive():
            thread.join()  # formal happens-before for _load_result visibility
            local_result = self._load_result
        # Every participating scheduler rank calls the synchronizer on every
        # pending poll, including ranks whose thread is still alive. This keeps
        # distributed collective order identical across ranks.
        result = self._synchronize_load_result(local_result)
        if result is None:
            return

        if not result.success:
            # A locally successful build is still staging-only. Discard staging
            # on every rank so the next load starts from the same clean state.
            cleanup = self._cancel_staged_load(corpus_id)
            cleanup_result = self._coordinate(cleanup)
            assert cleanup_result is not None
            if not cleanup_result.success:
                result.message += f" Cleanup also failed: {cleanup_result.message}"
            self._complete(result, recv_req)
            return

        assert local_result is not None
        try:
            self._worker.commit_corpus_load(corpus_id, local_result.loaded_token_count)
            local_commit = AddExternalCorpusReqOutput(
                success=True,
                corpus_id=corpus_id,
                message="External corpus commit succeeded.",
                loaded_token_count=local_result.loaded_token_count,
            )
        except Exception as e:
            local_commit = AddExternalCorpusReqOutput(
                success=False,
                corpus_id=corpus_id,
                message=f"External corpus commit failed: {e}",
            )

        commit_result = self._coordinate(local_commit)
        assert commit_result is not None
        if commit_result.success:
            # The commit-result collective is also the publication fence: no
            # rank returns to inference until every rank has published.
            self._complete(result, recv_req)
            return

        # Roll back ranks that published and cancel staging on ranks whose
        # commit failed. Synchronize cleanup before any scheduler can resume.
        cleanup = self._rollback_local_commit(corpus_id, local_commit.success)
        # This coordination is the cleanup fence, not a substitute for checking
        # whether rollback succeeded on every rank.
        cleanup_result = self._coordinate(cleanup)
        assert cleanup_result is not None
        if not cleanup_result.success:
            commit_result.message += f" Cleanup also failed: {cleanup_result.message}"
        self._complete(commit_result, recv_req)

    def _coordinate(
        self, result: Optional[AddExternalCorpusReqOutput]
    ) -> Optional[AddExternalCorpusReqOutput]:
        return self._synchronize_load_result(result)

    def _cancel_staged_load(self, corpus_id: str) -> AddExternalCorpusReqOutput:
        try:
            self._cancel_load(corpus_id)
            return AddExternalCorpusReqOutput(
                success=True,
                corpus_id=corpus_id,
                message="External corpus staging was cancelled.",
            )
        except Exception as e:
            return AddExternalCorpusReqOutput(
                success=False,
                corpus_id=corpus_id,
                message=f"Failed to cancel external corpus staging: {e}",
            )

    def _rollback_local_commit(
        self, corpus_id: str, was_published: bool
    ) -> AddExternalCorpusReqOutput:
        try:
            if was_published:
                self._worker.remove_external_corpus(corpus_id)
            else:
                self._cancel_load(corpus_id)
            return AddExternalCorpusReqOutput(
                success=True,
                corpus_id=corpus_id,
                message="External corpus rollback succeeded.",
            )
        except Exception as e:
            return AddExternalCorpusReqOutput(
                success=False,
                corpus_id=corpus_id,
                message=f"External corpus rollback failed: {e}",
            )

    def _complete(
        self,
        result: AddExternalCorpusReqOutput,
        recv_req: AddExternalCorpusReqInput,
    ) -> None:
        self._pending_load = None
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
        if (
            self._pending_load is not None
            and self._pending_load[0].corpus_id == recv_req.corpus_id
        ):
            return RemoveExternalCorpusReqOutput(
                success=False,
                message=(
                    f"Cannot remove corpus '{recv_req.corpus_id}' while its load "
                    "is pending."
                ),
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
