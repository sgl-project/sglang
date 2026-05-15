from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
    iter_control_batch_messages,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftTailSnapshot:
    """Stable verifier-side snapshot of the currently consumable draft tail."""

    request_id: str
    committed_len: int
    tail_tokens: list[int]
    raw_tail_len: int = 0
    raw_tail_tokens: list[int] = field(default_factory=list)


@dataclass
class RequestDraftTailState:
    """Store verifier-visible rolling draft tail metadata for one request."""

    drafter_rank: int
    # The verifier committed output length tracked by DraftSync/VerifyCommit.
    committed_len: int = 0
    # The only stale base allowed to append when bonus matches preserve a suffix.
    can_accept_prefix_len: int = 0
    tail_tokens: list[int] = field(default_factory=list)

    def consumable_tail_tokens(self) -> list[int]:
        return list(self.tail_tokens[:-1])

    def consumable_tail_len(self) -> int:
        return max(0, len(self.tail_tokens) - 1)


class DraftTailBuffer:
    """Verifier-side rolling draft tail state shared by scheduler and proxy."""

    def __init__(
        self,
        *,
        verifier_rank: int,
        required_tail_len: int,
    ) -> None:
        self.verifier_rank = int(verifier_rank)
        self.required_tail_len = max(0, int(required_tail_len))
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self._states: dict[str, RequestDraftTailState] = {}

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._states.clear()
            self._condition.notify_all()

    def has_request(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._states

    def get_committed_len(self, request_id: str) -> int | None:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return None
            return int(state.committed_len)

    def open_requests(self, messages: list[DraftSync]) -> None:
        if not messages:
            return
        with self._condition:
            for message in messages:
                self._open_request_locked(message)
            self._condition.notify_all()

    def _open_request_locked(self, message: DraftSync) -> None:
        committed_len = len(message.committed_output_ids)
        self._states[message.request_id] = RequestDraftTailState(
            drafter_rank=int(message.dst_drafter_rank),
            committed_len=committed_len,
            can_accept_prefix_len=committed_len,
        )

    def apply_verify_commits(self, messages: list[VerifyCommit]) -> None:
        if not messages:
            return
        with self._condition:
            for message in messages:
                self._apply_commit_locked(message)
            self._condition.notify_all()

    def _apply_commit_locked(self, message: VerifyCommit) -> dict | None:
        state = self._states.get(message.request_id)
        old_committed_len = int(message.pre_verify_committed_len)
        new_committed_len = int(message.bonus_token_pos) + 1
        # number of accepted tail draft tokens, bonus token not included
        accepted_tail_len = max(0, int(message.bonus_token_pos) - old_committed_len)
        if state is None:
            return {
                "request_id": message.request_id,
                "pre_committed_len": old_committed_len,
                "accepted_tail_len": accepted_tail_len,
                "raw_tail_len_before": 0,
                "bonus_token_id": int(message.bonus_token_id),
                "buffer_candidate_token_id": -1,
                "bonus_match": False,
                "preserved_suffix_len": 0,
                "tail_len_after": 0,
                "committed_len_after": 0,
            }

        # old_committed_len is the verifier output length before the forward
        # pass that produced this VerifyCommit. tail_tokens[0] corresponds to
        # absolute output position old_committed_len.
        raw_tail_len_before = len(state.tail_tokens)
        bonus_candidate_exists = 0 <= accepted_tail_len < raw_tail_len_before
        buffer_candidate_token_id = (
            int(state.tail_tokens[accepted_tail_len]) if bonus_candidate_exists else -1
        )
        bonus_match = bonus_candidate_exists and buffer_candidate_token_id == int(
            message.bonus_token_id
        )

        if state.tail_tokens and not (0 <= accepted_tail_len < len(state.tail_tokens)):
            raise RuntimeError(
                "Decoupled verify consumed all buffered draft tokens without a "
                "reserved bonus-token anchor: "
                f"request_id={message.request_id} "
                f"pre_verify_committed_len={old_committed_len} "
                f"bonus_token_pos={int(message.bonus_token_pos)} "
                f"accepted_tail_len={accepted_tail_len} "
                f"buffered_tail_len={len(state.tail_tokens)}"
            )

        remaining: list[int] = []
        can_accept_prefix_len = new_committed_len
        if bonus_match:
            remaining = state.tail_tokens[accepted_tail_len + 1 :]
            # if the bonus token matches, the verifier do not need to truncate the tail
            # Keep the previously accepted stale base. The preserved suffix may
            # itself have come from an older base, and in-flight stream outputs
            # from that base can still append contiguously.
            can_accept_prefix_len = int(state.can_accept_prefix_len)

        state.committed_len = new_committed_len
        state.can_accept_prefix_len = can_accept_prefix_len
        state.tail_tokens = remaining
        return {
            "request_id": message.request_id,
            "pre_committed_len": old_committed_len,
            "accepted_tail_len": accepted_tail_len,
            "raw_tail_len_before": raw_tail_len_before,
            "bonus_token_id": int(message.bonus_token_id),
            "buffer_candidate_token_id": buffer_candidate_token_id,
            "bonus_match": bonus_match,
            "preserved_suffix_len": len(remaining),
            "tail_len_after": len(state.tail_tokens),
            "committed_len_after": int(state.committed_len),
        }

    def close_requests(self, messages: list[DraftClose]) -> None:
        if not messages:
            return
        with self._condition:
            for message in messages:
                self._close_request_locked(message)
            self._condition.notify_all()

    def _close_request_locked(self, message: DraftClose) -> None:
        self._states.pop(message.request_id, None)

    def apply_control_batch(
        self,
        batch: DraftControlBatch,
        *,
        collect_stats: bool = False,
    ) -> dict | None:
        commit_stats: list[dict] = []
        with self._condition:
            for message in iter_control_batch_messages(batch):
                if isinstance(message, DraftSync):
                    self._open_request_locked(message)
                elif isinstance(message, VerifyCommit):
                    commit_stat = self._apply_commit_locked(message)
                    if collect_stats and commit_stat is not None:
                        commit_stats.append(commit_stat)
                elif isinstance(message, DraftClose):
                    self._close_request_locked(message)
            self._condition.notify_all()
        if not collect_stats:
            return None
        return {
            "commit_rids": [item["request_id"] for item in commit_stats],
            "pre_committed_lens_by_req": [
                item["pre_committed_len"] for item in commit_stats
            ],
            "accepted_tail_lens_by_req": [
                item["accepted_tail_len"] for item in commit_stats
            ],
            "raw_tail_lens_before_by_req": [
                item["raw_tail_len_before"] for item in commit_stats
            ],
            "bonus_token_ids_by_req": [item["bonus_token_id"] for item in commit_stats],
            "buffer_candidate_token_ids_by_req": [
                item["buffer_candidate_token_id"] for item in commit_stats
            ],
            "bonus_match_by_req": [item["bonus_match"] for item in commit_stats],
            "preserved_suffix_lens_by_req": [
                item["preserved_suffix_len"] for item in commit_stats
            ],
            "tail_lens_after_by_req": [item["tail_len_after"] for item in commit_stats],
            "committed_lens_after_by_req": [
                item["committed_len_after"] for item in commit_stats
            ],
        }

    def append_draft_stream_batch(
        self,
        batch: DraftTailStreamOutputBatch,
        *,
        collect_stats: bool = False,
    ) -> dict | None:
        if not batch.outputs:
            return None
        append_stats = self._new_append_stats(batch) if collect_stats else None
        with self._condition:
            for output in batch.outputs:
                result = self._push_one_locked(batch, output)
                if append_stats is not None:
                    self._record_append_result_locked(append_stats, output, result)
            if append_stats is not None:
                self._fill_append_after_lens_locked(append_stats)
            self._condition.notify_all()
        return append_stats

    def _push_one_locked(
        self,
        batch: DraftTailStreamOutputBatch,
        output: DraftTailStreamOutput,
    ) -> str:
        request_id = output.request_id
        base_committed_len = int(output.base_committed_len)
        token_pos = int(output.new_token_pos)
        token_id = int(output.new_token_id)
        src_drafter_rank = int(output.src_drafter_rank)
        dst_verifier_rank = int(output.dst_verifier_rank)
        batch_request_ids = [item.request_id for item in batch.outputs]
        batch_base_committed_lens = [
            int(item.base_committed_len) for item in batch.outputs
        ]
        batch_new_token_pos = [int(item.new_token_pos) for item in batch.outputs]
        batch_new_token_id = [int(item.new_token_id) for item in batch.outputs]

        if dst_verifier_rank != self.verifier_rank:
            raise RuntimeError(
                "Draft stream output targets the wrong verifier: "
                f"request_id={request_id} "
                f"verifier_rank={self.verifier_rank} "
                f"src_drafter_rank={src_drafter_rank} "
                f"dst_verifier_rank={dst_verifier_rank} "
                f"base_committed_len={base_committed_len} "
                f"token_pos={token_pos} "
                f"token_id={token_id} "
                f"batch_request_ids={batch_request_ids} "
                f"batch_base_committed_lens={batch_base_committed_lens} "
                f"batch_new_token_pos={batch_new_token_pos} "
                f"batch_new_token_id={batch_new_token_id}"
            )

        state = self._states.get(request_id)
        if state is None:
            # the verifier may have already closed this request, and the drafter still sends stream outputs
            return "unknown_request"

        state_committed_len = int(state.committed_len)
        can_accept_prefix_len = int(state.can_accept_prefix_len)
        tail_len_before = len(state.tail_tokens)
        # tail_tokens[0] corresponds to absolute position state.committed_len.
        # buffer_end_len is the first absolute position not yet present in the
        # buffer, so a normal append must use token_pos == buffer_end_len.
        buffer_end_len = state_committed_len + tail_len_before

        if src_drafter_rank != int(state.drafter_rank):
            raise RuntimeError(
                "Unexpected draft stream drafter rank: "
                f"request_id={request_id} "
                f"verifier_rank={self.verifier_rank} "
                f"src_drafter_rank={src_drafter_rank} "
                f"expected_drafter_rank={state.drafter_rank} "
                f"base_committed_len={base_committed_len} "
                f"state_committed_len={state_committed_len} "
                f"can_accept_prefix_len={can_accept_prefix_len} "
                f"token_pos={token_pos} "
                f"token_id={token_id} "
                f"tail_len_before={tail_len_before} "
                f"buffer_end_len={buffer_end_len} "
                f"tail_tokens={list(state.tail_tokens)} "
                f"batch_request_ids={batch_request_ids} "
                f"batch_base_committed_lens={batch_base_committed_lens} "
                f"batch_new_token_pos={batch_new_token_pos} "
                f"batch_new_token_id={batch_new_token_id}"
            )

        if base_committed_len > state_committed_len:
            raise RuntimeError(
                "Draft stream base is ahead of verifier state: "
                f"request_id={request_id} "
                f"verifier_rank={self.verifier_rank} "
                f"src_drafter_rank={src_drafter_rank} "
                f"expected_drafter_rank={state.drafter_rank} "
                f"base_committed_len={base_committed_len} "
                f"state_committed_len={state_committed_len} "
                f"can_accept_prefix_len={can_accept_prefix_len} "
                f"token_pos={token_pos} "
                f"token_id={token_id} "
                f"tail_len_before={tail_len_before} "
                f"buffer_end_len={buffer_end_len} "
                f"tail_tokens={list(state.tail_tokens)} "
                f"batch_request_ids={batch_request_ids} "
                f"batch_base_committed_lens={batch_base_committed_lens} "
                f"batch_new_token_pos={batch_new_token_pos} "
                f"batch_new_token_id={batch_new_token_id}"
            )

        if base_committed_len < can_accept_prefix_len:
            return "stale_base"

        if token_pos < state_committed_len:
            return "already_committed"

        if token_pos < buffer_end_len:
            existing_token_id = int(state.tail_tokens[token_pos - state_committed_len])
            if existing_token_id != token_id:
                raise RuntimeError(
                    "Draft stream token conflicts with buffered tail: "
                    f"request_id={request_id} "
                    f"verifier_rank={self.verifier_rank} "
                    f"src_drafter_rank={src_drafter_rank} "
                    f"expected_drafter_rank={state.drafter_rank} "
                    f"base_committed_len={base_committed_len} "
                    f"state_committed_len={state_committed_len} "
                    f"can_accept_prefix_len={can_accept_prefix_len} "
                    f"token_pos={token_pos} "
                    f"token_id={token_id} "
                    f"existing_token_id={existing_token_id} "
                    f"tail_len_before={tail_len_before} "
                    f"buffer_end_len={buffer_end_len} "
                    f"tail_tokens={list(state.tail_tokens)} "
                    f"batch_request_ids={batch_request_ids} "
                    f"batch_base_committed_lens={batch_base_committed_lens} "
                    f"batch_new_token_pos={batch_new_token_pos} "
                    f"batch_new_token_id={batch_new_token_id}"
                )
            return "duplicate"

        if token_pos > buffer_end_len:
            if base_committed_len == state_committed_len:
                raise RuntimeError(
                    "Draft stream token skips buffered tail: "
                    f"request_id={request_id} "
                    f"verifier_rank={self.verifier_rank} "
                    f"src_drafter_rank={src_drafter_rank} "
                    f"expected_drafter_rank={state.drafter_rank} "
                    f"base_committed_len={base_committed_len} "
                    f"state_committed_len={state_committed_len} "
                    f"can_accept_prefix_len={can_accept_prefix_len} "
                    f"token_pos={token_pos} "
                    f"token_id={token_id} "
                    f"tail_len_before={tail_len_before} "
                    f"buffer_end_len={buffer_end_len} "
                    f"tail_tokens={list(state.tail_tokens)} "
                    f"batch_request_ids={batch_request_ids} "
                    f"batch_base_committed_lens={batch_base_committed_lens} "
                    f"batch_new_token_pos={batch_new_token_pos} "
                    f"batch_new_token_id={batch_new_token_id}"
                )
            return "stale_gap"

        state.tail_tokens.append(token_id)
        return "appended"

    def _new_append_stats(self, batch: DraftTailStreamOutputBatch) -> dict:
        request_ids: list[str] = []
        index_by_request_id: dict[str, int] = {}
        for output in batch.outputs:
            if output.request_id in index_by_request_id:
                continue
            index_by_request_id[output.request_id] = len(request_ids)
            request_ids.append(output.request_id)
        return {
            "rids": request_ids,
            "_index_by_request_id": index_by_request_id,
            "draft_token_lens_by_req": [0] * len(request_ids),
            "appended_token_lens_by_req": [0] * len(request_ids),
            "num_appended_outputs": 0,
            "num_duplicate_outputs": 0,
            "num_stale_base_outputs": 0,
            "num_already_committed_outputs": 0,
            "num_stale_gap_outputs": 0,
            "num_unknown_request_outputs": 0,
        }

    def _record_append_result_locked(
        self,
        append_stats: dict,
        output: DraftTailStreamOutput,
        result: str,
    ) -> None:
        index = append_stats["_index_by_request_id"][output.request_id]
        append_stats["draft_token_lens_by_req"][index] += 1
        if result == "appended":
            append_stats["num_appended_outputs"] += 1
            append_stats["appended_token_lens_by_req"][index] += 1
        elif result == "duplicate":
            append_stats["num_duplicate_outputs"] += 1
        elif result == "stale_base":
            append_stats["num_stale_base_outputs"] += 1
        elif result == "already_committed":
            append_stats["num_already_committed_outputs"] += 1
        elif result == "stale_gap":
            append_stats["num_stale_gap_outputs"] += 1
        elif result == "unknown_request":
            append_stats["num_unknown_request_outputs"] += 1
        else:
            raise RuntimeError(f"Unexpected draft stream append result: {result}")

    def _fill_append_after_lens_locked(self, append_stats: dict) -> None:
        tail_lens_after_by_req: list[int] = []
        consumable_tail_lens_after_by_req: list[int] = []
        committed_lens_after_by_req: list[int] = []
        for request_id in append_stats["rids"]:
            state = self._states.get(request_id)
            if state is None:
                tail_lens_after_by_req.append(0)
                consumable_tail_lens_after_by_req.append(0)
                committed_lens_after_by_req.append(0)
                continue
            tail_lens_after_by_req.append(len(state.tail_tokens))
            consumable_tail_lens_after_by_req.append(state.consumable_tail_len())
            committed_lens_after_by_req.append(int(state.committed_len))
        append_stats["tail_lens_after_by_req"] = tail_lens_after_by_req
        append_stats["consumable_tail_lens_after_by_req"] = (
            consumable_tail_lens_after_by_req
        )
        append_stats["committed_lens_after_by_req"] = committed_lens_after_by_req
        append_stats.pop("_index_by_request_id", None)

    def _has_min_draft_tokens_locked(
        self, rids: list[str], min_draft_tokens: int
    ) -> bool:
        min_draft_tokens = max(0, int(min_draft_tokens))
        for rid in rids:
            state = self._states.get(rid)
            assert state, f"unexpected request_id={rid}"
            if len(state.tail_tokens) < min_draft_tokens:
                return False
        return True

    def _wait_for_draft_tokens_locked(
        self, rids: list[str], min_draft_tokens: int
    ) -> None:
        min_draft_tokens = max(0, int(min_draft_tokens))
        if min_draft_tokens <= 0:
            return
        while not self._closed and not self._has_min_draft_tokens_locked(
            rids, min_draft_tokens
        ):
            self._condition.wait()
        if self._closed:
            raise RuntimeError(
                "DraftTailBuffer closed while waiting for draft tail tokens."
            )

    def wait_for_draft_tokens(self, rids: list[str], min_draft_tokens: int) -> None:
        """Wait until every request has at least N raw draft tokens buffered."""
        with self._condition:
            self._wait_for_draft_tokens_locked(rids, min_draft_tokens)

    def get_draft_snapshots(
        self,
        reqs: list,
        *,
        allow_partial: bool = True,
        include_raw_tail_tokens: bool = False,
    ) -> list[DraftTailSnapshot]:
        with self._condition:
            if not allow_partial:
                min_raw_tail_len = (
                    self.required_tail_len + 1 if self.required_tail_len > 0 else 1
                )
                self._wait_for_draft_tokens_locked(
                    [req.rid for req in reqs], min_raw_tail_len
                )

            snapshots: list[DraftTailSnapshot] = []
            for req in reqs:
                state = self._states.get(req.rid)
                assert state, f"unexpected request_id={req.rid}"
                snapshots.append(
                    DraftTailSnapshot(
                        request_id=req.rid,
                        committed_len=int(state.committed_len),
                        tail_tokens=state.consumable_tail_tokens(),
                        raw_tail_len=len(state.tail_tokens),
                        raw_tail_tokens=(
                            list(state.tail_tokens) if include_raw_tail_tokens else []
                        ),
                    )
                )
            return snapshots
