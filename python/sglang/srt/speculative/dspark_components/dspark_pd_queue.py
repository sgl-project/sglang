"""Request-ordered P/D readiness for the synchronous speculative PP loop."""

from __future__ import annotations

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import (
    _apply_metadata_gate,
    _is_fake_transfer,
    _poll_with_failure_injection,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT


class DSparkPDQueueCoordinator:
    def __init__(self, world, server_args):
        self.world = world
        self.server_args = server_args

    def _exchange(self, label, value, error=None):
        records = self.world.all_gather_object((label, value, error))
        if any(record[0] != label or record[2] is not None for record in records):
            raise RuntimeError(f"DSpark PP P/D queue phase failed: {records}")
        return [record[1] for record in records]

    def agree(self, label, value):
        if any(item != value for item in self._exchange(label, value)):
            raise RuntimeError(f"DSpark PP P/D queue state differs: {label}")

    def minimum(self, label, values):
        records = self._exchange(label, tuple(values))
        if any(len(record) != len(values) for record in records):
            raise RuntimeError(f"DSpark PP P/D capacity shape differs: {label}")
        return tuple(min(items) for items in zip(*records, strict=True))

    def poll(self, label, entries, *, is_send, metadata_buffers=None, terminal=False):
        value, error = None, None
        try:
            reqs = list(entries) if is_send else [entry.req for entry in entries]
            pollers = [
                entry.disagg_kv_sender if is_send else entry.kv_receiver
                for entry in entries
            ]
            polls = _poll_with_failure_injection(pollers)
            if metadata_buffers is not None:
                _apply_metadata_gate(polls, entries, metadata_buffers, self.server_args)
            for index, (req, poll) in enumerate(zip(reqs, polls, strict=True)):
                if poll not in range(KVPoll.Failed, KVPoll.Success + 1):
                    raise ValueError("invalid P/D poll state")
                aborted = isinstance(req.finished_reason, FINISH_ABORT) and (
                    not terminal or poll in (KVPoll.Success, KVPoll.Failed)
                )
                if aborted or (
                    metadata_buffers is not None
                    and poll == KVPoll.Success
                    and not _is_fake_transfer(req, self.server_args)
                    and metadata_buffers.bootstrap_room[
                        entries[index].metadata_buffer_index, 0
                    ].item()
                    != (req.bootstrap_room or 0)
                ):
                    polls[index] = KVPoll.Failed
            value = (tuple((req.rid, req.bootstrap_room) for req in reqs), polls)
        except Exception as failure:  # noqa: BLE001 - Every stage must reach the fence.
            error = f"{type(failure).__name__}: {failure}"
        records = self._exchange(label, value, error)
        identities, _ = records[0]
        if any(record[0] != identities for record in records):
            raise RuntimeError(f"DSpark PP P/D request order differs: {label}")
        states = zip(*(r[1] for r in records), strict=True)
        # One failed stage cannot release another stage's live transfer buffers.
        # Keep the existing PP all-terminal rule before consuming transfer queues.
        return [
            (
                KVPoll.Transferring
                if terminal
                and any(item not in (KVPoll.Success, KVPoll.Failed) for item in items)
                else min(items)
            )
            for items in states
        ]
