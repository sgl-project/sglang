"""The PD consensus messages around each PP ring slot: the PP ranks agree on
the request ids each PD owner acts on, over the PP channel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Set

from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.runtime_context import get_disagg

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DisaggDecode
    from sglang.srt.disaggregation.prefill import DisaggPrefill
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
    from sglang.srt.managers.scheduler_components.pp_channel import PPChannel


def _send_consensus_bootstrapped_ids(
    channel: PPChannel,
    bmbs: List[List[str]],
    next_first_rank_mb_id: int,
    consensus_bootstrapped_rids: List[str],
    bootstrapped_rids: List[str],
):
    # 3 (Bootstrap): send the bootstrapped rids from the last stage to the first stage
    send_consensus_bootstrapped_work = []
    if channel.is_last_rank:
        if bmbs[next_first_rank_mb_id] is not None:
            consensus_bootstrapped_rids = bootstrapped_rids
            send_consensus_bootstrapped_work = channel.send_next(
                consensus_bootstrapped_rids
            )
    # 4 (Bootstrap): send the bootstrapped rids from a non-last rank to the next rank
    else:
        if consensus_bootstrapped_rids is not None:
            send_consensus_bootstrapped_work = channel.send_next(
                consensus_bootstrapped_rids
            )
    return send_consensus_bootstrapped_work, consensus_bootstrapped_rids


def _send_consensus_release_ids(
    channel: PPChannel,
    tmbs: List[List[str]],
    next_first_rank_mb_id: int,
    release_rids: List[str],
    transferred_rids: List[str],
):
    send_release_work = []
    if channel.is_last_rank:
        if tmbs[next_first_rank_mb_id] is not None:
            release_rids = transferred_rids
            send_release_work = channel.send_next(release_rids)
    # 4 (Release): send the release rids from non last rank to the next rank
    else:
        if release_rids is not None:
            send_release_work = channel.send_next(release_rids)
    return send_release_work, release_rids


def _route_aborts_to_bad(good_rids, bad_rids, aborted_rids):
    """Move aborted rids out of the good (intersection) set and into the
    bad (union) set, so PP consensus fails them uniformly on every rank.

    This also flushes aborted reqs that never reached good/bad consensus
    (e.g. stuck in Bootstrapping with a sender that has no working abort()).
    """
    if not aborted_rids:
        return good_rids, bad_rids
    good_rids = [rid for rid in good_rids if rid not in aborted_rids]
    bad_rids = list(set(bad_rids) | set(aborted_rids))
    return good_rids, bad_rids


class _DisaggPrefillRing:
    """The PD prefill messages around each ring slot: the PP ranks agree, in two
    passes, on the requests whose KV transfer ended before releasing them. The
    owner polls and releases; the driver callables move the core waiting list.
    """

    def __init__(
        self,
        owner: DisaggPrefill,
        channel: PPChannel,
        pp_loop_size: int,
        *,
        intake: Callable[[], None],
        schedule: Callable[[ScheduleSlot], Optional[ScheduleBatch]],
        release: Callable[[Optional[List[str]]], None],
    ):
        self.owner = owner
        self.channel = channel
        self.intake = intake
        self.schedule = schedule
        self.release = release
        # Per slot: the transferred ids it put to the consensus, or None.
        self.tmbs: List[Optional[List[str]]] = [None] * pp_loop_size
        # This slot's transferred ids, and the consensus received for the next slot.
        self.transferred_rids: Optional[List[str]] = None
        self.release_rids: Optional[List[str]] = None
        self.send_transfer_work: List[P2PWork] = []

    def _get_transferred_ids(self):
        # get the current stage transfer success
        if self.channel.is_first_rank:
            transferred_rids = self.owner.poll_transfer_rids()
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self.channel.recv_prev()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.owner.poll_transfer_rids()
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def take_ids(self, mb_id: int) -> None:
        self.intake()
        self.transferred_rids = self._get_transferred_ids()
        self.channel.commit(self.send_transfer_work)
        self.tmbs[mb_id] = self.transferred_rids

    def exchange_ids(self, next_slot: ScheduleSlot, next_first_rank_mb_id: int) -> None:
        send_release_work, _ = _send_consensus_release_ids(
            self.channel,
            self.tmbs,
            next_first_rank_mb_id,
            self.release_rids,
            self.transferred_rids,
        )
        self.release_rids = None
        if self.tmbs[next_slot.index] is not None:
            self.release_rids = self.channel.recv_prev()
        self.channel.commit(send_release_work)

    def after_commit(self, next_mb_id: int) -> None:
        if self.tmbs[next_mb_id] is not None:
            self.release(self.release_rids)

    def send_ids(self) -> None:
        self.send_transfer_work = self.channel.send_next(self.transferred_rids)


class _DisaggDecodeRing:
    """The PD decode messages around each ring slot: the PP ranks agree on the
    retracted requests to resume, the requests to preallocate, and the
    requests whose KV transfer ended. The owner polls and applies them."""

    def __init__(
        self,
        owner: DisaggDecode,
        channel: PPChannel,
        pp_loop_size: int,
        *,
        schedule: Callable[[ScheduleSlot], Optional[ScheduleBatch]],
        lora_ids_in_use: Callable[[], Set[Optional[str]]],
    ):
        self.owner = owner
        self.channel = channel
        self.schedule = schedule
        self.lora_ids_in_use = lora_ids_in_use
        # Per slot: the ids it put to each consensus, or None.
        self.rmbs: List[Optional[List[str]]] = [None] * pp_loop_size
        self.pmbs: List[Optional[List]] = [None] * pp_loop_size
        self.tmbs: List[Optional[List[str]]] = [None] * pp_loop_size
        # This slot's ids, and the consensus received for the next slot.
        self.retract_rids: Optional[List[str]] = None
        self.prealloc_rids: Optional[List] = None
        self.transferred_rids: Optional[List[str]] = None
        self.consensus_retract_rids: Optional[List[str]] = None
        self.consensus_prealloc_rids: Optional[List] = None
        self.release_rids: Optional[List[str]] = None
        self.send_retract_work: List[P2PWork] = []
        self.send_prealloc_work: List[P2PWork] = []
        self.send_transfer_work: List[P2PWork] = []

    def _get_retract_ids(self, mb_id: int):
        # communicate pre-consensus retracted reqs
        curr_retract_rids = self.owner.retracted_rids(mb_id)
        if self.channel.is_first_rank:
            # First rank, get all retracted req ids for the microbatch
            return curr_retract_rids
        else:
            # Other ranks, receive the retracted reqs info from the previous rank and ensure the consensus
            prev_retract_rids = self.channel.recv_prev()
            return list(set(prev_retract_rids) & set(curr_retract_rids))

    def _get_prealloc_ids(self):
        # communicate pre-consensus prealloc reqs
        if self.channel.is_first_rank:
            # First rank, pop the preallocated reqs from the prealloc queue
            good_prealloc_rids, bad_prealloc_rids = self.owner.poll_prealloc_rids()
        else:
            # Other ranks, receive the preallocated reqs info from the previous rank and ensure the consensus
            prev_prealloc_rids = self.channel.recv_prev()
            prev_good_prealloc_rids, prev_bad_prealloc_rids = prev_prealloc_rids
            curr_good_prealloc_rids, curr_bad_prealloc_rids = (
                self.owner.poll_prealloc_rids()
            )
            good_prealloc_rids = list(
                set(prev_good_prealloc_rids) & set(curr_good_prealloc_rids)
            )
            bad_prealloc_rids = list(
                set(prev_bad_prealloc_rids) | set(curr_bad_prealloc_rids)
            )
        # Same abort routing as the prefill bootstrap consensus above.
        aborted_rids = self.owner.aborted_prealloc_rids()
        good_prealloc_rids, bad_prealloc_rids = _route_aborts_to_bad(
            good_prealloc_rids, bad_prealloc_rids, aborted_rids
        )
        return [good_prealloc_rids, bad_prealloc_rids]

    def _get_transferred_ids(self):
        # get the current stage transfer success
        if self.channel.is_first_rank:
            transferred_rids = self.owner.poll_transfer_rids()
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self.channel.recv_prev()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.owner.poll_transfer_rids()
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def take_ids(self, mb_id: int) -> None:
        # reaching consensus through PP ranks
        self.retract_rids = self._get_retract_ids(mb_id)
        self.rmbs[mb_id] = self.retract_rids
        self.channel.commit(self.send_retract_work)

        self.prealloc_rids = self._get_prealloc_ids()
        self.pmbs[mb_id] = self.prealloc_rids
        self.channel.commit(self.send_prealloc_work)

        self.transferred_rids = self._get_transferred_ids()
        self.tmbs[mb_id] = self.transferred_rids
        self.channel.commit(self.send_transfer_work)

    def exchange_ids(self, next_slot: ScheduleSlot, next_first_rank_mb_id: int) -> None:
        """Pass this slot's ids on, and apply the next slot's consensus against
        the next slot's decoding requests: that slot schedules next and admits
        what the consensus readies."""
        owner = self.owner
        next_mb_id = next_slot.index
        # reach consensus on last rank and send to PP=0
        # otherwise, just pass along previous consensus
        send_retract_work, _ = _send_consensus_bootstrapped_ids(
            self.channel,
            self.rmbs,
            next_first_rank_mb_id,
            self.consensus_retract_rids,
            self.retract_rids,
        )
        send_prealloc_work, _ = _send_consensus_bootstrapped_ids(
            self.channel,
            self.pmbs,
            next_first_rank_mb_id,
            self.consensus_prealloc_rids,
            self.prealloc_rids,
        )
        send_release_work, _ = _send_consensus_release_ids(
            self.channel,
            self.tmbs,
            next_first_rank_mb_id,
            self.release_rids,
            self.transferred_rids,
        )

        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            owner.check_offload_progress()

        self.consensus_retract_rids = None
        if self.rmbs[next_mb_id] is not None:
            self.consensus_retract_rids = owner.apply_retract_consensus(
                self.channel.recv_prev(), next_slot
            )
        self.channel.commit(send_retract_work)

        self.consensus_prealloc_rids = None
        if self.pmbs[next_mb_id] is not None:
            self.consensus_prealloc_rids = owner.apply_prealloc_consensus(
                self.channel.recv_prev(),
                next_slot,
                lora_ids_in_use=self.lora_ids_in_use,
            )
        self.channel.commit(send_prealloc_work)

        self.release_rids = None
        if self.tmbs[next_mb_id] is not None:
            self.release_rids = owner.apply_transfer_consensus(
                self.channel.recv_prev(), next_slot
            )
        self.channel.commit(send_release_work)

    def after_commit(self, next_mb_id: int) -> None:
        pass

    def send_ids(self) -> None:
        self.send_retract_work = self.channel.send_next(self.retract_rids)
        self.send_prealloc_work = self.channel.send_next(self.prealloc_rids)
        self.send_transfer_work = self.channel.send_next(self.transferred_rids)
