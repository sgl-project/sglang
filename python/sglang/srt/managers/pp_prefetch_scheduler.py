"""PP-loop agreement for the native, ordered prefetch ACK stream.

Query still uses the controller's background collective. Completion only uses
TP/CP collectives there; the scheduler folds ACKs hop by hop and publishes the
common prefix on the next forward pass. No request queue, cache lease, capacity
reservation, policy evaluator or allocator is owned here.
"""

from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PrefetchResult:
    """Wire data only; each rank keeps its own native operation and buffers."""

    completed_tokens: int | None
    pool_hits: dict[str, int] | None
    completed_req: bool | None

    @classmethod
    def from_ack(cls, ack):
        return cls(ack.completed_tokens, ack.pool_hits, ack.completed_req)

    def fold(self, ack):
        # Query/allocation and the IO loop already establish ACK order and kind,
        # including the final ACK after local IO has stopped touching buffers.
        return PrefetchResult(
            min(self.completed_tokens, ack.completed_tokens)
            if self.completed_tokens is not None
            else None,
            {
                pool: min(self.pool_hits.get(pool, 0), ack.pool_hits.get(pool, 0))
                for pool in self.pool_hits.keys() | ack.pool_hits.keys()
            }
            if self.pool_hits is not None
            else None,
            self.completed_req,
        )


@dataclass(frozen=True)
class PrefetchUpdate:
    committed: tuple[PrefetchResult, ...] = ()
    # None: no proposal this beat. Empty tuple: a peer has no ACKs yet.
    proposed: tuple[PrefetchResult, ...] | None = None
    # Native storage reset/detach can overtake already-sent numeric results.
    generation: int = 0


class PPPrefetchConsensus:
    def __init__(self, controller):
        self.controller = controller
        self.rank = dist.get_rank(controller.pp_group)
        self.pending = deque()
        self.inflight = False
        self.tick = 0
        self.send_work = deque()
        self.generation = 0

    def reset(self):
        """Called only after native storage workers have stopped.

        Native reset/detach owns operation and buffer cleanup. Discard pending
        ACKs here; detach does not consume the native prefetch ACK queue.
        Keep transport cadence: old packets still need receiving, not applying.
        """
        cc = self.controller
        while not cc.pp_prefetch_ack_queue.empty():
            cc.pp_prefetch_ack_queue.get_nowait()
        self.pending.clear()
        self.inflight = False
        self.generation += 1

    def _drain_local(self):
        cc = self.controller
        count = torch.tensor(cc.pp_prefetch_ack_queue.qsize(), dtype=torch.int)
        groups = (
            (cc.attn_cp_group, cc.attn_tp_group)
            if cc.attn_cp_group is not None or cc.attn_tp_group is not None
            else (cc.tp_group,)
        )
        # These are scheduler groups, not the background completion groups.
        for group in groups:
            if group is not None and dist.get_world_size(group) > 1:
                dist.all_reduce(count, op=dist.ReduceOp.MIN, group=group)
        for _ in range(count.item()):
            self.pending.append(cc.pp_prefetch_ack_queue.get_nowait())

    def _publish(self, results):
        for result in results:
            ack = self.pending.popleft()
            ack.completed_tokens = result.completed_tokens
            ack.pool_hits = result.pool_hits
            self.controller._publish_prefetch_ack(ack)

    def advance(self, receive, send, return_delay):
        """One hop per PP beat, including idle beats. Never wait for local IO.

        PP0 receives the return from pp_loop_size beats ago, after the native
        pipeline has needed that beat's output, never the current fold.
        Peers truncate a proposal to their currently available ACKs;
        the remainder stays pending for a later trip. The commit precedes the
        native cache-event drain on every stage, preserving its PP0-led order.
        """
        self._drain_local()
        if self.rank == 0:
            committed = ()
            if self.tick >= return_delay:
                returned = receive()
                if (
                    returned.generation == self.generation
                    and returned.proposed is not None
                ):
                    committed = returned.proposed
                    self._publish(committed)
                    self.inflight = False
            proposed = None
            if self.pending and not self.inflight:
                proposed = tuple(PrefetchResult.from_ack(a) for a in self.pending)
                self.inflight = True
            update = PrefetchUpdate(committed, proposed, self.generation)
        else:
            update = receive()
            if update.generation != self.generation:
                update = PrefetchUpdate(generation=update.generation)
            else:
                self._publish(update.committed)
            if update.proposed is not None:
                update = PrefetchUpdate(
                    update.committed,
                    tuple(p.fold(a) for p, a in zip(update.proposed, self.pending)),
                    update.generation,
                )

        # In particular, the last stage must not wait for PP0 to receive the
        # preceding return before it can compute the next microbatch.
        if len(self.send_work) == return_delay:
            for work in self.send_work.popleft():
                work.work.wait()
        self.send_work.append(send(update))
        self.tick += 1
