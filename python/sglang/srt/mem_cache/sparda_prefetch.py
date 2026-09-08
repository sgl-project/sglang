"""One-layer SparDA KV prefetch runtime.

The runtime deliberately owns only asynchronous transfer ordering and the
lifetime of a page lease.  A cache implementation resolves logical predicted
blocks into host/device indices and supplies the corresponding lease.  This
keeps the protocol usable by HiCache without teaching the attention backend
about host-pool layout details.

The safety rule is simple: a ticket is not released until its completion event
has been synchronized.  Therefore cancellation, request cleanup, and page
reuse cannot expose a destination page while a previous copy is still in
flight.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
from typing import Any, Callable, Optional, Protocol, Sequence

from sglang.srt.mem_cache.l2_transfer import L2Transfer, TransferCompletion


class PrefetchTicketState(Enum):
    """Lifecycle states visible to the attention/cache integration."""

    SUBMITTED = auto()
    READY = auto()
    CONSUMED = auto()
    CANCELLED = auto()
    RELEASED = auto()


class PageLease(Protocol):
    """Lease held while a transfer may still write destination pages."""

    def release(self) -> None:
        """Make the leased pages eligible for reuse."""


@dataclass
class CallbackPageLease:
    """Small adapter for allocators that expose acquire/release callbacks."""

    release_callback: Callable[[], None]
    _released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self.release_callback()


@dataclass(frozen=True)
class ResolvedPrefetch:
    """A cache-resolved one-layer transfer plan.

    ``predicted_block_ids`` remain logical block IDs.  Physical indices are
    intentionally confined to this cache-side object and never cross into a
    remote storage API.
    """

    transfers: tuple[L2Transfer, ...]
    layer_num: int
    lease: Optional[PageLease] = None
    start_event: Any = None


@dataclass
class PrefetchTicket:
    """Handle for one request/layer/generation transfer."""

    request_id: str
    generation: int
    layer_id: int
    predicted_block_ids: tuple[int, ...]
    completion: Optional[TransferCompletion] = None
    lease: Optional[PageLease] = None
    state: PrefetchTicketState = PrefetchTicketState.SUBMITTED
    error: Optional[BaseException] = None
    completion_synchronized: bool = False
    _runtime: Optional[SparDAKVPrefetcher] = field(
        default=None, repr=False, compare=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def done(self) -> bool:
        """Return whether the ticket has reached a terminal state."""
        return self.state in {
            PrefetchTicketState.READY,
            PrefetchTicketState.CONSUMED,
            PrefetchTicketState.CANCELLED,
            PrefetchTicketState.RELEASED,
        }


class PrefetchResolver(Protocol):
    """Resolve logical forecast blocks into an in-process transfer plan."""

    def resolve(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        predicted_block_ids: Sequence[int],
        context: Any = None,
    ) -> Optional[ResolvedPrefetch]:
        """Return ``None`` when the prediction cannot be prefetched safely."""


class ForecastPageResolver(PrefetchResolver, Protocol):
    """Optional resolver extension that turns a forecast query into blocks."""

    def predict(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        forecast_query: Any,
        context: Any = None,
    ) -> Optional[Sequence[int]]:
        """Return logical block IDs or ``None`` to use the normal path."""


def _restrict_transfers_to_layer(
    transfers: Sequence[L2Transfer], target_layer_id: int
) -> list[L2Transfer]:
    """Make an existing transfer plan write only ``target_layer_id``.

    ``L2TransferEngine`` accepts a layer count and walks all layers below that
    count.  SparDA's first runtime only predicts one layer, so the runtime
    wraps the cache-provided mapper with a target check.  The original mapper
    still controls host-pool layout when the target is reached.
    """

    restricted: list[L2Transfer] = []
    for transfer in transfers:
        if not isinstance(transfer, L2Transfer):
            # Keep test doubles and compatible transfer implementations
            # opaque; the production engine receives L2Transfer instances.
            restricted.append(transfer)
            continue
        original_mapper = transfer.layer_mapper

        def layer_mapper(
            layer_id: int,
            *,
            original_mapper=original_mapper,
        ) -> Optional[int]:
            if layer_id != target_layer_id:
                return None
            if original_mapper is None:
                return layer_id
            return original_mapper(layer_id)

        restricted.append(transfer._replace(layer_mapper=layer_mapper))
    return restricted


class SparDAKVPrefetcher:
    """Thread-safe one-layer prefetch coordinator.

    The coordinator is intentionally independent of a particular cache tree.
    A resolver can be backed by HiCache, a test pool, or a future logical-key
    adapter.  A missing resolver result is a normal fallback, not an error.
    """

    def __init__(
        self,
        transfer_engine: Any,
        resolver: Optional[PrefetchResolver] = None,
    ) -> None:
        self._transfer_engine = transfer_engine
        self._resolver = resolver
        self._lock = threading.RLock()
        self._tickets: dict[tuple[str, int, int], PrefetchTicket] = {}
        self._latest_generation: dict[str, int] = {}
        self._metrics = Counter()

    def prefetch_forecast(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        predicted_block_ids: Sequence[int],
        *,
        context: Any = None,
    ) -> Optional[PrefetchTicket]:
        """Resolve and submit the next layer's predicted blocks."""
        if generation < 0:
            raise ValueError(f"generation must be non-negative, got {generation}")
        if layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {layer_id}")
        block_ids = tuple(int(block_id) for block_id in predicted_block_ids)
        if any(block_id < 0 for block_id in block_ids):
            raise ValueError("predicted block IDs must be non-negative")
        if not block_ids or self._resolver is None:
            self._metrics["fallback"] += 1
            return None

        with self._lock:
            latest = self._latest_generation.get(request_id)
            if latest is not None and generation < latest:
                self._metrics["stale_prediction"] += 1
                return None
            self._latest_generation[request_id] = max(generation, latest or 0)
            key = (request_id, generation, layer_id)
            previous = self._tickets.get(key)
            if previous is not None:
                if not self._cancel_locked(previous):
                    self._metrics["fallback"] += 1
                    return None

        resolved = self._resolver.resolve(
            request_id,
            generation,
            layer_id,
            block_ids,
            context,
        )
        if resolved is None or not resolved.transfers:
            self._metrics["fallback"] += 1
            return None
        return self.submit(
            request_id,
            generation,
            layer_id,
            block_ids,
            resolved,
        )

    def begin_request(self, request_id: str) -> int:
        """Start a new request generation and invalidate older tickets."""
        with self._lock:
            generation = self._latest_generation.get(request_id, -1) + 1
            self.invalidate_generation(request_id, generation)
        return generation

    def prefetch_forecast_query(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        forecast_query: Any,
        *,
        context: Any = None,
    ) -> Optional[PrefetchTicket]:
        """Run a resolver's optional selector and submit its logical blocks."""
        if self._resolver is None:
            self._metrics["fallback"] += 1
            return None
        predict = getattr(self._resolver, "predict", None)
        if predict is None:
            self._metrics["fallback"] += 1
            return None
        predicted_block_ids = predict(
            request_id,
            generation,
            layer_id,
            forecast_query,
            context,
        )
        if predicted_block_ids is None:
            self._metrics["fallback"] += 1
            return None
        return self.prefetch_forecast(
            request_id,
            generation,
            layer_id,
            predicted_block_ids,
            context=context,
        )

    def submit(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        predicted_block_ids: Sequence[int],
        resolved: ResolvedPrefetch,
    ) -> PrefetchTicket:
        """Submit a resolved plan to the existing HiCache transfer engine."""
        if generation < 0:
            raise ValueError(f"generation must be non-negative, got {generation}")
        if layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {layer_id}")
        if resolved.layer_num <= 0:
            raise ValueError(f"layer_num must be positive, got {resolved.layer_num}")
        key = (request_id, generation, layer_id)
        with self._lock:
            latest = self._latest_generation.get(request_id)
            if latest is not None and generation < latest:
                self._metrics["stale_prediction"] += 1
                if resolved.lease is not None:
                    resolved.lease.release()
                return PrefetchTicket(
                    request_id=request_id,
                    generation=generation,
                    layer_id=layer_id,
                    predicted_block_ids=tuple(predicted_block_ids),
                    state=PrefetchTicketState.CANCELLED,
                )
            self._latest_generation[request_id] = max(generation, latest or 0)
            previous = self._tickets.get(key)
            if previous is not None:
                if not self._cancel_locked(previous):
                    if resolved.lease is not None:
                        resolved.lease.release()
                    self._metrics["fallback"] += 1
                    return PrefetchTicket(
                        request_id=request_id,
                        generation=generation,
                        layer_id=layer_id,
                        predicted_block_ids=tuple(predicted_block_ids),
                        state=PrefetchTicketState.CANCELLED,
                    )

            try:
                # L2TransferEngine iterates from layer zero through
                # ``layer_num``.  Restrict every transfer to the predicted
                # target layer so a one-step forecast never copies unrelated
                # layers, even when the resolver returns a full-model layer
                # mapper.
                transfers = _restrict_transfers_to_layer(resolved.transfers, layer_id)
                completion = self._transfer_engine.submit_host_to_device(
                    transfers,
                    layer_num=max(resolved.layer_num, layer_id + 1),
                    start_event=resolved.start_event,
                )
            except BaseException:
                if resolved.lease is not None:
                    resolved.lease.release()
                raise

            ticket = PrefetchTicket(
                request_id=request_id,
                generation=generation,
                layer_id=layer_id,
                predicted_block_ids=tuple(predicted_block_ids),
                completion=completion,
                lease=resolved.lease,
                _runtime=self,
            )
            self._tickets[key] = ticket
            self._metrics["submitted"] += 1
            return ticket

    def wait(self, ticket: PrefetchTicket) -> bool:
        """Synchronize the copy event and make the ticket ready.

        Host synchronization is intentional here.  The caller may use the
        destination page immediately after this method returns, including on
        platforms whose event wait cannot be safely captured by a graph.
        """
        with ticket._lock:
            if ticket.state in {
                PrefetchTicketState.CONSUMED,
                PrefetchTicketState.RELEASED,
            }:
                return True
            completion = ticket.completion
            if ticket.state is PrefetchTicketState.CANCELLED and completion is None:
                return False

        try:
            if completion is not None:
                with ticket._lock:
                    synchronized = ticket.completion_synchronized
                finish_event = completion.finish_event
                if synchronized:
                    finish_event = None
                if finish_event is None:
                    # Some synchronous/test transfer providers do not expose an
                    # event.  Their completion is already observable here, so
                    # allow the lease to be released just like an event-backed
                    # completion.
                    with ticket._lock:
                        ticket.completion_synchronized = True
                else:
                    synchronize = getattr(finish_event, "synchronize", None)
                    if synchronize is not None:
                        synchronize()
                    else:
                        wait = getattr(finish_event, "wait", None)
                        if wait is not None:
                            wait()
                    with ticket._lock:
                        ticket.completion_synchronized = True
        except BaseException as exc:
            with ticket._lock:
                ticket.error = exc
            raise

        with ticket._lock:
            if ticket.state is PrefetchTicketState.SUBMITTED:
                ticket.state = PrefetchTicketState.READY
                self._metrics["completed"] += 1
            return ticket.state in {
                PrefetchTicketState.READY,
                PrefetchTicketState.CONSUMED,
            }

    def consume(self, ticket: PrefetchTicket) -> bool:
        """Wait for and consume a ticket, releasing its page lease."""
        if not self.wait(ticket):
            return False
        with ticket._lock:
            if ticket.state is PrefetchTicketState.READY:
                ticket.state = PrefetchTicketState.CONSUMED
                self._metrics["consumed"] += 1
        self._release_after_completion(ticket)
        self._remove_if_terminal(ticket)
        return True

    def wait_for_layer(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
    ) -> bool:
        """Wait for the ticket consumed by ``layer_id`` if one exists.

        A missing ticket is a normal cache miss/fallback.  A coordinator with
        no resolver returns ``True`` so the selection-only Phase1 behavior is
        unchanged; once a resolver is installed, ``False`` tells the model to
        use its current-query/full loading path instead of reading an
        unstaged forecast page.
        """
        key = (request_id, generation, layer_id)
        with self._lock:
            ticket = self._tickets.get(key)
        if ticket is None:
            self._metrics["miss"] += 1
            return self._resolver is None
        self._metrics["wait"] += 1
        return self.wait(ticket)

    def consume_for_layer(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
    ) -> bool:
        """Consume and release the ticket for a completed attention layer."""
        key = (request_id, generation, layer_id)
        with self._lock:
            ticket = self._tickets.get(key)
        if ticket is None:
            return True
        return self.consume(ticket)

    def cancel(self, ticket: PrefetchTicket) -> None:
        """Cancel a ticket without releasing pages before copy completion."""
        with self._lock:
            self._cancel_locked(ticket)

    def release(self, ticket: PrefetchTicket) -> None:
        """Idempotently release a consumed, ready, or cancelled ticket."""
        with self._lock:
            if ticket.state is PrefetchTicketState.SUBMITTED or (
                ticket.state is PrefetchTicketState.CANCELLED
                and ticket.lease is not None
            ):
                self._cancel_locked(ticket)
            else:
                if self._release_after_completion(ticket):
                    with ticket._lock:
                        ticket.state = PrefetchTicketState.RELEASED
                    self._remove_if_terminal(ticket)

    def cleanup_request(
        self,
        request_id: str,
        generation: Optional[int] = None,
    ) -> None:
        """Cancel all matching tickets during finish/reorder/eviction."""
        with self._lock:
            tickets = [
                ticket
                for ticket in self._tickets.values()
                if ticket.request_id == request_id
                and (generation is None or ticket.generation == generation)
            ]
            for ticket in tickets:
                self._cancel_locked(ticket)
            if generation is None:
                self._latest_generation.pop(request_id, None)

    def cleanup_all(self) -> None:
        """Cancel every outstanding ticket before cache pools are reset."""
        with self._lock:
            request_ids = {ticket.request_id for ticket in self._tickets.values()}
        for request_id in request_ids:
            self.cleanup_request(request_id)

    def invalidate_generation(self, request_id: str, generation: int) -> None:
        """Cancel older generations and make ``generation`` the newest one."""
        if generation < 0:
            raise ValueError(f"generation must be non-negative, got {generation}")
        with self._lock:
            current = self._latest_generation.get(request_id)
            if current is not None and generation < current:
                return
            self._latest_generation[request_id] = generation
            for ticket in list(self._tickets.values()):
                if ticket.request_id == request_id and ticket.generation < generation:
                    self._cancel_locked(ticket)

    def metrics(self) -> dict[str, int]:
        """Return an immutable snapshot of runtime counters."""
        with self._lock:
            return dict(self._metrics)

    def active_tickets(self) -> tuple[PrefetchTicket, ...]:
        """Return tickets retained for an unfinished request."""
        with self._lock:
            return tuple(self._tickets.values())

    def _cancel_locked(self, ticket: PrefetchTicket) -> bool:
        with ticket._lock:
            if ticket.state in {
                PrefetchTicketState.RELEASED,
                PrefetchTicketState.CONSUMED,
            } or (
                ticket.state is PrefetchTicketState.CANCELLED and ticket.lease is None
            ):
                return True
            ticket.state = PrefetchTicketState.CANCELLED
        self._metrics["cancelled"] += 1
        # synchronize before releasing the lease; otherwise a destination
        # page may be reused while the copy stream still writes it.
        try:
            self.wait(ticket)
        except BaseException:
            # Fail closed.  Keep the lease and ticket reachable so a later
            # cleanup attempt can retry the event synchronization.
            return False
        if self._release_after_completion(ticket):
            self._remove_if_terminal(ticket)
            return True
        return False

    def _release_after_completion(self, ticket: PrefetchTicket) -> bool:
        with ticket._lock:
            if ticket.completion is not None and not ticket.completion_synchronized:
                return False
            lease = ticket.lease
            if lease is None:
                return True
            ticket.lease = None
        lease.release()
        return True

    def _remove_if_terminal(self, ticket: PrefetchTicket) -> None:
        if ticket.state not in {
            PrefetchTicketState.CONSUMED,
            PrefetchTicketState.CANCELLED,
            PrefetchTicketState.RELEASED,
        }:
            return
        key = (ticket.request_id, ticket.generation, ticket.layer_id)
        with self._lock:
            if self._tickets.get(key) is ticket:
                self._tickets.pop(key, None)


class CallbackPrefetchResolver:
    """Resolver adapter for integrations that already own page resolution."""

    def __init__(
        self,
        callback: Callable[
            [str, int, int, Sequence[int], Any], Optional[ResolvedPrefetch]
        ],
    ) -> None:
        self._callback = callback

    def resolve(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        predicted_block_ids: Sequence[int],
        context: Any = None,
    ) -> Optional[ResolvedPrefetch]:
        return self._callback(
            request_id,
            generation,
            layer_id,
            predicted_block_ids,
            context,
        )


class TreeCachePrefetchResolver:
    """Bridge for cache implementations exposing public SparDA hooks.

    The hooks are intentionally optional.  A cache that cannot resolve a
    forecast safely returns ``None`` and the attention path continues with its
    normal loading/selection behavior.
    """

    def __init__(self, tree_cache: Any) -> None:
        self._tree_cache = tree_cache

    def is_available(self) -> bool:
        """Return whether the cache exposes both Phase2 resolver hooks."""
        return callable(getattr(self._tree_cache, "predict_sparda_blocks", None)) and (
            callable(getattr(self._tree_cache, "resolve_sparda_prefetch", None))
        )

    def predict(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        forecast_query: Any,
        context: Any = None,
    ) -> Optional[Sequence[int]]:
        callback = getattr(self._tree_cache, "predict_sparda_blocks", None)
        if callback is None:
            return None
        return callback(
            request_id,
            generation,
            layer_id,
            forecast_query,
            context,
        )

    def resolve(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        predicted_block_ids: Sequence[int],
        context: Any = None,
    ) -> Optional[ResolvedPrefetch]:
        callback = getattr(self._tree_cache, "resolve_sparda_prefetch", None)
        if callback is None:
            return None
        return callback(
            request_id,
            generation,
            layer_id,
            predicted_block_ids,
            context,
        )
