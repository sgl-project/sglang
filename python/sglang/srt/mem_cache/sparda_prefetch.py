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

import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, Sequence

from sglang.srt.mem_cache.l2_transfer import L2Transfer, TransferCompletion

logger = logging.getLogger(__name__)


class PrefetchTicketState(Enum):
    """Lifecycle states visible to the attention/cache integration."""

    SUBMITTED = auto()
    READY = auto()
    CONSUMED = auto()
    CANCELLED = auto()
    RELEASED = auto()


class PageLease(Protocol):
    """Lease held while a transfer may still write destination pages."""

    def mark_consumed(self) -> None:
        """Record that the attention consumer has finished using the pages."""

    def release(self) -> None:
        """Make the leased pages eligible for reuse."""


@dataclass
class CallbackPageLease:
    """Small adapter for allocators that expose acquire/release callbacks."""

    release_callback: Callable[[], None]
    consumed_callback: Optional[Callable[[], None]] = None
    _released: bool = False
    _releasing: bool = False
    _consumed: bool = False
    _consuming: bool = False
    _lock: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def mark_consumed(self) -> None:
        with self._lock:
            while self._releasing:
                self._lock.wait()
            if self._released or self._consumed or self._consuming:
                return
            self._consuming = True
        try:
            if self.consumed_callback is not None:
                self.consumed_callback()
        except BaseException:
            with self._lock:
                self._consuming = False
                self._lock.notify_all()
            raise
        with self._lock:
            self._consuming = False
            if not self._released:
                self._consumed = True
            self._lock.notify_all()

    def release(self) -> None:
        with self._lock:
            while self._consuming:
                self._lock.wait()
            if self._released or self._releasing:
                return
            self._releasing = True
        try:
            self.release_callback()
        except BaseException:
            with self._lock:
                self._releasing = False
                self._lock.notify_all()
            raise
        with self._lock:
            self._releasing = False
            self._released = True
            self._lock.notify_all()


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
    on_ready: Optional[Callable[[], Optional[bool]]] = None
    submit_callback: Optional[Callable[[], Optional[Any]]] = None
    safe_without_transfer: bool = False


class RemoteTransferCompletion:
    """Adapt a device-aware remote future to the local transfer contract."""

    def __init__(
        self,
        future: Any,
        on_result: Callable[[tuple[int, ...]], None],
    ) -> None:
        self.future = future
        self.on_result = on_result
        # SparDAKVPrefetcher waits on ``finish_event``.  The remote future
        # already imports and synchronizes the server's device event, so this
        # object is its event-like completion surface.
        self.finish_event = self
        self._lock = threading.Lock()
        self._synchronized = False

    def synchronize(self) -> None:
        """Wait for the remote H2D event and publish its hit mapping once."""
        with self._lock:
            if self._synchronized:
                return
            result = self.future.result()
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("remote sparse retrieve returned an invalid result")
            success, found_indices = result
            if not success:
                # The server returns no completion event when it did not copy
                # any destination page.  Publishing an empty hit set lets the
                # resolver reject the overlay and release its lease; a
                # transport exception remains fail-closed below.
                self.on_result(())
            else:
                self.on_result(tuple(int(index) for index in found_indices))
            self._synchronized = True


@dataclass
class PrefetchTicket:
    """Handle for one request/layer/generation transfer."""

    request_id: str
    generation: int
    layer_id: int
    predicted_block_ids: tuple[int, ...]
    completion: Optional[TransferCompletion] = None
    lease: Optional[PageLease] = None
    pending_resolved: Optional[ResolvedPrefetch] = field(
        default=None, repr=False, compare=False
    )
    state: PrefetchTicketState = PrefetchTicketState.SUBMITTED
    error: Optional[BaseException] = None
    completion_synchronized: bool = False
    ready_callback: Optional[Callable[[], Optional[bool]]] = field(
        default=None, repr=False, compare=False
    )
    ready_callback_called: bool = False
    _runtime: Optional[SparDAKVPrefetcher] = field(
        default=None, repr=False, compare=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def done(self) -> bool:
        """Return whether the ticket has reached a terminal state."""
        with self._lock:
            if self.state is PrefetchTicketState.READY:
                return True
            if self.state in {
                PrefetchTicketState.CONSUMED,
                PrefetchTicketState.RELEASED,
            }:
                return self.lease is None
            # A failed cancellation keeps the lease and ticket reachable so
            # cleanup can retry the event synchronization.
            return self.state is PrefetchTicketState.CANCELLED and self.lease is None


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
        *,
        submit_on_wait: bool = False,
    ) -> None:
        self._transfer_engine = transfer_engine
        self._resolver = resolver
        self._submit_on_wait = submit_on_wait
        self._lock = threading.RLock()
        self._tickets: dict[tuple[str, int, int], PrefetchTicket] = {}
        # A request/layer key can already be occupied by an older ticket when
        # a duplicate submit races its cleanup. Keep a losing lease in this
        # side table instead of dropping the only retry handle.
        self._orphan_tickets: dict[int, PrefetchTicket] = {}
        self._latest_generation: dict[str, int] = {}
        # Keep a tombstone for generations retired by request cleanup.  A
        # resolver runs outside the runtime lock, so cleanup can race with a
        # resolver that is about to return a transfer plan.
        self._retired_generations: dict[str, int] = {}
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
            if self._is_stale_generation_locked(request_id, generation):
                self._metrics["stale_prediction"] += 1
                return None
            self._latest_generation[request_id] = max(
                generation, latest if latest is not None else -1
            )
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
        if resolved is None:
            self._metrics["fallback"] += 1
            return None
        if (
            not resolved.transfers
            and resolved.submit_callback is None
            and not resolved.safe_without_transfer
        ):
            # The resolver may allocate a lease before discovering that the
            # selected blocks are already resident or otherwise unavailable.
            # There is no transfer completion to guard that lease, so release
            # it before taking the normal attention path.
            if resolved.lease is not None:
                self._retain_cancelled_lease(
                    request_id,
                    generation,
                    layer_id,
                    block_ids,
                    resolved,
                )
            self._metrics["fallback"] += 1
            return None
        return self.submit(
            request_id,
            generation,
            layer_id,
            block_ids,
            resolved,
            defer=self._submit_on_wait,
        )

    def begin_request(self, request_id: str) -> int:
        """Start a new request generation and invalidate older tickets."""
        with self._lock:
            generation = (
                max(
                    self._latest_generation.get(request_id, -1),
                    self._retired_generations.get(request_id, -1),
                )
                + 1
            )
            self._retired_generations.pop(request_id, None)
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
            logger.debug(
                "SparDA forecast unavailable: layer=%d reason=selector", layer_id
            )
            return None
        logger.debug(
            "SparDA forecast selected: layer=%d blocks=%d",
            layer_id,
            len(predicted_block_ids),
        )
        ticket = self.prefetch_forecast(
            request_id,
            generation,
            layer_id,
            predicted_block_ids,
            context=context,
        )
        if ticket is None:
            logger.debug("SparDA forecast unresolved: layer=%d reason=cache", layer_id)
        return ticket

    def submit(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        predicted_block_ids: Sequence[int],
        resolved: ResolvedPrefetch,
        *,
        defer: bool = False,
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
            if self._is_stale_generation_locked(request_id, generation):
                self._metrics["stale_prediction"] += 1
                return self._retain_cancelled_lease(
                    request_id=request_id,
                    generation=generation,
                    layer_id=layer_id,
                    predicted_block_ids=tuple(predicted_block_ids),
                    resolved=resolved,
                )
            self._latest_generation[request_id] = max(generation, latest or 0)
            previous = self._tickets.get(key)
            if previous is not None:
                if not self._cancel_locked(previous):
                    self._metrics["fallback"] += 1
                    return self._retain_cancelled_lease(
                        request_id=request_id,
                        generation=generation,
                        layer_id=layer_id,
                        predicted_block_ids=tuple(predicted_block_ids),
                        resolved=resolved,
                    )

            if defer:
                ticket = PrefetchTicket(
                    request_id=request_id,
                    generation=generation,
                    layer_id=layer_id,
                    predicted_block_ids=tuple(predicted_block_ids),
                    lease=resolved.lease,
                    pending_resolved=resolved,
                    ready_callback=resolved.on_ready,
                    _runtime=self,
                )
                self._tickets[key] = ticket
                self._metrics["deferred"] += 1
                return ticket

            ticket = PrefetchTicket(
                request_id=request_id,
                generation=generation,
                layer_id=layer_id,
                predicted_block_ids=tuple(predicted_block_ids),
                lease=resolved.lease,
                ready_callback=resolved.on_ready,
                _runtime=self,
            )
            # Install the ticket before invoking an external submit callback.
            # The callback may have accepted a remote lease before reporting
            # an error; keeping the ticket reachable lets request cleanup
            # retry cancellation/release instead of losing that lease.
            self._tickets[key] = ticket

            try:
                # L2TransferEngine iterates from layer zero through
                # ``layer_num``.  Restrict every transfer to the predicted
                # target layer so a one-step forecast never copies unrelated
                # layers, even when the resolver returns a full-model layer
                # mapper.
                transfers = _restrict_transfers_to_layer(resolved.transfers, layer_id)
                completion = None
                if transfers:
                    if self._transfer_engine is None:
                        raise RuntimeError(
                            "a transfer engine is required for resolved transfers"
                        )
                    completion = self._transfer_engine.submit_host_to_device(
                        transfers,
                        layer_num=max(resolved.layer_num, layer_id + 1),
                        start_event=resolved.start_event,
                    )
                elif resolved.submit_callback is not None:
                    completion = resolved.submit_callback()
                    if completion is None:
                        raise RuntimeError(
                            "prefetch submit callback returned no completion"
                        )
            except BaseException as exc:
                with ticket._lock:
                    ticket.error = exc
                    ticket.state = PrefetchTicketState.CANCELLED
                if self._release_after_completion(ticket):
                    self._remove_if_terminal(ticket)
                raise

            with ticket._lock:
                ticket.completion = completion
            self._metrics["submitted"] += 1
            logger.debug(
                "SparDA prefetch submitted: layer=%d blocks=%d",
                layer_id,
                len(ticket.predicted_block_ids),
            )
            return ticket

    def _start_deferred(self, ticket: PrefetchTicket) -> None:
        """Submit a demand-mode ticket immediately before its layer runs."""
        with self._lock:
            with ticket._lock:
                if ticket.state is not PrefetchTicketState.SUBMITTED:
                    return
                resolved = ticket.pending_resolved
                ticket.pending_resolved = None
            if resolved is None:
                return

            try:
                transfers = _restrict_transfers_to_layer(
                    resolved.transfers, ticket.layer_id
                )
                completion = None
                if transfers:
                    if self._transfer_engine is None:
                        raise RuntimeError(
                            "a transfer engine is required for resolved transfers"
                        )
                    completion = self._transfer_engine.submit_host_to_device(
                        transfers,
                        layer_num=max(resolved.layer_num, ticket.layer_id + 1),
                        start_event=resolved.start_event,
                    )
                elif resolved.submit_callback is not None:
                    completion = resolved.submit_callback()
                    if completion is None:
                        raise RuntimeError(
                            "prefetch submit callback returned no completion"
                        )
            except BaseException as exc:
                with ticket._lock:
                    ticket.error = exc
                    ticket.state = PrefetchTicketState.CANCELLED
                if self._release_after_completion(ticket):
                    self._remove_if_terminal(ticket)
                raise

            with ticket._lock:
                ticket.completion = completion
            self._metrics["submitted"] += 1
            logger.debug(
                "SparDA demand load submitted: layer=%d blocks=%d",
                ticket.layer_id,
                len(ticket.predicted_block_ids),
            )

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

        with ticket._lock:
            pending_resolved = ticket.pending_resolved
        if pending_resolved is not None:
            self._start_deferred(ticket)
            with ticket._lock:
                if ticket.state is PrefetchTicketState.CANCELLED:
                    return False
                completion = ticket.completion

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
                        else:
                            raise RuntimeError(
                                "transfer completion event must expose "
                                "synchronize() or wait()"
                            )
                    with ticket._lock:
                        ticket.completion_synchronized = True
        except BaseException as exc:
            with ticket._lock:
                ticket.error = exc
            raise

        with ticket._lock:
            should_activate = (
                ticket.state is PrefetchTicketState.SUBMITTED
                and ticket.ready_callback is not None
                and not ticket.ready_callback_called
            )
            if should_activate:
                ticket.ready_callback_called = True

        if should_activate:
            try:
                ready_result = ticket.ready_callback()
                if ready_result is False:
                    with ticket._lock:
                        ticket.state = PrefetchTicketState.CANCELLED
                    if self._release_after_completion(ticket):
                        self._remove_if_terminal(ticket)
                    return False
            except BaseException as exc:
                with ticket._lock:
                    ticket.ready_callback_called = False
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
            lease = ticket.lease
        if lease is not None:
            mark_consumed = getattr(lease, "mark_consumed", None)
            if mark_consumed is not None:
                mark_consumed()
        with ticket._lock:
            if ticket.state is PrefetchTicketState.READY:
                ticket.state = PrefetchTicketState.CONSUMED
                self._metrics["consumed"] += 1
        released = self._release_after_completion(ticket)
        if released:
            self._remove_if_terminal(ticket)
        return released

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
            logger.debug("SparDA prefetch miss: layer=%d", layer_id)
            return self._resolver is None
        self._metrics["wait"] += 1
        ready = self.wait(ticket)
        logger.debug("SparDA prefetch wait: layer=%d ready=%s", layer_id, ready)
        return ready

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
        consumed = self.consume(ticket)
        logger.debug(
            "SparDA prefetch consume: layer=%d consumed=%s", layer_id, consumed
        )
        return consumed

    def cancel_for_layer(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
    ) -> None:
        """Cancel a staged layer before attention falls back for the batch."""
        key = (request_id, generation, layer_id)
        with self._lock:
            ticket = self._tickets.get(key)
            if ticket is None:
                return
            self._cancel_locked(ticket)

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
    ) -> bool:
        """Cancel all matching tickets during finish/reorder/eviction."""
        all_released = True
        with self._lock:
            tickets = [
                ticket
                for ticket in tuple(self._tickets.values())
                + tuple(self._orphan_tickets.values())
                if ticket.request_id == request_id
                and (generation is None or ticket.generation == generation)
            ]
            for ticket in tickets:
                all_released = self._cancel_locked(ticket) and all_released
            if generation is None:
                latest = self._latest_generation.get(request_id)
                if latest is not None:
                    self._retired_generations[request_id] = max(
                        latest,
                        self._retired_generations.get(request_id, -1),
                    )
                self._latest_generation.pop(request_id, None)
            else:
                self._retired_generations[request_id] = max(
                    generation,
                    self._retired_generations.get(request_id, -1),
                )
                latest = self._latest_generation.get(request_id)
                if latest is not None and latest <= generation:
                    self._latest_generation.pop(request_id, None)
        return all_released

    def cleanup_all(self) -> bool:
        """Cancel every outstanding ticket before cache pools are reset."""
        with self._lock:
            request_ids = (
                {ticket.request_id for ticket in self._tickets.values()}
                | {ticket.request_id for ticket in self._orphan_tickets.values()}
                | set(self._latest_generation)
                | set(self._retired_generations)
            )
        all_released = True
        for request_id in request_ids:
            all_released = self.cleanup_request(request_id) and all_released
        return all_released

    def invalidate_generation(self, request_id: str, generation: int) -> None:
        """Cancel older generations and make ``generation`` the newest one."""
        if generation < 0:
            raise ValueError(f"generation must be non-negative, got {generation}")
        with self._lock:
            retired = self._retired_generations.get(request_id)
            if retired is not None and generation <= retired:
                return
            current = self._latest_generation.get(request_id)
            if current is not None and generation < current:
                return
            self._retired_generations.pop(request_id, None)
            self._latest_generation[request_id] = generation
            for ticket in list(self._tickets.values()) + list(
                self._orphan_tickets.values()
            ):
                if ticket.request_id == request_id and ticket.generation < generation:
                    self._cancel_locked(ticket)

    def metrics(self) -> dict[str, int]:
        """Return an immutable snapshot of runtime counters."""
        with self._lock:
            return dict(self._metrics)

    def active_tickets(self) -> tuple[PrefetchTicket, ...]:
        """Return tickets retained for an unfinished request."""
        with self._lock:
            return tuple(self._tickets.values()) + tuple(self._orphan_tickets.values())

    def offload_request_history(
        self,
        request: Any,
        *,
        keep_device_tokens: int,
        min_history_len: int = 0,
    ) -> bool:
        """Delegate active-request history placement to the cache resolver."""
        callback = getattr(self._resolver, "offload_request_history", None)
        if callback is None:
            return False
        return bool(
            callback(
                request,
                keep_device_tokens=keep_device_tokens,
                min_history_len=min_history_len,
            )
        )

    def restore_request(self, request: Any) -> bool:
        """Restore a request before scheduler/cache state is mutated."""
        callback = getattr(self._resolver, "restore_request", None)
        if callback is None:
            return False
        return bool(callback(request))

    def publish_compressed_index(
        self, request: Any, layer_id: int, levels: Sequence[Any]
    ) -> None:
        """Publish a cache-local compressed-key index for later host hits."""
        callback = getattr(self._resolver, "publish_compressed_index", None)
        if callback is not None:
            callback(request, layer_id, levels)

    def get_compressed_index(self, request: Any, layer_id: int):
        """Return a cache-local compressed-key index, if one is available."""
        callback = getattr(self._resolver, "get_compressed_index", None)
        if callback is None:
            return None
        return callback(request, layer_id)

    def _is_stale_generation_locked(self, request_id: str, generation: int) -> bool:
        latest = self._latest_generation.get(request_id)
        if latest is not None and generation < latest:
            return True
        retired = self._retired_generations.get(request_id)
        return retired is not None and generation <= retired

    def _retain_cancelled_lease(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        predicted_block_ids: Sequence[int],
        resolved: ResolvedPrefetch,
    ) -> PrefetchTicket:
        """Keep a failed cancellation/release reachable for request cleanup."""
        ticket = PrefetchTicket(
            request_id=request_id,
            generation=generation,
            layer_id=layer_id,
            predicted_block_ids=tuple(predicted_block_ids),
            lease=resolved.lease,
            state=PrefetchTicketState.CANCELLED,
            _runtime=self,
        )
        if ticket.lease is None:
            return ticket
        with self._lock:
            self._orphan_tickets[id(ticket)] = ticket
            if self._release_after_completion(ticket):
                self._remove_if_terminal(ticket)
            else:
                self._metrics["lease_release_failed"] += 1
        return ticket

    def _cancel_locked(self, ticket: PrefetchTicket) -> bool:
        with ticket._lock:
            if (
                ticket.state
                in {
                    PrefetchTicketState.RELEASED,
                }
                or (
                    ticket.state is PrefetchTicketState.CONSUMED
                    and ticket.lease is None
                )
                or (
                    ticket.state is PrefetchTicketState.CANCELLED
                    and ticket.lease is None
                )
            ):
                return True
            was_cancelled = ticket.state is PrefetchTicketState.CANCELLED
            ticket.state = PrefetchTicketState.CANCELLED
            has_completion = ticket.completion is not None
        if not was_cancelled:
            self._metrics["cancelled"] += 1
        # synchronize before releasing the lease; otherwise a destination
        # page may be reused while the copy stream still writes it.
        try:
            if has_completion:
                self.wait(ticket)
            else:
                # A ticket without a completion represents a synchronous
                # transfer plan (or a safe resident page), so no device event
                # remains that could write into the leased page.
                with ticket._lock:
                    ticket.completion_synchronized = True
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
        try:
            lease.release()
        except BaseException as exc:
            with ticket._lock:
                ticket.error = exc
            return False
        with ticket._lock:
            if ticket.lease is lease:
                ticket.lease = None
        return True

    def _remove_if_terminal(self, ticket: PrefetchTicket) -> None:
        with ticket._lock:
            if (
                ticket.state
                not in {
                    PrefetchTicketState.CONSUMED,
                    PrefetchTicketState.CANCELLED,
                    PrefetchTicketState.RELEASED,
                }
                or ticket.lease is not None
            ):
                return
        key = (ticket.request_id, ticket.generation, ticket.layer_id)
        with self._lock:
            if self._tickets.get(key) is ticket:
                self._tickets.pop(key, None)
            if self._orphan_tickets.get(id(ticket)) is ticket:
                self._orphan_tickets.pop(id(ticket), None)


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
        """Return whether the cache can resolve predicted pages safely."""
        availability = getattr(self._tree_cache, "sparda_prefetch_available", None)
        if availability is not None:
            return bool(availability())
        return callable(getattr(self._tree_cache, "resolve_sparda_prefetch", None))

    def offload_request_history(
        self,
        request: Any,
        *,
        keep_device_tokens: int,
        min_history_len: int = 0,
    ) -> bool:
        callback = getattr(self._tree_cache, "offload_sparda_request_history", None)
        if callback is None:
            return False
        return bool(
            callback(
                request,
                keep_device_tokens=keep_device_tokens,
                min_history_len=min_history_len,
            )
        )

    def restore_request(self, request: Any) -> bool:
        callback = getattr(self._tree_cache, "restore_sparda_request", None)
        if callback is None:
            return False
        return bool(callback(request))

    def publish_compressed_index(
        self, request: Any, layer_id: int, levels: Sequence[Any]
    ) -> None:
        callback = getattr(self._tree_cache, "publish_sparda_compressed_index", None)
        if callback is not None:
            callback(request, layer_id, levels)

    def get_compressed_index(self, request: Any, layer_id: int):
        callback = getattr(self._tree_cache, "get_sparda_compressed_index", None)
        if callback is None:
            return None
        return callback(request, layer_id)

    def predict(
        self,
        request_id: str,
        generation: int,
        layer_id: int,
        forecast_query: Any,
        context: Any = None,
    ) -> Optional[Sequence[int]]:
        callback = getattr(self._tree_cache, "predict_sparda_blocks", None)
        if callback is not None:
            predicted = callback(
                request_id,
                generation,
                layer_id,
                forecast_query,
                context,
            )
            if predicted is not None:
                return predicted

        selector_backend = getattr(context, "selector_backend", None)
        predictor = getattr(selector_backend, "predict_sparda_blocks", None)
        if predictor is None:
            return None
        forecast_batch = getattr(context, "forecast_batch", None)
        if forecast_batch is None:
            forecast_batch = forecast_query
        return predictor(
            forecast_batch,
            context.forward_batch,
            context.request_index,
            layer_id,
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
