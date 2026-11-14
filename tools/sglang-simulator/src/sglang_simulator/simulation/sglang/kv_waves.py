"""Turn Prefill EXTEND chunks into KV transfer waves and schedule them.

Every chunk that finishes after the Decode bootstrap becomes one page-aligned
wave, as ``send_kv_chunk`` sends it. Waves of one request are serial; waves of
different requests share ``max_inflight_waves`` FIFO slots on their transfer
queue, and queues share the bandwidth of the nodes they send from and to.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Optional, Sequence

from sglang_simulator.simulation.sglang.kv_transfer import KVTransferEstimator

# A Mooncake transfer queue's worker sends one chunk at a time.
DEFAULT_MAX_INFLIGHT_WAVES = 1
# SGLANG_DISAGGREGATION_QUEUE_SIZE: each prefill rank shards its sends over this
# many queues by destination, so one decode instance always uses the same queue.
DEFAULT_TRANSFER_QUEUES = 4


@dataclass(frozen=True)
class WaveOptions:
    # kv_transfer_config.chunked_prefill=False forces one wave per request; it
    # cannot enable chunking that serving disabled.
    enabled: bool
    page_size: int
    bytes_per_token: float
    bootstrap_alpha_s: float = 0.0


def resolve_wave_options(
    kv_transfer_config: dict,
    *,
    chunked_prefill_size: Optional[int],
    page_size: int,
    bytes_per_token: float,
) -> WaveOptions:
    serving_chunks = chunked_prefill_size is not None and chunked_prefill_size > 0
    return WaveOptions(
        enabled=serving_chunks
        and bool(kv_transfer_config.get("chunked_prefill", True)),
        page_size=max(int(page_size), 1),
        bytes_per_token=float(bytes_per_token),
        bootstrap_alpha_s=float(kv_transfer_config.get("bootstrap_alpha_s", 0.0)),
    )


@dataclass(frozen=True)
class PrefillChunk:
    """A finished EXTEND, or a device-cached prefix sent ahead of a forward.

    The KV of prompt positions ``[0, extend_end)`` exists at ``t``; the last
    chunk of a request finishes its prefill.
    """

    t: float
    extend_end: int


@dataclass(frozen=True)
class Wave:
    req_id: str
    wave_index: int
    emit: float
    tokens: int
    nbytes: int


@dataclass(frozen=True)
class RequestWaves:
    req_id: str
    waves: Sequence[Wave]
    prefill_end: float
    # The nodes the KV leaves and enters; queues through one node share it.
    src: Hashable = None
    dst: Hashable = None


@dataclass(frozen=True)
class ScheduledTransfer:
    req_id: str
    first_start: float
    last_done: float
    service_s: float
    excursion_s: float
    visible_s: float
    wave_tokens: list[int]

    @property
    def send_s(self) -> float:
        return self.service_s + self.excursion_s

    def as_stats(self) -> dict:
        return {
            "kv_xfer_visible_s": self.visible_s,
            "kv_xfer_n_waves": len(self.wave_tokens),
            "kv_xfer_service_s": self.service_s,
            "kv_xfer_excursion_s": self.excursion_s,
            "kv_xfer_send_s": self.send_s,
            "kv_xfer_wave_tokens": list(self.wave_tokens),
        }


def waves_from_chunks(
    chunks: Sequence[PrefillChunk],
    *,
    req_id: str,
    created_time: float,
    input_length: int,
    options: WaveOptions,
) -> list[Wave]:
    """The waves ``send_kv_chunk`` emits for one request's prompt."""
    if not chunks or input_length <= 0:
        return []
    page_size = options.page_size

    def wave(index: int, emit: float, start: int, end: int) -> Wave:
        # A wave moves whole pages, so a partial last page costs a full one.
        pages = -(-(end - start) // page_size)
        nbytes = round(pages * page_size * options.bytes_per_token)
        return Wave(req_id, index, emit, end - start, nbytes)

    # Chunks done before the Decode bootstrap are merged into the next send; a
    # prefill that finishes first sends everything once the bootstrap is done.
    bootstrap_done = created_time + options.bootstrap_alpha_s
    last_emit = max(chunks[-1].t, bootstrap_done)
    if not options.enabled:
        return [wave(0, last_emit, 0, input_length)]

    waves: list[Wave] = []
    sent = 0
    for chunk in chunks[:-1]:
        if chunk.t < bootstrap_done:
            continue
        filled = min(chunk.extend_end, input_length)
        end = filled - filled % page_size
        if end > sent:
            waves.append(wave(len(waves), chunk.t, sent, end))
            sent = end
    waves.append(wave(len(waves), last_emit, sent, input_length))
    return waves


@dataclass
class _InFlight:
    queue: Hashable
    item: RequestWaves
    index: int
    start: float
    data_start: float
    service: float
    excursion: float
    # Transfer time still to go (bytes and excursion) at the unshared rate.
    work: float
    shared: bool = False


def schedule_transfers(
    queues: Mapping[Hashable, Iterable[RequestWaves]],
    estimator: KVTransferEstimator,
    *,
    max_inflight_waves: int = DEFAULT_MAX_INFLIGHT_WAVES,
    visible_scale: float = 1.0,
) -> dict[str, ScheduledTransfer]:
    """Schedule waves on every transfer queue at once.

    Each queue admits the next wave of each request FIFO into
    ``max_inflight_waves`` slots, at the single-flow rate a profile is fitted
    at. While n queues move data out of a wave's ``src`` node, or into its
    ``dst`` node, the wave advances at 1/n of that rate; only the per-send alpha
    is not stretched. ``visible_scale`` scales only the wait Decode observes.
    """
    if max_inflight_waves < 1:
        raise ValueError("max_inflight_waves must be >= 1")
    if not 0.0 <= visible_scale <= 1.0:
        raise ValueError("visible_scale must be between 0 and 1")

    alpha = estimator.est_service_dur(0)
    by_id: dict[str, RequestWaves] = {}
    waiting: dict[Hashable, list[tuple[float, str, int]]] = {}
    for queue, requests in queues.items():
        heap = waiting.setdefault(queue, [])
        for item in requests:
            by_id[item.req_id] = item
            if item.waves:
                heap.append((item.waves[0].emit, item.req_id, 0))
        heapq.heapify(heap)
    free = {queue: max_inflight_waves for queue in waiting}
    in_flight: list[_InFlight] = []
    placed: dict[str, list[tuple[float, float, float, float]]] = {}
    now = min((heap[0][0] for heap in waiting.values() if heap), default=0.0)

    def rates() -> list[float]:
        moving = [f for f in in_flight if f.data_start <= now]
        senders: dict[Hashable, set] = {}
        receivers: dict[Hashable, set] = {}
        for f in moving:
            senders.setdefault(f.item.src, set()).add(f.queue)
            receivers.setdefault(f.item.dst, set()).add(f.queue)
        return [
            1.0 / max(len(senders[f.item.src]), len(receivers[f.item.dst]))
            if f.data_start <= now
            else 0.0
            for f in in_flight
        ]

    while in_flight or any(waiting.values()):
        for queue, heap in waiting.items():
            while free[queue] and heap and heap[0][0] <= now:
                _, req_id, index = heapq.heappop(heap)
                item = by_id[req_id]
                wave = item.waves[index]
                service = estimator.est_service_dur(wave.nbytes)
                excursion = estimator.est_excursion(
                    wave.nbytes, req_id=req_id, wave_index=wave.wave_index
                )
                in_flight.append(
                    _InFlight(
                        queue=queue,
                        item=item,
                        index=index,
                        start=now,
                        data_start=now + alpha,
                        service=service,
                        excursion=excursion,
                        work=service - alpha + excursion,
                    )
                )
                free[queue] -= 1

        rate = rates()
        ends = []
        for f, r in zip(in_flight, rate):
            if r == 0.0:
                ends.append(f.data_start)
            elif not f.shared and r == 1.0:
                ends.append(f.start + f.service + f.excursion)
            else:
                ends.append(now + f.work / r)
        candidates = ends + [
            heap[0][0] for queue, heap in waiting.items() if heap and free[queue]
        ]
        step = min(t for t in candidates if t >= now)

        for f, r in zip(in_flight, rate):
            if r:
                f.work = max(0.0, f.work - r * (step - now))
                f.shared |= r < 1.0
        now = step

        finished = [f for f, r, end in zip(in_flight, rate, ends) if r and end <= now]
        for f in finished:
            in_flight.remove(f)
            free[f.queue] += 1
            done = now
            service = done - f.start - f.excursion if f.shared else f.service
            placed.setdefault(f.item.req_id, []).append(
                (f.start, done, service, f.excursion)
            )
            if f.index + 1 < len(f.item.waves):
                release = max(f.item.waves[f.index + 1].emit, done)
                heapq.heappush(waiting[f.queue], (release, f.item.req_id, f.index + 1))

    scheduled = {}
    for req_id, intervals in placed.items():
        item = by_id[req_id]
        last_done = intervals[-1][1]
        scheduled[req_id] = ScheduledTransfer(
            req_id=req_id,
            first_start=intervals[0][0],
            last_done=last_done,
            service_s=math.fsum(interval[2] for interval in intervals),
            excursion_s=math.fsum(interval[3] for interval in intervals),
            visible_s=max(0.0, last_done - item.prefill_end) * visible_scale,
            wave_tokens=[wave.tokens for wave in item.waves],
        )
    return scheduled


@dataclass(frozen=True)
class KVTransferPlan:
    transfers: dict[str, ScheduledTransfer]
    estimator: KVTransferEstimator
    num_queues: int


def plan_kv_transfers(
    prefill_stats: Mapping[str, dict],
    routes: Mapping[str, tuple[int, int]],
    kv_layouts: Sequence[Optional[dict]],
    config: dict,
    *,
    accelerator_bandwidth: Optional[float] = None,
) -> KVTransferPlan:
    """Schedule the KV waves of every routed request after the Prefill pass.

    Decode instance ``j`` uses queue ``j % transfer_queues`` of its prefill
    instance. Queues share node bandwidth across decode instances, so no single
    instance can compute its own transfers.
    """
    transfer_queues = int(config.get("transfer_queues", DEFAULT_TRANSFER_QUEUES))
    if transfer_queues < 1:
        raise ValueError("kv_transfer_config.transfer_queues must be >= 1")
    estimator = KVTransferEstimator(config, accelerator_bandwidth)

    options: dict[int, WaveOptions] = {}
    queues: dict[tuple[int, int], list[RequestWaves]] = {}
    for req_id, (p_index, d_index) in routes.items():
        stat = prefill_stats.get(req_id)
        if stat is None:
            continue
        if p_index not in options:
            layout = kv_layouts[p_index]
            if not layout or layout["bytes_per_token"] <= 0:
                raise RuntimeError(
                    f"Prefill instance {p_index} reported no KV cache layout, "
                    f"so KV transfers cannot be sized: {layout!r}"
                )
            options[p_index] = resolve_wave_options(config, **layout)
        waves = waves_from_chunks(
            [PrefillChunk(**chunk) for chunk in stat["kv_prefill_chunks"]],
            req_id=req_id,
            created_time=stat["created_time"],
            input_length=stat["input_length"],
            options=options[p_index],
        )
        if waves:
            queues.setdefault((p_index, d_index % transfer_queues), []).append(
                RequestWaves(
                    req_id=req_id,
                    waves=waves,
                    prefill_end=stat["last_event_time"],
                    src=p_index,
                    dst=d_index,
                )
            )

    transfers = schedule_transfers(
        queues,
        estimator,
        max_inflight_waves=config.get("max_inflight_waves", DEFAULT_MAX_INFLIGHT_WAVES),
        visible_scale=config.get("visible_scale", 1.0),
    )
    return KVTransferPlan(transfers, estimator, len(queues))
