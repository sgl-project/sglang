"""Embedded Rust server lifecycle for the scheduler.

The Rust server replaces the Python api-server + `TokenizerManager` +
`DetokenizerManager` stack, running them as Rust threads inside the scheduler
process. This wrapper keeps all `SGLANG_RUST_SERVER` plumbing — startup,
CPU-core partitioning, the typed `server_args`, and control-response
routing — out of `scheduler.py`. The scheduler holds an `Optional[RustServer]`
and delegates to it.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import time
from array import array
from itertools import chain
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import msgspec
import orjson
import psutil

from sglang.srt.disaggregation.kv_events import select_kv_publisher_dp_rank
from sglang.srt.managers.utils import (
    MsgpackDecodeError,
    msgpack_decode_explained,
)
from sglang.srt.observability.startup_time import build_engine_startup_time
from sglang.srt.runtime_context import get_exec, get_mm, get_parallel, get_serving
from sglang.srt.rust_server.config import _build_server_args, _partition_cores
from sglang.srt.rust_server.multimodal import (
    MM_TRANSPORT_STATS,
    RUST_MM_FAMILIES,
    RustMmProcessor,
    RustMmSpec,
    settle_shm_buffers,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.flatten import (
    FlatPairColumns,
    NestedRowColumns,
    RaggedPairColumns,
)
from sglang.srt.utils.json_response import ORJSON_RESPONSE_OPTIONS
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq, BatchTokenIDOutput
    from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput, Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.rust_extensions._server import MmSpec, Server

logger = logging.getLogger(__name__)

# The extras families' header columns (`BatchHeader.out_lp_lens` through
# `hidden_poslens`). A frame without extras that sends a later column pads
# them empty; `push_generation` asserts the count whenever it sends extras.
_EXTRAS_HEADER_COLS = 12


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False  # a hostname; assume it resolves beyond this machine


def _rendezvous_is_loopback(dist_init_addr: Optional[str]) -> bool:
    """Whether every node rendezvoused on this machine (nodes emulated locally)."""
    if not dist_init_addr:
        return False
    try:
        return _is_loopback(NetworkAddress.parse(dist_init_addr).host)
    except ValueError:
        return False


class RustServer:
    """Owns the embedded multi-threaded Rust server (``sglang_server.Server``).

    The server owns the api-server, tokenizermanager, tokenizer, and detokenizer
    all implemented as Rust threads in scheduler process.
    """

    def __init__(
        self,
        server: Server,
        http_port: int,
        mm_spec: Optional[RustMmSpec] = None,
        max_per_poll: int = 256,
    ):
        self.server = server
        self.http_port = http_port
        self.mm_spec = mm_spec
        self._multimodal_enabled = mm_spec is not None
        self._max_per_poll = max_per_poll
        # The engine-level startup timings `/server_info` reports, set once
        # the scheduler finishes initializing (see `set_startup_time`).
        self.startup_time: Optional[Dict[str, Any]] = None
        # Batches whose customized_info could not be sent (see
        # `_customized_info_json`).
        self._dropped_customized_info = 0

    @classmethod
    def _health_generate_headers(cls, scheduler: Scheduler) -> Dict[str, str]:
        """Headers a healthy ``/health_generate`` carries; a deployment's
        router may read its metadata off the deep health probe."""
        return {}

    @classmethod
    def _default_max_new_tokens(cls, scheduler: Scheduler) -> Optional[int]:
        """What a ``/generate`` request that omits ``max_new_tokens`` asks for:
        ``SamplingParams``'s default, as the Python ``TokenizerManager`` leaves
        it. A subclass may return ``None``, no limit, which the scheduler
        clamps to what the context window leaves after the prompt."""
        return SamplingParams().max_new_tokens

    @classmethod
    def _server_kwargs(cls, scheduler: Scheduler) -> Dict[str, Any]:
        """Keyword arguments a model package's own ``Server`` class takes
        beyond the shared ones (see ``_load_extension``)."""
        return {}

    @classmethod
    def _load_extension(cls) -> ModuleType:
        from sglang.srt.rust_extensions import load_rust_extension

        return load_rust_extension("sglang.srt.rust_extensions._server")

    @classmethod
    def _resolve_multimodal(cls, scheduler: Scheduler) -> Optional[RustMmSpec]:
        """Resolve the built-in Rust MM pipeline for a multimodal model, and
        refuse a model it cannot serve. ``launch`` calls this before the
        listener binds, so a refused model never accepts a connection. A
        model package whose ``_start_multimodal`` starts its own pipeline
        overrides this to return ``None``; its server admits no generation
        request until that pipeline starts."""
        mm_spec = RustMmProcessor(
            server_args=scheduler.server_args,
            model_config=scheduler.model_config,
            processor=scheduler.processor,
        ).resolve_spec()
        if mm_spec is None:
            supported = sorted(
                set(chain.from_iterable(f.model_types for f in RUST_MM_FAMILIES))
            )
            raise RuntimeError(
                "SGLANG_RUST_SERVER=1: no Rust MM pipeline for "
                f"model_type={scheduler.model_config.hf_config.model_type!r} "
                f"(supported: {', '.join(supported)}; "
                "images only). Unset SGLANG_RUST_SERVER to serve this model."
            )
        return mm_spec

    def _start_multimodal(self, scheduler: Scheduler) -> None:
        """Start the Rust workers for the pipeline ``launch`` resolved."""
        assert self.mm_spec is not None
        self.server.start_mm_workers(
            self._build_mm_spec(self.mm_spec),
            get_mm().mm_processor_worker_num or RustMmProcessor.AUTO_MM_WORKERS,
        )

    @staticmethod
    def _server_core_budget(allowed_core_count: int, mm_workers: int) -> int:
        """Maximum cores available to the Rust frontend and MM workers."""
        return max(8, mm_workers + 4)

    @classmethod
    def _partition_cores(
        cls, mm_workers: int = 0
    ) -> tuple[Optional[List[int]], Optional[List[int]]]:
        return _partition_cores(
            mm_workers=mm_workers,
            server_core_budget=cls._server_core_budget,
        )

    @classmethod
    def _port_offset(cls) -> int:
        """This listener's offset from the base port based on per-node DP rank."""
        if get_exec().moe.is_ep_scale_joiner:
            # The joining TP group is entirely local to this node.
            tp_size_per_node = get_parallel().tp_size
        else:
            nnodes_per_pp_rank = max(get_parallel().nnodes // get_parallel().pp_size, 1)
            tp_size_per_node = get_parallel().tp_size // nnodes_per_pp_rank
        dp_group_width = get_parallel().attn_tp_size * get_parallel().attn_cp_size
        if not get_parallel().attn_dp_enabled and get_parallel().dp_size > 1:
            # Plain DP: one TP group per replica, every leader on this node.
            return get_parallel().dp_rank
        return (get_parallel().tp_rank % tp_size_per_node) // dp_group_width

    @classmethod
    def gather_server_endpoints(cls, hosts_listener: bool) -> List[Dict[str, Any]]:
        """Collective: every rank calls once; returns listener endpoints by DP rank."""
        host = get_serving().host
        if host in ("", "0.0.0.0", "::"):
            # An unspecified bind address is not reachable from another
            # node; advertise the address the interfaces answer on.
            host = get_local_ip_auto(fallback="127.0.0.1")
        # Nodes that rendezvous on loopback share this machine, so a loopback
        # listener reaches them; otherwise it is a likely misconfiguration.
        if get_parallel().nnodes > 1 and _is_loopback(host):
            dist_init_addr = get_parallel().dist_init_addr
            if not _rendezvous_is_loopback(dist_init_addr):
                logger.warning(
                    "SGLANG_RUST_SERVER: this rank listens on loopback %r but the "
                    "%d nodes rendezvous at %r; a node on another machine cannot "
                    "reach this rank's control endpoint",
                    host,
                    get_parallel().nnodes,
                    dist_init_addr,
                )
        config = None
        if hosts_listener:
            port = get_serving().port + cls._port_offset()
            config = {
                "dp_rank": select_kv_publisher_dp_rank(
                    get_parallel().attn_dp_size,
                    get_parallel().attn_dp_rank,
                    get_parallel().dp_rank,
                ),
                "endpoint": NetworkAddress(host, port).to_url(),
            }
        gathered = get_parallel().world_group.all_gather_object(config)
        return sorted(
            (entry for entry in gathered if entry is not None),
            key=lambda entry: entry["dp_rank"],
        )

    @classmethod
    def launch(cls, scheduler: Scheduler) -> RustServer:
        """Start the embedded Rust server threads and bind the listen port.

        The caller gates this (``SGLANG_RUST_SERVER`` + rank 0); this always
        creates. A model the server cannot serve is refused before the port
        is bound.
        """
        # Force turn off HF tokenizers rayon's unpinned global thread pool.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Preserve the DP startup log; ports use node-local offsets.
        dp_rank = (
            get_parallel().attn_dp_rank if get_parallel().num_dp_ranks > 1 else None
        )
        port_offset = cls._port_offset()
        listen_port = get_serving().port + port_offset
        listen_addr = NetworkAddress(get_serving().host, listen_port).to_host_port_str()

        _, server_cores = cls._partition_cores(
            mm_workers=(
                (get_mm().mm_processor_worker_num or RustMmProcessor.AUTO_MM_WORKERS)
                if scheduler.model_config.is_multimodal
                else 0
            )
        )

        # Multimodal models must have a Rust pipeline — there is no Python
        # fallback. The listener answers as soon as the server is constructed
        # (a multimodal one admits generation requests once `_start_multimodal`
        # starts its pool), so refuse a model without one first.
        mm_spec = (
            cls._resolve_multimodal(scheduler)
            if scheduler.model_config.is_multimodal
            else None
        )

        extension = cls._load_extension()
        server = extension.Server(
            _build_server_args(
                scheduler,
                extension=extension,
                health_generate_headers=cls._health_generate_headers(scheduler),
                default_max_new_tokens=cls._default_max_new_tokens(scheduler),
            ),
            # None runs unpinned; otherwise the list carries the pinning decision.
            cores=server_cores,
            port_offset=port_offset,
            **cls._server_kwargs(scheduler),
        )
        instance = cls(server, http_port=listen_port)

        if scheduler.model_config.is_multimodal:
            # New threads inherit the spawning thread's affinity, and this launch
            # thread holds the full mask. Narrow it while the MM threads are
            # created below (the Rust MM workers, and any a model package's
            # `_start_multimodal` starts) so they stay off the scheduler's
            # reserved cores, where MM preprocessing would preempt the scheduler
            # loop and inflate inter-token latency. Then give it back its own
            # mask: the scheduler thread itself is not pinned, as in the Python
            # frontend, so a busy neighbour on one core never holds it (and every
            # TP rank waiting on it at the next collective) in that core's queue.
            scheduler_cpus = None
            if server_cores is not None:
                try:
                    full_mask = os.sched_getaffinity(0)
                    os.sched_setaffinity(0, set(server_cores))
                    scheduler_cpus = full_mask
                except OSError as e:
                    logger.warning(
                        "rust server: cannot confine mm threads to server cores: %s", e
                    )
            instance.mm_spec = mm_spec
            instance._start_multimodal(scheduler)
            instance._multimodal_enabled = True
            if scheduler_cpus is not None:
                os.sched_setaffinity(0, scheduler_cpus)

        # Under DP every rank runs its own server on its own port, so the rank is
        # what tells two otherwise identical startup lines apart.
        dp_note = (
            ""
            if dp_rank is None
            else f" (DP rank {dp_rank}/{get_parallel().num_dp_ranks})"
        )
        logger.info(
            "SGLANG_RUST_SERVER enabled, Rust server listen on %s%s",
            listen_addr,
            dp_note,
        )

        return instance

    def _wrap_mm_result(self, buffers: dict) -> MultimodalProcessorOutput:
        """Wrap one request's ``mm.*`` buffers (``{name: numpy array |
        ShmBuffer}``) into the scheduler's ``MultimodalProcessorOutput``. Model
        packages with an external processor override this for their own
        item layout; the buffer names are the contract (see ``wrap_encoded``).
        A wrapper only translates: it may build ``ShmPointerMMData`` stubs from
        a ``ShmBuffer``'s name, and the segment stays valid for every TP rank
        because ``drain`` settles its ownership after the wrapper returns."""
        assert self.mm_spec is not None
        return RustMmProcessor.wrap_encoded(self.mm_spec, buffers)

    def set_startup_time(self, scheduler_startup_time: Dict[str, Any]) -> None:
        """Record the startup timings in the shape of Python's
        ``TokenizerManager.startup_time``. That frontend's end-to-end phase
        (``tokenizer_e2e``) runs from launch until the schedulers are ready;
        this one serves from inside the scheduler process, so it runs from
        the process's spawn until its initialization ends, now."""
        self.startup_time = build_engine_startup_time(
            [scheduler_startup_time],
            tokenizer_e2e=time.time() - psutil.Process().create_time(),
        )

    def mm_transport_stats(self) -> Dict[str, int]:
        """Snapshot of the zero-copy accounting at the multimodal hand-off
        (see :class:`~sglang.srt.rust_server.multimodal.MmTransportStats`);
        surfaced on ``/server_info`` as ``rust_mm_transport``."""
        return MM_TRANSPORT_STATS.snapshot()

    def wait_request(self, timeout_ms: int) -> None:
        """Block until a request is pushed into the in-process ring or the timeout
        elapses.
        """
        self.server.wait_request(timeout_ms)

    def drain(self, max_recv: int) -> List[Any]:
        """Ingress: non-blocking drain of the in-process ring → list of decoded
        request objects. The scheduler's request receiver calls this instead of
        polling the zmq socket when `rust_server_mode` is set.

        Each request arrives as a scalar msgpack `header` (with `input_ids`
        and `token_ids_logprob` left nil) plus its named **buffers** -- the one
        data plane every non-scalar payload uses. An inline buffer is a shaped
        numpy array that owns the Rust vector (nothing was copied to get here);
        a shm buffer is a `ShmBuffer` naming the segment to map after the TP
        broadcast. The header is `msgpack_decode`d (yielding the same
        `TokenizedGenerateReqInput` / control objects the zmq path produces, so
        the IPC schema is tracked automatically) and each buffer is attached
        under its name: `input_ids` as the `array("q")` the scheduler expects
        (the one copy left, forced by that type), `token_ids_logprob` as a
        list, and the `mm.*` set through :meth:`_wrap_mm_result`.
        `recv_requests` never waits: the ring drain is `try_recv` (returns the
        instant the ring is dry, capped at `max_recv`) -- same contract as
        `zmq.NOBLOCK`. Parking for work is :meth:`wait_request`, which does
        release the GIL.
        """
        limit = max_recv if max_recv > 0 else self._max_per_poll
        out = []
        for req in self.server.recv_requests(limit):
            buffers = dict(req.buffers)
            # Rust holds the unlink duty for every shm segment until this loop
            # settles the request: released to the stubs on admission, unlinked
            # on rejection. The wrapper never decides this (see `_wrap_mm_result`).
            try:
                obj = self._admit(req.header, buffers)
            except Exception:
                settle_shm_buffers(buffers, admitted=False)
                continue
            settle_shm_buffers(buffers, admitted=True)
            out.append(obj)
        return out

    def _admit(self, header: bytes, buffers: dict):
        """Decode and attach one drained request, or report the rejection to
        its client and raise."""
        try:
            obj = msgpack_decode_explained(header)
        except MsgpackDecodeError as e:
            # Return 400 for malformed request field (e.g. token_ids_logprob=[[0]].
            logger.warning(
                "rust ingress: dropping undecodable request %s: %s", e.rid, e.reason
            )
            if e.rid is not None:
                self.server.push_error(e.rid, f"invalid request: {e.reason}")
            raise
        try:
            self._attach_buffers(obj, buffers)
        except Exception as e:
            logger.warning(
                "rust ingress: rejecting request %s: %s: %s",
                obj.rid,
                type(e).__name__,
                e,
            )
            self.server.push_error(obj.rid, f"invalid request: {e}")
            raise
        return obj

    def _attach_buffers(self, obj, buffers: dict) -> None:
        """Attach one drained request's named buffers to its decoded header."""
        ids = buffers.get("input_ids")
        if ids is not None:  # generate request
            obj.input_ids = array("q")
            obj.input_ids.frombytes(memoryview(ids).cast("B"))
        token_ids_logprob = buffers.get("token_ids_logprob")
        if token_ids_logprob is not None:
            obj.token_ids_logprob = token_ids_logprob.tolist()
        if self._multimodal_enabled and "mm.meta" in buffers:
            # Wrapping the worker's buffers into tensors is the only Python
            # step of the Rust path; a text-only request on a multimodal
            # model carries no `mm.*` buffers.
            obj.mm_inputs = self._wrap_mm_result(buffers)

    def push_control_output(self, recv_req, output) -> None:
        """Push a control-request response through the egress ring to the waiting
        request (routed by rid), encoded as **msgpack** (the ring's message
        format).

        A msgspec struct is converted to a *named map* (``structs.asdict``, since
        the IPC structs are ``array_like`` and would otherwise lose field names)
        so the Rust api_server can shape it per-endpoint (e.g. /server_info)
        before rendering JSON to the client — keeping JSON formatting off the
        scheduler's GIL.
        """

        # Invariant: control requests always carry a rust-minted rid; without
        # one the response is unroutable, so fail loudly rather than drop it.
        assert recv_req.rid is not None, (
            f"control response without rid: {type(output).__name__}"
        )
        # No local try/except: a failed push propagates to run_scheduler_process's
        # outer handler, which logs the full traceback (scheduler-fatal either way).
        payload = (
            msgspec.structs.asdict(output)
            if isinstance(output, msgspec.Struct)
            else output
        )
        # enc_hook stringifies non-native types (paths, enums); JSON
        # rendering happens in Rust.
        encoded = msgspec.msgpack.encode(payload, enc_hook=str)

        self.server.push_control_result(recv_req.rid, encoded)

    def push_abort(self, abort: AbortReq, req: Req) -> None:
        """The terminal frame of a request the scheduler ended outside the output
        stream (aborted while queued, preempted, retracted or rejected), which
        Python's ``TokenizerManager._handle_abort_req`` builds from the same
        ``AbortReq``: a one-request batch carrying no new tokens and the abort
        finish reason. The counts are the request's own, so the frame does not
        reset the ones a stream already reported. The frame is marked as ended
        outside the stream: like ``_handle_abort_req``, the frontend records no
        request metrics for it."""
        finish_reason = abort.finished_reason or {
            "type": "abort",
            "message": abort.abort_message or "Abort in waiting queue",
        }
        header_cols = [
            [abort.rid],
            [finish_reason],
            [len(req.origin_input_ids)],
            [0],
            [req.cached_tokens],
            [req.reasoning_tokens],
        ]
        header_cols += [[]] * _EXTRAS_HEADER_COLS
        header_cols += [
            [abort.weight_versions] if abort.weight_versions is not None else [],
            [],  # image_tokens
            [],  # audio_tokens
            [],  # video_tokens
            [],  # decode_latencies
            # Ended outside the output stream: `_handle_abort_req` records no
            # request metrics for it.
            [True],
        ]
        header = msgspec.msgpack.encode(header_cols)
        if not self.server.push_decode_result_batch(header, [b""]):
            logger.warning(
                "Rust egress closed; dropped abort of %s during shutdown", abort.rid
            )

    def push_health_ack(self) -> None:
        """Acknowledge the health probes skipped while the scheduler was busy."""
        self.server.push_health_ack()

    def push_generation(
        self,
        payload: BatchTokenIDOutput,
        decode_latencies: Dict[int, float],
        cached_tokens_split: Optional[List[Tuple[int, int, int]]] = None,
        storage_backend: Optional[str] = None,
    ) -> None:
        """Egress redirect for generation output (replaces the zmq detokenizer).

        Push the WHOLE batch into the Rust egress ring as one frame (-> detokenizer
        shards -> client streams), mirroring the ingress ``input_ids`` split so the
        bulk numeric columns never go through msgpack:

          - ``header``: msgpack ``BatchHeader`` positional array — the per-request
            scalar columns (``rids, finish_reasons, prompt_tokens, tok_lens,
            cached_tokens, reasoning_tokens``) plus
            the shape metadata for the optional families (``*_lens`` element counts
            for the flat logprob columns, ``*_reqlens``/``*_poslens`` for the ragged
            and hidden ones), then the finished requests' weight-version
            spans, a multimodal model's media token counts,
            ``decode_latencies`` (by row: a finished non-streaming request's
            decode latency in seconds), the scheduler's ``customized_info``
            (per key, each request's values for its new output tokens) as one
            JSON text, which the frontend accumulates and reports in
            ``meta_info`` as Python's TokenizerManager does,
            ``cached_tokens_split`` (by row: the request's cached tokens per
            cache tier, ``(device, host, storage)``, when the cache has more
            tiers than the device) and, for the whole frame, the attached
            ``storage_backend``.
          - ``data``: the raw little-endian numeric buffer: the token ids
            first, as the 8-byte ``array("q")`` bytes the scheduler already
            holds, then every logprob / hidden column as 4-byte elements
            (``f32`` values, ``i32`` indices), concatenated in the order the
            Rust ``for_each_chunk`` reads them.

        Logprobs are columnar: output families are per-step deltas, input
        (prefill) families ride once on the first chunk. Ragged families (top-k,
        token-ids) flatten a per-position ``list[list]`` into flat ``val``/``idx``
        buffers plus a per-position ``lens`` vector (0 = null position). Hidden
        states flatten to rows of floats (one row per output position).
        """
        output_ids = payload.output_ids or []
        prompt_tokens = payload.prompt_tokens or []

        # Hot-path guard: almost no decode step wants logprobs / hidden states,
        # so only then pay the per-request flatten + buffer packing below.
        has_extra = bool(
            payload.output_token_logprobs_val
            or payload.input_token_logprobs_val
            or payload.output_top_logprobs_val
            or payload.input_top_logprobs_val
            or payload.output_token_ids_logprobs_val
            or payload.input_token_ids_logprobs_val
            or payload.output_hidden_states
        )

        # Runs on the scheduler's CUDA-launch thread every decode step, so each
        # Python-level pass over the batch costs inter-token latency: `rids` are
        # the plain rid strings (hashed to a routing key on the Rust side with a
        # per-process seed, off the GIL — not parsed; a rid is any string),
        # `finished_reasons` already `dict | None`, and `output_ids` entries are
        # always `array("q")` slices of `Req.output_ids` (never None), so both
        # `map`s stay in C and their bytes go out as-is: one memcpy per request,
        # no per-element repack. Measured at 256 single-token requests: 10 us,
        # against 18 us for `array("i", chain.from_iterable(...))`.
        rids = payload.rids
        finish_reasons = payload.finished_reasons
        tok_lens = list(map(len, output_ids))
        flat_ids = b"".join(map(array.tobytes, output_ids))

        # Column order here MUST match BatchHeader (header_cols) and
        # for_each_chunk's read order (data_cols); the extras contribution
        # is ordered by the `extras` tuple below.
        header_cols = [
            rids,
            finish_reasons,
            prompt_tokens,
            tok_lens,
            payload.cached_tokens,
            payload.reasoning_tokens,
        ]
        data_cols = [flat_ids]

        if has_extra:
            # The `extras` tuple is the SINGLE source of the extras column
            # order — it must match the Rust ``BatchHeader`` fields and
            # ``for_each_chunk``'s read order.
            #
            # TODO(perf): the per-request flatten assumes the logprob/hidden
            # columns are ragged, non-contiguous nested Python lists — which is
            # only an assumption. The scheduler moves these off the GPU with
            # `tensor.tolist()`, so revisit whether the upstream values are
            # still contiguous tensors; if so, ship raw bytes + a shape
            # descriptor and skip the flatten entirely.
            batch_size = len(rids)
            extras = (
                FlatPairColumns(
                    "output_token_logprobs",
                    payload.output_token_logprobs_val or [],
                    payload.output_token_logprobs_idx or [],
                ),
                FlatPairColumns(
                    "input_token_logprobs",
                    payload.input_token_logprobs_val or [],
                    payload.input_token_logprobs_idx or [],
                    first_none_to_nan=True,
                ),
                RaggedPairColumns(
                    "output_top_logprobs",
                    payload.output_top_logprobs_val or [],
                    payload.output_top_logprobs_idx or [],
                ),
                RaggedPairColumns(
                    "input_top_logprobs",
                    payload.input_top_logprobs_val or [],
                    payload.input_top_logprobs_idx or [],
                ),
                RaggedPairColumns(
                    "output_token_ids_logprobs",
                    payload.output_token_ids_logprobs_val or [],
                    payload.output_token_ids_logprobs_idx or [],
                ),
                RaggedPairColumns(
                    "input_token_ids_logprobs",
                    payload.input_token_ids_logprobs_val or [],
                    payload.input_token_ids_logprobs_idx or [],
                ),
                NestedRowColumns(
                    "output_hidden_states", payload.output_hidden_states or []
                ),
            )

            # Every column is all-or-nothing per payload — which is also what makes
            # a family's emptiness a reliable "nobody asked for this" signal.
            active = []
            for extra in extras:
                populated = False
                for name, col in extra.columns():
                    assert len(col) in (
                        0,
                        batch_size,
                    ), (
                        f"extras column {name}: {len(col)} entries for a batch of {batch_size}"
                    )
                    populated |= len(col) > 0
                if populated:
                    active.append(extra)

            # Flatten only the families someone asked for. `has_extra` above is a
            # per-FRAME guard, so one client enabling logprobs used to drag all
            # seven families through the per-request loop: at B=4096 that is 28,672
            # bound-method calls per decode step, materializing 12 columns of 4096
            # zeros nobody reads. Measured 0.37 ms -> 7.90 ms GIL-held per step,
            # i.e. 25-75% of a decode step added to the scheduler's critical path.
            #
            # Skipping `accept` leaves a family's buffers empty, which is exactly
            # the wire form the Rust decoder already treats as absent (`per_req_ok`
            # admits an empty column, `lens_i` reads 0 for every request). The
            # `header_cols`/`data_cols` loops below still walk all seven, so column
            # ORDER and arity are unchanged — an inactive family contributes empty
            # columns in place rather than disappearing.
            for extra in active:
                accept = extra.accept  # hoisted: this is the hottest loop here
                for i in range(batch_size):
                    accept(i)

            extras_start = len(header_cols)
            for extra in extras:
                header_cols += extra.header_cols()
                data_cols += extra.data_cols()
            # The padding below must line up with these columns.
            assert len(header_cols) - extras_start == _EXTRAS_HEADER_COLS, (
                "_EXTRAS_HEADER_COLS no longer matches the extras families"
            )

        # After the extras: the finished requests' weight-version spans (None
        # for the others), a multimodal model's media token counts, the
        # finished requests' decode latencies, the customized_info columns and
        # the cached tokens by tier, each empty when nothing in the batch
        # carries it, then the frame's storage backend. Only the latencies are
        # built here, for finished rows; the header stops at its last non-empty
        # column.
        latencies = []
        if decode_latencies:
            latencies = [None] * len(rids)
            for row, latency in decode_latencies.items():
                latencies[row] = latency
        media = self._multimodal_enabled
        trailing = [
            payload.weight_versions or [],
            (payload.image_tokens or []) if media else [],
            (payload.audio_tokens or []) if media else [],
            (payload.video_tokens or []) if media else [],
            latencies,
            [],  # abort_outputs, push_abort's column
            self._customized_info_json(payload.customized_info),
            cached_tokens_split or [],
            storage_backend,
        ]
        while trailing and not trailing[-1]:
            trailing.pop()
        if trailing:
            if not has_extra:
                header_cols += [[]] * _EXTRAS_HEADER_COLS
            header_cols += trailing

        header = msgspec.msgpack.encode(header_cols)
        # Pass the raw column list; the Rust side concatenates it into the frame
        # with the GIL released.
        if not self.server.push_decode_result_batch(header, data_cols):
            logger.warning(
                "Rust egress closed; dropped batch of %d requests during shutdown",
                len(rids),
            )

    def _customized_info_json(self, customized_info: Optional[dict]) -> bytes:
        """The batch's customized_info as the JSON text Python's responses
        carry it in (orjson, with `dumps_json`'s options), a value orjson has
        no form for as its `str`; empty when there is none. A batch even that
        cannot encode (say, an int past 64 bits) goes without it, logged, so a
        model's values never fail the scheduler step."""
        if not customized_info:
            return b""
        try:
            return orjson.dumps(
                customized_info, option=ORJSON_RESPONSE_OPTIONS, default=str
            )
        except orjson.JSONEncodeError as error:
            self._dropped_customized_info += 1
            count = self._dropped_customized_info
            if count & (count - 1) == 0:
                logger.warning(
                    "Dropped customized_info the Rust frontend cannot send "
                    "(%d batches so far): %s",
                    count,
                    error,
                )
            return b""

    @classmethod
    def _build_mm_spec(cls, spec: RustMmSpec) -> MmSpec:
        """The typed MM configuration for ``Server.start_mm_workers``: the
        :class:`RustMmSpec` fields the Rust pipeline consumes, as the Rust
        extension's own ``MmSpec`` class (same required-keyword contract as
        :func:`_build_server_args`; ``family`` / ``resample`` become the
        extension's ``MmFamily`` / ``MmResample`` enums)."""
        ext = cls._load_extension()
        family = {"qwen_vl": ext.MmFamily.QwenVl}[spec.family]
        resample = {"aten_u8": ext.MmResample.AtenU8, "pil": ext.MmResample.Pil}[
            spec.resample
        ]
        return ext.MmSpec(
            family=family,
            feature_shm=spec.feature_shm,
            image_token_id=spec.image_token_id,
            patch_size=spec.patch_size,
            merge_size=spec.merge_size,
            temporal_patch_size=spec.temporal_patch_size,
            min_pixels=spec.min_pixels,
            max_pixels=spec.max_pixels,
            image_mean=spec.image_mean,
            image_std=spec.image_std,
            resample=resample,
        )
