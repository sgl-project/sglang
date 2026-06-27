"""Embedded Rust server lifecycle for the scheduler.

Keeps all `SGLANG_RUST_SERVER` plumbing — startup, CPU-core partitioning, the
`server_args` blob, and control-response routing — out of `scheduler.py`. The
scheduler holds an `Optional[RustServer]` and delegates to it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from array import array
from typing import TYPE_CHECKING, Any, List, Optional

import msgspec

from sglang.srt.managers.io_struct import msgpack_decode

if TYPE_CHECKING:
    from sglang_server import Server

    from sglang.srt.managers.io_struct import BatchTokenIDOutput
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class RustServer:
    """Owns the embedded multi-threaded Rust server (``sglang_server.Server``).

    Started on the rank-0 scheduler when ``SGLANG_RUST_SERVER`` is set. The
    server owns the API server, TokenizerManager, Tokenizer, and Detokenizer
    (all Rust threads in this process) and produces ingress requests into an
    in-process ring drained via :attr:`recv_from_tokenizer`. Stages 1-5 never
    touch a ``PyObject``, so they run concurrently with the scheduler without
    contending for the GIL.
    """

    def __init__(self, server: Server, max_per_poll: int = 256):
        # The underlying `sglang_server.Server`.
        self.server = server
        # Cap for an unbounded (`<= 0`) drain limit.
        self._max_per_poll = max_per_poll
        # Per-rid egress chunk sequence counter (rust-server-specific state).
        self._seq_id: dict = {}

    @classmethod
    def maybe_create(cls, scheduler: Scheduler) -> RustServer:
        """Start the Rust server on the rank-0 scheduler if enabled, else ``None``."""

        # Lazy import: only required when the rust server is actually enabled, so
        # plain (non-rust) builds without the extension don't need it installed.
        from sglang_server import Server

        server_args = scheduler.server_args

        # Partition this rank's allowed CPU cores (already NUMA-local when
        # SGLANG_SET_CPU_AFFINITY / NUMA bind is on): reserve a few cores for
        # this scheduler's CUDA-launch / event-loop thread and hand the rest to
        # the Rust pools, so tokenize/detok never share cores with the
        # latency-critical launch thread.
        cores = cls._partition_cores()

        # Keep HF tokenizers (used by the Rust dynamo-tokenizers backend) off
        # rayon's unpinned global thread pool. The Rust pool parallelizes
        # tokenization *across* requests (one sequential encode per pinned core),
        # not *within* a call via encode_batch — so forcing sequential per-call
        # keeps rayon from spawning floating threads that would defeat core
        # pinning / NUMA isolation. setdefault so an explicit choice still wins.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # The bind address (host:port), tokenizer source/revision, and
        # tokenizer/detok worker counts all live in the dumped `server_args`
        # blob, so the Rust side resolves them itself (tokenizer_path falls back
        # to model_path there). Only the Python-computed core partition — which
        # isn't part of server_args — needs passing here.
        server = Server(
            pin_cores=cores is not None,
            cores=cores,
            server_args_json=cls._build_server_args(scheduler),
        )
        logger.info(
            "SGLANG_RUST_SERVER enabled: embedded Rust server listening on %s:%s",
            server_args.host,
            server_args.port,
        )
        return cls(server)

    def drain(self, max_recv: int) -> List[Any]:
        """Ingress: non-blocking drain of the in-process ring → list of decoded
        request objects. The scheduler's request receiver duck-types on this
        method (`hasattr(recv_from_tokenizer, "drain")`) to use the rust path
        instead of the zmq socket.

        The transfer is **columnar**: `recv_requests` returns scalar msgpack
        `headers` (with `input_ids` omitted) plus one concatenated raw int64
        `ids_buf` and per-request `lengths`, so the large `input_ids` tensor never
        goes through msgpack. Each header is `msgpack_decode`d (yielding the same
        `TokenizedGenerateReqInput` / control objects the zmq path produces, so
        the IPC schema is tracked automatically) and its `input_ids` slice is
        wrapped as the `array("q")` the scheduler expects. `recv_requests`
        releases the GIL for the drain + concat, so this never holds the GIL
        across a wait — same contract as `zmq.NOBLOCK`.
        """
        limit = max_recv if max_recv > 0 else self._max_per_poll
        headers, ids_buf, lengths = self.server.recv_requests(limit)
        if not headers:
            return []

        ids_view = memoryview(ids_buf)
        out = []
        pos = 0  # byte offset into ids_buf
        for header, n in zip(headers, lengths):
            obj = msgpack_decode(header)
            if n:  # generate request: attach its int64 ids slice as array("q")
                nbytes = n * 8
                ids = array("q")
                ids.frombytes(ids_view[pos : pos + nbytes])
                obj.input_ids = ids
                pos += nbytes
            out.append(obj)
        return out

    def push_control_output(self, recv_req, output) -> None:
        """Push a control-request response through the egress ring to the waiting
        request (routed by rid), encoded as **msgpack** (the ring's native
        format).

        A msgspec struct is converted to a *named map* (``structs.asdict``, since
        the IPC structs are ``array_like`` and would otherwise lose field names)
        so the Rust api_server can shape it per-endpoint (e.g. /server_info)
        before rendering JSON to the client — keeping JSON formatting off the
        scheduler's GIL.
        """

        try:
            payload = (
                msgspec.structs.asdict(output)
                if isinstance(output, msgspec.Struct)
                else output
            )
            # enc_hook stringifies non-native types (paths, enums); JSON
            # rendering happens in Rust.
            encoded = msgspec.msgpack.encode(payload, enc_hook=str)
            rid = getattr(recv_req, "rid", None) or "0"
            self.server.push_result(rid, encoded)
        except Exception as e:
            logger.warning(
                "rust control response push failed (%s): %s",
                type(output).__name__,
                e,
            )
            raise ValueError(
                f"rust control response push failed ({type(output).__name__}): {e}"
            )

    def push_generation(self, payload: BatchTokenIDOutput) -> None:
        """Egress redirect for generation output (replaces the zmq detokenizer).

        Fan the batch out into per-request ChunkEvents and push them into the
        Rust egress ring (-> detokenizer shard -> client stream). ChunkEvent is a
        positional msgpack array ``[rid, seq, token_ids, finish_reason,
        prompt_tokens]`` (which rmp-serde decodes from an array).
        ``output_ids[i]`` is the incremental new output tokens for this step;
        ``finished_reasons[i]``'s ``type`` becomes the chunk's finish reason;
        ``prompt_tokens[i]`` rides along so the egress can report usage.
        """
        output_ids = payload.output_ids or []
        prompt_tokens = payload.prompt_tokens or []
        for i, rid in enumerate(payload.rids):
            ids = output_ids[i] if i < len(output_ids) else None
            token_ids = list(ids) if ids is not None else []
            fr = payload.finished_reasons[i]
            finish = fr.get("type") if isinstance(fr, dict) else None
            n_prompt = prompt_tokens[i] if i < len(prompt_tokens) else 0

            seq = self._seq_id.get(rid, 0)
            self._seq_id[rid] = seq + 1
            chunk = msgspec.msgpack.encode([rid, seq, token_ids, finish, n_prompt])
            if not self.server.push_chunk(chunk):
                logger.warning("Rust egress ring full; dropped chunk for %s", rid)
            if finish is not None:
                self._seq_id.pop(rid, None)

    @staticmethod
    def _build_server_args(scheduler: Scheduler) -> str:
        """JSON blob of ``server_args`` handed to the Rust server.

        Recursively dumps the JSON-safe fields of ``server_args``, drilling into
        nested dataclasses, dicts, lists, and generic objects (via ``__dict__``).
        Binary, tensors, and non-serializable leaves (torch dtypes, callables)
        are dropped; cycles are guarded by an identity set. Read-only static
        config; live state goes through the request pipeline.

        The resolved ``model_config`` is attached to ``server_args`` first so the
        dump carries model-derived values (e.g. ``model_config.context_len``).
        """
        # Make the resolved model_config part of server_args so the recursive
        # dump includes it (context_len, architecture config, …).
        scheduler.server_args.model_config = scheduler.model_config

        drop = object()

        def to_json_safe(obj, seen):
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (bytes, bytearray, memoryview)):
                return drop
            if isinstance(obj, (list, tuple)):
                return [
                    s for s in (to_json_safe(v, seen) for v in obj) if s is not drop
                ]
            if isinstance(obj, dict):
                items = obj.items()
            elif hasattr(obj, "__dict__") and not callable(obj):
                # vars() captures declared dataclass fields *and* dynamically-set
                # attributes (e.g. the model_config attached to server_args, and
                # model_config.context_len set in __init__).
                items = ((k, v) for k, v in vars(obj).items() if not k.startswith("_"))
            elif dataclasses.is_dataclass(obj):  # slotted dataclass (no __dict__)
                items = (
                    (f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)
                )
            else:
                return drop  # opaque leaf (torch dtype, callable, …)
            oid = id(obj)
            if oid in seen:
                return drop
            seen.add(oid)
            out = {
                str(k): s
                for k, v in items
                for s in (to_json_safe(v, seen),)
                if s is not drop
            }
            seen.discard(oid)
            return out

        return json.dumps(to_json_safe(scheduler.server_args, set()))

    @staticmethod
    def _partition_cores() -> Optional[List[int]]:
        """Reserve launch cores for this scheduler and return the rest for the
        Rust server pools.

        Pins *this* (event-loop / CUDA-launch) thread to the reserved cores and
        returns the remaining allowed cores for the server. Both sets are a
        subset of this rank's NUMA-local cores (when affinity/NUMA bind is on),
        so the partition stays NUMA-local. Returns ``None`` (server runs
        unpinned, confined only by the process affinity) when the platform has
        no affinity API or too few cores to split.
        """
        if not hasattr(os, "sched_getaffinity"):
            return None
        try:
            allowed = sorted(os.sched_getaffinity(0))
        except OSError as e:
            logger.warning("rust server: cannot read cpu affinity: %s", e)
            return None

        # Need enough cores to reserve launch cores and still pin the pools.
        if len(allowed) < 4:
            logger.info(
                "rust server: only %d cores allowed; running pools unpinned",
                len(allowed),
            )
            return None

        # Keep a small slice for the launch loop; cap at 2 (the event loop is
        # effectively serial) and never take more than a quarter of the cores.
        reserve = min(2, len(allowed) // 4)
        launch_cores = allowed[:reserve]
        server_cores = allowed[reserve:]

        try:
            # pid 0 == this thread (the scheduler event-loop / launch thread).
            os.sched_setaffinity(0, set(launch_cores))
        except OSError as e:
            logger.warning("rust server: cannot pin launch thread: %s", e)
            return None

        logger.info(
            "rust server cores=%s, scheduler launch cores=%s",
            server_cores,
            launch_cores,
        )
        return server_cores
