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
import threading
from array import array
from typing import TYPE_CHECKING, Any, List, Optional

import msgspec

from sglang.srt.environ import envs
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

        # `maybe_create` runs once per DP group (gated on the group's first TP
        # rank). With dp_size>1 the frontends would otherwise all bind the same
        # `host:port`, so the address is per-DP-rank:
        #   * Mode B (standalone api-server): the rank runs **headless** on a TCP
        #     listener (`headless_server_bind`); a separate api-server process connects in.
        #   * Mode A (embedded): the rank serves HTTP on `port + dp_rank`; an
        #     external router / client LB fans out.
        #   * dp_size==1: embedded HTTP on the base port (`bind=None` → the Rust
        #     side resolves `host:port` from server_args).
        dp_rank = scheduler.ps.attn_dp_rank
        bind = None
        headless_server_bind = None
        if server_args.dp_size > 1 and envs.SGLANG_RUST_STANDALONE_API_SERVER.get():
            headless_server_bind = rust_headless_tcp_bind(server_args, dp_rank)
        elif server_args.dp_size > 1:
            bind = f"{server_args.host}:{server_args.port + dp_rank}"

        # The bind address, tokenizer source/revision, and tokenizer/detok worker
        # counts live in the dumped `server_args` blob, so the Rust side resolves
        # them itself. Only the Python-computed core partition and the per-DP-rank
        # `bind` / `tcp_bind` override (none part of server_args) pass here.
        server = Server(
            pin_cores=cores is not None,
            cores=cores,
            bind=bind,
            headless_server_bind=headless_server_bind,
            server_args_json=cls._build_server_args(scheduler),
        )
        logger.info(
            "SGLANG_RUST_SERVER enabled: Rust server (dp_rank=%s, %s) on %s",
            dp_rank,
            "headless TCP" if headless_server_bind else "embedded HTTP",
            headless_server_bind or bind or f"{server_args.host}:{server_args.port}",
        )
        # Mode B: the rank's headless listener is up (the `Server` above bound it),
        # so report the address the api-server should dial. Registration drives the
        # api-server's deferred pool connect (it has no deterministic endpoint
        # list); done on a daemon thread so scheduler startup isn't blocked.
        if headless_server_bind is not None:
            _register_headless_rank(server_args, dp_rank)
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
        """JSON blob of the scheduler's ``server_args`` for its embedded Rust
        server (carries the already-resolved ``model_config``)."""
        return RustServer._dump_server_args_json(
            scheduler.server_args, scheduler.model_config
        )

    @staticmethod
    def _dump_server_args_json(server_args, model_config) -> str:
        """JSON blob of ``server_args`` handed to the Rust server.

        Recursively dumps the JSON-safe fields of ``server_args``, drilling into
        nested dataclasses, dicts, lists, and generic objects (via ``__dict__``).
        Binary, tensors, and non-serializable leaves (torch dtypes, callables)
        are dropped; cycles are guarded by an identity set. Read-only static
        config; live state goes through the request pipeline.

        ``model_config`` is attached to ``server_args`` first so the dump carries
        model-derived values (e.g. ``model_config.context_len``). Shared by the
        embedded scheduler path and the standalone api-server process, which
        resolves its own ``ModelConfig`` (it has no scheduler).
        """
        # Make the resolved model_config part of server_args so the recursive
        # dump includes it (context_len, architecture config, …).
        server_args.model_config = model_config

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

        return json.dumps(to_json_safe(server_args, set()))

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


# --- Standalone api-server (Mode B): SGLANG_RUST_SERVER + ---------------------
# SGLANG_RUST_STANDALONE_API_SERVER + dp_size>1. The DP ranks run headless (TCP)
# and a single api-server process, spawned by the node-0 launcher, serves the
# HTTP API and connects to every rank. The api-server has no deterministic
# endpoint list: each rank reports the address to dial via `POST
# /internal/register`, so per-rank node IPs need no out-of-band discovery and the
# api-server connects its pool only once all ranks have registered.


def _rust_headless_port(server_args, dp_rank: int) -> int:
    """Per-DP-rank headless TCP port, offset from the HTTP ``port`` so it never
    collides with the api-server's ``port``."""
    return server_args.port + 1 + dp_rank


def rust_headless_tcp_bind(server_args, dp_rank: int) -> str:
    """Address DP rank ``dp_rank``'s headless listener **binds** to. Single-node:
    loopback. Multi-node (``nnodes > 1``): all interfaces, so the api-server on
    node 0 can reach ranks on other nodes — the concrete address it dials is the
    one the rank advertises (:func:`rust_headless_advertise_addr`)."""
    host = "0.0.0.0" if server_args.nnodes > 1 else "127.0.0.1"
    return f"{host}:{_rust_headless_port(server_args, dp_rank)}"


def rust_headless_advertise_addr(server_args, dp_rank: int) -> str:
    """Address the api-server should **dial** for DP rank ``dp_rank`` — the value
    the rank reports via registration. Single-node: loopback. Multi-node: this
    node's auto-detected reachable IP."""
    if server_args.nnodes > 1:
        from sglang.srt.utils.network import get_local_ip_auto

        host = get_local_ip_auto(fallback=server_args.host)
    else:
        host = "127.0.0.1"
    return f"{host}:{_rust_headless_port(server_args, dp_rank)}"


def _api_server_register_url(server_args) -> str:
    """URL of the node-0 standalone api-server's ``/internal/register`` endpoint.
    Node 0 is the ``dist_init_addr`` host (loopback for a single node)."""
    if server_args.dist_init_addr:
        from sglang.srt.utils.network import NetworkAddress

        host = NetworkAddress.parse(server_args.dist_init_addr).host
    else:
        host = "127.0.0.1"
    return f"http://{host}:{server_args.port}/internal/register"


def _register_headless_rank(server_args, dp_rank: int) -> None:
    """Report this headless DP rank's TCP endpoint to the standalone api-server so
    it can dial the rank and (once all ranks register) connect its pool.

    Runs on a daemon thread with bounded retry: the api-server (node 0) may not be
    listening yet, while this rank's own listener is already up — so the
    api-server connects as soon as it receives the registration.
    """
    url = _api_server_register_url(server_args)
    endpoint = rust_headless_advertise_addr(server_args, dp_rank)

    def _post() -> None:
        import time
        import urllib.error
        import urllib.request

        body = json.dumps({"dp_rank": dp_rank, "endpoint": endpoint}).encode()
        deadline = time.monotonic() + 120.0
        attempt = 0
        while True:
            attempt += 1
            try:
                req = urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp.read()
                logger.info(
                    "Rust headless rank %s registered %s -> %s", dp_rank, endpoint, url
                )
                return
            except (urllib.error.URLError, OSError) as e:
                if time.monotonic() >= deadline:
                    logger.error(
                        "Rust headless rank %s failed to register to %s after %d attempts: %s",
                        dp_rank,
                        url,
                        attempt,
                        e,
                    )
                    return
                time.sleep(0.2)

    threading.Thread(
        target=_post, daemon=True, name=f"rust-register-dp{dp_rank}"
    ).start()


def run_rust_api_server(server_args) -> None:
    """Entry point for the standalone Rust api-server (Mode B), run on a daemon
    thread of the node-0 launcher.

    Resolves its own ``ModelConfig`` (it has no scheduler), dumps server_args,
    and hands off to the Rust ``run_api_server``, which serves the OpenAI /
    ``/generate`` HTTP API plus ``/internal/register`` and connects to the DP
    ranks once they register. Blocks (with the GIL released) for the process
    lifetime.
    """
    import sglang_server

    from sglang.srt.configs.model_config import ModelConfig

    model_config = ModelConfig.from_server_args(server_args)
    server_args_json = RustServer._dump_server_args_json(server_args, model_config)

    bind = f"{server_args.host}:{server_args.port}"
    logger.info(
        "Rust standalone api-server: HTTP %s, awaiting %d DP-rank registration(s)",
        bind,
        server_args.dp_size,
    )
    sglang_server.run_api_server(bind, server_args_json=server_args_json)
