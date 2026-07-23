"""Embedded Rust server lifecycle for the scheduler.

The Rust server replaces the Python api-server + `TokenizerManager` +
`DetokenizerManager` stack (hence this module sits beside them in `managers/`),
running them as Rust threads inside the scheduler process. This wrapper keeps
all `SGLANG_RUST_SERVER` plumbing — startup, CPU-core partitioning, the
`server_args` blob, and control-response routing — out of `scheduler.py`. The
scheduler holds an `Optional[RustServer]` and delegates to it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from array import array
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import msgspec

from sglang.srt.environ import envs
from sglang.srt.managers.utils import (
    MsgpackDecodeError,
    msgpack_decode_explained,
)
from sglang.srt.utils.flatten import (
    FlatPairColumns,
    NestedRowColumns,
    RaggedPairColumns,
)
from sglang.srt.utils.hf_transformers.common import _resolve_local_or_cached_file
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang_server import Server

    from sglang.srt.managers.io_struct import BatchTokenIDOutput
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class MmProcessorHost:
    """Hosts the Python mm-processor stack for the Rust MM worker pool.

    The Rust ingress parks multimodal requests in its ``Encoding`` stage; a
    Rust-owned worker pool (``Server.start_mm_workers``) drains them and calls
    :meth:`handle_sync`, which runs the *same* model-specific ``mm_processor``
    + shared helper the Python TokenizerManager uses
    (``mm_request_processing.run_mm_processor_for_request``), then

      1. stores the processed ``(mm_inputs, token_type_ids)`` in the rid-keyed
         ``results`` table (consumed by :meth:`RustServer.drain`, which attaches
         it to the decoded ``TokenizedGenerateReqInput`` — by reference, no
         pickle hop), and
      2. returns the final placeholder-expanded ``input_ids`` to the Rust
         worker, which resumes the parked request (an exception rejects it as
         a 400).

    The store happens strictly *before* the return, and the scheduler can only
    drain the request after the Rust worker resumes it, so a drained mm request
    always finds its table entry.

    Threading: the Rust worker pool owns the dispatch threads (its size bounds
    in-flight requests so MM preprocessing can't starve the scheduler loop);
    this class only runs a private asyncio loop on one daemon thread because
    the processors' native interface is ``async``. Heavy work (media decode,
    HF processor tensor ops) runs in the processor's own executors /
    GIL-releasing C code.
    """

    # Rust mm-worker threads == max concurrently-processed mm requests
    # (GIL-contention bound).
    MM_WORKERS = 8

    def __init__(self, scheduler: Scheduler):
        # Lazy imports: this class is only instantiated for multimodal models
        # under SGLANG_RUST_SERVER.
        from sglang.srt.managers.multimodal_processor import (
            get_mm_processor,
            import_processors,
        )
        from sglang.srt.managers.tokenizer_manager import (
            _determine_tensor_transport_mode,
            _get_processor_wrapper,
        )
        from sglang.srt.utils.hf_transformers_utils import (
            get_tokenizer_from_processor,
        )

        self.server_args = scheduler.server_args
        self.model_config = scheduler.model_config
        self.max_req_input_len = getattr(scheduler, "max_req_input_len", None)

        # rid -> (mm_inputs, token_type_ids); popped by RustServer.drain.
        self.results: Dict[str, Tuple[Any, Optional[List[int]]]] = {}

        # Same processor stack the Python TokenizerManager builds
        # (init_tokenizer_and_processor, multimodal branch). The HF
        # AutoProcessor is reused from the scheduler's init_tokenizer (already
        # loaded, identical construction args) when available; under
        # skip_tokenizer_init the scheduler has none, so load one — the mm
        # processor still needs it to encode images.
        import_processors("sglang.srt.multimodal.processors")
        if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
            import_processors(mm_process_pkg, overwrite=True)
        _processor = getattr(scheduler, "processor", None) or _get_processor_wrapper(
            self.server_args
        )
        transport_mode = _determine_tensor_transport_mode(self.server_args)
        self.mm_processor = get_mm_processor(
            self.model_config.hf_config,
            self.server_args,
            _processor,
            transport_mode,
            model_config=self.model_config,
        )
        if self.server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer_from_processor(_processor)

        self.loop = asyncio.new_event_loop()
        threading.Thread(
            target=self.loop.run_forever, name="rust-mm-host-loop", daemon=True
        ).start()
        logger.info(
            "rust server: MM processor host started (processor=%s)",
            type(self.mm_processor).__name__,
        )

    def handle_sync(self, rid: str, payload: bytes) -> bytes:
        """Entry point called by the Rust MM workers: process one mm request and
        return the final ``input_ids`` as raw little-endian int64 bytes. An
        exception propagates to the worker, which rejects the request as a 400
        (matching the Python path). ``fut.result()`` releases the GIL while the
        coroutine runs on the host loop."""
        fut = asyncio.run_coroutine_threadsafe(self._process(rid, payload), self.loop)
        input_ids, mm_inputs, token_type_ids = fut.result()
        # Store BEFORE returning — the Rust worker resumes the request (and the
        # scheduler drains it) strictly after, so the entry is always there.
        self.results[rid] = (mm_inputs, token_type_ids)
        if hasattr(input_ids, "tolist"):  # tensor-shaped processor output
            input_ids = input_ids.tolist()
        return array("q", input_ids or []).tobytes()

    async def _process(self, rid: str, payload: bytes):
        """The tokenize-then-mm-process sequence of
        ``TokenizerManager._tokenize_one_request``, for one Rust-parked request."""
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.managers.mm_request_processing import (
            run_mm_processor_for_request,
        )

        text, input_ids, image_data, video_data, audio_data = msgspec.msgpack.decode(
            payload
        )
        obj = GenerateReqInput(
            rid=rid,
            text=text,
            input_ids=input_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
        )

        # Mirror _tokenize_one_request's pre-mm tokenize (the mm processor's
        # output overrides these ids with the placeholder-expanded prompt).
        input_text = obj.text
        if obj.input_ids is None:
            if self.tokenizer is None:
                raise ValueError(
                    "skip_tokenizer_init is set: multimodal text prompts require "
                    "input_ids"
                )
            if not input_text and obj.contains_mm_input():
                # Audio-only (e.g. Whisper): the mm processor provides input_ids.
                input_ids = []
            elif not getattr(self.tokenizer, "is_fast", False):
                input_ids = self.tokenizer.encode(input_text)
            else:
                input_ids = self.tokenizer(input_text)["input_ids"]
        else:
            input_ids = obj.input_ids

        return await run_mm_processor_for_request(
            obj=obj,
            input_text=input_text,
            input_ids=input_ids,
            token_type_ids=None,
            mm_processor=self.mm_processor,
            hf_architectures=self.model_config.hf_config.architectures,
            server_args=self.server_args,
            max_req_input_len=self.max_req_input_len,
            mm_receiver=None,
        )


class RustServer:
    """Owns the embedded multi-threaded Rust server (``sglang_server.Server``).

    The server owns the api-server, tokenizermanager, tokenizer, and detokenizer
    all implemented as Rust threads in scheduler process.
    """

    def __init__(
        self,
        server: Server,
        mm_host: Optional[MmProcessorHost] = None,
        max_per_poll: int = 256,
    ):
        self.server = server
        self.mm_host = mm_host
        self._max_per_poll = max_per_poll

    @classmethod
    def launch(cls, scheduler: Scheduler) -> RustServer:
        """Start the embedded Rust server threads and bind the listen port.

        The caller gates this (``SGLANG_RUST_SERVER`` + rank 0); this always
        creates.
        """

        # Lazy import: only required when the rust server is actually enabled.
        from sglang_server import Server

        # Force turn off HF tokenizers rayon's unpinned global thread pool.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        server_args = scheduler.server_args
        http_addr = f"{server_args.host}:{server_args.port}"
        launch_cores, server_cores = cls._partition_cores()

        server = Server(
            # None -> run unpinned; the list carries the pinning decision.
            cores=server_cores,
            http_addr=http_addr,
            server_args_json=cls._build_server_args(scheduler),
        )

        # Multimodal models get the in-process MM processor host; Rust worker
        # threads drive the same Python mm_processor the Python
        # TokenizerManager would.
        mm_host = None
        if scheduler.model_config.is_multimodal:
            mm_host = MmProcessorHost(scheduler)
            server.start_mm_workers(mm_host.handle_sync, MmProcessorHost.MM_WORKERS)

        # Narrow the scheduler thread only after the server threads are launched.
        if launch_cores is not None:
            try:
                # pid 0 == this thread (the scheduler event-loop / launch thread).
                os.sched_setaffinity(0, set(launch_cores))
            except OSError as e:
                logger.warning("rust server: cannot pin scheduler launch thread: %s", e)

        logger.info(
            "SGLANG_RUST_SERVER enabled, Rust server listen on %s",
            http_addr,
        )

        return cls(server, mm_host=mm_host)

    def wait_ingress(self, timeout_ms: int) -> None:
        """Block until a request is pushed into the in-process ring or the timeout
        elapses.
        """
        self.server.wait_ingress(timeout_ms)

    def drain(self, max_recv: int) -> List[Any]:
        """Ingress: non-blocking drain of the in-process ring → list of decoded
        request objects. The scheduler's request receiver calls this instead of
        polling the zmq socket when `rust_server_mode` is set.

        The transfer is **columnar**: `recv_requests` returns an `IngressBatch`
        of scalar msgpack `headers` (with `input_ids` omitted) plus one
        concatenated raw int64 `data` buffer and per-request `lengths`, so the
        large `input_ids` lists never go through msgpack. Each header is `msgpack_decode`d (yielding
        the same `TokenizedGenerateReqInput` / control objects the zmq path
        produces, so the IPC schema is tracked automatically) and its `input_ids`
        slice is wrapped as the `array("q")` the scheduler expects. `recv_requests`
        releases the GIL for the drain + concat, so this never holds the GIL
        across a wait — same contract as `zmq.NOBLOCK`.
        """
        limit = max_recv if max_recv > 0 else self._max_per_poll
        batch = self.server.recv_requests(limit)
        # Bind once: each attribute access converts the rust vec to a fresh list.
        headers, data, lengths = batch.headers, batch.data, batch.lengths
        if not headers:
            return []

        ids_view = memoryview(data)
        out = []
        pos = 0  # byte offset into ids_buf
        for header, n in zip(headers, lengths):
            nbytes = n * 8
            try:
                obj = msgpack_decode_explained(header)
            except MsgpackDecodeError as e:
                # Return 400 for malformed request field (e.g. token_ids_logprob=[[0]].
                logger.warning(
                    "rust ingress: dropping undecodable request %s: %s", e.rid, e.reason
                )
                if e.rid is not None:
                    self.server.push_error(e.rid, f"invalid request: {e.reason}")
                pos += nbytes
                continue
            if n:  # generate request: attach its int64 ids slice as array("q")
                ids = array("q")
                ids.frombytes(ids_view[pos : pos + nbytes])
                obj.input_ids = ids
                pos += nbytes
            if self.mm_host is not None:
                # A multimodal request's processed mm_inputs travel by reference
                # through the host's rid-keyed table (populated strictly
                # before the request was pushed to this ring) — no pickle hop.
                entry = self.mm_host.results.pop(getattr(obj, "rid", None), None)
                if entry is not None:
                    obj.mm_inputs, token_type_ids = entry
                    if token_type_ids is not None:
                        obj.token_type_ids = token_type_ids
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

        # Invariant: control requests always carry a rust-minted rid; without
        # one the response is unroutable, so fail loudly rather than drop it.
        assert (
            recv_req.rid is not None
        ), f"control response without rid: {type(output).__name__}"
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

        self.server.push_result(recv_req.rid, encoded)

    def push_generation(self, payload: BatchTokenIDOutput) -> None:
        """Egress redirect for generation output (replaces the zmq detokenizer).

        Fan the batch out into per-request chunks and push each into the Rust
        egress ring (-> detokenizer shard -> client stream). Each chunk is a
        ``(header, data)`` pair, mirroring the ingress ``input_ids`` split so the
        bulk numeric columns never go through msgpack:

          - ``header``: msgpack ``ChunkHeader`` positional array — scalars
            (``rid, seq, token_ids, finish_reason, prompt_tokens``) plus the shape
            metadata (``out_lp_n, in_lp_n`` element counts for the flat logprob
            columns, and the per-position ``lens`` vectors for the ragged / hidden
            columns).
          - ``data``: the raw little-endian numeric buffer — every column is a
            4-byte element (``f32`` values, ``i32`` indices), concatenated in the
            order the Rust ``decode_chunk_frame`` reads them.

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
        # the plain rid strings (parsed to u64 on the Rust side, off the GIL),
        # `finished_reasons` already `dict | None`, and `output_ids` entries are
        # always `array("i")` (never None) so `map(len)` and a bare
        # `chain.from_iterable` stay in C.
        rids = payload.rids
        finish_reasons = payload.finished_reasons
        tok_lens = list(map(len, output_ids))
        flat_ids = array("i", chain.from_iterable(output_ids))

        # Column order here MUST match BatchHeader (header_cols) and
        # decode_batch_frame's read order (data_cols); the extras contribution
        # is ordered by the `extras` tuple below.
        header_cols = [rids, finish_reasons, prompt_tokens, tok_lens]
        data_cols = [flat_ids.tobytes()]

        if has_extra:
            # The `extras` tuple is the SINGLE source of the extras column
            # order — it must match the Rust ``BatchHeader`` fields and
            # ``decode_batch_frame``'s read order.
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

            # Every column is all-or-nothing per payload.
            for extra in extras:
                for name, col in extra.columns():
                    assert len(col) in (
                        0,
                        batch_size,
                    ), f"extras column {name}: {len(col)} entries for a batch of {batch_size}"

            for i in range(batch_size):
                for extra in extras:
                    extra.accept(i)

            for extra in extras:
                header_cols += extra.header_cols()
                data_cols += extra.data_cols()

        header = msgspec.msgpack.encode(header_cols)
        # Pass the raw column list; the Rust side concatenates it into the frame
        # with the GIL released.
        if not self.server.push_batch(header, data_cols):
            logger.warning(
                "Rust egress closed; dropped batch of %d requests during shutdown",
                len(rids),
            )

    @staticmethod
    def _build_server_args(scheduler: Scheduler) -> str:
        """JSON blob of the scheduler's ``server_args`` for its embedded Rust
        server (carries the already-resolved ``model_config``)."""

        server_args = dict(vars(scheduler.server_args))
        model_config = dict(vars(scheduler.model_config))
        model_config["hf_config"] = None  # HF config is not JSON-serializable
        server_args["model_config"] = model_config
        # Launch-time facts Python's /server_info reports from scheduler_info /
        # the package — stamped here so the rust endpoint can serve them
        # statically (no scheduler round-trip).
        server_args["version"] = __version__
        server_args["max_total_num_tokens"] = scheduler.max_total_num_tokens

        # The Rust server only reads local files (hub-blind, so it can never
        # disagree with huggingface_hub about cache layout): resolve a repo-id
        # tokenizer_path to the cached tokenizer.json here. No network — the
        # scheduler's init_tokenizer already downloaded it.
        if not scheduler.server_args.skip_tokenizer_init:
            path = server_args["tokenizer_path"] or server_args["model_path"]
            if not os.path.exists(path):
                server_args["tokenizer_path"] = _resolve_local_or_cached_file(
                    path, "tokenizer.json", server_args["revision"]
                )

        return msgspec.json.encode(server_args, enc_hook=str).decode("utf-8")

    @staticmethod
    def _partition_cores() -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Split this rank's allowed cores into ``(launch_cores, server_cores)``.

        Pure computation — no affinity is changed here. Both sets are a subset
        of this rank's NUMA-local cores (when affinity/NUMA bind is on), so the
        partition stays NUMA-local. Returns ``(None, None)`` (server runs
        unpinned, confined only by the process affinity) when the platform has
        no affinity API or too few cores to split.
        """
        if not hasattr(os, "sched_getaffinity"):
            return None, None
        try:
            allowed = sorted(os.sched_getaffinity(0))
        except OSError as e:
            logger.warning("rust server: cannot read cpu affinity: %s", e)
            return None, None

        # Need enough cores to reserve launch cores and still pin the pools.
        if len(allowed) < 4:
            logger.info(
                "rust server: only %d cores allowed; running pools unpinned",
                len(allowed),
            )
            return None, None

        # Keep a small slice for the launch loop; cap at 2 (the event loop is
        # effectively serial) and never take more than a quarter of the cores.
        reserve = min(2, len(allowed) // 4)
        launch_cores = allowed[:reserve]
        server_cores = allowed[reserve:]
        logger.info(
            "rust server cores=%s, scheduler launch cores=%s",
            server_cores,
            launch_cores,
        )
        return launch_cores, server_cores
