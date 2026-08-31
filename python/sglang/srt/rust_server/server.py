"""Embedded Rust server lifecycle for the scheduler.

The Rust server replaces the Python api-server + `TokenizerManager` +
`DetokenizerManager` stack (hence this module sits beside them in `managers/`),
running them as Rust threads inside the scheduler process. This wrapper keeps
all `SGLANG_RUST_SERVER` plumbing — startup, CPU-core partitioning, the
typed `server_args` handoff, and control-response routing — out of `scheduler.py`. The
scheduler holds an `Optional[RustServer]` and delegates to it.
"""

from __future__ import annotations

import json
import logging
import os
from array import array
from itertools import chain
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import msgspec

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.utils import (
    MsgpackDecodeError,
    compute_num_reserved_tokens,
    msgpack_decode_explained,
)
from sglang.srt.rust_server.multimodal import (
    NATIVE_MM_FAMILIES,
    NativeMmHost,
    NativeMmSpec,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_mm,
    get_model,
    get_observability,
    get_serving,
)
from sglang.srt.utils.flatten import (
    FlatPairColumns,
    NestedRowColumns,
    RaggedPairColumns,
)
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import BatchTokenIDOutput
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.rust_extensions._server import MmSpec, Server, ServerArgs
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class RustServer:
    """Owns the embedded multi-threaded Rust server (``sglang_server.Server``).

    The server owns the api-server, tokenizermanager, tokenizer, and detokenizer
    all implemented as Rust threads in scheduler process.
    """

    def __init__(
        self,
        server: Server,
        mm_spec: Optional[NativeMmSpec] = None,
        max_per_poll: int = 256,
    ):
        self.server = server
        self.mm_spec = mm_spec
        self._max_per_poll = max_per_poll

    @classmethod
    def launch(cls, scheduler: Scheduler) -> RustServer:
        """Start the embedded Rust server threads and bind the listen port.

        The caller gates this (``SGLANG_RUST_SERVER`` + rank 0); this always
        creates.
        """
        from sglang.srt.rust_extensions import load_rust_extension

        Server = load_rust_extension("sglang.srt.rust_extensions._server").Server

        # Force turn off HF tokenizers rayon's unpinned global thread pool.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        server_args = scheduler.server_args
        # `TokenizerManager` merges these under each request's own sampling params
        # (`{**preferred, **obj.sampling_params}`), and this server replaces that
        # manager wholesale — so honouring the flag is not implemented here yet.
        # Refuse rather than run: silently dropping it means generating with
        # sampling the operator did not configure, and `/get_model_info` would go on
        # advertising values no request ever receives.
        if get_serving().preferred_sampling_params:
            raise ValueError(
                "SGLANG_RUST_SERVER does not yet apply --preferred-sampling-params "
                "(the Python TokenizerManager merges it into every request; the rust "
                "ingress has no equivalent). Launch without SGLANG_RUST_SERVER, or "
                "drop --preferred-sampling-params and send those values per request."
            )
        http_addr = f"{get_serving().host}:{get_serving().port}"

        # Per-DP-rank HTTP port with client load balancing. `None` when DP is off,
        # so the rank is not conflated with rank 0 of a one-rank group.
        dp_rank = scheduler.ps.attn_dp_rank if scheduler.ps.dp_size > 1 else None
        if dp_rank is not None:
            http_addr = f"{get_serving().host}:{get_serving().port + dp_rank}"

        launch_cores, server_cores = cls._partition_cores(
            mm_workers=(
                (get_mm().mm_processor_worker_num or NativeMmHost.AUTO_MM_WORKERS)
                if scheduler.model_config.is_multimodal
                else 0
            )
        )

        server = Server(
            cls._build_server_args(scheduler),
            # None -> run unpinned; the list carries the pinning decision.
            cores=server_cores,
            http_addr=http_addr,
        )

        # Multimodal models must have a native Rust pipeline — there is no Python
        # fallback.
        mm_spec = None
        if scheduler.model_config.is_multimodal:
            # New threads inherit the spawning thread's affinity, and this launch
            # thread still holds the full mask. Narrow it first so every MM thread
            # created below (the processor's executors, the Rust MM workers) stays
            # off the scheduler's reserved cores, where MM preprocessing would
            # preempt the scheduler loop and inflate inter-token latency.
            if server_cores is not None:
                try:
                    os.sched_setaffinity(0, set(server_cores))
                except OSError as e:
                    logger.warning(
                        "rust server: cannot confine mm threads to server cores: %s", e
                    )
            mm_host = NativeMmHost(
                server_args=server_args,
                model_config=scheduler.model_config,
                processor=scheduler.processor,
            )
            mm_spec = mm_host.resolve_native_spec()
            if mm_spec is None:
                supported = sorted(
                    set(chain.from_iterable(f.model_types for f in NATIVE_MM_FAMILIES))
                )
                raise RuntimeError(
                    "SGLANG_RUST_SERVER=1: no native Rust MM pipeline for "
                    f"model_type={scheduler.model_config.hf_config.model_type!r} "
                    f"(supported: {', '.join(supported)}; "
                    "images only). Unset SGLANG_RUST_SERVER to serve this model."
                )
            server.start_mm_workers(cls._build_mm_spec(mm_spec), mm_host.mm_workers)

        # Narrow the scheduler thread only after the server threads are launched.
        if launch_cores is not None:
            try:
                # pid 0 == this thread (the scheduler event-loop / launch thread).
                os.sched_setaffinity(0, set(launch_cores))
            except OSError as e:
                logger.warning("rust server: cannot pin scheduler launch thread: %s", e)

        # Under DP every rank runs its own server on its own port, so the rank is
        # what tells two otherwise identical startup lines apart.
        dp_note = (
            "" if dp_rank is None else f" (DP rank {dp_rank}/{scheduler.ps.dp_size})"
        )
        logger.info(
            "SGLANG_RUST_SERVER enabled, Rust server listen on %s%s",
            http_addr,
            dp_note,
        )

        return cls(server, mm_spec=mm_spec)

    def wait_request(self, timeout_ms: int) -> None:
        """Block until a request is pushed into the in-process ring or the timeout
        elapses.
        """
        self.server.wait_request(timeout_ms)

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
        never waits: the ring drain is `try_recv` (returns the instant the ring
        is dry, capped at `max_recv`) and the rest is one memcpy per header
        plus one for the concatenated ids — same contract as `zmq.NOBLOCK`.
        Parking for work is :meth:`wait_request`, which does release the GIL.
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
            if self.mm_spec is not None and isinstance(obj, TokenizedGenerateReqInput):
                # The buffers rode the Rust sidecar, parked before the ring push;
                # wrapping them into tensors is the only Python step of the native
                # path. `None` for a text-only request on a multimodal model.
                native = self.server.take_mm(obj.rid)
                if native is not None:
                    obj.mm_inputs = NativeMmHost.build_native_mm(self.mm_spec, native)
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

        self.server.push_control_result(recv_req.rid, encoded)

    def push_generation(self, payload: BatchTokenIDOutput) -> None:
        """Egress redirect for generation output (replaces the zmq detokenizer).

        Push the WHOLE batch into the Rust egress ring as one frame (-> detokenizer
        shards -> client streams), mirroring the ingress ``input_ids`` split so the
        bulk numeric columns never go through msgpack:

          - ``header``: msgpack ``BatchHeader`` positional array — the per-request
            scalar columns (``rids, finish_reasons, prompt_tokens, tok_lens``) plus
            the shape metadata for the optional families (``*_lens`` element counts
            for the flat logprob columns, ``*_reqlens``/``*_poslens`` for the ragged
            and hidden ones).
          - ``data``: the raw little-endian numeric buffer — every column is a
            4-byte element (``f32`` values, ``i32`` indices), concatenated in the
            order the Rust ``for_each_chunk`` reads them.

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
        # always `array("i")` (never None) so `map(len)` and a bare
        # `chain.from_iterable` stay in C.
        rids = payload.rids
        finish_reasons = payload.finished_reasons
        tok_lens = list(map(len, output_ids))
        flat_ids = array("i", chain.from_iterable(output_ids))

        # Column order here MUST match BatchHeader (header_cols) and
        # for_each_chunk's read order (data_cols); the extras contribution
        # is ordered by the `extras` tuple below.
        header_cols = [rids, finish_reasons, prompt_tokens, tok_lens]
        data_cols = [flat_ids.tobytes()]

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
                    ), f"extras column {name}: {len(col)} entries for a batch of {batch_size}"
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

            for extra in extras:
                header_cols += extra.header_cols()
                data_cols += extra.data_cols()

        header = msgspec.msgpack.encode(header_cols)
        # Pass the raw column list; the Rust side concatenates it into the frame
        # with the GIL released.
        if not self.server.push_decode_result_batch(header, data_cols):
            logger.warning(
                "Rust egress closed; dropped batch of %d requests during shutdown",
                len(rids),
            )

    @staticmethod
    def _build_mm_spec(spec: NativeMmSpec) -> MmSpec:
        """The typed MM handoff for ``Server.start_mm_workers``: the
        :class:`NativeMmSpec` fields the Rust pipeline consumes, as the Rust
        extension's own ``MmSpec`` class (same required-keyword contract as
        :meth:`_build_server_args`; ``family`` / ``resample`` become the
        extension's ``MmFamily`` / ``MmResample`` enums)."""
        from sglang.srt.rust_extensions import load_rust_extension

        ext = load_rust_extension("sglang.srt.rust_extensions._server")
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

    @staticmethod
    def _build_server_args(scheduler: Scheduler) -> ServerArgs:
        """The typed launch handoff for the scheduler's embedded Rust server:
        the ``server_args`` fields it reads, the already-resolved
        ``model_config``, and launch-time facts — as the Rust extension's own
        ``ServerArgs`` class. Its constructor takes every field as a required
        keyword (see ``rust/sglang-server/src/message/config.rs``), so a
        missing, extra or mistyped field fails here at boot rather than
        running on a silently-defaulted knob."""
        from sglang.srt.rust_extensions import load_rust_extension

        ext = load_rust_extension("sglang.srt.rust_extensions._server")

        sa = resolving_view(scheduler.server_args)
        mc = scheduler.model_config
        disaggregation_mode = {
            "null": ext.DisaggregationMode.Null,
            "prefill": ext.DisaggregationMode.Prefill,
            "decode": ext.DisaggregationMode.Decode,
        }[get_disagg().disaggregation_mode]
        return ext.ServerArgs(
            model_path=get_model().model_path,
            served_model_name=get_serving().served_model_name,
            tokenizer_path=get_serving().tokenizer_path,
            revision=get_model().revision,
            load_format=get_model().load_format,
            weight_version=get_serving().weight_version,
            host=get_serving().host,
            port=get_serving().port,
            log_level=get_observability().log_level,
            log_level_http=get_observability().log_level_http,
            chat_template=get_serving().chat_template,
            tool_call_parser=get_serving().tool_call_parser,
            reasoning_parser=get_serving().reasoning_parser,
            stream_response_default_include_usage=get_serving().stream_response_default_include_usage,
            tokenizer_worker_num=get_serving().tokenizer_worker_num,
            detokenizer_worker_num=get_serving().detokenizer_worker_num,
            skip_tokenizer_init=get_serving().skip_tokenizer_init,
            incremental_streaming_output=get_serving().incremental_streaming_output,
            disaggregation_mode=disaggregation_mode,
            model_config=ext.ModelConfig(
                context_len=mc.context_len,
                vocab_size=mc.vocab_size,
                is_multimodal=mc.is_multimodal,
                # Resolved default sampling params (generation_config.json when
                # `--sampling-defaults model`, {} otherwise). The rust server
                # consumes these for omitted temperature/top_p in chat
                # conversions instead of hard-coding the OpenAI terminal
                # defaults.
                default_sampling_params=ext.DefaultSamplingParams(
                    **mc.get_default_sampling_params()
                ),
            ),
            # `preferred_sampling_params` is deliberately absent: `launch`
            # refuses to start when it is set, so the Rust server never needs it.
            preferred_sampling_params=(
                json.dumps(get_serving().preferred_sampling_params)
                if get_serving().preferred_sampling_params is not None
                else None
            ),
            allow_auto_truncate=get_serving().allow_auto_truncate,
            enable_return_hidden_states=sa.enable_return_hidden_states,
            # Not a `server_args` field: `TokenizerManager` derives it, and the
            # rust ingress needs the same number for its total-token check.
            num_reserved_tokens=compute_num_reserved_tokens(),
            # Launch-time facts Python's /server_info reports from
            # scheduler_info / the package — stamped here so the rust endpoint
            # can serve them statically (no scheduler round-trip).
            version=__version__,
            max_total_num_tokens=scheduler.max_total_num_tokens,
        )

    @staticmethod
    def _partition_cores(
        mm_workers: int = 0,
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
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
        # Bound the pool instead of taking the whole remainder: this rank's
        # allowed cores are usually the entire NUMA node, shared with the sibling
        # TP ranks' processes, so an unbounded mask lets MM preprocessing bursts
        # preempt a sibling's CUDA-launch thread and inflate every rank's forward
        # through the TP collectives. Measured on Qwen3.5-35B TP4 at one 720p
        # image per request: ~20 ms of ViT wall time on the worst sibling, gone
        # once bounded. The budget covers the CPU-hot threads (MM workers, plus
        # the I/O-shaped tokenizer/ingress/egress/api ones that are rarely all hot
        # at once) and leaves the rest of the node to the scheduler ranks.
        pool_budget = max(8, mm_workers + 4)
        server_cores = allowed[reserve : reserve + pool_budget]
        logger.info(
            "rust server cores=%s, scheduler launch cores=%s",
            server_cores,
            launch_cores,
        )
        return launch_cores, server_cores
