"""Embedded Rust server lifecycle for the scheduler.

Keeps all `SGLANG_RUST_SERVER` plumbing — startup, CPU-core partitioning, the
`server_args` blob, and control-response routing — out of `scheduler.py`. The
scheduler holds an `Optional[RustServer]` and delegates to it.
"""

from __future__ import annotations

import logging
import os
import re
from array import array
from itertools import chain
from typing import TYPE_CHECKING, Any, List, Optional

import msgspec

from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
)

if TYPE_CHECKING:
    from sglang_server import Server

    from sglang.srt.managers.io_struct import BatchTokenIDOutput
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _flatten_ragged(per_pos_val, per_pos_idx):
    """Flatten a per-position ``list[Optional[list]]`` (top-k / token-ids
    logprobs) into flat ``val``/``idx`` buffers plus a per-position ``lens``
    vector for the columnar egress wire. A falsy (None/empty) position
    contributes no values and a ``0`` length — the Rust side reshapes it back to
    a ``null`` position, matching ``detokenize_top_logprobs_tokens``.
    """
    flat_val: List[float] = []
    flat_idx: List[int] = []
    lens: List[int] = []
    if not per_pos_val:
        return flat_val, flat_idx, lens
    per_pos_idx = per_pos_idx or []
    for p, pv in enumerate(per_pos_val):
        if pv:
            pi = per_pos_idx[p] if p < len(per_pos_idx) else []
            flat_val.extend(pv)
            flat_idx.extend(pi or [])
            lens.append(len(pv))
        else:
            lens.append(0)
    return flat_val, flat_idx, lens


def _flatten_floats(x):
    """Recursively flatten a (possibly nested) float structure into a flat list
    of floats — handles the ``float | list[float]`` union inside a hidden-state
    chunk."""
    if isinstance(x, (int, float)):
        return [float(x)]
    out: List[float] = []
    for e in x:
        out.extend(_flatten_floats(e))
    return out


# Tag string -> struct field names, in tagged-array order.
_TAG_TO_FIELDS = {
    cls.__struct_config__.tag: cls.__struct_fields__
    for cls in (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
}

# Leading ``$[<n>]`` in a msgspec ValidationError path, e.g. ``$[12][0]``.
_ARRAY_INDEX_RE = re.compile(r"\$\[(\d+)\]")


def _explain_decode_failure(header: bytes, err: Exception) -> tuple[Optional[str], str]:
    """Recover the rid and a human-readable message from a header the *typed*
    decode rejected.
    """
    msg = str(err)
    try:
        arr = msgspec.msgpack.decode(header)
    except Exception:
        return None, msg
    if not (isinstance(arr, (list, tuple)) and arr):
        return None, msg
    rid = str(arr[1]) if len(arr) > 1 and arr[1] is not None else None
    fields = _TAG_TO_FIELDS.get(arr[0])
    if fields is not None:
        m = _ARRAY_INDEX_RE.search(msg)
        if m is not None:
            idx = int(m.group(1))
            if 1 <= idx <= len(fields):
                msg = f"{msg[:m.start()]}$.{fields[idx - 1]}{msg[m.end():]}"
    return rid, msg


def _flatten_hidden(hs):
    """Flatten one request's hidden states into a flat ``val`` buffer plus a
    per-row ``lens`` vector (one row per output position). Each top-level element
    becomes a single row; the Rust side reshapes back to ``list[list[float]]``,
    matching ``meta_info["hidden_states"]``'s common per-position-vector shape.
    """
    vals: List[float] = []
    lens: List[int] = []
    if not hs:
        return vals, lens
    for row in hs:
        flat = _flatten_floats(row)
        vals.extend(flat)
        lens.append(len(flat))
    return vals, lens


class RustServer:
    """Owns the embedded multi-threaded Rust server (``sglang_server.Server``).

    The server owns the api-server, tokenizermanager, tokenizer, and detokenizer
    all implemented as Rust threads in scheduler process.
    """

    def __init__(self, server: Server, max_per_poll: int = 256):
        self.server = server
        self._max_per_poll = max_per_poll

    @classmethod
    def maybe_create(cls, scheduler: Scheduler) -> RustServer:
        """Start the Rust server on the rank-0 scheduler if enabled, else ``None``."""

        # Lazy import: only required when the rust server is actually enabled.
        from sglang_server import Server

        # Force turn off HF tokenizers rayon's unpinned global thread pool.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        server_args = scheduler.server_args
        bind = f"{server_args.host}:{server_args.port}"
        cores = cls._partition_cores()

        server = Server(
            pin_cores=cores is not None,
            cores=cores,
            bind=bind,
            server_args_json=cls._build_server_args(scheduler),
        )

        logger.info(
            "SGLANG_RUST_SERVER enabled, Rust server listen on %s",
            bind,
        )

        return cls(server)

    def wait_ingress(self, timeout_ms: int) -> None:
        """Block until a request is pushed into the in-process ring or the timeout
        elapses.
        """
        self.server.wait_ingress(timeout_ms)

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
            nbytes = n * 8
            try:
                obj = msgpack_decode(header)
            except Exception as e:
                # Return 400 for malformed request field (e.g. token_ids_logprob=[[0]].
                rid, reason = _explain_decode_failure(header, e)
                logger.warning(
                    "rust ingress: dropping undecodable request %s: %s", rid, reason
                )
                if rid is not None:
                    self.server.push_error(rid, f"invalid request: {reason}")
                pos += nbytes
                continue
            if n:  # generate request: attach its int64 ids slice as array("q")
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
        out_lp_val = payload.output_token_logprobs_val or []
        out_lp_idx = payload.output_token_logprobs_idx or []
        in_lp_val = payload.input_token_logprobs_val or []
        in_lp_idx = payload.input_token_logprobs_idx or []
        out_top_val = payload.output_top_logprobs_val or []
        out_top_idx = payload.output_top_logprobs_idx or []
        in_top_val = payload.input_top_logprobs_val or []
        in_top_idx = payload.input_top_logprobs_idx or []
        out_tid_val = payload.output_token_ids_logprobs_val or []
        out_tid_idx = payload.output_token_ids_logprobs_idx or []
        in_tid_val = payload.input_token_ids_logprobs_val or []
        in_tid_idx = payload.input_token_ids_logprobs_idx or []
        hidden = getattr(payload, "output_hidden_states", None) or []

        def at(seq_, j):
            return seq_[j] if j < len(seq_) else None

        # Hot-path guard: the overwhelming majority of decode steps request no
        # logprobs / hidden states. Only then do we run the per-request flatten +
        # raw-buffer packing; otherwise the loop stays as cheap as plain token
        # streaming (a small header + empty data), which keeps this off the
        # scheduler/CUDA-launch thread's critical path.
        has_extra = bool(
            out_lp_val
            or in_lp_val
            or out_top_val
            or in_top_val
            or out_tid_val
            or in_tid_val
            or hidden
        )

        # Both paths ship the WHOLE batch in one frame: columnar scalars via a
        # single msgpack header + one concatenated raw `data` buffer + one
        # `push_batch` FFI. The tm-egress dispatcher fans it out per rid (routing
        # is by rid, so the old per-rid load-balance seq is no longer tracked
        # here). Collapsing N per-request encodes + N FFI crossings to one is what
        # keeps 4k-16k-request PD decode steps off the scheduler's critical path.
        rids = payload.rids
        # Ship the WHOLE finish-reason dict per request (type + matched + message
        # + status_code + err_type + length).
        finish_reasons = [
            (fr if isinstance(fr, dict) else None) for fr in payload.finished_reasons
        ]
        # `chain.from_iterable` flattens the per-request id arrays in C (no
        # Python-level per-token loop); `tok_lens` splits them back out in Rust.
        tok_lens = [len(x) if x else 0 for x in output_ids]
        flat_ids = array("i", chain.from_iterable(x or () for x in output_ids))

        # Column order here MUST match BatchHeader (header_cols) and
        # decode_batch_frame's read order (data_cols).
        header_cols = [rids, finish_reasons, list(prompt_tokens), tok_lens]
        data_cols = [flat_ids.tobytes()]

        if has_extra:
            # Rare: at least one request wants logprobs/hidden states. Append the
            # numeric columns concatenated across all requests (column-major).
            # Flat families carry a per-request element count; ragged families and
            # hidden states carry a per-request position/row count + a flat
            # per-position length stream.
            olp_v, olp_i, out_lp_lens = array("f"), array("i"), []
            ilp_v, ilp_i, in_lp_lens = array("f"), array("i"), []
            ot_v, ot_i, ot_pos, ot_req = array("f"), array("i"), [], []
            it_v, it_i, it_pos, it_req = array("f"), array("i"), [], []
            od_v, od_i, od_pos, od_req = array("f"), array("i"), [], []
            id_v, id_i, id_pos, id_req = array("f"), array("i"), [], []
            h_v, h_pos, h_req = array("f"), [], []

            # TODO(perf): this per-request flatten assumes the logprob/hidden
            # columns are ragged, non-contiguous nested Python lists — which is
            # only an assumption. The scheduler moves these off the GPU with
            # `tensor.tolist()`, so revisit whether the underlying values are
            # still contiguous tensors upstream. If so, ship them as raw bytes
            # (`tensor.contiguous().numpy().tobytes()`) + a shape descriptor and
            # skip this flatten entirely, making the extras path loop-free too.
            for i in range(len(rids)):
                olv = at(out_lp_val, i) or []
                olp_v.extend(olv)
                olp_i.extend(at(out_lp_idx, i) or [])
                out_lp_lens.append(len(olv))
                ilv = at(in_lp_val, i) or []
                ilp_v.extend(ilv)
                ilp_i.extend(at(in_lp_idx, i) or [])
                in_lp_lens.append(len(ilv))
                otv, oti, otl = _flatten_ragged(at(out_top_val, i), at(out_top_idx, i))
                ot_v.extend(otv)
                ot_i.extend(oti)
                ot_pos.extend(otl)
                ot_req.append(len(otl))
                itv, iti, itl = _flatten_ragged(at(in_top_val, i), at(in_top_idx, i))
                it_v.extend(itv)
                it_i.extend(iti)
                it_pos.extend(itl)
                it_req.append(len(itl))
                odv, odi, odl = _flatten_ragged(at(out_tid_val, i), at(out_tid_idx, i))
                od_v.extend(odv)
                od_i.extend(odi)
                od_pos.extend(odl)
                od_req.append(len(odl))
                idv, idi, idl = _flatten_ragged(at(in_tid_val, i), at(in_tid_idx, i))
                id_v.extend(idv)
                id_i.extend(idi)
                id_pos.extend(idl)
                id_req.append(len(idl))
                hv, hlens = _flatten_hidden(at(hidden, i))
                h_v.extend(hv)
                h_pos.extend(hlens)
                h_req.append(len(hlens))

            header_cols += [
                out_lp_lens,
                in_lp_lens,
                ot_req,
                ot_pos,
                it_req,
                it_pos,
                od_req,
                od_pos,
                id_req,
                id_pos,
                h_req,
                h_pos,
            ]
            data_cols += [
                olp_v.tobytes(),
                olp_i.tobytes(),
                ilp_v.tobytes(),
                ilp_i.tobytes(),
                ot_v.tobytes(),
                ot_i.tobytes(),
                it_v.tobytes(),
                it_i.tobytes(),
                od_v.tobytes(),
                od_i.tobytes(),
                id_v.tobytes(),
                id_i.tobytes(),
                h_v.tobytes(),
            ]

        header = msgspec.msgpack.encode(header_cols)
        if not self.server.push_batch(header, b"".join(data_cols)):
            logger.warning(
                "Rust egress ring full; dropped batch of %d requests", len(rids)
            )

    @staticmethod
    def _build_server_args(scheduler: Scheduler) -> str:
        """JSON blob of the scheduler's ``server_args`` for its embedded Rust
        server (carries the already-resolved ``model_config``)."""

        server_args = dict(vars(scheduler.server_args))
        model_config = dict(vars(scheduler.model_config))
        model_config["hf_config"] = None  # HF config is not JSON-serializable
        server_args["model_config"] = model_config

        return msgspec.json.encode(server_args, enc_hook=str).decode("utf-8")

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
