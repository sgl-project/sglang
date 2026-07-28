"""Embedded Rust server lifecycle for the scheduler.

The Rust server replaces the Python api-server + `TokenizerManager` +
`DetokenizerManager` stack (hence this module sits beside them in `managers/`),
running them as Rust threads inside the scheduler process. This wrapper keeps
all `SGLANG_RUST_SERVER` plumbing — startup, CPU-core partitioning, the
`server_args` blob, and control-response routing — out of `scheduler.py`. The
scheduler holds an `Optional[RustServer]` and delegates to it.
"""

from __future__ import annotations

import logging
import os
from array import array
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import msgspec

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.utils import (
    MsgpackDecodeError,
    compute_num_reserved_tokens,
    msgpack_decode_explained,
)
from sglang.srt.utils.flatten import (
    FlatPairColumns,
    NestedRowColumns,
    RaggedPairColumns,
)
from sglang.srt.utils.hf_transformers.common import resolve_local_or_cached_file
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.io_struct import BatchTokenIDOutput
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server._core import Server
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class NativeMmHost:
    """Builds and validates the native Rust MM pipeline for one model.

    Construction loads the same model-specific ``mm_processor`` stack the
    Python TokenizerManager would build — not to process requests (the Rust
    worker pool does that natively, GIL-free), but as the source of truth
    :meth:`resolve_native_spec` validates and resolves the pipeline parameters
    from. At drain time :meth:`build_native_mm` wraps the Rust-produced
    buffers into the scheduler's ``MultimodalProcessorOutput``.

    There is no Python fallback: a model without a native spec fails at
    launch, and per-request inputs outside the pipeline's scope (video/audio,
    precomputed features, undecodable images) are rejected by the Rust worker
    as a 400.
    """

    # Rust mm-worker threads when --mm-processor-worker-num is 0 ("model-
    # specific default"). The workers are GIL-free, so unlike the Python
    # processor pool more than one is always usable.
    AUTO_MM_WORKERS = 8

    # Model types whose image-only M-RoPE matches the native fast path.
    NATIVE_QWEN_MODEL_TYPES = frozenset(
        ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe")
    )

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        processor: Any = None,
    ):
        # Lazy imports: this class is only instantiated for multimodal models
        # under SGLANG_RUST_SERVER.
        from sglang.srt.managers.multimodal_processor import (
            get_mm_processor,
            import_processors,
        )
        from sglang.srt.managers.tokenizer_manager import (
            determine_tensor_transport_mode,
            get_processor_wrapper,
        )

        self.server_args = server_args
        self.model_config = model_config
        # Rust mm-worker threads == max concurrently-processed mm requests.
        self.mm_workers = server_args.mm_processor_worker_num or self.AUTO_MM_WORKERS

        # Same processor stack the Python TokenizerManager builds
        # (init_tokenizer_and_processor, multimodal branch). The HF
        # AutoProcessor is reused from the caller's init_tokenizer (already
        # loaded, identical construction args) when available.
        import_processors("sglang.srt.multimodal.processors")
        if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
            import_processors(mm_process_pkg, overwrite=True)
        _processor = processor or get_processor_wrapper(self.server_args)
        transport_mode = determine_tensor_transport_mode(self.server_args)
        self.mm_processor = get_mm_processor(
            self.model_config.hf_config,
            self.server_args,
            _processor,
            transport_mode,
            model_config=self.model_config,
        )
        self._processor = _processor
        # Set by resolve_native_spec() when the model family has a pure-Rust
        # pipeline; consumed by build_native_mm (the drain-time adapter).
        self._native: Optional[Dict[str, Any]] = None

    def resolve_native_spec(self) -> Optional[str]:
        """JSON spec for the Rust-native MM pipeline, or ``None`` when the
        model has no native pipeline (the launch gate turns that into a hard
        error). Only resolved settings are carried (patch geometry, pixel
        limits, normalization, token ids) — never the HF config. Conservative
        by design: any unrecognized knob disables the native path rather than
        approximating it."""
        import json

        spec = self._resolve_qwen_spec()
        if spec is None:
            spec = self._resolve_mimo_spec()
        if spec is None:
            return None
        logger.info(
            "rust server: native MM pipeline enabled (family=%s)", spec["family"]
        )
        return json.dumps(spec)

    def _resolve_qwen_spec(self) -> Optional[Dict[str, Any]]:
        from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

        hf_config = self.model_config.hf_config
        if (
            type(self.mm_processor) is not QwenVLImageProcessor
            or getattr(hf_config, "model_type", None)
            not in self.NATIVE_QWEN_MODEL_TYPES
        ):
            return None
        ip = getattr(self._processor, "image_processor", None)
        if type(ip).__name__ not in (
            "Qwen2VLImageProcessor",
            "Qwen2VLImageProcessorFast",
        ):
            return None

        # `--mm-process-config {"image": {...}}`: only pixel-limit overrides
        # are mirrored natively; anything else disables the native pipeline.
        image_overrides = dict(
            (self.server_args.mm_process_config or {}).get("image", {})
        )
        if not set(image_overrides) <= {"min_pixels", "max_pixels"}:
            return None

        size = getattr(ip, "size", None) or {}
        min_pixels = image_overrides.get(
            "min_pixels", getattr(ip, "min_pixels", None) or size.get("shortest_edge")
        )
        max_pixels = image_overrides.get(
            "max_pixels", getattr(ip, "max_pixels", None) or size.get("longest_edge")
        )
        try:
            spec = {
                "family": "qwen_vl",
                "feature_shm": self._use_feature_shm(),
                "image_token_id": hf_config.image_token_id,
                "patch_size": ip.patch_size,
                "merge_size": ip.merge_size,
                "temporal_patch_size": ip.temporal_patch_size,
                "min_pixels": int(min_pixels),
                "max_pixels": int(max_pixels),
                "image_mean": [float(x) for x in ip.image_mean],
                "image_std": [float(x) for x in ip.image_std],
            }
        except (AttributeError, TypeError):  # missing/odd processor attrs
            return None
        self._native = {
            **spec,
            "vision_start_token_id": getattr(hf_config, "vision_start_token_id", None),
            "vision_end_token_id": getattr(hf_config, "vision_end_token_id", None),
            "video_token_id": getattr(hf_config, "video_token_id", None),
            "feature_dim": 3
            * spec["temporal_patch_size"]
            * spec["patch_size"]
            * spec["patch_size"],
        }
        return spec

    def _resolve_mimo_spec(self) -> Optional[Dict[str, Any]]:
        """MiMo-V2 family (``model_type="mimo_v2"``, e.g. MiMo-V2.5): images
        only, 1-D-rope checkpoints only. Parameters come from the resolved
        ``MiMoProcessor`` the Python TokenizerManager would drive — the single
        source that already merged ``processor_config``/``vision_config`` —
        so the native pipeline cannot drift from the Python one. Pixel-limit
        knobs in ``--mm-process-config`` are ignored on both paths (the MiMo
        processor never reads them), so they don't disable the native
        pipeline the way they would for Qwen."""
        from sglang.srt.multimodal.processors.mimo_v2 import (
            _QWEN2VL_PIXEL_MEAN,
            _QWEN2VL_PIXEL_STD,
            MiMoV2Processor,
        )

        hf_config = self.model_config.hf_config
        if (
            type(self.mm_processor) is not MiMoV2Processor
            or getattr(hf_config, "model_type", None) != "mimo_v2"
        ):
            return None
        mp = self.mm_processor.mimo_processor
        # The native pipeline emits the rope_type="rope" position contract
        # ([3, len] arange, delta 0). An mrope checkpoint would need the Qwen
        # get_rope_index path instead; none exists to validate against, so
        # refuse rather than approximate.
        if mp.rope_type != "rope":
            return None

        spec = {
            "family": "mimo_v2",
            "feature_shm": self._use_feature_shm(),
            "image_token_id": mp.image_token_id,
            "patch_size": mp.patch_size,
            "merge_size": mp.merge_size,
            "temporal_patch_size": mp.temporal_patch_size,
            "min_pixels": int(mp.default_image_processor_kwargs["min_pixels"]),
            "max_pixels": int(mp.default_image_processor_kwargs["max_pixels"]),
            # 0..255-scale constants: MiMo standardizes unrescaled floats.
            "image_mean": _QWEN2VL_PIXEL_MEAN.flatten().tolist(),
            "image_std": _QWEN2VL_PIXEL_STD.flatten().tolist(),
        }
        self._native = {
            **spec,
            "vision_start_token_id": mp.vision_start_token_id,
            "vision_end_token_id": mp.vision_end_token_id,
            "video_token_id": mp.video_token_id,
            "feature_dim": 3
            * spec["temporal_patch_size"]
            * spec["patch_size"]
            * spec["patch_size"],
        }
        return spec

    def _use_feature_shm(self) -> bool:
        """Park native feature buffers in POSIX shm instead of the sidecar.

        On when — and only when — the drained request is broadcast across TP
        ranks AND the receiver's ``unwrap_shm_features`` will materialize the
        stubs (its gates: non-default tensor transport, no
        ``skip_tokenizer_init``). With inline tensors the whole feature buffer
        (~20 MB/image) rides ``broadcast_pyobj`` serially on the scheduler
        loop; ranks 1..n then start the TP-sharded ViT ~30 ms after rank 0 and
        every rank stalls at the first collective for exactly that skew. With
        shm the broadcast carries a ~100-byte stub and all ranks map +
        materialize in parallel behind the receiver's barrier — the same
        transport the Python TokenizerManager uses. Single-rank serving keeps
        the zero-copy inline path (shm would only add a copy per image).
        """
        from sglang.srt.managers.tokenizer_manager import (
            determine_tensor_transport_mode,
        )

        return (
            self.server_args.tp_size > 1
            and determine_tensor_transport_mode(self.server_args) != "default"
            and not self.server_args.skip_tokenizer_init
        )

    def build_native_mm(self, entry: tuple):
        """Drain-time adapter: wrap the Rust-produced buffers into the
        scheduler's ``MultimodalProcessorOutput``. Only tensor wrapping happens
        here — all processing (load, resize, patchify, token expansion, M-RoPE)
        already ran in Rust. This runs on the scheduler loop, so it must stay
        copy-free AND hash-free: the numpy arrays from ``take_mm`` own the Rust
        buffers (zero copy), ``torch.from_numpy`` only wraps them, and each
        item's ``hash`` is the worker-precomputed feature hash so the
        scheduler's ``set_pad_value`` skips ``hash_feature``. Any per-byte work
        here (memcpy, sha256 — tens of MB per image-heavy request) measurably
        stalls every running request's inter-token latency."""
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalProcessorOutput,
        )

        native = self._native
        features_arr, shm_names, grids, hashes, offsets, mrope_arr, mrope_delta = entry
        if shm_names is None:
            features = torch.from_numpy(features_arr.reshape(-1, native["feature_dim"]))
        items = []
        row = 0
        for index, ((t, h, w), item_hash, offset) in enumerate(
            zip(grids, hashes, offsets)
        ):
            n = t * h * w
            if shm_names is None:
                feature = features[row : row + n]
            else:
                # The worker parked this item's buffer in a named POSIX
                # segment (see `_use_feature_shm`). Build the stub in its
                # post-`__setstate__` form: rank 0 never pickle-roundtrips its
                # own copy, and `materialize()` needs the mapped view.
                # Ownership of the unlink moved here with `take_mm`.
                feature = ShmPointerMMData.__new__(ShmPointerMMData)
                feature.__setstate__(
                    {
                        "shm_name": shm_names[index],
                        "shape": (n, native["feature_dim"]),
                        "dtype": torch.float32,
                        "precomputed_hash": item_hash,
                    }
                )
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=feature,
                    hash=item_hash,
                    offsets=[tuple(offset)],
                    model_specific_data={
                        "image_grid_thw": torch.tensor([[t, h, w]], dtype=torch.long)
                    },
                )
            )
            row += n
        if envs.SGLANG_MM_PRECOMPUTE_HASH.get():
            for item in items:
                item.set_pad_value()
        mrope_positions = torch.from_numpy(mrope_arr.reshape(3, -1))
        return MultimodalProcessorOutput(
            mm_items=items,
            im_token_id=native["image_token_id"],
            im_start_id=native["vision_start_token_id"],
            im_end_id=native["vision_end_token_id"],
            video_token_id=native["video_token_id"],
            mrope_positions=mrope_positions,
            mrope_position_delta=torch.tensor([[mrope_delta]], dtype=torch.long),
        )


class RustServer:
    """Owns the embedded multi-threaded Rust server (``sglang_server.Server``).

    The server owns the api-server, tokenizermanager, tokenizer, and detokenizer
    all implemented as Rust threads in scheduler process.
    """

    def __init__(
        self,
        server: Server,
        mm_host: Optional[NativeMmHost] = None,
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
        from sglang.srt.server._core import Server

        # Force turn off HF tokenizers rayon's unpinned global thread pool.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        server_args = scheduler.server_args
        # `TokenizerManager` merges these under each request's own sampling params
        # (`{**preferred, **obj.sampling_params}`), and this server replaces that
        # manager wholesale — so honouring the flag is not implemented here yet.
        # Refuse rather than run: silently dropping it means generating with
        # sampling the operator did not configure, and `/get_model_info` would go on
        # advertising values no request ever receives.
        if server_args.preferred_sampling_params:
            raise ValueError(
                "SGLANG_RUST_SERVER does not yet apply --preferred-sampling-params "
                "(the Python TokenizerManager merges it into every request; the rust "
                "ingress has no equivalent). Launch without SGLANG_RUST_SERVER, or "
                "drop --preferred-sampling-params and send those values per request."
            )
        http_addr = f"{server_args.host}:{server_args.port}"
        launch_cores, server_cores = cls._partition_cores(
            mm_workers=(
                (server_args.mm_processor_worker_num or NativeMmHost.AUTO_MM_WORKERS)
                if scheduler.model_config.is_multimodal
                else 0
            )
        )

        server = Server(
            # None -> run unpinned; the list carries the pinning decision.
            cores=server_cores,
            http_addr=http_addr,
            server_args_json=cls._build_server_args(scheduler),
        )

        # Multimodal models must have a native Rust pipeline — there is no
        # Python fallback. The host resolves/validates the spec from the same
        # processor stack the Python TokenizerManager would build; the Rust
        # worker pool then processes every mm request natively.
        mm_host = None
        if scheduler.model_config.is_multimodal:
            # New threads inherit the spawning thread's CPU affinity, and this
            # launch thread still has the full mask here. Narrow it to the
            # server cores first so every MM thread created below — the
            # processor's executors and the Rust MM workers — stays off the
            # scheduler's reserved launch cores: MM preprocessing scheduled
            # onto those cores preempts the scheduler loop and inflates
            # inter-token latency.
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
            spec = mm_host.resolve_native_spec()
            if spec is None:
                raise RuntimeError(
                    "SGLANG_RUST_SERVER=1: no native Rust MM pipeline for "
                    f"model_type={scheduler.model_config.hf_config.model_type!r} "
                    "(supported: "
                    f"{', '.join(sorted(NativeMmHost.NATIVE_QWEN_MODEL_TYPES))}, "
                    "mimo_v2; images only). Unset SGLANG_RUST_SERVER to serve "
                    "this model."
                )
            server.start_mm_workers(spec, mm_host.mm_workers)

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
            if self.mm_host is not None and isinstance(obj, TokenizedGenerateReqInput):
                # Natively processed: the buffers rode a Rust sidecar (stored
                # strictly before the ring push); wrap them into tensors here —
                # the only Python step of the native path. `None` for text-only
                # requests on a multimodal model.
                native = self.server.take_mm(obj.rid)
                if native is not None:
                    obj.mm_inputs = self.mm_host.build_native_mm(native)
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
        # Not a `server_args` field: `TokenizerManager` derives it, and the rust
        # ingress needs the same number for its total-token check.
        server_args["num_reserved_tokens"] = compute_num_reserved_tokens(
            scheduler.server_args
        )
        server_args["max_total_num_tokens"] = scheduler.max_total_num_tokens

        # The Rust server only reads local files (hub-blind, so it can never
        # disagree with huggingface_hub about cache layout): resolve a repo-id
        # tokenizer_path to the cached tokenizer.json here. No network — the
        # scheduler's init_tokenizer already downloaded it.
        if not scheduler.server_args.skip_tokenizer_init:
            path = server_args["tokenizer_path"] or server_args["model_path"]
            if not os.path.exists(path):
                server_args["tokenizer_path"] = resolve_local_or_cached_file(
                    path, "tokenizer.json", server_args["revision"]
                )

        return msgspec.json.encode(server_args, enc_hook=str).decode("utf-8")

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
        # Bound the pool set instead of taking the whole remainder: this
        # rank's "allowed" cores are usually the entire NUMA node, which the
        # *sibling* TP ranks' processes share — an unbounded mask lets MM
        # preprocessing bursts land on (and preempt) a sibling's CUDA-launch
        # thread, inflating every rank's forward via the TP collectives.
        # Measured on Qwen3.5-35B TP4 with one 720p image per request: the
        # unbounded mask cost the worst sibling ~20 ms of ViT wall time;
        # bounding the pools removed it. The budget covers the CPU-hot
        # threads (MM workers + tokenizer/ingress/egress/api, which are
        # I/O-shaped and rarely all hot at once); everything else on the node
        # is left to the scheduler ranks.
        pool_budget = max(8, mm_workers + 4)
        server_cores = allowed[reserve : reserve + pool_budget]
        logger.info(
            "rust server cores=%s, scheduler launch cores=%s",
            server_cores,
            launch_cores,
        )
        return launch_cores, server_cores
