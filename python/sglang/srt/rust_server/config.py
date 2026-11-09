"""Configuration and CPU placement for the embedded Rust server."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import socket
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.disaggregation.kv_events import select_kv_publisher_dp_rank
from sglang.srt.entrypoints.server_utils import (
    build_model_info_fields,
    build_server_info,
)
from sglang.srt.entrypoints.v1_loads import (
    _accelerator_name,
    _num_accelerators_per_dp_rank,
)
from sglang.srt.environ import envs
from sglang.srt.managers.load_snapshot import ShmLoadSnapshotWriter
from sglang.srt.managers.utils import compute_num_reserved_tokens
from sglang.srt.observability.metrics_collector import tokenizer_metric_labels
from sglang.srt.observability.request_logger import (
    WHITELISTED_HEADERS,
    RequestLogger,
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)
from sglang.srt.observability.utils import generate_buckets
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_lora,
    get_memory,
    get_mm,
    get_model,
    get_observability,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.watermarking.core import redact_watermark_secrets
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.rust_extensions._server import ServerArgs
    from sglang.srt.server_args import ServerArgs as PyServerArgs

logger = logging.getLogger(__name__)


def _build_server_args(
    scheduler: Scheduler,
    *,
    extension: Optional[ModuleType] = None,
    health_generate_headers: Optional[Dict[str, str]] = None,
    default_max_new_tokens: Optional[int] = SamplingParams().max_new_tokens,
    request_class: Optional[type] = None,
    server_added_custom_params: Tuple[str, ...] = (),
    port_offset: int = 0,
) -> ServerArgs:
    """The typed launch configuration for the scheduler's embedded Rust server:
    the ``server_args`` fields it reads, the already-resolved
    ``model_config``, and launch-time facts — as the Rust extension's own
    ``ServerArgs`` class. Its constructor takes every field as a required
    keyword (see ``rust/sglang-server/src/message/config.rs``), so a
    missing, extra or mistyped field fails here at boot rather than
    running on a silently-defaulted knob."""
    from sglang.srt.rust_extensions import load_rust_extension

    ext = extension or load_rust_extension("sglang.srt.rust_extensions._server")

    sa = resolving_view(scheduler.server_args)
    mc = scheduler.model_config
    disaggregation_mode = {
        "null": ext.DisaggregationMode.Null,
        "prefill": ext.DisaggregationMode.Prefill,
        "decode": ext.DisaggregationMode.Decode,
    }[get_disagg().disaggregation_mode]
    grpc_port = (
        None
        if get_serving().smg_grpc_mode or get_serving().grpc_mode
        else get_serving().grpc_port
    )
    deployment_info = {
        "disaggregation_bootstrap_port": get_disagg().disaggregation_bootstrap_port,
        "node_rank": get_parallel().node_rank,
        "nnodes": get_parallel().nnodes,
        "dp_size": get_parallel().num_dp_ranks,
        "dp_rank": select_kv_publisher_dp_rank(
            get_parallel().attn_dp_size,
            get_parallel().attn_dp_rank,
            get_parallel().dp_rank,
        ),
        "dist_init_addr": get_parallel().dist_init_addr,
        "base_gpu_id": get_device().base_gpu_id,
        "page_size": get_schedule().page_size,
        "dcp_size": get_parallel().dcp_size,
        "max_running_requests": get_schedule().max_running_requests,
        "max_prefill_tokens": get_schedule().max_prefill_tokens,
        "enable_dp_attention": get_parallel().attn_dp_size > 1,
        "speculative_algorithm": get_spec().speculative_algorithm,
        "enable_hierarchical_cache": get_memory().enable_hierarchical_cache,
        "hicache_ratio": get_memory().hicache_ratio,
        "hicache_size": get_memory().hicache_size,
        "hicache_write_policy": get_memory().hicache_write_policy,
        # Python's scheduler handshake only computes startup_time after all
        # components have initialized, but the Rust frontend must start earlier
        # to serve the disaggregation bootstrap registry. Read the already-
        # initialized KV publisher directly instead of calling get_init_info()
        # on a partially initialized scheduler.
        "hicache_host_total_tokens": None,
        "kv_event_sources": scheduler.kv_events_publisher.local_kv_event_sources(
            scheduler.page_size * get_parallel().dcp_size
        ),
        # Every Rust listener of the deployment (`RustServer.gather_server_endpoints`).
        "rust_server_endpoints": scheduler.rust_server_endpoints,
        "node_listeners": _node_listeners(grpc_port=grpc_port),
    }
    # `/v1/loads` reads this rank's slot of the scheduler's shared-memory
    # snapshots directly, from the file on this node even under multi-node DP;
    # only a forced ZMQ transport leaves no file to read.
    shm_writer = scheduler.load_snapshot_writer
    if not isinstance(shm_writer, ShmLoadSnapshotWriter):
        shm_writer = None
    return ext.ServerArgs(
        model_path=get_model().model_path,
        served_model_name=get_serving().served_model_name,
        tokenizer_path=scheduler.rust_server_tokenizer_path(),
        public_tokenizer_path=get_serving().tokenizer_path,
        revision=get_model().revision,
        load_format=get_model().load_format,
        weight_version=get_serving().weight_version,
        hicache_storage_backend=get_memory().hicache_storage_backend,
        hicache_storage_backend_extra_config=get_memory().hicache_storage_backend_extra_config,
        hicache_storage_prefetch_policy=get_memory().hicache_storage_prefetch_policy,
        hicache_write_policy=get_memory().hicache_write_policy,
        host=get_serving().host,
        port=get_serving().port,
        # The shared field is also populated for legacy SMG mode; only forward
        # it when it selects SGLang's native gRPC transport.
        grpc_port=grpc_port,
        grpc_response_timeout_secs=get_serving().grpc_response_timeout_secs,
        api_key=get_serving().api_key,
        admin_api_key=get_serving().admin_api_key,
        log_level=get_observability().log_level,
        log_level_http=get_observability().log_level_http,
        chat_template=get_serving().chat_template,
        default_chat_template_kwargs=json.dumps(
            get_serving().default_chat_template_kwargs or {}
        ),
        tool_call_parser=get_serving().tool_call_parser,
        reasoning_parser=get_serving().reasoning_parser,
        stream_response_default_include_usage=get_serving().stream_response_default_include_usage,
        enable_cache_report=get_serving().enable_cache_report,
        tokenizer_worker_num=get_serving().tokenizer_worker_num,
        detokenizer_worker_num=get_serving().detokenizer_worker_num,
        skip_tokenizer_init=get_serving().skip_tokenizer_init,
        skip_server_warmup=get_serving().skip_server_warmup,
        gc_threshold=get_device().gc_threshold,
        health_generate_headers=dict(health_generate_headers or {}),
        default_priority_value=(
            get_schedule().default_priority_value
            if get_schedule().enable_priority_scheduling
            else None
        ),
        enable_request_header_overrides=envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get(),
        enable_request_decompression=envs.SGLANG_ENABLE_REQUEST_DECOMPRESSION.get(),
        incremental_streaming_output=get_serving().incremental_streaming_output,
        disaggregation_mode=disaggregation_mode,
        model_config=ext.ModelConfig(
            context_len=mc.context_len,
            vocab_size=mc.vocab_size,
            is_multimodal=mc.is_multimodal,
            model_type=getattr(mc.hf_config, "model_type", None),
            # Resolved default sampling params (generation_config.json when
            # `--sampling-defaults model`, {} otherwise). The rust server
            # consumes these for omitted temperature/top_p in chat
            # conversions instead of hard-coding the OpenAI terminal
            # defaults.
            default_sampling_params=ext.DefaultSamplingParams(
                **mc.get_default_sampling_params()
            ),
        ),
        preferred_sampling_params=(
            json.dumps(get_serving().preferred_sampling_params)
            if get_serving().preferred_sampling_params is not None
            else None
        ),
        default_max_new_tokens=default_max_new_tokens,
        lora_ids={
            ref.lora_name: ref.lora_id
            for ref in (get_lora().lora_paths or [])
            if ref.lora_name is not None
        },
        enable_strict_thinking=get_serving().enable_strict_thinking,
        # A list of pairs keeps the configured order, which decides the error
        # when several modalities are over their limit.
        limit_mm_data_per_request=list(
            (get_mm().limit_mm_data_per_request or {}).items()
        ),
        allow_auto_truncate=get_serving().allow_auto_truncate,
        enable_return_hidden_states=sa.enable_return_hidden_states,
        # Not a `server_args` field: `TokenizerManager` derives it, and the
        # rust ingress needs the same number for its total-token check.
        num_reserved_tokens=compute_num_reserved_tokens(),
        tokenizer_model_max_length=_tokenizer_model_max_length(
            scheduler.tokenizer, mc.context_len
        ),
        # Launch-time facts Python's /server_info and /model_info report from
        # scheduler_info / the package / the model — stamped here so the rust
        # endpoints can serve them statically (no scheduler round-trip).
        version=__version__,
        max_total_num_tokens=scheduler.max_total_num_tokens,
        deployment_info=json.dumps(deployment_info),
        launch_server_info=_launch_server_info(scheduler),
        launch_model_info=json.dumps(
            msgspec_to_builtins(build_model_info_fields(mc, scheduler.server_args)),
            default=str,
        ),
        load_snapshot_path=shm_writer.path if shm_writer else None,
        load_snapshot_dp_size=shm_writer.dp_size if shm_writer else 0,
        load_snapshot_dp_rank=shm_writer.dp_rank if shm_writer else 0,
        accelerator=_accelerator_name(),
        num_accelerators=_num_accelerators_per_dp_rank(),
        metrics=(
            _build_metrics_config(ext) if get_observability().enable_metrics else None
        ),
        request_log=(
            ext.RequestLogConfig(
                **request_log_settings(
                    get_observability().log_requests_level,
                    request_class,
                    get_observability().log_requests_format,
                ),
                **_request_log_targets(
                    get_observability().log_requests_target, port_offset
                ),
            )
            if get_observability().log_requests
            else None
        ),
        crash_dump=bool(get_observability().crash_dump_folder),
        server_added_custom_params=list(server_added_custom_params),
    )


def request_log_settings(
    level: int, request_class: Optional[type], log_format: str = "text"
) -> Dict[str, Any]:
    """``RequestLogger``'s settings for ``--log-requests-level`` ``level`` and
    ``--log-requests-format`` ``log_format``, for the Rust server to render the
    lines Python logs: the level's metadata, ``request_class``'s fields in
    declaration order with each default rendered as the format renders a
    value, and from level 2 the OpenAI request models' fields."""
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )
    from sglang.srt.managers.io_struct import GenerateReqInput

    request_class = request_class or GenerateReqInput
    logger = RequestLogger(
        log_requests=True,
        log_requests_level=level,
        log_requests_format=log_format,
        log_requests_target=None,
    )
    max_length, skip_names, out_skip_names = logger.metadata

    def default(field: dataclasses.Field) -> Any:
        if field.default is not dataclasses.MISSING:
            return field.default
        return field.default_factory()

    def render(value: Any) -> str:
        if log_format == "json":
            return json.dumps(
                _transform_data_for_logging(value, max_length), ensure_ascii=False
            )
        return _dataclass_to_string_truncated(value, max_length)

    openai_requests = level >= 2
    batch_item_fields, batch_shared_fields = _batch_item_fields(request_class)
    return dict(
        max_length=max_length,
        skip_names=sorted(skip_names or ()),
        out_skip_names=sorted(out_skip_names or ()),
        json=log_format == "json",
        class_name=request_class.__name__,
        fields=[
            (field.name, render(default(field)))
            for field in dataclasses.fields(request_class)
        ],
        batch_item_fields=batch_item_fields,
        batch_shared_fields=batch_shared_fields,
        headers=list(WHITELISTED_HEADERS),
        exceeded_ms=logger.log_exceeded_ms,
        openai_requests=openai_requests,
        chat_completion_fields=(
            _openai_model_fields(ChatCompletionRequest) if openai_requests else []
        ),
        completion_fields=(
            _openai_model_fields(CompletionRequest) if openai_requests else []
        ),
    )


def _openai_model_fields(model: type) -> List[Tuple[str, Optional[str]]]:
    """What ``model.model_dump(exclude_none=True)`` lists of a request: each
    dumped field in declaration order, with its default as JSON (``None``
    when it is required or ``None``)."""
    from pydantic_core import to_jsonable_python

    fields = []
    for name, field in model.model_fields.items():
        if field.exclude:
            continue
        default = (
            None
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        fields.append(
            (name, None if default is None else json.dumps(to_jsonable_python(default)))
        )
    return fields


def _request_log_targets(targets: Optional[List[str]], rank: int) -> Dict[str, Any]:
    """``create_log_targets``' targets for the listener at node-local DP
    offset ``rank``: stdout, and in each target directory (created here) the
    file ``{hostname}_{rank}.log``, which the Rust server rotates hourly. A
    directory listed twice gets one file: two rotators of one file would
    rename it under each other."""
    if not targets:
        return dict(stdout=True, files=[])
    files = {}
    for directory in targets:
        if directory.lower() == "stdout":
            continue
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{socket.gethostname()}_{rank}.log")
        files.setdefault(os.path.realpath(path), path)
    return dict(
        stdout=any(target.lower() == "stdout" for target in targets),
        files=list(files.values()),
    )


def _batch_item_fields(request_class: type) -> Tuple[List[str], List[str]]:
    """The fields a batch item's object (``obj[i]``) takes from a normalized
    batch: its own entry of a per-item field, or a shared field's value; the
    rest keep their defaults. Read off ``request_class.__getitem__`` itself by
    giving each field a marker per item."""
    item_fields, shared_fields = [], []
    for field in dataclasses.fields(request_class):
        if not field.init:
            continue
        batch = request_class(text=["", ""])
        try:
            batch.normalize_batch_and_arguments()
        except Exception as e:
            raise RuntimeError(
                f"cannot read which fields a batch item of {request_class.__name__} "
                "keeps: normalizing a two-prompt batch failed"
            ) from e
        markers = [object(), object()]
        setattr(batch, field.name, markers)
        try:
            value = getattr(batch[0], field.name)
        except Exception:  # an item helper that cannot take the marker
            continue
        if value is markers[0]:
            item_fields.append(field.name)
        elif value is markers:
            shared_fields.append(field.name)
    return item_fields, shared_fields


def _build_metrics_config(ext: ModuleType):
    """What the Rust server needs to record ``TokenizerMetricsCollector``'s
    families as Python does, and where every process's samples live."""
    obs = get_observability()

    def token_buckets(rule: Optional[List[str]]) -> Optional[List[float]]:
        # The default rule keeps the Rust side's copy of Python's defaults.
        if not rule or rule[0] == "default":
            return None
        return generate_buckets(rule, [])

    return ext.MetricsConfig(
        # Every process writes its prometheus_client samples under this
        # directory (set before the schedulers start); `/metrics` merges them.
        multiproc_dir=os.environ["PROMETHEUS_MULTIPROC_DIR"],
        labels=tokenizer_metric_labels(),
        priority_label=get_schedule().enable_priority_scheduling,
        bucket_time_to_first_token=obs.bucket_time_to_first_token,
        bucket_inter_token_latency=obs.bucket_inter_token_latency,
        bucket_e2e_request_latency=obs.bucket_e2e_request_latency,
        bucket_prompt_tokens=token_buckets(obs.prompt_tokens_buckets),
        bucket_generation_tokens=token_buckets(obs.generation_tokens_buckets),
    )


def _node_listeners(*, grpc_port: Optional[int]) -> List[Dict[str, Any]]:
    """`DeploymentInfo.node_listeners`: every Rust listener on this listener's
    node, in port-offset order, so each of them reports the same table."""
    parallel = get_parallel()
    nnodes_per_tp_group = max(parallel.nnodes // parallel.pp_size, 1)
    tp_size_per_node = parallel.tp_size // nnodes_per_tp_group
    first_tp_rank = tp_size_per_node * (parallel.node_rank % nnodes_per_tp_group)
    node_tp_ranks = range(first_tp_rank, first_tp_rank + tp_size_per_node)
    # Each attention DP group hosts a listener on its first rank (CP=TP=0).
    dp_group_width = parallel.attn_tp_size * parallel.attn_cp_size
    leader_tp_ranks = [tp for tp in node_tp_ranks if tp % dp_group_width == 0]
    # Plain DP launches every replica on these leaders; otherwise only this one.
    if not parallel.attn_dp_enabled and parallel.dp_size > 1:
        replicas = range(parallel.dp_size)
    else:
        replicas = [parallel.dp_rank]
    # Each listener reports this as its own `dp_rank`.
    listener_dp_ranks = [
        select_kv_publisher_dp_rank(
            parallel.attn_dp_size, tp_rank // dp_group_width, dp_rank
        )
        for dp_rank in replicas
        for tp_rank in leader_tp_ranks
    ]
    # RustServer.launch offsets a listener by the node's listeners before it.
    return [
        {
            "port_offset": offset,
            "dp_rank": dp_rank,
            "http_port": get_serving().port + offset,
            "grpc_port": None if grpc_port is None else grpc_port + offset,
        }
        for offset, dp_rank in enumerate(listener_dp_ranks)
    ]


def _launch_server_info(scheduler: Scheduler) -> str:
    """The `/server_info` fields Python's HTTP server reports that are fixed
    at launch, as a JSON object: the resolved `server_args`,
    `launch_command`, `kv_events`, and the scheduler's init info less what is
    only known once initialization finishes (`startup_time`, which the Rust
    server reports per request, and the gRPC `kv_event_sources`). Secrets are
    redacted as Python's HTTP and gRPC frontends redact them."""
    scheduler_info = {
        "status": "ready",
        "max_total_num_tokens": scheduler.max_total_num_tokens,
        "max_req_input_len": scheduler.max_req_input_len,
    }
    info = msgspec_to_builtins(
        redact_watermark_secrets(
            build_server_info(scheduler.server_args, scheduler_info)
        )
    )
    # The Rust listener speaks HTTP/1 only, whatever `--enable-http2` says; a
    # router reads this key to pick its protocol.
    info["enable_http2"] = False
    _redact_api_keys(info, scheduler.server_args)
    return json.dumps(info, default=str)


_API_KEY_FIELDS = ("api_key", "admin_api_key")


def _redact_api_keys(info: Dict[str, Any], server_args: PyServerArgs) -> None:
    """Keep the listener's keys out of `/server_info`'s `launch_command`,
    which only the API key (or nothing) guards: reporting them would hand the
    admin key to every API-key holder. `build_server_info` has replaced the
    key fields in `info` already, so the values come from `server_args`."""
    keys = [getattr(server_args, name) for name in _API_KEY_FIELDS]
    keys = [key for key in keys if key]
    command = info.get("launch_command")
    if not command:
        return
    # The command joins argv. First each key's value, whole and longest first
    # (one key may contain the other), which catches a value holding spaces...
    for key in sorted(keys, key=len, reverse=True):
        command = re.sub(rf"(?<=[ =]){re.escape(key)}(?= |$)", "<redacted>", command)
    # ...then whatever follows a key flag, which catches a value a later flag
    # overrode. argparse takes any unambiguous prefix of a flag.
    flags = [f"--{name.replace('_', '-')}" for name in _API_KEY_FIELDS]
    words, redact_next = [], False
    for word in command.split(" "):
        flag, eq, _ = word.partition("=")
        if redact_next:
            word, redact_next = "<redacted>", False
        elif len(flag) > 2 and any(f.startswith(flag) for f in flags):
            if eq:
                word = f"{flag}=<redacted>"
            else:
                redact_next = True
        words.append(word)
    info["launch_command"] = " ".join(words)


def _tokenizer_model_max_length(tokenizer: Any, context_len: int) -> int:
    """`/v1/tokenize`'s `max_model_len`, as `OpenAIServingTokenize` reports
    it: the tokenizer's `model_max_length`, or the context length when that is
    not a 64-bit integer (an unset HF value is a 1e30 sentinel). The Rust
    field is signed, so the bound stops at 2**63."""
    max_model_len = getattr(tokenizer, "model_max_length", -1)
    if isinstance(max_model_len, int) and -(2**63) <= max_model_len < 2**63:
        return max_model_len
    return context_len


def _frontend_cpu_key() -> Optional[Tuple[str, Tuple[int, ...]]]:
    """This process's ``(hostname, allowed cores)``: listeners with equal keys
    would partition the same cores, or ``None`` without an affinity API."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return socket.gethostname(), tuple(sorted(os.sched_getaffinity(0)))
    except OSError:
        return None


def _frontend_cpu_share(
    cpu_keys: List[Optional[Tuple[str, Tuple[int, ...]]]], rank: int
) -> Tuple[int, int]:
    """``(index, count)`` of ``rank`` among the ranks whose key equals its own,
    in rank order; ``(0, 1)`` for a rank that contributed no key."""
    own_key = cpu_keys[rank]
    if own_key is None:
        return 0, 1
    peers = [peer for peer, key in enumerate(cpu_keys) if key == own_key]
    return peers.index(rank), len(peers)


def _partition_cores(
    mm_workers: int = 0,
    server_core_budget: Optional[Callable[[int, int], int]] = None,
    cpu_share: Tuple[int, int] = (0, 1),
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Split this rank's allowed cores into ``(launch_cores, server_cores)``.

    ``launch_cores`` are left out of ``server_cores``, so the scheduler thread
    always finds cores no frontend thread can take; the scheduler thread itself
    is not pinned to them. Pure computation — no affinity is changed here.
    Both sets are a subset of this rank's NUMA-local cores (when affinity/NUMA
    bind is on), so the partition stays NUMA-local. Returns ``(None, None)``
    (server runs unpinned, confined only by the process affinity) when the
    platform has no affinity API or too few cores to split.

    ``cpu_share`` is ``(index, count)`` from ``_frontend_cpu_share``: under
    attention DP every DP rank hosts a listener, and ranks launched with one
    affinity mask would otherwise all pin their frontends onto the same first
    cores of it. The mask is cut into ``count`` contiguous blocks and only
    block ``index`` is partitioned.
    """
    if not hasattr(os, "sched_getaffinity"):
        return None, None
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except OSError as e:
        logger.warning("rust server: cannot read cpu affinity: %s", e)
        return None, None

    index, count = cpu_share
    if count > 1:
        block = len(allowed) // count
        if block >= 4:
            allowed = allowed[index * block : (index + 1) * block]
        else:
            logger.warning(
                "rust server: %d cores shared by %d listeners; their frontend "
                "cores overlap",
                len(allowed),
                count,
            )

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
    pool_budget = (
        server_core_budget(len(allowed), mm_workers)
        if server_core_budget is not None
        else max(8, mm_workers + 4)
    )
    server_cores = allowed[reserve : reserve + pool_budget]
    logger.info(
        "rust server cores=%s, cores kept free for the scheduler=%s",
        server_cores,
        launch_cores,
    )
    return launch_cores, server_cores
