"""Runtime identity and lookup capacity for remote Engram storage."""

import hashlib
import json
import os
import socket


def make_remote_config(options, heads):
    from mooncake.engram import EngramStoreConfig

    from sglang.srt.runtime_context import (
        get_device,
        get_exec,
        get_model,
        get_parallel,
        get_schedule,
        get_spec,
        process_model_config,
    )

    parallel, schedule = get_parallel(), get_schedule()
    if parallel.pp_size != 1:
        raise ValueError(
            "deepep_rdma currently requires PP1 with matching table owners"
        )
    if parallel.dp_size != 1:
        raise ValueError("deepep_rdma currently requires one model replica per engine")
    if parallel.launch_world_size % parallel.nnodes:
        raise ValueError("Engram requires uniform GPU caller placement across nodes")
    callers = parallel.launch_world_size // parallel.nnodes
    caller = parallel.tp_rank % callers
    model, device = get_model(), get_device()
    model_key = json.dumps([model.model_path, model.revision], separators=(",", ":"))
    node = json.dumps(
        [
            os.environ.get("MOONCAKE_HOST_ID", "").strip() or socket.gethostname(),
            device.base_gpu_id,
            device.gpu_id_step,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        ],
        separators=(",", ":"),
    )
    node_id = hashlib.sha256(node.encode()).hexdigest()
    graph = get_exec().graph.cuda_graph_config
    decode = graph.decode.max_bs or 1
    prefill_graph = graph.prefill.max_bs or 1
    prefill = schedule.chunked_prefill_size
    if prefill is None or prefill < 1:
        prefill = max(schedule.max_prefill_tokens, process_model_config().context_len)
    # KV sizing caps its default request estimate at 4096 before memory limits.
    draft = get_spec().speculative_num_draft_tokens or 1
    decode_tokens = max(schedule.max_running_requests or 4096, decode) * draft
    tokens = max(
        prefill,
        prefill_graph,
        decode_tokens,
    )
    if schedule.enable_mixed_chunk:
        tokens = max(tokens, prefill + decode_tokens)
    # Global-TP Engram lookup gathers attention-DP token IDs before lookup.
    capacity = tokens * max(parallel.attn_dp_size, 1) * heads
    configured = options.get("max_lookup_rows", capacity)
    if configured < capacity:
        raise ValueError(
            f"max_lookup_rows must cover at least {capacity} submitted IDs"
        )
    settings = dict(options)
    settings["max_lookup_rows"] = configured
    if os.environ.get("NCCL_CUMEM_ENABLE") != "1":
        raise ValueError(
            "deepep_rdma requires NCCL_CUMEM_ENABLE=1 before SGLang launch"
        )
    return EngramStoreConfig(
        backend="deepep_rdma",
        node_id=node_id,
        model_key=model_key,
        caller_id=caller,
        num_callers=callers,
        **settings,
    )
