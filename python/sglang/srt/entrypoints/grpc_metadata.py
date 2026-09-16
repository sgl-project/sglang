"""Server metadata shared by the full and follower-only native gRPC servers."""

import json
from typing import Any, Dict

from sglang.srt.runtime_context import describe_kv_events_publisher, get_serving
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins


def get_server_info_json(server_args, scheduler_info: Dict[str, Any]) -> str:
    """Preserve the leader's server-info schema on both node roles.

    ``kv_event_sources`` comes from scheduler readiness and lists only actual
    node-local publishers (an empty list means none). Each entry carries the
    global DP rank, local subscription endpoint, topic and logical block size.
    The existing ``kv_events`` descriptor remains unchanged for older clients;
    its global DP count must not be used to infer node-local ownership.
    ``node_rank``, ``nnodes``, ``dp_size`` and ``dist_init_addr`` are already in
    the resolved server args, including the shared engine rendezvous address.
    """
    result = server_args.resolved_dict()
    result["launch_command"] = server_args.launch_command
    result.update(scheduler_info)
    result["kv_events"] = describe_kv_events_publisher(server_args)
    return json.dumps(msgspec_to_builtins(result), default=str)


def start_follower_grpc_server(server_args, scheduler_info: Dict[str, Any]):
    """Expose GetServerInfo only, without constructing a TokenizerManager.

    The snapshot is taken after scheduler readiness. This does not launch a
    sidecar or enable inference/control RPCs.
    """
    serving = get_serving()
    if serving.grpc_port is None or serving.smg_grpc_mode or serving.grpc_mode:
        return None

    from sglang.srt.rust_extensions import load_rust_extension

    grpc_native = load_rust_extension("sglang.srt.rust_extensions._grpc")
    return grpc_native.start_metadata_server(
        host=serving.host,
        port=serving.grpc_port,
        server_info_json=get_server_info_json(server_args, scheduler_info),
    )
