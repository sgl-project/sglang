from __future__ import annotations

import argparse
import ast
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import ray
import sglang as sgl
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.utils.network import is_valid_ipv6_address

ACTOR_NAME = "draft"
RAY_NAMESPACE = "dspec"
DEFAULT_PROMPT_COLUMN_CANDIDATES = [
    "prompt",
    "messages",
    "chat",
    "conversations",
    "text",
    "question",
    "instruction",
    "input",
    "query",
]
DAPO_MATH_17K_DEFAULT_PROMPT_COLUMN = "prompt"


def format_tcp_address(ip: str, port: int | str) -> str:
    """Return a ZMQ TCP endpoint, preserving IPv6 bracket formatting."""
    host = f"[{ip}]" if is_valid_ipv6_address(ip) else ip
    return f"tcp://{host}:{port}"


@dataclass
class DemoRayRuntime:
    """Track a locally started Ray head so callers can clean it up reliably."""

    address: str
    namespace: str
    head_process: subprocess.Popen | None = None

    def build_init_kwargs(self) -> dict[str, object]:
        """Return common keyword arguments for connecting to this Ray runtime."""
        return {
            "address": self.address,
            "namespace": self.namespace,
            "ignore_reinit_error": True,
            "log_to_driver": True,
            "logging_level": "ERROR",
        }

    def stop(self) -> None:
        """Terminate the local Ray head process if this runtime started one."""
        if self.head_process is None:
            return
        if self.head_process.poll() is not None:
            return
        self.head_process.terminate()
        try:
            self.head_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.head_process.kill()
            self.head_process.wait(timeout=10)


@dataclass
class DecoupledSpecEndpointConfig:
    """Bind/connect endpoint config for one decoupled-spec instance."""

    bind_endpoint: str
    connect_endpoints: list[str]
    rank: int


@dataclass
class DecoupledSpecTopology:
    """Endpoint topology and actor handles for a decoupled-spec run."""

    drafter_configs: list[DecoupledSpecEndpointConfig]
    verifier_configs: list[DecoupledSpecEndpointConfig]
    draft_actors: list[Any] | None = None
    cleanup_handles: list[Any] | None = None


def _pick_free_local_port() -> int:
    """Ask the OS for a currently free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _start_local_ray_head(port: int) -> subprocess.Popen:
    """Start a blocking local Ray head process on the requested port."""
    command = [
        sys.executable,
        "-m",
        "ray",
        "start",
        "--head",
        f"--port={port}",
        "--node-ip-address=127.0.0.1",
        "--include-dashboard=false",
        "--disable-usage-stats",
        "--block",
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


def init_demo_ray(namespace: str) -> DemoRayRuntime:
    """Start a private local Ray runtime and connect the current process to it."""
    port = _pick_free_local_port()
    address = f"127.0.0.1:{port}"
    head_process = _start_local_ray_head(port)
    deadline = time.monotonic() + 30
    last_error = None

    while time.monotonic() < deadline:
        try:
            ray.init(
                address=address,
                namespace=namespace,
                ignore_reinit_error=True,
                log_to_driver=True,
                logging_level="ERROR",
            )
            return DemoRayRuntime(
                address=address,
                namespace=namespace,
                head_process=head_process,
            )
        except Exception as exc:
            last_error = exc
            if head_process.poll() is not None:
                break
            time.sleep(0.5)

    head_process.terminate()
    try:
        head_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        head_process.kill()
        head_process.wait(timeout=10)
    raise RuntimeError(
        f"Failed to start demo Ray head at {address}: {last_error!r}"
    ) from last_error


def _new_local_ipc_endpoint() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    return f"ipc://{path}"


def create_local_decoupled_spec_topology(
    num_drafters: int = 1,
    num_verifiers: int = 1,
) -> DecoupledSpecTopology:
    """Create local IPC endpoints and return them as engine-ready lists."""
    if num_drafters <= 0:
        raise ValueError("num_drafters must be positive")
    if num_verifiers <= 0:
        raise ValueError("num_verifiers must be positive")
    control_endpoints = [_new_local_ipc_endpoint() for _ in range(num_drafters)]
    result_endpoints = [_new_local_ipc_endpoint() for _ in range(num_verifiers)]
    return DecoupledSpecTopology(
        drafter_configs=[
            DecoupledSpecEndpointConfig(
                bind_endpoint=endpoint,
                connect_endpoints=list(result_endpoints),
                rank=rank,
            )
            for rank, endpoint in enumerate(control_endpoints)
        ],
        verifier_configs=[
            DecoupledSpecEndpointConfig(
                bind_endpoint=endpoint,
                connect_endpoints=list(control_endpoints),
                rank=rank,
            )
            for rank, endpoint in enumerate(result_endpoints)
        ],
    )


@ray.remote
class LocalDraftActor:
    """Ray actor that hosts a local single-node decoupled draft engine."""

    def __init__(
        self,
        *,
        model_path: str,
        gpu_ids: list[str],
        tp_size: int,
        speculative_num_steps: int,
        bind_endpoint: str,
        connect_endpoints: list[str],
        rank: int,
    ):
        """Pin GPUs and construct the draft `sgl.Engine` for local demos."""
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        self.drafter = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            speculative_algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_steps + 1,
            decoupled_spec_bind_endpoint=bind_endpoint,
            decoupled_spec_connect_endpoints=connect_endpoints,
            decoupled_spec_rank=rank,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
        )

    def ready(self) -> bool:
        """Signal that the local draft engine has initialized successfully."""
        return True

    def shutdown(self) -> None:
        """Shutdown the local draft engine owned by this actor."""
        self.drafter.shutdown()


def get_visible_gpu_ids() -> list[str]:
    """Return visible GPU ids, honoring CUDA_VISIBLE_DEVICES when it is set."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    gpu_count = torch.cuda.device_count()
    return [str(i) for i in range(gpu_count)]


def allocate_demo_gpus(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Split visible GPUs between draft and target engines for local demos."""
    visible_gpu_ids = get_visible_gpu_ids()
    total_visible_gpus = len(visible_gpu_ids)
    required_gpus = args.draft_tp_size + args.target_tp_size
    if total_visible_gpus == 0:
        raise RuntimeError("No visible CUDA GPUs found for decoupled spec demo")
    if args.draft_tp_size <= 0 or args.target_tp_size <= 0:
        raise ValueError("draft-tp-size and target-tp-size must both be positive")
    if required_gpus > total_visible_gpus:
        raise ValueError(
            "Insufficient visible GPUs for the demo: "
            f"need {required_gpus}, but only {total_visible_gpus} visible ({visible_gpu_ids})"
        )

    draft_gpu_ids = visible_gpu_ids[: args.draft_tp_size]
    target_gpu_ids = visible_gpu_ids[
        args.draft_tp_size : args.draft_tp_size + args.target_tp_size
    ]
    return draft_gpu_ids, target_gpu_ids


def _get_drafter_debug_env_vars() -> dict[str, str]:
    """Collect decoupled-spec debug environment variables for Ray actors."""
    return get_decoupled_spec_actor_env_vars()


def get_decoupled_spec_actor_env_vars(
    args: argparse.Namespace | None = None,
) -> dict[str, str]:
    """Collect decoupled-spec environment variables for Ray actors."""
    env_vars: dict[str, str] = {}
    if args is not None and hasattr(args, "decoupled_spec_allow_partial"):
        env_vars["SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL"] = (
            "1" if args.decoupled_spec_allow_partial else "0"
        )
    for env_name in (
        "SGLANG_DECOUPLED_SPEC_DEBUG",
        "SGLANG_DECOUPLED_SPEC_TRACE_DIR",
        "SGLANG_DECOUPLED_SPEC_SUMMARY_INTERVAL",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            env_vars[env_name] = env_value
    return env_vars


def launch_drafter_actor(
    args: argparse.Namespace,
    endpoint_config: DecoupledSpecEndpointConfig | None = None,
    *,
    topology: DecoupledSpecTopology | None = None,
) -> ray.actor.ActorHandle:
    """Launch the local draft actor and block until it reports readiness."""
    if topology is not None:
        endpoint_config = topology.drafter_configs[0]
    elif endpoint_config is None:
        topology = create_local_decoupled_spec_topology()
        endpoint_config = topology.drafter_configs[0]

    actor_options = dict(
        name=ACTOR_NAME,
        num_gpus=args.draft_tp_size,
        max_concurrency=128,
    )
    debug_env_vars = _get_drafter_debug_env_vars()
    if debug_env_vars:
        actor_options["runtime_env"] = {"env_vars": debug_env_vars}
    actor = LocalDraftActor.options(**actor_options).remote(
        model_path=args.draft_model_path,
        gpu_ids=args.draft_gpu_ids,
        tp_size=args.draft_tp_size,
        speculative_num_steps=args.num_speculative_steps,
        bind_endpoint=endpoint_config.bind_endpoint,
        connect_endpoints=endpoint_config.connect_endpoints,
        rank=endpoint_config.rank,
    )
    ray.get(actor.ready.remote())
    return actor


def launch_verifier(
    target_model_path: str,
    target_tp_size: int,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    endpoint_config: DecoupledSpecEndpointConfig | None = None,
    mamba_scheduler_strategy: str | None = None,
    *,
    topology: DecoupledSpecTopology | None = None,
) -> sgl.Engine:
    """Construct a local decoupled verify engine connected to the draft actor."""
    if topology is not None:
        endpoint_config = topology.verifier_configs[0]
    elif endpoint_config is None:
        topology = create_local_decoupled_spec_topology()
        endpoint_config = topology.verifier_configs[0]

    engine_kwargs: dict[str, Any] = dict(
        model_path=target_model_path,
        tp_size=target_tp_size,
        speculative_algorithm="DECOUPLED_VERIFY",
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        decoupled_spec_bind_endpoint=endpoint_config.bind_endpoint,
        decoupled_spec_connect_endpoints=endpoint_config.connect_endpoints,
        decoupled_spec_rank=endpoint_config.rank,
        disable_radix_cache=True,
    )
    if mamba_scheduler_strategy is not None:
        engine_kwargs["mamba_scheduler_strategy"] = mamba_scheduler_strategy
    return sgl.Engine(**engine_kwargs)


def _sort_gpu_ids(gpu_ids: list[Any]) -> list[str]:
    """Sort Ray/CUDA GPU ids numerically when possible and return strings."""

    def sort_key(value: Any) -> tuple[int, Any]:
        """Build a stable sort key for numeric and non-numeric GPU ids."""
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)

    return [str(value) for value in sorted(gpu_ids, key=sort_key)]


def _get_assigned_gpu_ids_from_ray() -> list[str]:
    """Read the GPU ids assigned to the current Ray actor."""
    context = ray.get_runtime_context()
    accelerator_ids = getattr(context, "get_accelerator_ids", lambda: {})()
    gpu_ids = accelerator_ids.get("GPU", [])
    if gpu_ids:
        return _sort_gpu_ids(gpu_ids)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        return _sort_gpu_ids(
            [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
        )
    return []


def pin_actor_to_assigned_gpus(expected_num_gpus: int) -> list[str]:
    """Set CUDA_VISIBLE_DEVICES to the GPUs Ray assigned to this actor."""
    gpu_ids = _get_assigned_gpu_ids_from_ray()
    if expected_num_gpus > 0 and len(gpu_ids) < expected_num_gpus:
        raise RuntimeError(
            f"Ray assigned {len(gpu_ids)} GPUs to actor, expected at least "
            f"{expected_num_gpus}: {gpu_ids}"
        )
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    return gpu_ids


def reserve_tcp_port(preferred_port: int | None = None) -> tuple[int, socket.socket]:
    """Bind and hold a TCP port, returning both the port and lock socket."""

    def bind_port(port: int) -> socket.socket:
        """Bind a listening socket to all interfaces on a specific port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.listen(1)
        return sock

    if preferred_port is not None:
        return preferred_port, bind_port(preferred_port)

    for _ in range(256):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("0.0.0.0", 0))
        probe.listen(1)
        candidate_port = int(probe.getsockname()[1])
        probe.close()
        try:
            return candidate_port, bind_port(candidate_port)
        except OSError:
            continue

    raise RuntimeError("failed to reserve a TCP port")


@ray.remote
class PortActor:
    """Ray actor used to reserve bootstrap ports on a specific placement node."""

    def __init__(self):
        """Initialize without a reservation; callers reserve ports explicitly."""
        self._reserved_socket: socket.socket | None = None

    def reserve_port(self, preferred_port: int | None = None) -> dict[str, Any]:
        """Reserve a TCP port on this actor's node and report host and port."""
        self.release_port()
        port, sock = reserve_tcp_port(preferred_port)
        self._reserved_socket = sock
        return {
            "host": ray.util.get_node_ip_address(),
            "port": port,
        }

    def release_port(self) -> bool:
        """Release any port reservation currently held by this actor."""
        if self._reserved_socket is not None:
            self._reserved_socket.close()
            self._reserved_socket = None
        return True


def create_result_endpoint_from_pg(
    pg,
    *,
    avoid_port: int | None = None,
    preferred_port: int | None = None,
) -> str:
    """Reserve a verifier result endpoint on target placement-group rank 0."""
    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
    for _ in range(16):
        actor = PortActor.options(
            num_cpus=0,
            scheduling_strategy=scheduling_strategy,
        ).remote()
        try:
            reservation = ray.get(actor.reserve_port.remote(preferred_port))
            host = reservation["host"]
            port = int(reservation["port"])
            ray.get(actor.release_port.remote())
        finally:
            ray.kill(actor, no_restart=True)

        if avoid_port is None or port != avoid_port:
            return format_tcp_address(host, port)
        if preferred_port is not None:
            raise RuntimeError(
                f"preferred result endpoint port {preferred_port} conflicts with "
                f"avoid_port {avoid_port}"
            )

    raise RuntimeError("failed to reserve a result endpoint port")


@ray.remote
class DraftActor:
    """Ray actor that hosts a draft engine for Ray/multi-node benchmark runs."""

    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
        speculative_num_steps: int,
        verifier_result_endpoints: list[str],
        rank: int,
        control_port: int | None = None,
        deterministic: bool = False,
        decoupled_spec_trace_dir: str | None = None,
    ):
        """Pin GPUs, allocate a control endpoint, and create the draft engine."""
        self.assigned_gpu_ids = pin_actor_to_assigned_gpus(tp_size)
        reserved_control_socket: socket.socket | None = None
        if control_port is None:
            control_port, reserved_control_socket = reserve_tcp_port()
        control_endpoint = format_tcp_address(
            ray.util.get_node_ip_address(),
            control_port,
        )
        if reserved_control_socket is not None:
            reserved_control_socket.close()
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            speculative_algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_steps + 1,
            decoupled_spec_bind_endpoint=control_endpoint,
            decoupled_spec_connect_endpoints=verifier_result_endpoints,
            decoupled_spec_rank=rank,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_deterministic_inference=deterministic,
            decoupled_spec_trace_dir=decoupled_spec_trace_dir,
        )
        self.control_endpoint = control_endpoint

    def ready(self) -> dict[str, Any]:
        """Return actor metadata once the remote draft engine is ready."""
        return {
            "assigned_gpu_ids": self.assigned_gpu_ids,
            "control_endpoint": self.control_endpoint,
        }

    def shutdown(self) -> bool:
        """Shutdown the remote draft engine owned by this actor."""
        self.engine.shutdown()
        return True


def launch_draft_actors(
    args: argparse.Namespace,
    verifier_result_endpoints: list[str],
    control_port: int | None = None,
) -> tuple[list[Any], list[str]]:
    """Launch draft actors and collect their control bind endpoints."""
    actors = []
    control_endpoints = []
    actor_env_vars = get_decoupled_spec_actor_env_vars(args)
    for replica_index in range(args.num_draft_replicas):
        actor_options: dict[str, Any] = dict(
            num_gpus=args.draft_tp_size,
            num_cpus=1,
            max_concurrency=128,
        )
        if actor_env_vars:
            actor_options["runtime_env"] = {"env_vars": actor_env_vars}
        actor = DraftActor.options(**actor_options).remote(
            model_path=args.draft_model_path,
            tp_size=args.draft_tp_size,
            speculative_num_steps=args.num_speculative_steps,
            verifier_result_endpoints=verifier_result_endpoints,
            rank=replica_index,
            control_port=control_port if replica_index == 0 else None,
            deterministic=args.deterministic,
            decoupled_spec_trace_dir=args.decoupled_spec_trace_dir,
        )
        actors.append(actor)
    ready_infos = ray.get([actor.ready.remote() for actor in actors])
    control_endpoints.extend(info["control_endpoint"] for info in ready_infos)
    return actors, control_endpoints


def create_remote_decoupled_spec_topology(
    args: argparse.Namespace,
    pg,
    *,
    avoid_port: int | None = None,
    preferred_result_port: int | None = None,
    preferred_control_port: int | None = None,
) -> DecoupledSpecTopology:
    """Create Ray/multi-node decoupled-spec endpoints and draft actors."""
    result_endpoint = create_result_endpoint_from_pg(
        pg,
        avoid_port=avoid_port,
        preferred_port=preferred_result_port,
    )
    draft_actors, control_endpoints = launch_draft_actors(
        args,
        [result_endpoint],
        control_port=preferred_control_port,
    )
    return DecoupledSpecTopology(
        drafter_configs=[
            DecoupledSpecEndpointConfig(
                bind_endpoint=endpoint,
                connect_endpoints=[result_endpoint],
                rank=rank,
            )
            for rank, endpoint in enumerate(control_endpoints)
        ],
        verifier_configs=[
            DecoupledSpecEndpointConfig(
                bind_endpoint=result_endpoint,
                connect_endpoints=control_endpoints,
                rank=0,
            )
        ],
        draft_actors=draft_actors,
    )


@ray.remote
class TargetActor:
    """Ray actor that hosts either a decoupled verifier or normal decode engine."""

    def __init__(
        self,
        *,
        mode: str,
        model_path: str,
        tp_size: int,
        ep_size: int | None = None,
        moe_a2a_backend: str | None = None,
        mamba_scheduler_strategy: str | None = None,
        nnodes: int,
        node_rank: int,
        dist_init_addr: str | None,
        batch_size: int,
        speculative_num_steps: int | None = None,
        bind_endpoint: str | None = None,
        connect_endpoints: list[str] | None = None,
        rank: int | None = None,
        deterministic: bool = False,
        decoupled_spec_trace_dir: str | None = None,
    ):
        """Pin GPUs and initialize the target engine for one node rank."""
        self.mode = mode
        self.node_rank = node_rank
        self.assigned_gpu_ids = pin_actor_to_assigned_gpus(max(tp_size // nnodes, 1))
        if node_rank >= 1:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

        engine_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            tp_size=tp_size,
            nnodes=nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            enable_deterministic_inference=deterministic,
            decoupled_spec_trace_dir=decoupled_spec_trace_dir,
        )
        if ep_size is not None:
            engine_kwargs["ep_size"] = ep_size
        if moe_a2a_backend is not None:
            engine_kwargs["moe_a2a_backend"] = moe_a2a_backend
        if mamba_scheduler_strategy is not None:
            engine_kwargs["mamba_scheduler_strategy"] = mamba_scheduler_strategy
        if mode == "decoupled_spec":
            engine_kwargs.update(
                speculative_algorithm="DECOUPLED_VERIFY",
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_steps + 1,
                decoupled_spec_bind_endpoint=bind_endpoint,
                decoupled_spec_connect_endpoints=connect_endpoints,
                decoupled_spec_rank=rank,
                disable_radix_cache=True,
            )
        elif mode == "decode":
            engine_kwargs["disable_piecewise_cuda_graph"] = True
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.engine = sgl.Engine(**engine_kwargs)

    def ready(self) -> dict[str, Any]:
        """Return actor metadata once the target engine has initialized."""
        return {
            "mode": self.mode,
            "node_rank": self.node_rank,
            "assigned_gpu_ids": self.assigned_gpu_ids,
        }

    def generate_batch(
        self,
        prompt_input_ids: list[list[int]],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run generation on rank 0 and return raw outputs."""
        if self.node_rank != 0:
            raise RuntimeError("generate_batch must be called on node rank 0")

        outputs = self.engine.generate(
            input_ids=prompt_input_ids,
            sampling_params=sampling_params,
        )
        if not isinstance(outputs, list):
            outputs = [outputs]
        return {"outputs": outputs}

    def shutdown(self) -> bool:
        """Shutdown the target engine owned by this actor."""
        self.engine.shutdown()
        return True

def infer_prompt_column(
    available_columns: list[str],
) -> str:
    """Choose a prompt column from common names in a parquet schema."""
    for candidate in DEFAULT_PROMPT_COLUMN_CANDIDATES:
        if candidate in available_columns:
            return candidate

    raise ValueError(
        "Unable to auto-detect the prompt column. "
        f"Available columns: {available_columns}"
    )


def resolve_dapo_math_17k_prompt_column(
    available_columns: list[str],
    prompt_column: str | None = None,
) -> str:
    """Validate and return the DAPO-Math-17k column that stores chat prompts."""
    selected_column = prompt_column or DAPO_MATH_17K_DEFAULT_PROMPT_COLUMN
    if selected_column not in available_columns:
        raise ValueError(
            "dataset-format=dapo_math_17k requires a prompt-style column with "
            "chat messages. "
            f"Requested column: {selected_column!r}. "
            f"Available columns: {available_columns}"
        )
    return selected_column


def _looks_like_chat_message(value: Any) -> bool:
    """Return whether a value resembles a single chat message dict."""
    return isinstance(value, dict) and "role" in value and "content" in value


def _is_chat_message_list(value: Any) -> bool:
    """Return whether a value is a non-empty list of chat message dicts."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_looks_like_chat_message(item) for item in value)
    )


def _flatten_message_content(content: Any) -> str:
    """Flatten string or multimodal-style message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_segments: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_segments.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_segments.append(item["text"])
        return "".join(text_segments)
    return str(content)


def _messages_to_fallback_text(messages: list[dict[str, Any]]) -> str:
    """Render chat messages into a simple role-prefixed text fallback."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _flatten_message_content(message.get("content", ""))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_real_verify_acceptance_stats(
    meta_info: dict[str, Any],
) -> tuple[float | None, float | None, int, int, int]:
    """Extract real draft-token acceptance metrics from SGLang meta_info."""
    verify_ct = meta_info.get("spec_verify_ct")
    accepted_tokens = meta_info.get("spec_accept_token_num")
    draft_tokens = meta_info.get("spec_draft_token_num")
    if verify_ct is None or accepted_tokens is None or draft_tokens is None:
        return None, None, 0, 0, 0

    verify_ct = int(verify_ct)
    accepted_tokens = int(accepted_tokens)
    draft_tokens = int(draft_tokens)
    if verify_ct <= 0 or draft_tokens <= 0:
        return None, None, 0, 0, 0

    # Acclen reports accepted draft tokens only; the verifier-sampled bonus
    # token is intentionally excluded.
    accept_length = accepted_tokens / verify_ct
    accept_rate = accepted_tokens / draft_tokens
    return accept_length, accept_rate, accepted_tokens, draft_tokens, verify_ct


def _maybe_parse_json_prompt(value: Any) -> Any:
    """Parse prompt strings that contain JSON or Python literal structures."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value


_CHATML_ROLE_PATTERN = re.compile(r"<\|im_start\|>(system|user|assistant)\n")


def _maybe_append_chatml_generation_prompt(
    prompt: str, *, enable_thinking: bool = False
) -> str:
    """Ensure ChatML text prompts end with an assistant generation prefix."""
    stripped = prompt.rstrip()
    if "<|im_start|>" not in stripped or "<|im_end|>" not in stripped:
        return prompt

    role_matches = list(_CHATML_ROLE_PATTERN.finditer(stripped))
    if not role_matches:
        return prompt

    last_role = role_matches[-1].group(1)
    thinking_suffix = "<think>\n" if enable_thinking else ""

    # The prompt already ends with an assistant generation prefix.
    if last_role == "assistant" and not stripped.endswith("<|im_end|>"):
        if enable_thinking and not stripped.endswith("<think>"):
            return stripped + thinking_suffix
        return stripped

    # ChatML user/system turns should terminate with an assistant prefix for generation.
    if last_role in {"system", "user"} and stripped.endswith("<|im_end|>"):
        return stripped + "\n<|im_start|>assistant\n" + thinking_suffix

    return prompt


def _build_chat_template_renderer(model_path: str, *, enable_thinking: bool = False):
    """Create a tokenizer-backed chat-template renderer when available."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return None
    if not getattr(tokenizer, "chat_template", None):
        return None

    def render(messages: list[dict[str, Any]]) -> str:
        """Render chat messages through the loaded tokenizer chat template."""
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = _maybe_append_chatml_generation_prompt(
                prompt, enable_thinking=enable_thinking
            )
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("tokenizer.apply_chat_template returned an empty prompt")
        return prompt

    return render


def _normalize_prompt(
    value: Any,
    row_index: int,
    column_name: str,
    chat_template_renderer,
    *,
    enable_thinking: bool = False,
) -> str:
    """Normalize a raw dataset value into the final string prompt."""
    if value is None:
        raise ValueError(
            f"Row {row_index} in column {column_name!r} is null, cannot build a prompt."
        )
    value = _maybe_parse_json_prompt(value)
    if isinstance(value, str):
        return _maybe_append_chatml_generation_prompt(
            value, enable_thinking=enable_thinking
        )
    if _is_chat_message_list(value):
        if chat_template_renderer is not None:
            try:
                return chat_template_renderer(value)
            except Exception:
                pass
        return _messages_to_fallback_text(value)
    return str(value)


def _build_dapo_math_17k_prompt(
    row: dict[str, Any],
    *,
    row_index: int,
    prompt_column: str,
    chat_template_renderer,
    enable_thinking: bool = False,
) -> str:
    """Build one prompt from a DAPO-Math-17k parquet row."""
    if prompt_column not in row:
        raise ValueError(
            f"Row {row_index} is missing DAPO-Math prompt column {prompt_column!r}."
        )
    return _normalize_prompt(
        row.get(prompt_column),
        row_index,
        prompt_column,
        chat_template_renderer,
        enable_thinking=enable_thinking,
    )


def load_prompt_batch(
    parquet_path: str,
    target_model_path: str,
    offset: int,
    batch_size: int,
    prompt_column: str | None,
    dataset_format: str,
    disable_chat_template: bool,
    enable_thinking: bool,
) -> tuple[str, list[str], int]:
    """Load and normalize a batch of prompts from a parquet dataset."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read parquet prompts. "
            "Please install it in the current Python environment."
        ) from exc

    if offset < 0:
        raise ValueError("offset must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    if offset >= total_rows:
        raise ValueError(
            f"offset {offset} is out of range for {parquet_path}; total rows: {total_rows}"
        )

    column_names = parquet_file.schema_arrow.names
    if dataset_format == "dapo_math_17k":
        selected_column = resolve_dapo_math_17k_prompt_column(
            column_names,
            prompt_column=prompt_column,
        )
        prompt_column_label = f"dapo_math_17k[{selected_column}]"
        read_columns = [selected_column]
    else:
        selected_column = prompt_column or infer_prompt_column(column_names)
        if selected_column not in column_names:
            raise ValueError(
                f"prompt column {selected_column!r} not found. "
                f"Available columns: {column_names}"
            )
        prompt_column_label = selected_column
        read_columns = [selected_column]
    chat_template_renderer = None
    if not disable_chat_template:
        chat_template_renderer = _build_chat_template_renderer(
            target_model_path, enable_thinking=enable_thinking
        )

    prompts: list[str] = []
    current_row = 0
    remaining_skip = offset
    reader_batch_size = max(batch_size, 1024)

    for record_batch in parquet_file.iter_batches(
        batch_size=reader_batch_size,
        columns=read_columns,
    ):
        if dataset_format == "dapo_math_17k":
            batch_rows = record_batch.to_pylist()
        else:
            batch_rows = record_batch.column(0).to_pylist()

        if remaining_skip >= len(batch_rows):
            remaining_skip -= len(batch_rows)
            current_row += len(batch_rows)
            continue

        start_index = remaining_skip
        end_index = min(len(batch_rows), start_index + (batch_size - len(prompts)))
        for local_index in range(start_index, end_index):
            row_index = current_row + local_index
            if dataset_format == "dapo_math_17k":
                prompts.append(
                    _build_dapo_math_17k_prompt(
                        batch_rows[local_index],
                        row_index=row_index,
                        prompt_column=selected_column,
                        chat_template_renderer=chat_template_renderer,
                        enable_thinking=enable_thinking,
                    )
                )
            else:
                prompts.append(
                    _normalize_prompt(
                        batch_rows[local_index],
                        row_index,
                        selected_column,
                        chat_template_renderer,
                        enable_thinking=enable_thinking,
                    )
                )

        current_row += len(batch_rows)
        remaining_skip = 0
        if len(prompts) >= batch_size:
            break

    if not prompts:
        raise ValueError(
            f"No prompts were loaded from {parquet_path} using column {prompt_column_label!r}."
        )
    return prompt_column_label, prompts, total_rows
