from __future__ import annotations

import argparse
import ast
import json
import os
import re
import socket
from dataclasses import dataclass
from typing import Any

import ray
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

import sglang as sgl
from sglang.srt.utils.network import is_valid_ipv6_address

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
        "CUDA_LAUNCH_BLOCKING",
        "SGLANG_DECOUPLED_SPEC_TRACE_DIR",
        "SGLANG_DECOUPLED_SPEC_SUMMARY_INTERVAL",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            env_vars[env_name] = env_value
    return env_vars


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


def reserve_tcp_port(
    preferred_port: int | None = None,
    avoid_ports: set[int] | None = None,
) -> tuple[int, socket.socket]:
    """Bind and hold a TCP port, returning both the port and lock socket."""
    avoid_ports = set(avoid_ports or ())

    def bind_port(port: int) -> socket.socket:
        """Bind a listening socket to all interfaces on a specific port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.listen(1)
        return sock

    if preferred_port is not None:
        if preferred_port in avoid_ports:
            raise RuntimeError(
                f"preferred port {preferred_port} conflicts with avoided ports "
                f"{sorted(avoid_ports)}"
            )
        return preferred_port, bind_port(preferred_port)

    for _ in range(256):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("0.0.0.0", 0))
        probe.listen(1)
        candidate_port = int(probe.getsockname()[1])
        probe.close()
        if candidate_port in avoid_ports:
            continue
        try:
            return candidate_port, bind_port(candidate_port)
        except OSError:
            continue

    raise RuntimeError("failed to reserve a TCP port")


def _get_alive_gpu_nodes() -> list[dict[str, Any]]:
    """Return alive Ray nodes with currently available accelerator capacity."""
    available_resources = ray._private.state.available_resources_per_node()
    node_infos = {node["NodeID"]: node for node in ray.nodes() if node.get("Alive")}

    candidates = []
    for node_id, node in node_infos.items():
        node_resources = available_resources.get(node_id, {})
        available_gpus = int(node_resources.get("GPU", node_resources.get("NPU", 0)))
        if available_gpus <= 0:
            continue
        candidates.append(
            {
                "node_id": node_id,
                "node_ip": node["NodeManagerAddress"],
                "available_gpus": available_gpus,
            }
        )

    candidates.sort(key=lambda item: (-item["available_gpus"], item["node_ip"]))
    return candidates


def plan_draft_placement(args: argparse.Namespace) -> list[str]:
    """Plan drafter actor node placement from currently available Ray GPUs."""
    if args.draft_tp_size <= 0:
        raise ValueError("draft-tp-size must be positive")
    if args.num_draft_replicas is None or args.num_draft_replicas <= 0:
        raise ValueError("num-draft-replicas must be positive")

    candidate_nodes = _get_alive_gpu_nodes()
    total_capacity = sum(
        node["available_gpus"] // args.draft_tp_size for node in candidate_nodes
    )
    if total_capacity < args.num_draft_replicas:
        raise ValueError(
            "Not enough free GPUs for drafters after reserving verifier resources: "
            f"need {args.num_draft_replicas} replicas with "
            f"tp_size={args.draft_tp_size}, "
            f"capacity is {total_capacity}"
        )

    remaining = args.num_draft_replicas
    node_assignments: list[str] = []
    for node in candidate_nodes:
        capacity = node["available_gpus"] // args.draft_tp_size
        take = min(capacity, remaining)
        node_assignments.extend([node["node_id"]] * take)
        remaining -= take
        if remaining == 0:
            break

    if remaining != 0:
        raise ValueError(
            f"Unable to place {args.num_draft_replicas} drafters with "
            f"tp_size={args.draft_tp_size}"
        )
    return node_assignments


@ray.remote
class PortActor:
    """Ray actor used to reserve bootstrap ports on a specific placement node."""

    def __init__(self):
        """Initialize without a reservation; callers reserve ports explicitly."""
        self._reserved_socket: socket.socket | None = None

    def reserve_port(
        self,
        preferred_port: int | None = None,
        avoid_ports: list[int] | None = None,
    ) -> dict[str, Any]:
        """Reserve a TCP port on this actor's node and report host and port."""
        self.release_port()
        port, sock = reserve_tcp_port(
            preferred_port,
            avoid_ports=set(avoid_ports or ()),
        )
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
    avoid_ports: set[int] | None = None,
    preferred_port: int | None = None,
) -> str:
    """Reserve a verifier result endpoint on target placement-group rank 0."""
    avoid_ports = set(avoid_ports or ())
    if avoid_port is not None:
        avoid_ports.add(avoid_port)
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
            reservation = ray.get(
                actor.reserve_port.remote(preferred_port, sorted(avoid_ports))
            )
            host = reservation["host"]
            port = int(reservation["port"])
            ray.get(actor.release_port.remote())
        finally:
            ray.kill(actor, no_restart=True)

        if port not in avoid_ports:
            return format_tcp_address(host, port)
        if preferred_port is not None:
            raise RuntimeError(
                f"preferred result endpoint port {preferred_port} conflicts with "
                f"avoid_ports {sorted(avoid_ports)}"
            )

    raise RuntimeError("failed to reserve a result endpoint port")


def create_endpoint_on_node(
    node_id: str,
    *,
    avoid_ports: set[int] | None = None,
    preferred_port: int | None = None,
) -> str:
    """Reserve an endpoint on the requested Ray node."""
    avoid_ports = set(avoid_ports or ())
    scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    for _ in range(16):
        actor = PortActor.options(
            num_cpus=0,
            scheduling_strategy=scheduling_strategy,
        ).remote()
        try:
            reservation = ray.get(
                actor.reserve_port.remote(preferred_port, sorted(avoid_ports))
            )
            host = reservation["host"]
            port = int(reservation["port"])
            ray.get(actor.release_port.remote())
        finally:
            ray.kill(actor, no_restart=True)

        if port not in avoid_ports:
            return format_tcp_address(host, port)
        if preferred_port is not None:
            raise RuntimeError(
                f"preferred endpoint port {preferred_port} conflicts with "
                f"avoid_ports {sorted(avoid_ports)}"
            )

    raise RuntimeError("failed to reserve an endpoint port")


@ray.remote
class DraftActor:
    """Ray actor that hosts a draft engine for Ray/multi-node benchmark runs."""

    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
        speculative_num_steps: int,
        bind_endpoint: str,
        connect_endpoints: list[str],
        rank: int,
        deterministic: bool = False,
        decoupled_spec_trace_dir: str | None = None,
    ):
        """Pin GPUs and create the draft engine with a preplanned endpoint."""
        self.assigned_gpu_ids = pin_actor_to_assigned_gpus(tp_size)
        self.engine = sgl.Engine(
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
            enable_deterministic_inference=deterministic,
            decoupled_spec_trace_dir=decoupled_spec_trace_dir,
        )
        self.control_endpoint = bind_endpoint

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
    node_assignments: list[str],
    endpoint_configs: list[DecoupledSpecEndpointConfig],
) -> list[Any]:
    """Launch draft actors at planned nodes with planned endpoint configs."""
    if len(node_assignments) != args.num_draft_replicas:
        raise ValueError(
            f"node_assignments has {len(node_assignments)} entries, expected "
            f"{args.num_draft_replicas}"
        )
    if len(endpoint_configs) != args.num_draft_replicas:
        raise ValueError(
            f"endpoint_configs has {len(endpoint_configs)} entries, expected "
            f"{args.num_draft_replicas}"
        )

    actors = []
    actor_env_vars = get_decoupled_spec_actor_env_vars(args)
    for node_id, endpoint_config in zip(
        node_assignments,
        endpoint_configs,
        strict=True,
    ):
        actor_options: dict[str, Any] = dict(
            num_gpus=args.draft_tp_size,
            num_cpus=1,
            max_concurrency=128,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        )
        if actor_env_vars:
            actor_options["runtime_env"] = {"env_vars": actor_env_vars}
        actor = DraftActor.options(**actor_options).remote(
            model_path=args.draft_model_path,
            tp_size=args.draft_tp_size,
            speculative_num_steps=args.num_speculative_steps,
            bind_endpoint=endpoint_config.bind_endpoint,
            connect_endpoints=endpoint_config.connect_endpoints,
            rank=endpoint_config.rank,
            deterministic=args.deterministic,
            decoupled_spec_trace_dir=args.decoupled_spec_trace_dir,
        )
        actors.append(actor)
    ray.get([actor.ready.remote() for actor in actors])
    return actors


def create_remote_decoupled_spec_topology(
    args: argparse.Namespace,
    verifier_pgs,
    *,
    avoid_ports: set[int] | None = None,
    preferred_result_ports: list[int | None] | None = None,
    preferred_control_ports: list[int | None] | None = None,
) -> DecoupledSpecTopology:
    """Create Ray/multi-node decoupled-spec endpoints and draft actors."""
    if not isinstance(verifier_pgs, list):
        verifier_pgs = [verifier_pgs]
    if not verifier_pgs:
        raise ValueError("at least one verifier placement group is required")
    if preferred_result_ports is not None and len(preferred_result_ports) != len(
        verifier_pgs
    ):
        raise ValueError(
            f"preferred_result_ports has {len(preferred_result_ports)} entries, "
            f"expected {len(verifier_pgs)}"
        )
    if (
        preferred_control_ports is not None
        and len(preferred_control_ports) != args.num_draft_replicas
    ):
        raise ValueError(
            f"preferred_control_ports has {len(preferred_control_ports)} entries, "
            f"expected {args.num_draft_replicas}"
        )

    used_ports = set(avoid_ports or ())
    node_assignments = plan_draft_placement(args)

    result_endpoints = []
    for verifier_rank, pg in enumerate(verifier_pgs):
        preferred_result_port = (
            preferred_result_ports[verifier_rank]
            if preferred_result_ports is not None
            else None
        )
        result_endpoint = create_result_endpoint_from_pg(
            pg,
            avoid_ports=used_ports,
            preferred_port=preferred_result_port,
        )
        result_endpoints.append(result_endpoint)
        try:
            used_ports.add(int(result_endpoint.rsplit(":", 1)[1]))
        except ValueError:
            pass

    control_endpoints = []
    for draft_rank, node_id in enumerate(node_assignments):
        preferred_control_port = (
            preferred_control_ports[draft_rank]
            if preferred_control_ports is not None
            else None
        )
        control_endpoint = create_endpoint_on_node(
            node_id,
            avoid_ports=used_ports,
            preferred_port=preferred_control_port,
        )
        control_endpoints.append(control_endpoint)
        try:
            used_ports.add(int(control_endpoint.rsplit(":", 1)[1]))
        except ValueError:
            pass

    drafter_configs = [
        DecoupledSpecEndpointConfig(
            bind_endpoint=endpoint,
            connect_endpoints=result_endpoints,
            rank=rank,
        )
        for rank, endpoint in enumerate(control_endpoints)
    ]
    verifier_configs = [
        DecoupledSpecEndpointConfig(
            bind_endpoint=endpoint,
            connect_endpoints=control_endpoints,
            rank=rank,
        )
        for rank, endpoint in enumerate(result_endpoints)
    ]
    draft_actors = launch_draft_actors(args, node_assignments, drafter_configs)
    return DecoupledSpecTopology(
        drafter_configs=drafter_configs,
        verifier_configs=verifier_configs,
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
    """Extract draft-token acceptance metrics from current SGLang meta_info."""
    verify_ct = meta_info.get("spec_verify_ct")
    accepted_tokens = meta_info.get("spec_accepted_drafts")
    draft_tokens = meta_info.get("spec_proposed_drafts")
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
