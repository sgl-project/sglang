#!/usr/bin/env python3
"""
Run decoupled speculative decoding for either an input prompt or a prompt dataset.

By default, this compares decoupled speculative decoding against normal decode.
Use `--skip-decode` to run decoupled speculation only and `--show-responses` to
print full response text. When `--output-dir` is set, JSON output records full
prompt and response text.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)

try:
    from . import common
except ImportError:
    import common

DEFAULT_RAY_NAMESPACE = "dspec"
DEFAULT_PROMPT_COLUMN_CANDIDATES = common.DEFAULT_PROMPT_COLUMN_CANDIDATES
_RUNTIME_IMPORTS_READY = False


def _ensure_runtime_imports() -> None:
    """Import Ray and decoupled-spec helpers after argparse handles --help."""
    global _RUNTIME_IMPORTS_READY
    global create_remote_decoupled_spec_topology
    global PlacementGroupSchedulingStrategy
    global PortActor
    global TargetActor
    global get_decoupled_spec_actor_env_vars
    global placement_group
    global ray
    global remove_placement_group

    if _RUNTIME_IMPORTS_READY:
        return

    import ray as ray_module
    from ray.util.placement_group import placement_group as ray_placement_group
    from ray.util.placement_group import (
        remove_placement_group as ray_remove_placement_group,
    )
    from ray.util.scheduling_strategies import (
        PlacementGroupSchedulingStrategy as RayPlacementGroupSchedulingStrategy,
    )

    ray = ray_module
    placement_group = ray_placement_group
    remove_placement_group = ray_remove_placement_group
    PlacementGroupSchedulingStrategy = RayPlacementGroupSchedulingStrategy
    create_remote_decoupled_spec_topology = common.create_remote_decoupled_spec_topology
    PortActor = common.PortActor
    TargetActor = common.TargetActor
    get_decoupled_spec_actor_env_vars = common.get_decoupled_spec_actor_env_vars
    _RUNTIME_IMPORTS_READY = True


PromptSample = common.PromptSample
ModeMetrics = common.ModeMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run decoupled speculation on one prompt or a parquet prompt batch, "
            "optionally comparing against normal decode."
        )
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Single prompt to generate from. When --batch-size is greater than 1, "
            "the prompt is repeated to fill the batch. Mutually exclusive with "
            "--dataset-path."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        "--parquet-path",
        dest="dataset_path",
        default=None,
        help="Path to the parquet dataset.",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help=(
            "Prompt column in the parquet file. If omitted, common names are "
            f"searched in order: {DEFAULT_PROMPT_COLUMN_CANDIDATES}."
        ),
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "codeforces_raw", "dapo_math_17k"],
        default="auto",
        help=(
            "How to interpret the parquet rows. "
            "'auto' reads one prompt-like column. "
            "'codeforces_raw' builds a prompt from Codeforces problem fields and, "
            "when enabled, renders it through the model tokenizer's chat template. "
            "'dapo_math_17k' reads the DAPO-Math-17k structured prompt messages "
            "and renders them through the target model chat template."
        ),
    )
    parser.add_argument(
        "--code-language",
        choices=["python", "py", "cpp", "c++"],
        default="python",
        help=(
            "Target language used when --dataset-format=codeforces_raw. "
            "Ignored for normal prompt-column datasets."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        "--bs",
        dest="batch_size",
        type=int,
        default=1,
        help=(
            "Number of valid prompts to run in one generate call. When using "
            "multiple verifier replicas, prompts are distributed as evenly as "
            "possible across verifier replicas."
        ),
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Disable tokenizer.apply_chat_template for chat-style prompt objects.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking-style generation when building chat prompts for "
            "models such as Qwen3/Qwen3.5. Disabled by default."
        ),
    )
    parser.add_argument(
        "--context-length",
        "--max-new-tokens",
        dest="context_length",
        type=int,
        required=True,
        help="Generation length. This is passed as max_new_tokens.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Optional prompt token upper bound. Prompts over this limit are skipped.",
    )
    parser.add_argument(
        "--target-model-path",
        required=True,
        help="Target/verifier model path.",
    )
    parser.add_argument(
        "--draft-model-path",
        required=True,
        help="Draft model path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path used for prompt length filtering. Defaults to target model.",
    )
    parser.add_argument("--target-tp-size", type=int, required=True)
    parser.add_argument(
        "--target-ep-size",
        type=int,
        default=None,
        help="Expert parallel size for the target/verifier engine.",
    )
    parser.add_argument(
        "--target-moe-a2a-backend",
        default=None,
        help="MoE A2A backend for the target/verifier engine, e.g. deepep.",
    )
    parser.add_argument(
        "--target-mamba-scheduler-strategy",
        "--mamba-scheduler-strategy",
        dest="target_mamba_scheduler_strategy",
        choices=["auto", "no_buffer", "extra_buffer"],
        default=None,
        help=(
            "Mamba scheduler strategy for the target/verifier engine. "
            "Decoupled verifier and drafter engines disable radix cache, so "
            "the default no_buffer strategy is normally sufficient."
        ),
    )
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--num-speculative-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic inference for both decoupled drafter and "
            "verifier engines."
        ),
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Set sampling_params.ignore_eos=True for both decoupled speculative "
            "decoding and normal decoding. Disabled by default."
        ),
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster address. Use 'auto' for an existing cluster or local fallback on nnodes=1.",
    )
    parser.add_argument("--ray-namespace", default=DEFAULT_RAY_NAMESPACE)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument(
        "--n-gpu-per-node",
        type=int,
        default=None,
        help=(
            "GPU count available on each Ray node. Used with --nnodes to bound "
            "the total verifier and drafter GPU budgets."
        ),
    )
    parser.add_argument(
        "--verify-ngpus",
        dest="verify_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for verifier replicas. If omitted, all GPUs "
            "not reserved by --draft-ngpus are used for verifier replicas."
        ),
    )
    parser.add_argument(
        "--draft-ngpus",
        dest="draft_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for all drafter replicas. The number of "
            "drafters is derived as draft_ngpus / draft_tp_size."
        ),
    )
    parser.add_argument(
        "--dist-init-addr",
        default=None,
        help=(
            "Optional SGLang distributed init address override. If omitted for "
            "multi-node runs, the script uses each verifier placement group's "
            "rank-0 host."
        ),
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=None,
        help=(
            "Base port for this run. With V verifier replicas, spec dist-init "
            "uses base..base+V-1, decode uses base+V..base+2V-1, verifier "
            "result endpoints use base+2V..base+3V-1, and drafter control "
            "endpoints start at base+3V."
        ),
    )
    parser.add_argument(
        "--num-draft-replicas",
        dest="num_draft_replicas",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory to write per-mode CSV/JSON outputs. "
            "A normal comparison run writes decoupled-spec.csv, "
            "decoupled-spec.json, decode.csv, and decode.json."
        ),
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Only run decoupled speculation and skip the normal decode baseline.",
    )
    parser.add_argument(
        "--show-responses",
        action="store_true",
        help=(
            "Print full response text in the terminal. When --output-dir is set, "
            "full prompt and response text is always included in per-mode JSON."
        ),
    )
    parser.add_argument(
        "--decoupled-spec-trace-dir",
        default=None,
        help="Directory for decoupled speculative decoding CSV trace files.",
    )
    return parser.parse_args()


def load_prompt_samples(
    args: argparse.Namespace,
) -> tuple[str, list[PromptSample], int]:
    return common.load_prompt_samples(args)


def _parse_host_port(addr: str) -> tuple[str, int | None]:
    if addr.count(":") == 1:
        host, raw_port = addr.rsplit(":", 1)
        if raw_port:
            return host, int(raw_port)
    return addr, None


def _normalize_layout_host(host: str) -> str:
    """Normalize host strings used only for layout display/grouping."""
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _host_from_endpoint(endpoint: str) -> str:
    """Extract the host from a tcp://host:port endpoint."""
    endpoint = endpoint.removeprefix("tcp://")
    if endpoint.startswith("["):
        end = endpoint.find("]")
        if end != -1:
            return _normalize_layout_host(endpoint[1:end])
    host, _ = _parse_host_port(endpoint)
    return _normalize_layout_host(host)


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def derive_dist_init_addr(
    args: argparse.Namespace,
    *,
    port_offset: int = 0,
) -> str | None:
    if args.nnodes == 1 and args.dist_init_addr is None:
        if args.dist_init_port is not None:
            return f"127.0.0.1:{args.dist_init_port + port_offset}"
        return f"127.0.0.1:{_pick_free_local_port()}"

    if args.dist_init_addr is None:
        raise ValueError("dist-init-addr is required when nnodes > 1")

    host, parsed_port = _parse_host_port(args.dist_init_addr)
    base_port = args.dist_init_port if args.dist_init_port is not None else parsed_port
    if base_port is None:
        raise ValueError(
            "dist-init-addr must include a port or dist-init-port must be set"
        )

    return f"{host}:{base_port + port_offset}"


def derive_dist_init_addr_from_pg(
    args: argparse.Namespace,
    pg,
    *,
    port_offset: int = 0,
) -> str | None:
    if args.dist_init_addr is not None or args.nnodes == 1:
        return derive_dist_init_addr(args, port_offset=port_offset)

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
    actor = PortActor.options(
        num_cpus=0,
        scheduling_strategy=scheduling_strategy,
    ).remote()
    try:
        preferred_port = (
            args.dist_init_port + port_offset
            if args.dist_init_port is not None
            else None
        )
        reservation = ray.get(actor.reserve_port.remote(preferred_port))
        host = reservation["host"]
        port = int(reservation["port"])
        ray.get(actor.release_port.remote())
    finally:
        ray.kill(actor, no_restart=True)

    return f"{host}:{port}"


def init_ray(address: str, namespace: str, nnodes: int) -> None:
    init_kwargs = dict(
        address=address,
        namespace=namespace,
        ignore_reinit_error=True,
        log_to_driver=True,
        logging_level=logging.ERROR,
    )
    try:
        ray.init(**init_kwargs)
    except Exception:
        if address != "auto" or nnodes != 1:
            raise
        ray.init(
            namespace=namespace,
            ignore_reinit_error=True,
            log_to_driver=True,
            logging_level=logging.ERROR,
        )


def derive_target_layout(args: argparse.Namespace) -> tuple[int, int]:
    for candidate_nnodes in range(1, args.nnodes + 1):
        if args.target_tp_size % candidate_nnodes != 0:
            continue
        target_gpus_per_node = args.target_tp_size // candidate_nnodes
        if target_gpus_per_node <= args.n_gpu_per_node:
            return candidate_nnodes, target_gpus_per_node

    raise ValueError(
        f"target-tp-size ({args.target_tp_size}) cannot be packed evenly across up to "
        f"{args.nnodes} nodes with {args.n_gpu_per_node} GPUs per node"
    )


def validate_resources(args: argparse.Namespace) -> tuple[int, int]:
    if args.nnodes <= 0:
        raise ValueError("nnodes must be positive")
    if args.target_tp_size <= 0:
        raise ValueError("target-tp-size must be positive")
    if args.target_ep_size is not None and args.target_ep_size <= 0:
        raise ValueError("target-ep-size must be positive when set")
    if args.draft_tp_size <= 0:
        raise ValueError("draft-tp-size must be positive")
    if args.verify_ngpus is not None and args.verify_ngpus <= 0:
        raise ValueError("verify-ngpus must be positive when set")
    if args.draft_ngpus is not None and args.draft_ngpus <= 0:
        raise ValueError("draft-ngpus must be positive when set")
    if args.num_draft_replicas is not None and args.num_draft_replicas <= 0:
        raise ValueError("num-draft-replicas must be positive when set")
    if args.draft_ngpus is None:
        args.draft_ngpus = args.draft_tp_size * (args.num_draft_replicas or 1)
    if args.draft_ngpus % args.draft_tp_size != 0:
        raise ValueError(
            f"draft-ngpus ({args.draft_ngpus}) must be divisible by "
            f"draft-tp-size ({args.draft_tp_size})"
        )
    derived_num_draft_replicas = args.draft_ngpus // args.draft_tp_size
    if (
        args.num_draft_replicas is not None
        and args.num_draft_replicas != derived_num_draft_replicas
    ):
        raise ValueError(
            "num-draft-replicas must match draft-ngpus / draft-tp-size "
            f"({derived_num_draft_replicas}) when --draft-ngpus is set"
        )
    args.num_draft_replicas = derived_num_draft_replicas
    if args.verify_ngpus is not None and args.verify_ngpus % args.target_tp_size != 0:
        raise ValueError(
            f"verify-ngpus ({args.verify_ngpus}) must be divisible by "
            f"target-tp-size ({args.target_tp_size})"
        )

    if args.n_gpu_per_node is None:
        if args.nnodes != 1:
            raise ValueError("n-gpu-per-node is required when nnodes > 1")
        args.n_gpu_per_node = (
            args.verify_ngpus or args.target_tp_size
        ) + args.draft_ngpus
    if args.n_gpu_per_node <= 0:
        raise ValueError("n-gpu-per-node must be positive")

    total_cluster_gpus = args.n_gpu_per_node * args.nnodes
    if args.verify_ngpus is None:
        args.verify_ngpus = total_cluster_gpus - args.draft_ngpus
    if args.draft_ngpus + args.verify_ngpus > total_cluster_gpus:
        raise ValueError(
            f"verify-ngpus + draft-ngpus ({args.verify_ngpus} + "
            f"{args.draft_ngpus}) exceeds nnodes*n-gpu-per-node "
            f"({total_cluster_gpus})"
        )
    if args.verify_ngpus <= 0:
        raise ValueError(
            f"draft-ngpus ({args.draft_ngpus}) must leave GPUs for at least "
            "one verifier replica"
        )
    if args.verify_ngpus % args.target_tp_size != 0:
        raise ValueError(
            f"verify-ngpus ({args.verify_ngpus}) must be divisible by "
            f"target-tp-size ({args.target_tp_size})"
        )
    args.num_verifier_replicas = args.verify_ngpus // args.target_tp_size

    target_nnodes, target_gpus_per_node = derive_target_layout(args)

    if args.draft_tp_size > args.n_gpu_per_node:
        raise ValueError(
            f"each draft actor needs {args.draft_tp_size} GPUs on one node, "
            f"but n-gpu-per-node is only {args.n_gpu_per_node}"
        )

    ray_gpus = int(ray.cluster_resources().get("GPU", 0))
    if ray_gpus and total_cluster_gpus > ray_gpus:
        raise ValueError(
            f"Ray cluster reports {ray_gpus} GPUs, but this run requires "
            f"{total_cluster_gpus}"
        )

    alive_target_nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive")
        and float(node.get("Resources", {}).get("GPU", 0)) >= target_gpus_per_node
    ]
    if len(alive_target_nodes) < target_nnodes:
        raise ValueError(
            f"Ray cluster has {len(alive_target_nodes)} alive GPU nodes with at "
            f"least {target_gpus_per_node} GPUs, but target needs {target_nnodes} nodes"
        )

    return target_nnodes, target_gpus_per_node


def create_target_placement_group(target_nnodes: int, target_gpus_per_node: int):
    bundles = [{"CPU": 1, "GPU": target_gpus_per_node} for _ in range(target_nnodes)]
    strategy = "PACK" if target_nnodes == 1 else "STRICT_SPREAD"
    pg = placement_group(bundles, strategy=strategy)
    ray.get(pg.ready())
    return pg


def create_target_placement_groups(
    num_replicas: int,
    target_nnodes: int,
    target_gpus_per_node: int,
):
    return [
        create_target_placement_group(target_nnodes, target_gpus_per_node)
        for _ in range(num_replicas)
    ]


def _get_pg_bundle_hosts(pg, num_bundles: int) -> list[str]:
    hosts = []
    for bundle_index in range(num_bundles):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_index,
        )
        actor = PortActor.options(
            num_cpus=0,
            scheduling_strategy=scheduling_strategy,
        ).remote()
        try:
            info = ray.get(actor.get_node_info.remote())
            hosts.append(info["host"])
        finally:
            ray.kill(actor, no_restart=True)
    return hosts


def print_decoupled_spec_layout(
    *,
    args: argparse.Namespace,
    target_nnodes: int,
    target_gpus_per_node: int,
    verifier_pgs: list[Any],
    topology: Any,
) -> None:
    node_hosts = sorted(
        {
            _normalize_layout_host(node["NodeManagerAddress"])
            for node in ray.nodes()
            if node.get("Alive") and node.get("NodeManagerAddress")
        }
    )
    node_layout: dict[str, list[str]] = {host: [] for host in node_hosts}

    for verifier_rank, pg in enumerate(verifier_pgs):
        bundle_hosts = _get_pg_bundle_hosts(pg, target_nnodes)
        for bundle_index, raw_host in enumerate(bundle_hosts):
            host = _normalize_layout_host(raw_host)
            node_layout.setdefault(host, [])
            label = f"verifier{verifier_rank}(tp={args.target_tp_size}"
            if target_nnodes > 1:
                label += f", bundle={bundle_index}"
            label += f", gpus={target_gpus_per_node})"
            node_layout[host].append(label)

    for drafter_rank, config in enumerate(topology.drafter_configs):
        host = _host_from_endpoint(config.bind_endpoint)
        node_layout.setdefault(host, [])
        node_layout[host].append(f"drafter{drafter_rank}(tp={args.draft_tp_size})")

    print("=== decoupled_spec_layout ===")
    print(f"nnodes: {args.nnodes}")
    print(
        f"nverifier={args.num_verifier_replicas}, "
        f"tp_size={args.target_tp_size}, "
        f"target_nnodes={target_nnodes}, "
        f"target_gpus_per_node={target_gpus_per_node}"
    )
    print(f"ndrafter={args.num_draft_replicas}, tp_size={args.draft_tp_size}")
    for node_index, host in enumerate(sorted(node_layout), start=1):
        items = ", ".join(node_layout[host]) if node_layout[host] else "idle"
        print(f"node{node_index} ({host}): {items}")


def launch_target_actors(
    *,
    args: argparse.Namespace,
    mode: str,
    dist_init_addr: str | None,
    target_nnodes: int,
    target_gpus_per_node: int,
    pg,
    bind_endpoint: str | None = None,
    connect_endpoints: list[str] | None = None,
    rank: int | None = None,
) -> list[Any]:
    actor_env_vars = get_decoupled_spec_actor_env_vars()
    actors = []
    for node_rank in range(target_nnodes):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=node_rank,
        )
        actor_options: dict[str, Any] = dict(
            num_gpus=target_gpus_per_node,
            num_cpus=1,
            scheduling_strategy=scheduling_strategy,
        )
        if actor_env_vars:
            actor_options["runtime_env"] = {"env_vars": actor_env_vars}
        actor = TargetActor.options(**actor_options).remote(
            mode=mode,
            model_path=args.target_model_path,
            tp_size=args.target_tp_size,
            ep_size=args.target_ep_size,
            moe_a2a_backend=args.target_moe_a2a_backend,
            mamba_scheduler_strategy=args.target_mamba_scheduler_strategy,
            nnodes=target_nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            speculative_num_steps=args.num_speculative_steps,
            bind_endpoint=bind_endpoint,
            connect_endpoints=connect_endpoints,
            rank=rank,
            deterministic=args.deterministic,
            decoupled_spec_trace_dir=args.decoupled_spec_trace_dir,
        )
        actors.append(actor)

    ray.get([actor.ready.remote() for actor in actors])
    return actors


def shutdown_actors(actors: list[Any]) -> None:
    if not actors:
        return
    try:
        ray.get([actor.shutdown.remote() for actor in actors], timeout=60)
    except Exception as exc:
        logger.warning("actor shutdown failed: %s", exc)
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass

collect_mode_metrics = common.collect_mode_metrics


def _split_indices(num_items: int, num_shards: int) -> list[list[int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    base_shard_size, remainder = divmod(num_items, num_shards)
    shards: list[list[int]] = []
    start = 0
    for shard_index in range(num_shards):
        shard_size = base_shard_size + (1 if shard_index < remainder else 0)
        end = start + shard_size
        shards.append(list(range(start, end)))
        start = end
    return shards


def run_mode(
    *,
    args: argparse.Namespace,
    mode: str,
    prompt_input_ids: list[list[int]],
    sampling_params: dict[str, Any],
    prompt_samples: list[PromptSample],
    dist_init_addrs: list[str | None],
    target_nnodes: int,
    target_gpus_per_node: int,
    pgs: list[Any] | None = None,
    endpoint_configs: list[Any] | None = None,
    include_output_text: bool = True,
) -> ModeMetrics:
    target_actor_groups: list[list[Any]] = []
    owns_pgs = pgs is None
    num_replicas = (
        len(endpoint_configs) if endpoint_configs is not None else len(dist_init_addrs)
    )
    if num_replicas <= 0:
        raise ValueError("run_mode requires at least one target replica")
    if len(dist_init_addrs) != num_replicas:
        raise ValueError(
            f"dist_init_addrs has {len(dist_init_addrs)} entries, expected {num_replicas}"
        )
    if pgs is not None and len(pgs) != num_replicas:
        raise ValueError(f"pgs has {len(pgs)} entries, expected {num_replicas}")

    replica_indices = _split_indices(len(prompt_samples), num_replicas)
    verifier_assignments = [0] * len(prompt_samples)
    for replica_index, indices in enumerate(replica_indices):
        for index in indices:
            verifier_assignments[index] = replica_index
    outputs_by_index: list[dict[str, Any] | None] = [None] * len(prompt_samples)
    try:
        if pgs is None:
            pgs = create_target_placement_groups(
                num_replicas,
                target_nnodes,
                target_gpus_per_node,
            )

        for replica_index in range(num_replicas):
            endpoint_config = (
                endpoint_configs[replica_index]
                if endpoint_configs is not None
                else None
            )
            actors = launch_target_actors(
                args=args,
                mode=mode,
                dist_init_addr=dist_init_addrs[replica_index],
                target_nnodes=target_nnodes,
                target_gpus_per_node=target_gpus_per_node,
                pg=pgs[replica_index],
                bind_endpoint=(
                    endpoint_config.bind_endpoint
                    if endpoint_config is not None
                    else None
                ),
                connect_endpoints=(
                    endpoint_config.connect_endpoints
                    if endpoint_config is not None
                    else None
                ),
                rank=endpoint_config.rank if endpoint_config is not None else None,
            )
            target_actor_groups.append(actors)

        result_refs = []
        for replica_index, indices in enumerate(replica_indices):
            if not indices:
                continue
            shard_input_ids = [prompt_input_ids[index] for index in indices]
            result_refs.append(
                (
                    indices,
                    target_actor_groups[replica_index][0].generate_batch.remote(
                        shard_input_ids,
                        sampling_params,
                    ),
                )
            )

        for indices, result_ref in result_refs:
            result = ray.get(result_ref)
            shard_outputs = result["outputs"]
            if len(shard_outputs) != len(indices):
                raise RuntimeError(
                    f"{mode} returned {len(shard_outputs)} outputs for "
                    f"{len(indices)} prompts on one replica"
                )
            for index, output in zip(indices, shard_outputs, strict=True):
                outputs_by_index[index] = output
    finally:
        for actors in target_actor_groups:
            shutdown_actors(actors)
        if owns_pgs and pgs is not None:
            for pg in pgs:
                remove_placement_group(pg)

    if any(output is None for output in outputs_by_index):
        missing = [
            index for index, output in enumerate(outputs_by_index) if output is None
        ]
        raise RuntimeError(f"{mode} did not return outputs for indices {missing}")

    return collect_mode_metrics(
        mode=mode,
        outputs=[output for output in outputs_by_index if output is not None],
        prompt_samples=prompt_samples,
        verifier_assignments=verifier_assignments,
        include_output_text=include_output_text,
    )

build_result = common.build_result
write_output_files = common.write_output_files
print_summary = common.print_summary


def main() -> None:
    args = parse_args()
    _ensure_runtime_imports()

    prompt_column, prompt_samples, total_rows = load_prompt_samples(args)
    prompt_input_ids = [list(sample.prompt_input_ids) for sample in prompt_samples]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.context_length,
        "ignore_eos": args.ignore_eos,
    }

    draft_actors: list[Any] = []
    spec_pgs = []
    try:
        init_ray(args.ray_address, args.ray_namespace, args.nnodes)
        target_nnodes, target_gpus_per_node = validate_resources(args)

        spec_pgs = create_target_placement_groups(
            args.num_verifier_replicas,
            target_nnodes,
            target_gpus_per_node,
        )
        spec_dist_init_addrs = [
            derive_dist_init_addr_from_pg(args, pg, port_offset=replica_index)
            for replica_index, pg in enumerate(spec_pgs)
        ]
        spec_dist_init_ports = {
            port
            for addr in spec_dist_init_addrs
            if addr is not None
            for _, port in [_parse_host_port(addr)]
            if port is not None
        }
        num_verifiers = args.num_verifier_replicas
        reserved_dist_init_ports = set(spec_dist_init_ports)
        if not args.skip_decode:
            if args.dist_init_port is not None:
                reserved_dist_init_ports.update(
                    args.dist_init_port + num_verifiers + replica_index
                    for replica_index in range(num_verifiers)
                )
            elif args.dist_init_addr is not None:
                _, base_port = _parse_host_port(args.dist_init_addr)
                if base_port is not None:
                    reserved_dist_init_ports.update(
                        base_port + num_verifiers + replica_index
                        for replica_index in range(num_verifiers)
                    )
        preferred_result_ports = (
            [args.dist_init_port + 2 * num_verifiers + i for i in range(num_verifiers)]
            if args.dist_init_port is not None
            else None
        )
        preferred_control_ports = (
            [
                args.dist_init_port + 3 * num_verifiers + i
                for i in range(args.num_draft_replicas)
            ]
            if args.dist_init_port is not None
            else None
        )
        topology = create_remote_decoupled_spec_topology(
            args,
            spec_pgs,
            avoid_ports=reserved_dist_init_ports,
            preferred_result_ports=preferred_result_ports,
            preferred_control_ports=preferred_control_ports,
        )
        print_decoupled_spec_layout(
            args=args,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            verifier_pgs=spec_pgs,
            topology=topology,
        )
        draft_actors = topology.draft_actors or []
        spec_metrics = run_mode(
            args=args,
            mode="decoupled_spec",
            prompt_input_ids=prompt_input_ids,
            sampling_params=sampling_params,
            prompt_samples=prompt_samples,
            dist_init_addrs=spec_dist_init_addrs,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            pgs=spec_pgs,
            endpoint_configs=topology.verifier_configs,
            include_output_text=True,
        )
        shutdown_actors(draft_actors)
        draft_actors = []

        decode_metrics = None
        if not args.skip_decode:
            decode_dist_init_addrs = [
                derive_dist_init_addr_from_pg(
                    args,
                    pg,
                    port_offset=args.num_verifier_replicas + replica_index,
                )
                for replica_index, pg in enumerate(spec_pgs)
            ]
            decode_metrics = run_mode(
                args=args,
                mode="decode",
                prompt_input_ids=prompt_input_ids,
                sampling_params=sampling_params,
                prompt_samples=prompt_samples,
                dist_init_addrs=decode_dist_init_addrs,
                target_nnodes=target_nnodes,
                target_gpus_per_node=target_gpus_per_node,
                pgs=spec_pgs,
                include_output_text=True,
            )

        result = build_result(
            args=args,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            prompt_column=prompt_column,
            total_rows=total_rows,
            prompt_samples=prompt_samples,
            spec_metrics=spec_metrics,
            decode_metrics=decode_metrics,
        )
        print_summary(result)
        if args.output_dir:
            print("output_files:")
            for output_path in write_output_files(result, args.output_dir):
                print(f"  {output_path}")
    finally:
        shutdown_actors(draft_actors)
        for pg in spec_pgs:
            remove_placement_group(pg)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
