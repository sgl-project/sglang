#!/usr/bin/env python3
"""
Run decoupled speculative decoding on a single node without Ray.

This keeps the same user-facing workload arguments as
multi-node.py, but removes Ray/multi-node launch. The drafter
engines run in local child processes and the verifier engine runs in this
process, which makes it convenient to launch under nsys.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from . import common
except ImportError:
    import common

LOCAL_HOST = "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run decoupled speculation on a single node without Ray, "
            "optionally comparing against a decode or MTP baseline."
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
            f"searched in order: {common.DEFAULT_PROMPT_COLUMN_CANDIDATES}."
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
            "Number of valid prompts to run in one generate call. Single-node "
            "mode supports one verifier replica."
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
        "--n-gpu-per-node",
        type=int,
        default=None,
        help=(
            "GPU count available on this node. If omitted, it is derived from "
            "--verify-ngpus and --draft-ngpus."
        ),
    )
    parser.add_argument(
        "--verify-ngpus",
        dest="verify_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for the verifier. Single-node mode supports "
            "one verifier replica, so this must equal --target-tp-size when set."
        ),
    )
    parser.add_argument(
        "--draft-ngpus",
        dest="draft_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for all drafters. The number of drafters is "
            "derived as draft_ngpus / draft_tp_size."
        ),
    )
    parser.add_argument(
        "--dist-init-addr",
        default=None,
        help=(
            "Optional verifier distributed init address override. If omitted, "
            "the script derives one from --dist-init-port or a free local port."
        ),
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=None,
        help=(
            "Base port for this run. Spec dist-init uses base, baseline uses "
            "base+1, verifier result endpoint uses base+2, and drafter control "
            "endpoints start at base+3."
        ),
    )
    parser.add_argument(
        "--reserved-ports",
        default=None,
        help=(
            "Comma- or whitespace-separated list of pre-reserved local ports. "
            "When set, ports do not need to be contiguous. They are consumed in "
            "order: spec dist-init, optional baseline dist-init, verifier result, "
            "then drafter control endpoints. This cannot be combined with "
            "--dist-init-addr or --dist-init-port."
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
            "A baseline comparison run writes decoupled-spec.csv/json plus "
            "<baseline>.csv/json."
        ),
    )
    parser.add_argument(
        "--baseline",
        choices=["decode", "mtp", "none"],
        default="decode",
        help=(
            "Baseline to run after decoupled speculation. 'mtp' uses SGLang's "
            "builtin colocated, serial draft-verify MTP/EAGLE path."
        ),
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
        "--spec-trace-dir",
        default=None,
        help="Directory for speculative decoding CSV trace files.",
    )
    parser.add_argument(
        "--draft-ready-timeout-s",
        type=float,
        default=900.0,
        help="Seconds to wait for all local draft engines to finish loading.",
    )
    return parser.parse_args()


def _visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    return [str(index) for index in range(256)]


def get_decoupled_spec_actor_env_vars() -> dict[str, str]:
    env_vars: dict[str, str] = {
        "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL": os.environ.get(
            "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL", "1"
        )
    }
    for env_name in (
        "CUDA_LAUNCH_BLOCKING",
        "SGLANG_DECOUPLED_SPEC_TRACE_DIR",
        "SGLANG_DECOUPLED_SPEC_SUMMARY_INTERVAL",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            env_vars[env_name] = env_value
    return env_vars


def _child_env_for_gpus(gpus: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    for name, value in get_decoupled_spec_actor_env_vars().items():
        env[name] = value
    return env


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _pick_free_port_block(
    num_ports: int,
    *,
    avoid_ports: set[int] | None = None,
) -> int:
    avoid_ports = avoid_ports or set()
    for _ in range(256):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((LOCAL_HOST, 0))
            base_port = int(sock.getsockname()[1])
        candidate_ports = {base_port + offset for offset in range(num_ports)}
        if candidate_ports & avoid_ports:
            continue
        if all(_port_available(port) for port in candidate_ports):
            return base_port
    raise RuntimeError(f"failed to find a free local block of {num_ports} ports")


def _port_from_dist_init_addr(addr: str | None) -> int | None:
    if addr is None:
        return None
    try:
        return int(addr.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def _format_tcp_endpoint(port: int) -> str:
    return f"tcp://{LOCAL_HOST}:{port}"


def _parse_reserved_ports(raw_ports: str | None) -> list[int]:
    if raw_ports is None:
        return []
    ports: list[int] = []
    for raw_port in raw_ports.replace(",", " ").split():
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError(f"invalid reserved port: {raw_port!r}") from exc
        if port <= 0 or port > 65535:
            raise ValueError(f"reserved port out of range: {port}")
        ports.append(port)
    if len(set(ports)) != len(ports):
        raise ValueError(f"reserved ports must be unique: {ports}")
    return ports


def _allocate_local_ports(
    args: argparse.Namespace,
    *,
    num_verifiers: int,
    num_drafters: int,
) -> tuple[str, str | None, str, list[str], list[int] | None]:
    reserved_ports = _parse_reserved_ports(args.reserved_ports)
    if reserved_ports:
        if args.dist_init_addr is not None or args.dist_init_port is not None:
            raise ValueError(
                "--reserved-ports cannot be combined with --dist-init-addr or "
                "--dist-init-port"
            )
        required_ports = 1 + (1 if args.baseline != "none" else 0) + 1 + num_drafters
        if len(reserved_ports) < required_ports:
            raise ValueError(
                f"--reserved-ports provides {len(reserved_ports)} ports, but this "
                f"run needs {required_ports}: spec dist-init, "
                f"{'baseline dist-init, ' if args.baseline != 'none' else ''}"
                "verifier result, and drafter control endpoints"
            )
        usable_ports = [port for port in reserved_ports if _port_available(port)]
        skipped_ports = [port for port in reserved_ports if port not in usable_ports]
        if len(usable_ports) < required_ports:
            raise ValueError(
                f"--reserved-ports has only {len(usable_ports)} currently "
                f"available ports after skipping in-use ports {skipped_ports}, "
                f"but this run needs {required_ports}"
            )
        if skipped_ports:
            print(
                f"skipping_in_use_reserved_ports: {skipped_ports}",
                flush=True,
            )
        port_iter = iter(usable_ports)
        spec_dist_init_addr = f"{LOCAL_HOST}:{next(port_iter)}"
        baseline_dist_init_addr = (
            f"{LOCAL_HOST}:{next(port_iter)}" if args.baseline != "none" else None
        )
        result_endpoint = _format_tcp_endpoint(next(port_iter))
        control_endpoints = [
            _format_tcp_endpoint(next(port_iter)) for _ in range(num_drafters)
        ]
        return (
            spec_dist_init_addr,
            baseline_dist_init_addr,
            result_endpoint,
            control_endpoints,
            reserved_ports,
        )

    explicit_dist_init_port = _port_from_dist_init_addr(args.dist_init_addr)
    base_port = args.dist_init_port or _pick_free_port_block(
        3 + num_drafters,
        avoid_ports=(
            {explicit_dist_init_port} if explicit_dist_init_port is not None else None
        ),
    )
    spec_dist_init_addr = args.dist_init_addr or f"{LOCAL_HOST}:{base_port}"
    baseline_dist_init_addr = f"{LOCAL_HOST}:{base_port + num_verifiers}"
    result_endpoint = _format_tcp_endpoint(base_port + 2 * num_verifiers)
    control_endpoints = [
        _format_tcp_endpoint(base_port + 3 * num_verifiers + index)
        for index in range(num_drafters)
    ]
    return (
        spec_dist_init_addr,
        baseline_dist_init_addr,
        result_endpoint,
        control_endpoints,
        None,
    )


def _uses_bind_connect_endpoint_args() -> bool:
    try:
        from sglang.srt.server_args import ServerArgs
    except Exception:
        return True
    return hasattr(ServerArgs, "decoupled_spec_bind_endpoint")


def _add_decoupled_endpoint_kwargs(
    engine_kwargs: dict[str, Any],
    *,
    bind_endpoint: str,
    connect_endpoints: list[str],
    control_endpoints: list[str],
    result_endpoints: list[str],
    rank: int,
) -> None:
    if _uses_bind_connect_endpoint_args():
        engine_kwargs.update(
            decoupled_spec_bind_endpoint=bind_endpoint,
            decoupled_spec_connect_endpoints=connect_endpoints,
            decoupled_spec_rank=rank,
        )
    else:
        engine_kwargs.update(
            decoupled_spec_control_endpoints=control_endpoints,
            decoupled_spec_result_endpoints=result_endpoints,
        )


def validate_single_node_args(args: argparse.Namespace) -> None:
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

    if args.verify_ngpus is None:
        args.verify_ngpus = args.target_tp_size
    if args.verify_ngpus != args.target_tp_size:
        raise ValueError(
            "single-node.py currently supports one verifier replica, so "
            "verify-ngpus must equal target-tp-size"
        )

    if args.n_gpu_per_node is None:
        args.n_gpu_per_node = args.verify_ngpus + args.draft_ngpus
    if args.n_gpu_per_node <= 0:
        raise ValueError("n-gpu-per-node must be positive")
    if args.verify_ngpus + args.draft_ngpus > args.n_gpu_per_node:
        raise ValueError(
            f"verify-ngpus + draft-ngpus ({args.verify_ngpus} + "
            f"{args.draft_ngpus}) exceeds n-gpu-per-node ({args.n_gpu_per_node})"
        )

    visible_gpus = _visible_gpu_ids()
    if len(visible_gpus) < args.n_gpu_per_node:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES exposes {len(visible_gpus)} GPUs, but this run "
            f"requires {args.n_gpu_per_node}"
        )

    args.num_verifier_replicas = 1
    args.nnodes = 1
    args.ray_address = None
    args.ray_namespace = None
    args.target_gpus = visible_gpus[: args.verify_ngpus]
    draft_start = args.verify_ngpus
    draft_end = draft_start + args.draft_ngpus
    args.draft_gpus = visible_gpus[draft_start:draft_end]


def run_draft_engine_process(
    *,
    rank: int,
    gpu_ids: list[str],
    model_path: str,
    tp_size: int,
    speculative_num_steps: int,
    bind_endpoint: str,
    connect_endpoints: list[str],
    deterministic: bool,
    spec_trace_dir: str | None,
    ready_queue,
    stop_reader,
) -> None:
    os.environ.update(_child_env_for_gpus(gpu_ids))
    try:
        import sglang as sgl

        engine_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            tp_size=tp_size,
            speculative_algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_steps + 1,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_deterministic_inference=deterministic,
            spec_trace_dir=spec_trace_dir,
        )
        _add_decoupled_endpoint_kwargs(
            engine_kwargs,
            bind_endpoint=bind_endpoint,
            connect_endpoints=connect_endpoints,
            control_endpoints=[bind_endpoint],
            result_endpoints=connect_endpoints,
            rank=rank,
        )
        engine = sgl.Engine(**engine_kwargs)
        ready_queue.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "gpus": gpu_ids,
                "bind_endpoint": bind_endpoint,
            }
        )
        try:
            stop_reader.recv()
        except EOFError:
            pass
    except Exception:
        ready_queue.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "gpus": gpu_ids,
                "error": traceback.format_exc(),
            }
        )
        raise
    finally:
        try:
            stop_reader.close()
        except Exception:
            pass
        if "engine" in locals():
            engine.shutdown()


def wait_for_drafts(
    processes: list[mp.Process],
    ready_queue,
    timeout_s: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    pending = {index for index in range(len(processes))}
    ready: list[dict[str, Any]] = []
    while pending:
        for index in list(pending):
            process = processes[index]
            if not process.is_alive() and process.exitcode is not None:
                raise RuntimeError(
                    f"draft process index={index} exited early with "
                    f"exitcode={process.exitcode}"
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for draft engines: {pending}")
        try:
            message = ready_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        rank = int(message.get("rank", -1))
        if "error" in message:
            raise RuntimeError(
                f"draft rank {rank} failed to start:\n{message['error']}"
            )
        if rank in pending:
            pending.remove(rank)
            ready.append(message)
            print(
                "draft_ready: "
                f"rank={rank} pid={message['pid']} "
                f"gpus={','.join(message['gpus'])} "
                f"endpoint={message['bind_endpoint']}",
                flush=True,
            )
    ready.sort(key=lambda item: int(item["rank"]))
    return ready


def start_draft_engines(
    args: argparse.Namespace,
    control_endpoints: list[str],
    result_endpoints: list[str],
) -> tuple[list[mp.Process], list[Any]]:
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    stop_senders = []
    processes: list[mp.Process] = []
    try:
        for rank, endpoint in enumerate(control_endpoints):
            start = rank * args.draft_tp_size
            end = start + args.draft_tp_size
            gpu_ids = args.draft_gpus[start:end]
            stop_reader, stop_sender = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=run_draft_engine_process,
                kwargs=dict(
                    rank=rank,
                    gpu_ids=gpu_ids,
                    model_path=args.draft_model_path,
                    tp_size=args.draft_tp_size,
                    speculative_num_steps=args.num_speculative_steps,
                    bind_endpoint=endpoint,
                    connect_endpoints=result_endpoints,
                    deterministic=args.deterministic,
                    spec_trace_dir=args.spec_trace_dir,
                    ready_queue=ready_queue,
                    stop_reader=stop_reader,
                ),
                name=f"decoupled-draft-rank-{rank}",
            )
            process.start()
            stop_reader.close()
            stop_senders.append(stop_sender)
            processes.append(process)
        wait_for_drafts(processes, ready_queue, args.draft_ready_timeout_s)
        return processes, stop_senders
    except Exception:
        shutdown_draft_engines(processes, stop_senders)
        raise


def shutdown_draft_engines(processes: list[mp.Process], stop_senders) -> None:
    if not processes:
        return
    print("stopping_draft_engines...", flush=True)
    for stop_sender in stop_senders or []:
        try:
            stop_sender.send("stop")
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            try:
                stop_sender.close()
            except Exception:
                pass
    for process in processes:
        process.join(timeout=60)
    for process in processes:
        if process.is_alive():
            print(
                f"terminating_draft_engine: name={process.name} pid={process.pid}",
                flush=True,
            )
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            print(
                f"killing_draft_engine: name={process.name} pid={process.pid}",
                flush=True,
            )
            process.kill()
    for process in processes:
        process.join(timeout=10)


def create_verifier_engine(
    args: argparse.Namespace,
    *,
    result_endpoint: str,
    control_endpoints: list[str],
    dist_init_addr: str,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.target_gpus)
    for name, value in get_decoupled_spec_actor_env_vars().items():
        os.environ[name] = value

    import sglang as sgl

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.target_model_path,
        tp_size=args.target_tp_size,
        dist_init_addr=dist_init_addr,
        speculative_algorithm="DECOUPLED_VERIFY",
        speculative_num_steps=args.num_speculative_steps,
        speculative_num_draft_tokens=args.num_speculative_steps + 1,
        disable_radix_cache=True,
        enable_deterministic_inference=args.deterministic,
        spec_trace_dir=args.spec_trace_dir,
        log_level="info",
    )
    _add_decoupled_endpoint_kwargs(
        engine_kwargs,
        bind_endpoint=result_endpoint,
        connect_endpoints=control_endpoints,
        control_endpoints=control_endpoints,
        result_endpoints=[result_endpoint],
        rank=0,
    )
    if args.target_ep_size is not None:
        engine_kwargs["ep_size"] = args.target_ep_size
    if args.target_moe_a2a_backend is not None:
        engine_kwargs["moe_a2a_backend"] = args.target_moe_a2a_backend
    return sgl.Engine(**engine_kwargs)


def create_decode_engine(args: argparse.Namespace, *, dist_init_addr: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.target_gpus)

    import sglang as sgl

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.target_model_path,
        tp_size=args.target_tp_size,
        dist_init_addr=dist_init_addr,
        enable_deterministic_inference=args.deterministic,
        disable_overlap_schedule=True,
        spec_trace_dir=args.spec_trace_dir,
        log_level="info",
    )
    if args.target_ep_size is not None:
        engine_kwargs["ep_size"] = args.target_ep_size
    if args.target_moe_a2a_backend is not None:
        engine_kwargs["moe_a2a_backend"] = args.target_moe_a2a_backend
    return sgl.Engine(**engine_kwargs)


def create_mtp_engine(args: argparse.Namespace, *, dist_init_addr: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.target_gpus)

    import sglang as sgl

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.target_model_path,
        tp_size=args.target_tp_size,
        dist_init_addr=dist_init_addr,
        speculative_algorithm="EAGLE",
        speculative_num_steps=args.num_speculative_steps,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=args.num_speculative_steps + 1,
        enable_deterministic_inference=args.deterministic,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        mamba_scheduler_strategy="no_buffer",
        spec_trace_dir=args.spec_trace_dir,
        log_level="info",
    )
    if args.target_ep_size is not None:
        engine_kwargs["ep_size"] = args.target_ep_size
    if args.target_moe_a2a_backend is not None:
        engine_kwargs["moe_a2a_backend"] = args.target_moe_a2a_backend
    return sgl.Engine(**engine_kwargs)


def run_engine_generate(
    engine,
    *,
    input_ids: list[list[int]],
    sampling_params: dict[str, Any],
) -> list[dict[str, Any]]:
    outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    if not isinstance(outputs, list):
        outputs = [outputs]
    return outputs


def _collect_mode_metrics(
    *,
    mode: str,
    outputs: list[dict[str, Any]],
    prompt_samples,
):
    return common.collect_mode_metrics(
        mode=mode,
        outputs=outputs,
        prompt_samples=prompt_samples,
        verifier_assignments=[0] * len(prompt_samples),
        include_output_text=True,
    )


def _write_summary_json(result: dict[str, Any], output_dir: str) -> Path:
    summary_path = Path(output_dir).expanduser() / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    args = parse_args()
    validate_single_node_args(args)

    prompt_column, prompt_samples, total_rows = common.load_prompt_samples(args)
    prompt_input_ids = [list(sample.prompt_input_ids) for sample in prompt_samples]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.context_length,
        "ignore_eos": args.ignore_eos,
    }

    num_verifiers = args.num_verifier_replicas
    num_drafters = args.num_draft_replicas
    (
        spec_dist_init_addr,
        baseline_dist_init_addr,
        result_endpoint,
        control_endpoints,
        reserved_ports,
    ) = _allocate_local_ports(
        args,
        num_verifiers=num_verifiers,
        num_drafters=num_drafters,
    )

    print("local_decoupled_spec_topology:", flush=True)
    print(f"  target_gpus: {args.target_gpus}", flush=True)
    print(f"  draft_gpus: {args.draft_gpus}", flush=True)
    if reserved_ports is not None:
        print(f"  reserved_ports: {reserved_ports}", flush=True)
    print(f"  verifier_result_endpoint: {result_endpoint}", flush=True)
    print(f"  draft_control_endpoints: {control_endpoints}", flush=True)
    print(f"  verifier_dist_init_addr: {spec_dist_init_addr}", flush=True)

    draft_processes: list[mp.Process] = []
    draft_stop_senders = None
    verifier_engine = None
    baseline_engine = None
    try:
        draft_processes, draft_stop_senders = start_draft_engines(
            args,
            control_endpoints=control_endpoints,
            result_endpoints=[result_endpoint],
        )
        print("creating_verifier_engine...", flush=True)
        verifier_engine = create_verifier_engine(
            args,
            result_endpoint=result_endpoint,
            control_endpoints=control_endpoints,
            dist_init_addr=spec_dist_init_addr,
        )
        print("running_decoupled_spec_generate...", flush=True)
        spec_outputs = run_engine_generate(
            verifier_engine,
            input_ids=prompt_input_ids,
            sampling_params=sampling_params,
        )
        print("decoupled_spec_generate_done", flush=True)
        spec_metrics = _collect_mode_metrics(
            mode="decoupled_spec",
            outputs=spec_outputs,
            prompt_samples=prompt_samples,
        )
        print("shutting_down_verifier_engine...", flush=True)
        verifier_engine.shutdown()
        verifier_engine = None
        shutdown_draft_engines(draft_processes, draft_stop_senders)
        draft_processes = []
        draft_stop_senders = None

        baseline_metrics = None
        if args.baseline != "none":
            print(f"creating_{args.baseline}_engine...", flush=True)
            assert baseline_dist_init_addr is not None
            if args.baseline == "decode":
                baseline_engine = create_decode_engine(
                    args,
                    dist_init_addr=baseline_dist_init_addr,
                )
            elif args.baseline == "mtp":
                baseline_engine = create_mtp_engine(
                    args,
                    dist_init_addr=baseline_dist_init_addr,
                )
            else:
                raise ValueError(f"Unsupported baseline: {args.baseline}")
            print(f"running_{args.baseline}_generate...", flush=True)
            baseline_outputs = run_engine_generate(
                baseline_engine,
                input_ids=prompt_input_ids,
                sampling_params=sampling_params,
            )
            print(f"{args.baseline}_generate_done", flush=True)
            baseline_metrics = _collect_mode_metrics(
                mode=args.baseline,
                outputs=baseline_outputs,
                prompt_samples=prompt_samples,
            )

        result = common.build_result(
            args=args,
            target_nnodes=1,
            target_gpus_per_node=args.target_tp_size,
            prompt_column=prompt_column,
            total_rows=total_rows,
            prompt_samples=prompt_samples,
            spec_metrics=spec_metrics,
            baseline_metrics=baseline_metrics,
        )
        result["config"]["runner"] = "single_node_no_ray"
        result["config"]["target_gpus"] = args.target_gpus
        result["config"]["draft_gpus"] = args.draft_gpus
        result["config"]["verifier_result_endpoint"] = result_endpoint
        result["config"]["draft_control_endpoints"] = control_endpoints
        result["config"]["verifier_dist_init_addr"] = spec_dist_init_addr
        if baseline_dist_init_addr is not None:
            result["config"]["baseline_dist_init_addr"] = baseline_dist_init_addr
        if reserved_ports is not None:
            result["config"]["reserved_ports"] = reserved_ports

        common.print_summary(result)
        if args.output_dir:
            print("output_files:")
            for output_path in common.write_output_files(result, args.output_dir):
                print(f"  {output_path}")
            print(f"  {_write_summary_json(result, args.output_dir)}")
    finally:
        if baseline_engine is not None:
            baseline_engine.shutdown()
        if verifier_engine is not None:
            verifier_engine.shutdown()
        shutdown_draft_engines(draft_processes, draft_stop_senders)


if __name__ == "__main__":
    main()
