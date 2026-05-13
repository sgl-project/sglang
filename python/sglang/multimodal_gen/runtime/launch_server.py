# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import dataclasses
import multiprocessing as mp
import os
import pickle
import shutil
import signal
import sys
import threading
import time
from multiprocessing.connection import Connection

import psutil
import uvicorn
import zmq

from sglang.multimodal_gen.runtime.disaggregation.diffusion_server import (
    DiffusionServer,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
from sglang.multimodal_gen.runtime.managers.gpu_worker import run_scheduler_process
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    prepare_server_args,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.common import is_port_available
from sglang.multimodal_gen.runtime.utils.logging_utils import configure_logger, logger
from sglang.multimodal_gen.runtime.warmup_utils import build_server_warmup_reqs
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info


def _find_available_port(
    start: int = 10000, avoid: set[int] | None = None, max_attempts: int = 100
) -> int:
    """Find an available port starting from *start*, skipping ports in *avoid*."""
    if avoid is None:
        avoid = set()
    port = max(1024, min(start, 65535))
    for _ in range(max_attempts):
        if port not in avoid and is_port_available(port):
            return port
        port += 1
        if port > 65535:
            port = 1024
    raise RuntimeError(
        f"No available port found after {max_attempts} attempts (start={start})"
    )


def _build_disagg_calibration_reqs(
    server_args: ServerArgs, temp_dirs: list[str] | None = None
) -> list[Req]:
    return build_server_warmup_reqs(server_args, temp_dirs=temp_dirs)


def _run_disagg_startup_calibration(
    frontend_endpoint: str,
    server_args: ServerArgs,
) -> None:
    temp_dirs: list[str] = []
    warmup_reqs = _build_disagg_calibration_reqs(server_args, temp_dirs=temp_dirs)
    if not warmup_reqs:
        return

    context = zmq.Context(io_threads=1)
    sock = context.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, max(1000, int(server_args.disagg_timeout * 1000)))
    sock.setsockopt(zmq.SNDTIMEO, 10000)
    sock.connect(frontend_endpoint)

    try:
        for idx, req in enumerate(warmup_reqs, start=1):
            logger.info(
                "Running disagg startup calibration request %d/%d via %s",
                idx,
                len(warmup_reqs),
                frontend_endpoint,
            )
            sock.send(pickle.dumps([req]))
            reply = sock.recv_multipart()
            output_batch = pickle.loads(reply[-1]) if reply else None
            if getattr(output_batch, "error", None):
                raise RuntimeError(
                    f"Disagg startup calibration failed: {output_batch.error}"
                )
            time.sleep(0.5)
    finally:
        sock.close(linger=0)
        context.destroy(linger=0)
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _wait_for_disagg_role_registration(
    diffusion_server: DiffusionServer,
    *,
    expected_encoders: int,
    expected_denoisers: int,
    expected_decoders: int,
    timeout_s: float,
    poll_interval_s: float = 0.1,
) -> None:
    logger.info(
        "Waiting for standalone role registration before startup calibration: "
        "encoder=%d, denoiser=%d, decoder=%d",
        expected_encoders,
        expected_denoisers,
        expected_decoders,
    )
    deadline = time.monotonic() + timeout_s

    while True:
        stats = diffusion_server.get_stats()
        actual_encoders = int(stats.get("encoder_peers", 0))
        actual_denoisers = int(stats.get("denoiser_peers", 0))
        actual_decoders = int(stats.get("decoder_peers", 0))

        if (
            actual_encoders == expected_encoders
            and actual_denoisers == expected_denoisers
            and actual_decoders == expected_decoders
        ):
            logger.info(
                "All standalone role instances registered; starting startup calibration"
            )
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "Timed out waiting for standalone role registration before startup "
                "calibration "
                f"(encoder {actual_encoders}/{expected_encoders}, "
                f"denoiser {actual_denoisers}/{expected_denoisers}, "
                f"decoder {actual_decoders}/{expected_decoders})"
            )

        time.sleep(min(poll_interval_s, remaining))


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def _terminate_processes(processes: list[mp.Process], timeout_s: float = 5.0) -> None:
    """Best-effort cleanup for spawned worker processes."""
    for process in processes:
        try:
            process.terminate()
        except Exception:
            pass

    deadline = time.time() + timeout_s
    for process in processes:
        remaining = max(0.0, deadline - time.time())
        try:
            process.join(timeout=remaining)
        except Exception:
            pass


def _spawn_disagg_worker_group(
    pool_ctx: mp.context.BaseContext,
    worker_ids: list[int],
    role_args: ServerArgs,
    process_name_builder,
    group_label: str,
) -> list[mp.Process]:
    """Spawn all ranks for a disagg role instance before waiting for readiness."""
    processes: list[mp.Process] = []
    ready_readers: list[tuple[int, Connection]] = []
    ready_writers: list[Connection] = []
    role_device = (
        role_args.resolved_role_device()
        if hasattr(role_args, "resolved_role_device")
        else getattr(role_args, "disagg_role_device", "auto")
    )
    if role_device == "auto":
        role_device = "cpu" if getattr(role_args, "num_gpus", 1) <= 0 else "cuda"
    platform_override = "cpu" if role_device == "cpu" else "cuda"
    override_env_key = "SGLANG_DIFFUSION_PLATFORM_OVERRIDE"

    try:
        for rank_idx, worker_id in enumerate(worker_ids):
            reader, writer = pool_ctx.Pipe(duplex=False)
            ready_readers.append((rank_idx, reader))
            ready_writers.append(writer)

            process = pool_ctx.Process(
                target=_run_disagg_role_process,
                args=(worker_id, rank_idx, rank_idx, role_args, writer, [], []),
                name=process_name_builder(rank_idx),
                daemon=True,
            )
            # The child imports current_platform before GPUWorker can inspect
            # role args. Set the override only while spawning this role worker
            # so a CPU encoder can coexist with CUDA denoiser/decoder workers.
            old_platform_override = os.environ.get(override_env_key)
            os.environ[override_env_key] = platform_override
            try:
                logger.debug(
                    "%s rank %d: spawning with platform override %s",
                    group_label,
                    rank_idx,
                    platform_override,
                )
                process.start()
            finally:
                if old_platform_override is None:
                    os.environ.pop(override_env_key, None)
                else:
                    os.environ[override_env_key] = old_platform_override
            processes.append(process)

        logger.info(
            "%s: started ranks=%s on worker_ids=%s; waiting for readiness",
            group_label,
            list(range(len(worker_ids))),
            worker_ids,
        )

        for writer in ready_writers:
            writer.close()

        for rank_idx, reader in ready_readers:
            try:
                data = reader.recv()
            except EOFError as exc:
                if rank_idx < len(processes):
                    process = processes[rank_idx]
                    try:
                        process.join(timeout=0.1)
                    except Exception:
                        pass
                    exitcode = process.exitcode
                else:
                    exitcode = None
                raise RuntimeError(
                    f"{group_label} rank {rank_idx} exited before reporting ready "
                    f"(exitcode={exitcode})."
                ) from exc
            finally:
                reader.close()

            if data.get("status") != "ready":
                raise RuntimeError(
                    f"{group_label} rank {rank_idx} failed to initialize: {data}"
                )

        return processes
    except Exception:
        _terminate_processes(processes)
        raise
    finally:
        for writer in ready_writers:
            try:
                writer.close()
            except Exception:
                pass
        for _, reader in ready_readers:
            try:
                reader.close()
            except Exception:
                pass


def launch_server(server_args: ServerArgs, launch_http_server: bool = True):
    """
    Args:
        launch_http_server: False for offline local mode
    """
    configure_logger(server_args)

    # Start a new server with multiple worker processes
    logger.info("Starting server...")

    num_gpus = server_args.num_gpus
    processes = []

    # Pipes for master to talk to slaves
    task_pipes_to_slaves_w = []
    task_pipes_to_slaves_r = []
    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        task_pipes_to_slaves_r.append(r)
        task_pipes_to_slaves_w.append(w)

    # Pipes for slaves to talk to master
    result_pipes_from_slaves_w = []
    result_pipes_from_slaves_r = []
    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        result_pipes_from_slaves_r.append(r)
        result_pipes_from_slaves_w.append(w)

    # Launch all worker processes
    master_port = server_args.master_port or _find_available_port(
        start=server_args.scheduler_port + 100,
        avoid={server_args.scheduler_port},
    )
    scheduler_pipe_readers = []
    scheduler_pipe_writers = []

    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_writers.append(writer)
        if i == 0:  # Master worker
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    i,  # rank
                    master_port,
                    server_args,
                    writer,
                    None,  # No task pipe to read from master
                    None,  # No result pipe to write to master
                    task_pipes_to_slaves_w,
                    result_pipes_from_slaves_r,
                ),
                name=f"sglang-diffusionWorker-{i}",
                daemon=True,
            )
        else:  # Slave workers
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    i,  # rank
                    master_port,
                    server_args,
                    writer,
                    None,  # No task pipe to read from master
                    None,  # No result pipe to write to master
                    task_pipes_to_slaves_r[i - 1],
                    result_pipes_from_slaves_w[i - 1],
                ),
                name=f"sglang-diffusionWorker-{i}",
                daemon=True,
            )
        scheduler_pipe_readers.append(reader)
        process.start()
        processes.append(process)

    # Wait for all workers to be ready
    scheduler_infos = []
    for writer in scheduler_pipe_writers:
        writer.close()

    # Close unused pipe ends in parent process
    for p in task_pipes_to_slaves_w:
        p.close()
    for p in task_pipes_to_slaves_r:
        p.close()
    for p in result_pipes_from_slaves_w:
        p.close()
    for p in result_pipes_from_slaves_r:
        p.close()

    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            processes[i].join()
            logger.error(f"Exit code: {processes[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)
        reader.close()

    logger.debug("All workers are ready")

    if launch_http_server:
        logger.info("Starting FastAPI server.")
        if server_args.webui:
            logger.info("Launch FastAPI server in another process because of webui.")
            http_server_process = mp.Process(
                target=launch_http_server_only,
                args=(server_args,),
                name=f"sglang-diffusion-webui",
                daemon=True,
            )
            http_server_process.start()
        else:
            launch_http_server_only(server_args)

    return processes


def launch_pool_disagg_server(
    server_args: ServerArgs,
    encoder_gpus: list[list[int]],
    denoiser_gpus: list[list[int]],
    decoder_gpus: list[list[int]],
    launch_http_server: bool = True,
):
    """Launch a pool-based disaggregated server with N:M:K independent role instances.

    DiffusionServer orchestrates the full pipeline, dispatching at every
    role transition (Encoder → Denoiser → Decoder).

    Args:
        server_args: Base server configuration
        encoder_gpus: List of GPU ID lists, one per encoder instance.
            e.g., [[0], [2]] for 2 encoder instances on GPUs 0 and 2.
        denoiser_gpus: List of GPU ID lists, one per denoiser instance.
            e.g., [[1], [3]] for 2 denoiser instances.
        decoder_gpus: List of GPU ID lists, one per decoder instance.
            e.g., [[0], [2]] for 2 decoder instances (can share with encoder).
        launch_http_server: Whether to launch the HTTP server.

    Example:
        launch_pool_disagg_server(server_args,
            encoder_gpus=[[0], [2]],
            denoiser_gpus=[[1], [3]],
            decoder_gpus=[[0], [2]],
        )
    """
    configure_logger(server_args)

    num_encoders = len(encoder_gpus)
    num_denoisers = len(denoiser_gpus)
    num_decoders = len(decoder_gpus)
    logger.info(
        "Starting pool disagg server: %d encoder(s), %d denoiser(s), %d decoder(s)...",
        num_encoders,
        num_denoisers,
        num_decoders,
    )

    host = server_args.host or "127.0.0.1"

    def find_port(start):
        return _find_available_port(start)

    # Allocate endpoints
    port_cursor = server_args.scheduler_port + 3000

    # Per-instance work endpoints (instance binds PULL, DS connects PUSH)
    encoder_work_endpoints = []
    for i in range(num_encoders):
        p = find_port(port_cursor)
        encoder_work_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    denoiser_work_endpoints = []
    for i in range(num_denoisers):
        p = find_port(port_cursor)
        denoiser_work_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    decoder_work_endpoints = []
    for i in range(num_decoders):
        p = find_port(port_cursor)
        decoder_work_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    encoder_control_endpoints = []
    for i in range(num_encoders):
        p = find_port(port_cursor)
        encoder_control_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    denoiser_control_endpoints = []
    for i in range(num_denoisers):
        p = find_port(port_cursor)
        denoiser_control_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    decoder_control_endpoints = []
    for i in range(num_decoders):
        p = find_port(port_cursor)
        decoder_control_endpoints.append(f"tcp://{host}:{p}")
        port_cursor = p + 1

    # Per-role-type result endpoints (DS binds PULL, instances connect PUSH)
    # Use deterministic convention: scheduler_port + {1,2,3}
    base_port = server_args.scheduler_port
    encoder_result_ep = f"tcp://{host}:{base_port + 1}"
    denoiser_result_ep = f"tcp://{host}:{base_port + 2}"
    decoder_result_ep = f"tcp://{host}:{base_port + 3}"

    logger.info(
        "Pool endpoints allocated: %d work + 3 result endpoints",
        num_encoders + num_denoisers + num_decoders,
    )

    # Launch all role instances
    all_processes = []

    role_configs = [
        (
            RoleType.ENCODER,
            encoder_gpus,
            encoder_work_endpoints,
            encoder_control_endpoints,
            encoder_result_ep,
        ),
        (
            RoleType.DENOISER,
            denoiser_gpus,
            denoiser_work_endpoints,
            denoiser_control_endpoints,
            denoiser_result_ep,
        ),
        (
            RoleType.DECODER,
            decoder_gpus,
            decoder_work_endpoints,
            decoder_control_endpoints,
            decoder_result_ep,
        ),
    ]

    try:
        for role_type, gpu_lists, work_eps, control_eps, result_ep in role_configs:
            for inst_idx, gpu_ids in enumerate(gpu_lists):
                is_cpu_instance = role_type == RoleType.ENCODER and len(gpu_ids) == 0
                if len(gpu_ids) == 0 and not is_cpu_instance:
                    raise ValueError(
                        f"Empty GPU list is only supported for encoder CPU instances, got {role_type.value}[{inst_idx}]"
                    )

                num_role_workers = 1 if is_cpu_instance else len(gpu_ids)
                role_device = (
                    "cpu" if is_cpu_instance else server_args.disagg_role_device
                )

                # Per-role parallelism: use explicit overrides if set, else None (auto-derive)
                role_par = server_args.get_role_parallelism(role_type)

                role_overrides = {
                    "disagg_role": role_type,
                    "disagg_mode": True,
                    "disagg_instance_id": inst_idx,
                    "disagg_role_device": role_device,
                    "gpu_ids": None if is_cpu_instance else gpu_ids,
                    "pool_work_endpoint": work_eps[inst_idx],
                    "pool_control_endpoint": control_eps[inst_idx],
                    "pool_control_advertised_endpoint": control_eps[inst_idx],
                    "pool_result_endpoint": result_ep,
                    "num_gpus": num_role_workers,
                    "warmup": role_type == RoleType.ENCODER,
                    "scheduler_port": find_port(port_cursor),
                    "master_port": find_port(port_cursor + 100),
                    # Per-role parallelism (None = auto-derive from num_gpus)
                    "tp_size": role_par["tp_size"],
                    "sp_degree": role_par["sp_degree"],
                    "ulysses_degree": role_par["ulysses_degree"],
                    "ring_degree": role_par["ring_degree"],
                }
                port_cursor = role_overrides["master_port"] + 100

                base_dict = {
                    f.name: getattr(server_args, f.name)
                    for f in dataclasses.fields(server_args)
                }
                base_dict.update(role_overrides)
                base_dict.pop("pipeline_config", None)
                role_args = ServerArgs.from_kwargs(**base_dict)

                pool_ctx = mp.get_context("spawn")
                worker_ids = [0] if is_cpu_instance else gpu_ids
                processes = _spawn_disagg_worker_group(
                    pool_ctx=pool_ctx,
                    worker_ids=worker_ids,
                    role_args=role_args,
                    process_name_builder=lambda rank_idx, role_type=role_type, inst_idx=inst_idx: (
                        f"sglang-pool-{role_type.value}-{inst_idx}-r{rank_idx}"
                    ),
                    group_label=f"Pool {role_type.value}[{inst_idx}]",
                )
                all_processes.extend(processes)

                logger.info(
                    "Pool %s[%d] ready on %s %s (work=%s, control=%s)",
                    role_type.value.upper(),
                    inst_idx,
                    "CPU worker(s)" if is_cpu_instance else "GPU(s)",
                    worker_ids,
                    work_eps[inst_idx],
                    control_eps[inst_idx],
                )
    except Exception:
        _terminate_processes(all_processes)
        raise

    logger.info("All pool role instances ready")

    # Start DiffusionServer
    frontend_endpoint = f"tcp://{host}:{server_args.scheduler_port}"

    diffusion_server = None
    try:
        diffusion_server = DiffusionServer(
            frontend_endpoint=frontend_endpoint,
            encoder_work_endpoints=encoder_work_endpoints,
            denoiser_work_endpoints=denoiser_work_endpoints,
            decoder_work_endpoints=decoder_work_endpoints,
            encoder_result_endpoint=encoder_result_ep,
            denoiser_result_endpoint=denoiser_result_ep,
            decoder_result_endpoint=decoder_result_ep,
            dispatch_policy_name=server_args.disagg_dispatch_policy,
            timeout_s=float(server_args.disagg_timeout),
            downstream_wait_timeout_s=float(server_args.disagg_downstream_wait_timeout),
            max_slots_per_instance=server_args.disagg_max_slots_per_instance,
        )
        diffusion_server.start()
        _wait_for_disagg_role_registration(
            diffusion_server,
            expected_encoders=len(encoder_gpus),
            expected_denoisers=len(denoiser_gpus),
            expected_decoders=len(decoder_gpus),
            timeout_s=float(server_args.disagg_timeout),
        )
        if server_args.warmup:
            _run_disagg_startup_calibration(frontend_endpoint, server_args)

        if launch_http_server:
            logger.info(
                "Starting FastAPI server (connected to DiffusionServer at port %d).",
                server_args.scheduler_port,
            )
            launch_http_server_only(server_args)
    except Exception:
        if diffusion_server is not None:
            try:
                diffusion_server.stop()
            except Exception:
                logger.exception("Failed to stop DiffusionServer during cleanup")
        _terminate_processes(all_processes)
        raise

    return all_processes


def _run_disagg_role_process(
    gpu_id: int,
    _local_rank: int,
    rank: int,
    server_args: ServerArgs,
    pipe_writer: Connection,
    task_pipes: list,
    result_pipes: list,
):
    """Entry point for a disagg role process.

    Uses the physical GPU index (gpu_id) as local_rank so that
    torch.cuda.set_device(local_rank) selects the correct GPU.
    This avoids relying on CUDA_VISIBLE_DEVICES remapping, which
    may not work if CUDA was pre-initialized in the parent process.
    """
    run_scheduler_process(
        local_rank=gpu_id,
        rank=rank,
        master_port=server_args.master_port,
        server_args=server_args,
        pipe_writer=pipe_writer,
        task_pipe_r=None,
        result_pipe_w=None,
        task_pipes_to_slaves=task_pipes,
        result_pipes_from_slaves=result_pipes,
    )


def launch_http_server_only(server_args):
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang-diffusion")
        trace_set_thread_info("DiffHTTPServer")

    # set for endpoints to access global_server_args
    set_global_server_args(server_args)
    app = create_app(server_args)
    uvicorn.run(
        app,
        use_colors=True,
        log_level=server_args.log_level,
        host=server_args.host,
        port=server_args.port,
        reload=False,
    )


def parse_url_string(url_str: str) -> list[str]:
    """Parse a semicolon-separated URL string into a list.

    Example: "tcp://10.0.0.1:35000;tcp://10.0.0.2:35000" -> ["tcp://...", "tcp://..."]
    """
    return [u.strip() for u in url_str.split(";") if u.strip()]


def launch_disagg_server(server_args: ServerArgs):
    """Launch DiffusionServer head node + HTTP server (--disagg-role server).

    No GPU workers are spawned. Connects to remote role instances
    specified by --encoder-urls, --denoiser-urls, --decoder-urls.

    Result endpoints use deterministic convention:
        encoder result: scheduler_port + 1
        denoiser result: scheduler_port + 2
        decoder result: scheduler_port + 3
    """
    configure_logger(server_args)

    for name, val in [
        ("--encoder-urls", server_args.encoder_urls),
        ("--denoiser-urls", server_args.denoiser_urls),
        ("--decoder-urls", server_args.decoder_urls),
    ]:
        if val is None:
            raise ValueError(f"{name} is required for --disagg-role server")

    host = server_args.host or "127.0.0.1"
    base_port = server_args.scheduler_port

    encoder_work_endpoints = parse_url_string(server_args.encoder_urls)
    denoiser_work_endpoints = parse_url_string(server_args.denoiser_urls)
    decoder_work_endpoints = parse_url_string(server_args.decoder_urls)

    encoder_result_ep = f"tcp://{host}:{base_port + 1}"
    denoiser_result_ep = f"tcp://{host}:{base_port + 2}"
    decoder_result_ep = f"tcp://{host}:{base_port + 3}"

    frontend_endpoint = f"tcp://{host}:{base_port}"

    logger.info(
        "Starting DiffusionServer: %d encoder(s), %d denoiser(s), %d decoder(s)",
        len(encoder_work_endpoints),
        len(denoiser_work_endpoints),
        len(decoder_work_endpoints),
    )
    logger.info("  Frontend: %s", frontend_endpoint)
    logger.info("  Encoder work endpoints: %s", encoder_work_endpoints)
    logger.info("  Denoiser work endpoints: %s", denoiser_work_endpoints)
    logger.info("  Decoder work endpoints: %s", decoder_work_endpoints)
    logger.info(
        "  Result endpoints: encoder=%s, denoiser=%s, decoder=%s",
        encoder_result_ep,
        denoiser_result_ep,
        decoder_result_ep,
    )

    diffusion_server = DiffusionServer(
        frontend_endpoint=frontend_endpoint,
        encoder_work_endpoints=encoder_work_endpoints,
        denoiser_work_endpoints=denoiser_work_endpoints,
        decoder_work_endpoints=decoder_work_endpoints,
        encoder_result_endpoint=encoder_result_ep,
        denoiser_result_endpoint=denoiser_result_ep,
        decoder_result_endpoint=decoder_result_ep,
        dispatch_policy_name=server_args.disagg_dispatch_policy,
        timeout_s=float(server_args.disagg_timeout),
        downstream_wait_timeout_s=float(server_args.disagg_downstream_wait_timeout),
        max_slots_per_instance=server_args.disagg_max_slots_per_instance,
    )
    try:
        diffusion_server.start()
        _wait_for_disagg_role_registration(
            diffusion_server,
            expected_encoders=len(encoder_work_endpoints),
            expected_denoisers=len(denoiser_work_endpoints),
            expected_decoders=len(decoder_work_endpoints),
            timeout_s=float(server_args.disagg_timeout),
        )
        if server_args.warmup:
            _run_disagg_startup_calibration(frontend_endpoint, server_args)

        logger.info(
            "Starting HTTP server (connected to DiffusionServer at port %d).",
            base_port,
        )
        launch_http_server_only(server_args)
    except Exception:
        try:
            diffusion_server.stop()
        except Exception:
            logger.exception("Failed to stop DiffusionServer during cleanup")
        raise


def launch_disagg_role(server_args: ServerArgs):
    """Launch a standalone disaggregated role instance (--disagg-role encoder/denoising/decoder).

    The instance:
    1. Binds its work PULL socket on tcp://0.0.0.0:{scheduler_port}
    2. Connects its result PUSH socket to the DiffusionServer head node
       (derived from --disagg-server-addr + role offset)
    3. Spawns GPU worker processes for the assigned role.
    """
    configure_logger(server_args)

    role_type = server_args.disagg_role
    if server_args.disagg_server_addr is None:
        raise ValueError(
            "--disagg-server-addr is required for --disagg-role " f"{role_type.value}"
        )

    # Derive endpoints
    work_endpoint = server_args.derive_pool_work_endpoint()
    control_endpoint = server_args.derive_pool_control_endpoint()
    control_advertised_endpoint = server_args.derive_pool_control_advertised_endpoint()
    result_endpoint = server_args.derive_pool_result_endpoint()

    logger.info(
        "Starting disagg role: %s, num_gpus=%d",
        role_type.value,
        server_args.num_gpus,
    )
    logger.info("  Work endpoint (bind): %s", work_endpoint)
    logger.info(
        "  Control endpoint (bind/advertise): %s / %s",
        control_endpoint,
        control_advertised_endpoint,
    )
    logger.info("  Result endpoint (connect): %s", result_endpoint)
    logger.info(
        "  P2P: hostname=%s, ib_device=%s, pool_size=%d",
        server_args.disagg_p2p_hostname,
        server_args.disagg_ib_device,
        server_args.disagg_transfer_pool_size,
    )

    # Build role-specific ServerArgs
    # Use a different port for the scheduler's internal ROUTER socket to avoid
    # conflicting with the pool work PULL socket (both bind on scheduler_port).
    internal_scheduler_port = _find_available_port(
        start=server_args.scheduler_port + 100,
        avoid={server_args.scheduler_port},
    )

    role_par = server_args.get_role_parallelism(role_type)
    role_overrides = {
        "disagg_role": role_type,
        "disagg_mode": True,
        "pool_work_endpoint": work_endpoint,
        "pool_control_endpoint": control_endpoint,
        "pool_control_advertised_endpoint": control_advertised_endpoint,
        "pool_result_endpoint": result_endpoint,
        "warmup": role_type == RoleType.ENCODER,
        "scheduler_port": internal_scheduler_port,
        # Per-role parallelism (None = auto-derive from num_gpus)
        "tp_size": role_par["tp_size"],
        "sp_degree": role_par["sp_degree"],
        "ulysses_degree": role_par["ulysses_degree"],
        "ring_degree": role_par["ring_degree"],
    }
    if server_args.gpu_ids is not None:
        role_overrides["num_gpus"] = len(server_args.gpu_ids)

    base_dict = {
        f.name: getattr(server_args, f.name) for f in dataclasses.fields(server_args)
    }
    base_dict.update(role_overrides)
    base_dict.pop("pipeline_config", None)
    role_args = ServerArgs.from_kwargs(**base_dict)

    # Spawn GPU worker processes
    is_cpu_role = (
        role_type == RoleType.ENCODER and role_args.resolved_role_device() == "cpu"
    )
    pool_ctx = mp.get_context("spawn")

    explicit_gpu_ids = role_args.gpu_ids
    if is_cpu_role:
        if explicit_gpu_ids:
            raise ValueError("--gpu-ids cannot be used when encoder role runs on CPU.")
        worker_ids = [0]
    elif explicit_gpu_ids is not None:
        if not explicit_gpu_ids:
            raise ValueError("--gpu-ids cannot be empty for CUDA disagg roles.")
        worker_ids = explicit_gpu_ids
    else:
        worker_ids = [
            server_args.base_gpu_id + rank_idx
            for rank_idx in range(max(role_args.num_gpus, 1))
        ]
    num_workers = len(worker_ids)
    processes = _spawn_disagg_worker_group(
        pool_ctx=pool_ctx,
        worker_ids=worker_ids,
        role_args=role_args,
        process_name_builder=lambda rank_idx, role_type=role_type: (
            f"sglang-{role_type.value}-r{rank_idx}"
        ),
        group_label=f"Role {role_type.value}",
    )

    logger.info(
        "Role %s ready (%d worker(s), worker_ids=%s, work=%s, control=%s, device=%s)",
        role_type.value.upper(),
        num_workers,
        worker_ids,
        work_endpoint,
        control_endpoint,
        role_args.resolved_role_device(),
    )

    # Block until interrupted
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Role %s shutting down.", role_type.value)


def dispatch_launch(server_args: ServerArgs):
    """Route to the correct launch function based on --disagg-role."""
    role = server_args.disagg_role
    if role == RoleType.MONOLITHIC:
        launch_server(server_args)
    elif role == RoleType.SERVER:
        launch_disagg_server(server_args)
    elif role in (RoleType.ENCODER, RoleType.DENOISER, RoleType.DECODER):
        launch_disagg_role(server_args)
    else:
        raise ValueError(f"Unknown disagg_role: {role}")


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        dispatch_launch(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
