# SPDX-License-Identifier: Apache-2.0
"""Weight Cache Daemon — a persistent process that holds post-quantized,
TP-sharded model weights in GPU memory and serves them via CUDA IPC handles.

Each GPU runs one daemon process for its TP rank. The daemon:
1. Loads model weights from disk (full pipeline: disk → TP shard → quantize)
2. Exports every parameter/buffer as a CUDA IPC handle
3. Serves handles over a Unix socket to requesting engine processes
4. Validates CacheConfig compatibility before serving

Usage:
    # Single-node: launch all TP rank daemons with a single command:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 4 \
        --load-format auto --dtype auto --quantization fp8

    # Multi-node: run on each node with --nnodes and --node-rank:
    # Node 0:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 16 \
        --nnodes 2 --node-rank 0 \
        --dist-init-method tcp://node0-ip:29500

    # Node 1:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model --tp-size 16 \
        --nnodes 2 --node-rank 1 \
        --dist-init-method tcp://node0-ip:29500

    # Or launch a single daemon for a specific rank:
    python -m sglang.srt.weight_cache.daemon \
        --model-path /path/to/model \
        --gpu-id 0 --tp-size 4 --tp-rank 0 \
        --dist-init-method tcp://127.0.0.1:29500
"""

import logging
import os
import signal
import socket
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import publish
from sglang.srt.utils import MultiprocessingSerializer

from .protocol import (
    CacheConfig,
    check_ipc_parallelism,
    check_ipc_quant_support,
    compute_env_stamp,
    compute_global_rank,
    compute_local_gpu_id,
    get_quant_method_name,
    get_resolved_model_revision,
    hash_loader_extra_config,
    hash_quant_config,
    normalize_model_path_for_cache,
    recv_msg,
    send_msg,
)
from .registry import (
    CacheIdentity,
    DaemonRegistration,
    FileWeightCacheRegistry,
    new_daemon_id,
)

logger = logging.getLogger(__name__)

# Per-connection timeout for the serial serve loop. A client exchange is tiny
# (a config dict + IPC handle metadata), so this generous bound never trips a
# healthy client, yet guarantees one hung/dead peer can't stall the other
# engine ranks indefinitely.
CLIENT_CONNECTION_TIMEOUT = 30.0


class WeightCacheDaemon:
    """Persistent GPU weight cache for a single TP rank.

    Holds the complete post-quantization state_dict in GPU memory and
    serves CUDA IPC handles to engine processes via Unix socket.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        dp_size: int = 1,
        ep_size: int = 1,
        load_format: str = "auto",
        dtype: str = "auto",
        quantization: Optional[str] = None,
        model_loader_extra_config: str = "{}",
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        dist_init_method: Optional[str] = None,
        weight_cache_runtime_dir: Optional[str] = None,
        weight_cache_namespace: str = "default",
        daemon_id: Optional[str] = None,
        force: bool = False,
    ):
        check_ipc_parallelism(dp_size, ep_size, where="daemon")
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.dp_size = dp_size
        self.ep_size = ep_size
        self.load_format = load_format
        self.dtype = dtype
        self.quantization = quantization
        self.model_loader_extra_config = model_loader_extra_config
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.dist_init_method = dist_init_method
        self.daemon_id = daemon_id or new_daemon_id()
        self.force = force
        self.registry = FileWeightCacheRegistry(
            weight_cache_runtime_dir, namespace=weight_cache_namespace
        )
        self.identity: Optional[CacheIdentity] = None
        self.registration: Optional[DaemonRegistration] = None
        self.device_uuid: Optional[str] = None
        self.socket_path: Optional[str] = None

        self.model = None
        self.config: Optional[CacheConfig] = None
        # name -> {"handle": base64_str, "shape": list, "dtype": str, "is_param": bool}
        self.state_entries: Dict[str, Dict[str, Any]] = {}
        self._gpu_work_started = False

    def _init_distributed(self, server_args, model_config):
        """Initialize the distributed backend required for model loading.

        Uses the same world_size/rank formula as the engine:
            world_size = tp_size * pp_size
            rank = tp_size * pp_rank + tp_rank
        """
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )

        if model_parallel_is_initialized():
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Distributed already initialized, skipping"
            )
            return

        # Initialize distributed environment
        import torch.distributed as dist

        if not dist.is_initialized():
            if self.dist_init_method is None:
                # Fallback: auto-assign a port. This only works for single-process.
                import socket as sock_mod

                with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    free_port = s.getsockname()[1]
                self.dist_init_method = f"tcp://127.0.0.1:{free_port}"

            init_distributed_environment(
                world_size=self.tp_size * self.pp_size,
                rank=compute_global_rank(self.tp_size, self.pp_rank, self.tp_rank),
                distributed_init_method=self.dist_init_method,
                local_rank=self.gpu_id,
                backend=current_platform.get_torch_distributed_backend_str(),
            )

        initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            expert_model_parallel_size=self.ep_size,
        )

        # Initialize DP attention state (required by some models like Qwen3 MoE)
        from sglang.srt.layers.dp_attention import initialize_dp_attention

        initialize_dp_attention(server_args, model_config)

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Distributed backend initialized (tp_size={self.tp_size}, "
            f"pp_size={self.pp_size}, "
            f"world_size={self.tp_size * self.pp_size})"
        )

    def load(self):
        """Full loading pipeline: disk → TP shard → quantize → export IPC handles."""
        # CUDA IPC weight sharing relies on torch's _share_cuda_ handle export,
        # which only exists on CUDA-alike platforms (CUDA / ROCm). Fail loud here
        # instead of dying deep inside the export with an opaque error.
        if not current_platform.is_cuda_alike():
            raise RuntimeError(
                f"[WeightCacheDaemon] the weight cache daemon requires a CUDA-alike "
                f"platform (CUDA or ROCm) for CUDA IPC weight sharing, but the "
                f"active platform device type is {current_platform.device_type!r}. "
                f"Disable the weight cache (--weight-cache-mode off)."
            )
        # expandable_segments makes torch's caching allocator hand out memory
        # that cannot be exported via _share_cuda_, so the IPC export below would
        # die mid-way with an opaque CUDA error. Fail fast with an actionable
        # message before touching the device.
        self._assert_ipc_compatible_allocator()
        current_platform.set_device(current_platform.get_device(self.gpu_id))
        self._gpu_work_started = True

        # Reduce thread contention during multi-process loading
        torch.set_num_threads(1)

        # Lazy imports to avoid circular dependencies and speed up startup
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.loader import get_model_loader
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(
            model_path=self.model_path,
            dtype=self.dtype,
            quantization=self.quantization,
            trust_remote_code=self.trust_remote_code,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            load_format=self.load_format,
            model_loader_extra_config=self.model_loader_extra_config,
        )
        # ServerArgs resolves source-specific "auto" values (for example
        # mistral, remote, or runai_streamer). The daemon must fingerprint and
        # load with that exact resolved value or every client will mismatch.
        self.load_format = str(
            getattr(server_args.load_format, "value", server_args.load_format)
        )
        publish(server_args, role="weight_cache_daemon")

        # Initialize distributed backend for model loading
        # (must be done after server_args and model_config are available)
        # Build model config first, then init distributed
        model_config = ModelConfig(
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            dtype=self.dtype,
            quantization=self.quantization,
        )

        # Build cache config fingerprint BEFORE loading the model.
        # Loading may mutate hf_config.quantization_config (e.g. via
        # process_weights_after_loading), which would produce a different
        # hash than what the engine computes from the original config.
        # ModelConfig always exposes hf_config/quantization directly;
        # quantization_config is the only genuinely-optional attribute.
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(
            self.quantization or model_config.quantization
        )
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)

        # Give unsupported methods the actionable allowlist error before
        # attempting to serialize vendor-specific quantization objects.
        check_ipc_quant_support(quant_method, quant_config, where="daemon")

        self.config = CacheConfig(
            model_path=normalize_model_path_for_cache(self.model_path),
            model_arch=(
                model_config.hf_config.architectures[0]
                if model_config.hf_config.architectures
                else ""
            ),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            quant_method=quant_method,
            quant_config_hash=hash_quant_config(quant_config),
            dtype=str(model_config.dtype),
            revision=self.revision or "",
            resolved_revision=get_resolved_model_revision(model_config),
            load_format=self.load_format,
            model_loader_extra_config_hash=hash_loader_extra_config(
                self.model_loader_extra_config
            ),
            trust_remote_code=self.trust_remote_code,
            **compute_env_stamp(self.gpu_id),
        )

        # Reserve the exact physical-GPU + config + namespace identity before
        # expensive model loading. Logical rank never participates in the path.
        self.device_uuid = current_platform.get_device_uuid(self.gpu_id)
        identity = self.registry.identity_for(self.config, self.device_uuid)
        socket_path = self.registry.socket_path(identity)
        self.registry.claim(
            identity,
            pid=os.getpid(),
            daemon_id=self.daemon_id,
            force=self.force,
        )
        self.identity = identity
        self.socket_path = socket_path

        # Initialize distributed backend (requires server_args + model_config)
        self._init_distributed(server_args, model_config)

        # Build load config
        load_config = LoadConfig(
            load_format=self.load_format,
            model_loader_extra_config=self.model_loader_extra_config,
            tp_rank=self.tp_rank,
        )

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Loading model from disk: {self.model_path}"
        )
        tic = time.perf_counter()

        # DefaultModelLoader includes TP sharding and quant post-processing.
        loader = get_model_loader(load_config=load_config, model_config=model_config)
        self.model = loader.load_model(
            model_config=model_config,
            device_config=DeviceConfig(current_platform.device_type, self.gpu_id),
        )

        elapsed = time.perf_counter() - tic
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Model loaded from disk in {elapsed:.2f}s"
        )

        # Ensure every post-processing kernel has retired before we export the
        # memory: clients map these tensors read-only via IPC and would otherwise
        # risk observing half-written weights.
        current_platform.synchronize()

        # Export all parameters and buffers as IPC handles
        self._export_state()

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Exported {len(self.state_entries)} tensors as IPC handles. "
            f"Ready to serve."
        )

    @staticmethod
    def _assert_ipc_compatible_allocator() -> None:
        """Reject allocator configs incompatible with CUDA IPC export.

        The expandable-segments allocator returns memory that cannot be shared
        through torch's _share_cuda_ handle, which would make the export fail
        partway with an opaque error. Detect it up front and fail loud.
        """
        for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            conf = os.environ.get(var, "")
            for field in conf.split(","):
                key, _, value = field.partition(":")
                if (
                    key.strip() == "expandable_segments"
                    and value.strip().lower() == "true"
                ):
                    raise RuntimeError(
                        f"[WeightCacheDaemon] {var} sets expandable_segments:True, "
                        f"which is incompatible with CUDA IPC weight sharing: the "
                        f"expandable-segments allocator hands out memory that cannot "
                        f"be exported via _share_cuda_, so the IPC handle export "
                        f"would fail mid-way. Unset expandable_segments for the "
                        f"weight cache daemon process (it can stay enabled for the "
                        f"engine itself)."
                    )

    def _export_state(self):
        """Export model parameters and buffers as CUDA IPC handles.

        This includes both persistent buffers (in state_dict) and non-persistent
        buffers (e.g. rotary embedding cos_sin_cache) so the engine can fully
        reconstruct the model state via zero-copy IPC.
        """
        self.state_entries.clear()

        # remove_duplicate=False so tied weights are recognized as parameters
        # under every name. state_dict() below emits both tied keys, and with the
        # deduped set the duplicate would be mis-registered as a buffer, not a
        # parameter, on the client.
        param_names = set(
            name for name, _ in self.model.named_parameters(remove_duplicate=False)
        )
        state_dict_names = set(self.model.state_dict().keys())

        # Export all items from state_dict (parameters + persistent buffers)
        for name, tensor in self.model.state_dict().items():
            ipc_handle = MultiprocessingSerializer.serialize(
                tensor.data, output_str=True
            )
            self.state_entries[name] = {
                "handle": ipc_handle,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": name in param_names,
            }

        # Also export non-persistent buffers (not in state_dict but needed
        # for inference, e.g. rotary embedding cos_sin_cache)
        non_persistent_count = 0
        for name, buf in self.model.named_buffers():
            if name not in state_dict_names:
                ipc_handle = MultiprocessingSerializer.serialize(
                    buf.data, output_str=True
                )
                self.state_entries[name] = {
                    "handle": ipc_handle,
                    "shape": list(buf.shape),
                    "dtype": str(buf.dtype).replace("torch.", ""),
                    "is_param": False,
                }
                non_persistent_count += 1

        # Log total size
        total_bytes = sum(
            entry["handle"].__len__() if hasattr(entry["handle"], "__len__") else 0
            for entry in self.state_entries.values()
        )
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] "
            f"Exported {len(self.state_entries)} tensors "
            f"({non_persistent_count} non-persistent buffers), "
            f"serialized handle size ~{total_bytes / 1024 / 1024:.1f} MB"
        )

    def serve(self):
        """Block and serve IPC handles over Unix socket."""
        if self.identity is None or self.config is None or self.socket_path is None:
            raise RuntimeError("weight cache daemon must load and claim before serve")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            sock.bind(self.socket_path)
        finally:
            os.umask(old_umask)
        sock.listen(8)
        sock.settimeout(1.0)  # Allow periodic shutdown check
        # Publish only after bind+listen. Atomic registry replacement is the
        # readiness signal, so clients never discover a half-started daemon.
        self.registration = self.registry.publish(
            self.identity,
            config=self.config,
            socket_path=self.socket_path,
            pid=os.getpid(),
            daemon_id=self.daemon_id,
        )

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] Listening on {self.socket_path}"
        )

        self._running = True

        def _signal_handler(signum, frame):
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] Received signal "
                f"{signum}, shutting down"
            )
            self._running = False

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            while self._running:
                try:
                    conn, _ = sock.accept()
                    # The listen-socket timeout above only bounds accept(); the
                    # accepted connection is blocking by default. Since we serve
                    # connections serially, a client that connects but never sends
                    # (or dies mid-send) would block recv_msg forever and stall
                    # every other engine rank. Bound each exchange instead.
                    conn.settimeout(CLIENT_CONNECTION_TIMEOUT)
                    try:
                        self._handle_connection(conn)
                    except Exception as e:
                        logger.error(
                            f"[WeightCacheDaemon gpu={self.gpu_id}] "
                            f"Error handling connection: {e}",
                            exc_info=True,
                        )
                    finally:
                        conn.close()
                except socket.timeout:
                    continue
        finally:
            sock.close()
            self.registry.unpublish(self.identity, daemon_id=self.daemon_id)
            self.registration = None
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                "Stopped accepting connections"
            )

    def _daemon_metadata(self) -> Dict[str, Any]:
        if self.registration is None:
            raise RuntimeError("weight cache daemon is not registered")
        return {
            "daemon_id": self.registration.daemon_id,
            "device_uuid": self.registration.identity.device_uuid,
            "config_fingerprint": self.registration.identity.config_fingerprint,
            "pid": self.registration.pid,
            "process_start_time": self.registration.process_start_time,
        }

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        req = recv_msg(conn)

        if req.get("type") == "query_config":
            # Client asks for config without requesting handles
            send_msg(
                conn,
                {
                    "status": "ok",
                    "config": self.config.to_dict(),
                    "daemon": self._daemon_metadata(),
                },
            )

        elif req.get("type") == "fetch_state":
            # Client requests full state with IPC handles
            engine_config = CacheConfig.from_dict(req["config"])
            if not self.config.matches(engine_config):
                # Log detailed mismatch info for debugging
                daemon_dict = self.config.to_dict()
                engine_dict = engine_config.to_dict()
                mismatches = {
                    k: (daemon_dict.get(k), engine_dict.get(k))
                    for k in daemon_dict
                    if daemon_dict.get(k) != engine_dict.get(k)
                }
                logger.warning(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] "
                    f"Config mismatch: {mismatches}"
                )
                send_msg(
                    conn, {"status": "mismatch", "daemon_config": self.config.to_dict()}
                )
                return

            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Serving {len(self.state_entries)} IPC handles to engine"
            )
            send_msg(
                conn,
                {
                    "status": "ok",
                    "config": self.config.to_dict(),
                    "entries": self.state_entries,
                    "daemon": self._daemon_metadata(),
                },
            )

        elif req.get("type") == "ping":
            send_msg(conn, {"status": "ok"})

        else:
            send_msg(
                conn,
                {
                    "status": "error",
                    "message": f"Unknown request type: {req.get('type')}",
                },
            )

    def shutdown(self):
        """Stop serving and leave exported tensors alive until process exit."""
        self._running = False
        if dist.is_initialized():
            dist.destroy_process_group()
        # Do not free exported tensors or release the claim while this PID is
        # still live. Process exit releases GPU state atomically from clients'
        # liveness perspective; the next claimant reclaims the dead record.


def run_weight_cache_daemon(
    model_path: str,
    gpu_id: int,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_size: int = 1,
    pp_rank: int = 0,
    dp_size: int = 1,
    ep_size: int = 1,
    load_format: str = "auto",
    dtype: str = "auto",
    quantization: Optional[str] = None,
    model_loader_extra_config: str = "{}",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    dist_init_method: Optional[str] = None,
    weight_cache_runtime_dir: Optional[str] = None,
    weight_cache_namespace: str = "default",
    daemon_id: Optional[str] = None,
    force: bool = False,
):
    """Entry point for running a weight cache daemon process."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s [Daemon gpu={gpu_id} tp_rank={tp_rank}] "
            "%(levelname)s %(message)s"
        ),
    )

    # Die if our parent (the engine or the standalone launcher that spawned us)
    # dies, even on SIGKILL/OOM-kill. Without this an orphaned daemon keeps a
    # full weight copy pinned in GPU memory and its live registry claim blocks
    # the next launch.
    from sglang.srt.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()

    daemon = WeightCacheDaemon(
        model_path=model_path,
        gpu_id=gpu_id,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=pp_size,
        pp_rank=pp_rank,
        dp_size=dp_size,
        ep_size=ep_size,
        load_format=load_format,
        dtype=dtype,
        quantization=quantization,
        model_loader_extra_config=model_loader_extra_config,
        trust_remote_code=trust_remote_code,
        revision=revision,
        dist_init_method=dist_init_method,
        weight_cache_runtime_dir=weight_cache_runtime_dir,
        weight_cache_namespace=weight_cache_namespace,
        daemon_id=daemon_id,
        force=force,
    )
    try:
        daemon.load()
        daemon.serve()
    finally:
        daemon.shutdown()


def launch_weight_cache_daemons(
    model_path: str,
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
    ep_size: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    base_gpu_id: int = 0,
    gpu_id_step: int = 1,
    load_format: str = "auto",
    dtype: str = "auto",
    quantization: Optional[str] = None,
    model_loader_extra_config: str = "{}",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    dist_init_method: Optional[str] = None,
    timeout: int = 1800,
    force: bool = False,
    weight_cache_runtime_dir: Optional[str] = None,
    weight_cache_namespace: str = "default",
):
    """Launch weight cache daemon processes for this node's PP×TP ranks.

    For single-node (nnodes=1): spawns pp_size * tp_size daemons.
    For multi-node (nnodes>1): spawns this node's share of PP×TP daemons,
    mapping local gpu_id to the correct global (pp_rank, tp_rank).

    Uses subprocess.Popen instead of multiprocessing.Process to avoid
    initializing CUDA in the parent process, which can degrade CUDA IPC
    performance in child processes.

    Usage (single-node):
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 4

    Usage (multi-node, run on each node):
        # Node 0:
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 16 \\
            --nnodes 2 --node-rank 0 \\
            --dist-init-method tcp://node0-ip:29500

        # Node 1:
        python -m sglang.srt.weight_cache.daemon \\
            --model-path /path/to/model --tp-size 16 \\
            --nnodes 2 --node-rank 1 \\
            --dist-init-method tcp://node0-ip:29500
    """
    import socket as sock_mod
    import subprocess
    import sys

    # Replicate _calculate_rank_ranges logic from engine.py
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )
    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )

    if nnodes > 1 and dist_init_method is None:
        raise ValueError(
            "dist_init_method is required for multi-node weight cache daemons. "
            "Use --dist-init-method tcp://<node0-ip>:<port> to specify the "
            "rendezvous address accessible from all nodes."
        )

    # Auto-allocate a free port for the distributed init method
    if dist_init_method is None:
        with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        dist_init_method = f"tcp://127.0.0.1:{free_port}"

    python_path = sys.executable
    daemon_module = "sglang.srt.weight_cache.daemon"

    registry = FileWeightCacheRegistry(
        weight_cache_runtime_dir, namespace=weight_cache_namespace
    )
    procs = []
    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            gpu_id = compute_local_gpu_id(
                pp_rank,
                tp_rank,
                pp_size_per_node,
                tp_size_per_node,
                base_gpu_id=base_gpu_id,
                gpu_id_step=gpu_id_step,
            )
            child_daemon_id = new_daemon_id()
            cmd = [
                python_path,
                "-m",
                daemon_module,
                "--model-path",
                model_path,
                "--gpu-id",
                str(gpu_id),
                "--tp-size",
                str(tp_size),
                "--tp-rank",
                str(tp_rank),
                "--dp-size",
                str(dp_size),
                "--ep-size",
                str(ep_size),
                "--pp-size",
                str(pp_size),
                "--pp-rank",
                str(pp_rank),
                "--load-format",
                load_format,
                "--dtype",
                dtype,
                "--dist-init-method",
                dist_init_method,
                "--weight-cache-namespace",
                weight_cache_namespace,
                "--daemon-id",
                child_daemon_id,
            ]
            if weight_cache_runtime_dir:
                cmd += ["--weight-cache-runtime-dir", weight_cache_runtime_dir]
            if force:
                cmd += ["--force"]
            if quantization:
                cmd += ["--quantization", quantization]
            if model_loader_extra_config and model_loader_extra_config != "{}":
                cmd += ["--model-loader-extra-config", model_loader_extra_config]
            if trust_remote_code:
                cmd += ["--trust-remote-code"]
            if revision:
                cmd += ["--revision", revision]

            proc = subprocess.Popen(cmd)
            procs.append((proc, child_daemon_id))
            logger.info(
                f"Launched weight cache daemon gpu={gpu_id} "
                f"pp_rank={pp_rank} tp_rank={tp_rank} pid={proc.pid}"
            )

    # Wait for all daemons on this node to become ready
    num_daemons = len(procs)
    check_interval = 2
    start_time = time.time()

    def _terminate_and_reap_daemons() -> None:
        for child, _ in procs:
            if child.poll() is None:
                child.terminate()
        killed = []
        for child, _ in procs:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                killed.append(child)
        for child in killed:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error(
                    "Weight cache daemon pid=%s remained unreaped after SIGKILL",
                    child.pid,
                )

    for proc, child_daemon_id in procs:
        while (
            registry.find_registration(daemon_id=child_daemon_id, pid=proc.pid) is None
        ):
            time.sleep(check_interval)
            if time.time() - start_time > timeout:
                logger.error(
                    f"Weight cache daemon pid={proc.pid} "
                    f"did not become ready within {timeout}s"
                )
                _terminate_and_reap_daemons()
                raise TimeoutError(
                    f"Weight cache daemon pid={proc.pid} "
                    f"did not become ready within {timeout}s"
                )
            # Check if any daemon exited prematurely
            for p, _ in procs:
                retcode = p.poll()
                if retcode is not None:
                    logger.error(
                        f"Weight cache daemon exited prematurely with code {retcode}"
                    )
                    _terminate_and_reap_daemons()
                    raise RuntimeError(
                        f"Weight cache daemon exited prematurely with code {retcode}"
                    )
        logger.info(f"Weight cache daemon pid={proc.pid} is ready")

    logger.info(
        f"All {num_daemons} weight cache daemons on node {node_rank} are ready "
        f"(pp_ranks={pp_rank_range.start}..{pp_rank_range.stop - 1}, "
        f"tp_ranks={tp_rank_range.start}..{tp_rank_range.stop - 1}, "
        f"dist_init_method={dist_init_method})"
    )

    # Monitor daemons — poll all of them and, the moment any one exits,
    # terminate the rest and raise. A serial proc.wait() would not notice a
    # mid-list death (e.g. procs[1] dying while procs[0] is still alive) until
    # the earlier proc happened to exit, and it never surfaced the failure.
    exited = None
    try:
        while exited is None:
            for proc, _ in procs:
                if proc.poll() is not None:
                    exited = proc
                    break
            else:
                time.sleep(1)
                continue
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down daemons")
    finally:
        _terminate_and_reap_daemons()
        logger.info("All weight cache daemons have been terminated")

    if exited is not None:
        raise RuntimeError(
            f"Weight cache daemon (pid={exited.pid}) exited with code "
            f"{exited.returncode}; terminated the remaining daemons."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGLang Weight Cache Daemon")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU device ID for a single daemon. "
        "If omitted, launches daemons for all TP ranks (0..tp_size-1).",
    )
    parser.add_argument(
        "--tp-rank",
        type=int,
        default=None,
        help="TP rank for a single daemon. "
        "If omitted, launches daemons for all TP ranks.",
    )
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--pp-rank", type=int, default=0, help="Pipeline parallel rank")
    parser.add_argument("--nnodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument(
        "--base-gpu-id",
        type=int,
        default=0,
        help="GPU id of this node's first rank (mirrors the engine's "
        "--base-gpu-id). Used to place daemons on the same GPUs the engine "
        "ranks will use.",
    )
    parser.add_argument(
        "--gpu-id-step",
        type=int,
        default=1,
        help="Stride between consecutive ranks' GPU ids (mirrors the engine's "
        "--gpu-id-step).",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Rank of this node (0-indexed). Required for multi-node.",
    )
    parser.add_argument("--load-format", default="auto", help="Weight load format")
    parser.add_argument("--dtype", default="auto", help="Model dtype")
    parser.add_argument("--quantization", default=None, help="Quantization method")
    parser.add_argument(
        "--model-loader-extra-config",
        default="{}",
        help="Extra config for model loader (JSON string)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None, help="Model revision")
    parser.add_argument(
        "--weight-cache-runtime-dir",
        default=None,
        help="Private node-local directory for weight-cache registry and sockets.",
    )
    parser.add_argument(
        "--weight-cache-namespace",
        default="default",
        help="Optional namespace included in every cache identity.",
    )
    parser.add_argument("--daemon-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dist-init-method",
        default=None,
        help="Distributed init method (e.g. tcp://node0-ip:29500). "
        "Auto-assigned for single-node when launching all ranks. "
        "Required for multi-node (nnodes > 1) and must be accessible "
        "from all nodes. Also required for tp_size > 1 when launching "
        "a single rank.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help=(
            "Timeout in seconds to wait for all daemons to become ready (default: 1800)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Kill and replace a live daemon with the same cache identity.",
    )
    args = parser.parse_args()

    if args.gpu_id is not None or args.tp_rank is not None:
        # Single-rank mode: launch one daemon for the specified rank
        gpu_id = args.gpu_id if args.gpu_id is not None else args.tp_rank
        tp_rank = args.tp_rank if args.tp_rank is not None else args.gpu_id
        run_weight_cache_daemon(
            model_path=args.model_path,
            gpu_id=gpu_id,
            tp_size=args.tp_size,
            tp_rank=tp_rank,
            pp_size=args.pp_size,
            pp_rank=args.pp_rank,
            dp_size=args.dp_size,
            ep_size=args.ep_size,
            load_format=args.load_format,
            dtype=args.dtype,
            quantization=args.quantization,
            model_loader_extra_config=args.model_loader_extra_config,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            dist_init_method=args.dist_init_method,
            weight_cache_runtime_dir=args.weight_cache_runtime_dir,
            weight_cache_namespace=args.weight_cache_namespace,
            daemon_id=args.daemon_id,
            force=args.force,
        )
    else:
        # Multi-rank mode: launch daemons for this node's TP ranks
        launch_weight_cache_daemons(
            model_path=args.model_path,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            dp_size=args.dp_size,
            ep_size=args.ep_size,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            base_gpu_id=args.base_gpu_id,
            gpu_id_step=args.gpu_id_step,
            load_format=args.load_format,
            dtype=args.dtype,
            quantization=args.quantization,
            model_loader_extra_config=args.model_loader_extra_config,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            dist_init_method=args.dist_init_method,
            timeout=args.timeout,
            force=args.force,
            weight_cache_runtime_dir=args.weight_cache_runtime_dir,
            weight_cache_namespace=args.weight_cache_namespace,
        )
