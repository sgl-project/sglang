# SPDX-License-Identifier: Apache-2.0
"""Weight Cache Daemon — a persistent process that holds post-quantized,
TP-sharded model weights in GPU memory and serves them via pluggable transports.

Each GPU runs one daemon process for its TP rank. The daemon:
1. Loads model weights from disk (full pipeline: disk → TP shard → quantize)
2. Exports parameters, buffers, and immutable derived weight state
3. Serves transport entries over a Unix socket to requesting engine processes
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

import argparse
import dataclasses
import logging
import multiprocessing
import os
import signal
import socket
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, publish

from .derived_weight_state import collect_derived_weight_state
from .protocol import (
    CacheConfig,
    check_ipc_quant_support,
    cleanup_stale_daemon_files,
    compute_env_stamp,
    compute_global_rank,
    compute_local_gpu_id,
    get_quant_method_name,
    get_ready_path,
    get_socket_path,
    hash_quant_config,
    recv_msg,
    send_msg,
)
from .transport import choose_daemon_transport_backend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# Per-connection timeout for the serial serve loop. A client exchange is tiny
# (a config dict + IPC handle metadata), so this generous bound never trips a
# healthy client, yet guarantees one hung/dead peer can't stall the other
# engine ranks indefinitely.
CLIENT_CONNECTION_TIMEOUT = 30.0


@dataclasses.dataclass
class WeightCacheDaemonArgs:
    """Daemon-private worker identity and standalone launcher controls.

    ``gpu_id``, ``tp_rank``, ``pp_rank``, and ``dist_init_method`` identify a
    worker after the shared server configuration has been resolved. ``timeout``
    and ``force`` apply only when this entrypoint launches and monitors a local
    daemon group. None has a corresponding ServerArgs field with these
    per-worker semantics.
    """

    gpu_id: Optional[int] = None
    tp_rank: Optional[int] = None
    pp_rank: int = 0
    dist_init_method: Optional[str] = None
    timeout: int = 1800
    force: bool = False
    weight_heterogeneous_transfer_mode: Optional[str] = None
    weight_heterogeneous_transfer_host: Optional[str] = None
    weight_heterogeneous_transfer_port: int = 31999
    weight_heterogeneous_transfer_registry_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weight_heterogeneous_transfer_mode not in (None, "source", "target"):
            raise ValueError(
                "weight heterogeneous transfer mode must be None, 'source', or 'target'"
            )
        if self.weight_heterogeneous_transfer_mode is None:
            if self.weight_heterogeneous_transfer_host is not None or (
                self.weight_heterogeneous_transfer_registry_url is not None
            ):
                raise ValueError(
                    "heterogeneous transfer host/registry URL require an active mode"
                )
            return
        if not self.weight_heterogeneous_transfer_host:
            raise ValueError("active heterogeneous transfer mode requires a host")
        if self.weight_heterogeneous_transfer_port <= 0:
            raise ValueError("heterogeneous transfer port must be positive")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--gpu-id",
            type=int,
            default=None,
            help="GPU device ID for a single daemon. If omitted, launches all local ranks.",
        )
        parser.add_argument(
            "--tp-rank",
            type=int,
            default=None,
            help="TP rank for a single daemon. If omitted, launches all local ranks.",
        )
        parser.add_argument("--pp-rank", type=int, default=0)
        parser.add_argument(
            "--dist-init-method",
            default=None,
            help="Daemon distributed init method (for example tcp://host:29500).",
        )
        parser.add_argument("--timeout", type=int, default=1800)
        parser.add_argument("--force", action="store_true")
        parser.add_argument(
            "--weight-heterogeneous-transfer-mode",
            choices=("source", "target"),
            default=None,
            help="Heterogeneous transfer role. Omit to disable transfer.",
        )
        parser.add_argument(
            "--weight-heterogeneous-transfer-host",
            default=None,
            help="Source bind host or target source-registry host.",
        )
        parser.add_argument(
            "--weight-heterogeneous-transfer-port",
            type=int,
            default=31999,
            help="Manifest registry bind/connect port.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "WeightCacheDaemonArgs":
        return cls(
            gpu_id=args.gpu_id,
            tp_rank=args.tp_rank,
            pp_rank=args.pp_rank,
            dist_init_method=args.dist_init_method,
            timeout=args.timeout,
            force=args.force,
            weight_heterogeneous_transfer_mode=(
                args.weight_heterogeneous_transfer_mode
            ),
            weight_heterogeneous_transfer_host=(
                args.weight_heterogeneous_transfer_host
            ),
            weight_heterogeneous_transfer_port=(
                args.weight_heterogeneous_transfer_port
            ),
        )


class WeightCacheDaemon:
    """Persistent GPU weight cache for a single TP rank.

    Holds the complete post-quantization state_dict in GPU memory and
    serves CUDA IPC handles to engine processes via Unix socket.
    """

    def __init__(
        self,
        server_args: "ServerArgs",
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dist_init_method: Optional[str] = None,
        daemon_args: Optional[WeightCacheDaemonArgs] = None,
    ):
        self.server_args = server_args
        self.daemon_args = daemon_args or WeightCacheDaemonArgs()
        cfg = resolving_view(server_args)
        self.model_path = cfg.model_path
        self.gpu_id = gpu_id
        self.nnodes = cfg.nnodes
        self.tp_size = cfg.tp_size
        self.tp_rank = tp_rank
        self.pp_size = cfg.pp_size
        self.pp_rank = pp_rank
        self.dp_size = cfg.dp_size
        self.ep_size = cfg.ep_size
        self.moe_dp_size = cfg.moe_dp_size
        self.enable_dp_attention = cfg.enable_dp_attention
        self.enable_dp_lm_head = cfg.enable_dp_lm_head
        self.attn_cp_size = cfg.attn_cp_size
        self.moe_dense_tp_size = cfg.moe_dense_tp_size
        self.moe_a2a_backend = cfg.moe_a2a_backend
        self.deepep_mode = cfg.deepep_mode
        self.load_format = cfg.load_format
        self.dtype = cfg.dtype
        self.quantization = cfg.quantization
        self.model_loader_extra_config = cfg.model_loader_extra_config
        self.trust_remote_code = cfg.trust_remote_code
        self.revision = cfg.revision
        self.dist_init_method = dist_init_method
        self.weight_heterogeneous_transfer_mode = (
            self.daemon_args.weight_heterogeneous_transfer_mode
        )
        self.weight_heterogeneous_transfer_registry_url = (
            self.daemon_args.weight_heterogeneous_transfer_registry_url
        )
        if (
            self.weight_heterogeneous_transfer_mode is not None
            and not self.weight_heterogeneous_transfer_registry_url
        ):
            raise ValueError(
                "heterogeneous weight transfer requires a manifest registry URL"
            )

        self.device_uuid = current_platform.get_device_uuid(gpu_id)
        self.socket_path = get_socket_path(self.device_uuid)
        self.ready_path = get_ready_path(self.device_uuid)

        self.model = None
        self.config: Optional[CacheConfig] = None
        # name -> transport-specific tensor entry metadata (shape/dtype/is_param + payload metadata)
        self.state_entries: Dict[str, Dict[str, Any]] = {}
        self.preloaded_weights_bytes = 0
        self.transport_backend = None
        self.weight_transfer_session: Optional[Any] = None
        self.weight_heterogeneous_transfer_stats: Optional[Dict[str, int]] = None

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
                moe_a2a_backend=self.moe_a2a_backend,
            )

        initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            expert_model_parallel_size=self.ep_size,
            attention_data_parallel_size=(
                self.dp_size if self.enable_dp_attention else 1
            ),
            attention_context_model_parallel_size=self.attn_cp_size,
            moe_data_model_parallel_size=self.moe_dp_size,
        )

        # Initialize DP attention state before model construction.
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

        # Reduce thread contention during multi-process loading
        torch.set_num_threads(1)

        # Lazy imports to avoid circular dependencies and speed up startup
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.loader import get_model_loader

        server_args = self.server_args
        publish(server_args, role="weight_cache_daemon")

        from sglang.srt.layers.moe import initialize_moe_config

        initialize_moe_config()

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

        # Build cache config fingerprint before loading the model.
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

        if self.weight_heterogeneous_transfer_mode is not None:
            from .weight_heterogeneous_transfer import (
                validate_weight_heterogeneous_transfer_configuration,
            )

            validate_weight_heterogeneous_transfer_configuration(
                tp_size=self.tp_size,
                dp_size=self.dp_size,
                ep_size=self.ep_size,
                pp_size=self.pp_size,
                enable_dp_attention=self.enable_dp_attention,
                moe_dp_size=self.moe_dp_size,
                attn_cp_size=self.attn_cp_size,
                nnodes=self.nnodes,
                quantization=self.quantization or model_config.quantization,
            )

        # Refuse unsupported quant methods before creating distributed groups
        # or touching model weights.
        check_ipc_quant_support(quant_method, quant_config, where="daemon")

        # The initialized groups are the authority for rank identity. This
        # avoids maintaining a second copy of the model-parallel hierarchy.
        self._init_distributed(server_args, model_config)
        self._initialize_eplb_expert_location_metadata(model_config)
        moe_dp_rank = get_parallel().moe_dp_rank
        moe_ep_rank = get_parallel().moe_ep_rank
        self.config = CacheConfig(
            model_path=self.model_path,
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
            moe_dp_size=self.moe_dp_size,
            moe_dp_rank=moe_dp_rank,
            moe_ep_rank=moe_ep_rank,
            enable_dp_attention=self.enable_dp_attention,
            enable_dp_lm_head=self.enable_dp_lm_head,
            attn_cp_size=self.attn_cp_size,
            moe_dense_tp_size=self.moe_dense_tp_size,
            moe_a2a_backend=self.moe_a2a_backend,
            quant_method=quant_method,
            quant_config_hash=hash_quant_config(quant_config),
            dtype=str(model_config.dtype),
            revision=self.revision or "",
            **compute_env_stamp(),
        )

        current_platform.empty_cache()
        memory_before_load = torch.cuda.memory_reserved(self.gpu_id)

        # Build load config
        load_config = LoadConfig(
            load_format=(
                "dummy"
                if self.weight_heterogeneous_transfer_mode == "target"
                else self.load_format
            ),
            model_loader_extra_config=(
                {}
                if self.weight_heterogeneous_transfer_mode == "target"
                else self.model_loader_extra_config
            ),
            tp_rank=self.tp_rank,
        )

        if self.weight_heterogeneous_transfer_mode == "target":
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
                f"Allocating target parallel layout: {self.model_path}"
            )
        else:
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
                f"Loading model from disk: {self.model_path}"
            )
        tic = time.perf_counter()

        # Load model using DefaultModelLoader (includes TP sharding + quant post-process)
        loader = get_model_loader(load_config=load_config, model_config=model_config)
        source_load_capture = None
        if self.weight_heterogeneous_transfer_mode == "source":
            from .weight_load_recorder import capture_weight_load_plan

            with capture_weight_load_plan(loader) as source_load_capture:
                self.model = loader.load_model(
                    model_config=model_config,
                    device_config=DeviceConfig(
                        current_platform.device_type, self.gpu_id
                    ),
                )
        else:
            self.model = loader.load_model(
                model_config=model_config,
                device_config=DeviceConfig(current_platform.device_type, self.gpu_id),
            )

        elapsed = time.perf_counter() - tic
        if self.weight_heterogeneous_transfer_mode == "target":
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
                f"Model allocated in {elapsed:.2f}s"
            )
        else:
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
                f"Model loaded from disk in {elapsed:.2f}s"
            )

        # Ensure every post-processing kernel has retired before we export the
        # memory: clients map these tensors read-only via IPC and would otherwise
        # risk observing half-written weights.
        current_platform.synchronize()
        current_platform.empty_cache()
        self.preloaded_weights_bytes = max(
            0, torch.cuda.memory_reserved(self.gpu_id) - memory_before_load
        )

        if self.weight_heterogeneous_transfer_mode is not None:
            from .weight_heterogeneous_transfer import WeightTransferSession

            self.weight_transfer_session = WeightTransferSession.create(
                model=self.model,
                model_config=model_config,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                pp_size=self.pp_size,
                pp_rank=self.pp_rank,
                ep_size=self.ep_size,
                dp_size=(self.dp_size if self.enable_dp_attention else 1),
                gpu_id=self.gpu_id,
                revision=(
                    self.revision
                    or getattr(model_config.hf_config, "_commit_hash", None)
                    or "local"
                ),
                is_source_daemon=(
                    self.weight_heterogeneous_transfer_mode == "source"
                ),
                load_plan=(
                    source_load_capture.plan
                    if source_load_capture is not None
                    else None
                ),
            )

        if self.weight_heterogeneous_transfer_mode == "source":
            from .weight_heterogeneous_transfer import (
                register_weight_manifest,
            )

            register_weight_manifest(
                self.weight_heterogeneous_transfer_registry_url,
                global_rank=compute_global_rank(
                    self.tp_size, self.pp_rank, self.tp_rank
                ),
                device_uuid=self.device_uuid,
                session=self.weight_transfer_session,
            )
            logger.info(
                "[WeightCacheDaemon gpu=%s] Registered source weight manifest at %s",
                self.gpu_id,
                self.weight_heterogeneous_transfer_registry_url,
            )

        if self.weight_heterogeneous_transfer_mode == "target":
            from .weight_heterogeneous_transfer import (
                transfer_weights_from_source_daemons,
            )

            self.weight_heterogeneous_transfer_stats = (
                transfer_weights_from_source_daemons(
                    target_session=self.weight_transfer_session,
                    model=self.model,
                    device_uuid=self.device_uuid,
                    registry_url=self.weight_heterogeneous_transfer_registry_url,
                )
            )
            logger.info(
                "[WeightCacheDaemon gpu=%s tp_rank=%s] Received heterogeneous "
                "weights from %s: %s",
                self.gpu_id,
                self.tp_rank,
                self.weight_heterogeneous_transfer_registry_url,
                self.weight_heterogeneous_transfer_stats,
            )
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
        """Export model state entries through the selected transport backend."""
        self.state_entries.clear()

        # remove_duplicate=False so tied weights are recognized as parameters
        # under every name. state_dict() below emits both tied keys, and with the
        # deduped set the duplicate would be mis-registered as a buffer, not a
        # parameter, on the client.
        param_names = set(
            name for name, _ in self.model.named_parameters(remove_duplicate=False)
        )
        state_dict_names = set(self.model.state_dict().keys())
        state_tensors: Dict[str, Tuple[torch.Tensor, bool]] = {}

        # Export all items from state_dict (parameters + persistent buffers)
        for name, tensor in self.model.state_dict().items():
            state_tensors[name] = (tensor.data, name in param_names)

        # Also export non-persistent buffers (not in state_dict but needed
        # for inference, e.g. rotary embedding cos_sin_cache)
        non_persistent_count = 0
        for name, buf in self.model.named_buffers():
            if name not in state_dict_names:
                state_tensors[name] = (buf.data, False)
                non_persistent_count += 1

        # These exact post-load values cannot be reconstructed generically on
        # the client because their layout depends on the model/backend.
        plain_tensors, plain_values = collect_derived_weight_state(self.model)
        for name, tensor in plain_tensors.items():
            if name in state_tensors:
                raise RuntimeError(
                    f"weight-cache derived tensor name collides with state: {name}"
                )
            state_tensors[name] = (tensor.data, False)

        self.transport_backend = choose_daemon_transport_backend(state_tensors)
        self.state_entries = self.transport_backend.prepare_export(state_tensors)
        for name in plain_tensors:
            self.state_entries[name]["is_plain_tensor"] = True
        for name, value in plain_values.items():
            if name in self.state_entries:
                raise RuntimeError(
                    f"weight-cache plain value name collides with state: {name}"
                )
            self.state_entries[name] = {
                "is_plain_value": True,
                "value": value,
            }

        # Log approximate serialized metadata size (not payload-backed bytes).
        # Only the handle blob carries real weight, so measure it directly:
        # stringifying every entry would allocate a copy of all handles.
        total_bytes = sum(
            len(handle)
            for handle in (entry.get("handle") for entry in self.state_entries.values())
            if isinstance(handle, (str, bytes, bytearray))
        )
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] "
            f"Exported {len(self.state_entries)} state entries "
            f"({non_persistent_count} non-persistent buffers), "
            f"({len(plain_tensors)} plain derived tensors), "
            f"({len(plain_values)} plain derived values), "
            f"transport={self.transport_backend.name}, "
            f"metadata size ~{total_bytes / 1024 / 1024:.1f} MB"
        )

    def _initialize_eplb_expert_location_metadata(self, model_config) -> None:
        """Build the same initial physical expert layout as the engine."""
        if not self.server_args.enable_eplb:
            return

        from sglang.srt.eplb.expert_location import (
            compute_initial_expert_location_metadata,
            set_global_expert_location_metadata,
        )

        set_global_expert_location_metadata(
            compute_initial_expert_location_metadata(
                model_config=model_config,
                moe_ep_rank=get_parallel().moe_ep_rank,
            )
        )

    def serve(self):
        """Block and serve IPC handles over Unix socket."""
        # Do NOT unlink an existing socket here: stale-file cleanup is the launch
        # path's job (cleanup_stale_daemon_files refuses to remove a socket whose
        # .ready still points at a live PID). A leftover live socket makes bind()
        # fail loudly instead of silently stealing another daemon's socket.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            sock.bind(self.socket_path)
        finally:
            os.umask(old_umask)
        sock.listen(8)
        sock.settimeout(1.0)  # Allow periodic shutdown check

        # Write ready file
        with open(self.ready_path, "w") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"config={self.config.to_dict()}\n")
            if self.weight_heterogeneous_transfer_stats is not None:
                f.write(
                    "weight_heterogeneous_transfer="
                    f"{self.weight_heterogeneous_transfer_stats}\n"
                )

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] Listening on {self.socket_path}"
        )

        self._running = True

        def _signal_handler(signum, frame):
            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] Received signal {signum}, shutting down"
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
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if os.path.exists(self.ready_path):
                os.unlink(self.ready_path)
            logger.info(f"[WeightCacheDaemon gpu={self.gpu_id}] Shutdown complete")

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection."""
        req = recv_msg(conn)

        if req.get("type") == "query_config":
            # Client asks for config without requesting handles
            send_msg(conn, {"status": "ok", "config": self.config.to_dict()})

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
                f"Serving {len(self.state_entries)} tensors via "
                f"{self.transport_backend.name} transport"
            )
            self.transport_backend.send_fetch_state_response(
                conn,
                config=self.config.to_dict(),
                entries=self.state_entries,
                # PID so the client can watch daemon liveness: if this
                # process dies while clients hold IPC mappings, their
                # param.data (and any CUDA-graph-captured addresses) dangle.
                pid=os.getpid(),
                preloaded_weights_bytes=self.preloaded_weights_bytes,
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
        """Release GPU memory and clean up."""
        if self.weight_transfer_session is not None:
            self.weight_transfer_session.close()
            self.weight_transfer_session = None
        if dist.is_initialized():
            dist.destroy_process_group()
        if self.model is not None:
            del self.model
            self.model = None
        self.state_entries.clear()
        current_platform.empty_cache()
        self._running = False


def run_weight_cache_daemon(
    server_args: "ServerArgs",
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dist_init_method: Optional[str] = None,
    daemon_args: Optional[WeightCacheDaemonArgs] = None,
):
    """Entry point for running a weight cache daemon process."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Daemon gpu={gpu_id} tp_rank={tp_rank}] %(levelname)s %(message)s",
    )

    # Die if our parent (the engine or the standalone launcher that spawned us)
    # dies, even on SIGKILL/OOM-kill. Without this an orphaned daemon keeps a
    # full weight copy pinned in GPU memory and its live-PID .ready file blocks
    # the next launch — the opposite of fast recovery.
    from sglang.srt.utils import kill_itself_when_parent_died

    kill_itself_when_parent_died()

    daemon = WeightCacheDaemon(
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dist_init_method=dist_init_method,
        daemon_args=daemon_args,
    )

    try:
        daemon.load()
        daemon.serve()
    finally:
        daemon.shutdown()


def spawn_weight_cache_daemon(
    server_args: "ServerArgs",
    *,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dist_init_method: str,
    daemon_args: Optional[WeightCacheDaemonArgs] = None,
):
    """Start one daemon from the complete resolved server configuration."""
    ctx = multiprocessing.get_context("spawn")
    process_args = (server_args, gpu_id, tp_rank, pp_rank, dist_init_method)
    if daemon_args is not None:
        process_args += (daemon_args,)
    proc = ctx.Process(
        target=run_weight_cache_daemon,
        args=process_args,
    )
    proc.start()
    return proc


def _prepare_weight_heterogeneous_transfer(
    server_args: "ServerArgs",
    daemon_args: WeightCacheDaemonArgs,
    *,
    expected_rank_count: int,
):
    """Validate the phase-one contract and resolve the manifest registry."""
    if daemon_args.weight_heterogeneous_transfer_mode is None:
        return None, None

    from .weight_heterogeneous_transfer import (
        validate_weight_heterogeneous_transfer_configuration,
    )

    cfg = resolving_view(server_args)
    validate_weight_heterogeneous_transfer_configuration(
        tp_size=cfg.tp_size,
        dp_size=cfg.dp_size,
        ep_size=cfg.ep_size,
        pp_size=cfg.pp_size,
        enable_dp_attention=cfg.enable_dp_attention,
        moe_dp_size=cfg.moe_dp_size,
        attn_cp_size=cfg.attn_cp_size,
        nnodes=cfg.nnodes,
        quantization=cfg.quantization,
    )

    if daemon_args.weight_heterogeneous_transfer_registry_url:
        return None, daemon_args.weight_heterogeneous_transfer_registry_url
    if daemon_args.weight_heterogeneous_transfer_mode == "source":
        from .weight_manifest_server import WeightManifestServer

        manifest_server = WeightManifestServer(
            host=daemon_args.weight_heterogeneous_transfer_host,
            port=daemon_args.weight_heterogeneous_transfer_port,
            expected_rank_count=expected_rank_count,
        )
        from sglang.srt.utils.network import NetworkAddress

        try:
            connect_host = {
                "0.0.0.0": "127.0.0.1",
                "::": "::1",
            }.get(daemon_args.weight_heterogeneous_transfer_host)
            if connect_host is None:
                connect_host = daemon_args.weight_heterogeneous_transfer_host
            registry_address = NetworkAddress(connect_host, manifest_server.port)
        except BaseException:
            manifest_server.close()
            raise
        return (
            manifest_server,
            f"tcp://{registry_address.to_host_port_str()}",
        )
    from sglang.srt.utils.network import NetworkAddress

    source_address = NetworkAddress(
        daemon_args.weight_heterogeneous_transfer_host,
        daemon_args.weight_heterogeneous_transfer_port,
    )
    return None, f"tcp://{source_address.to_host_port_str()}"


def _cleanup_weight_cache_daemon_launch(procs, weight_manifest_server) -> None:
    """Stop every launched daemon and release the optional registry."""
    try:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
    finally:
        if weight_manifest_server is not None:
            weight_manifest_server.close()


def launch_weight_cache_daemons(
    server_args: "ServerArgs",
    dist_init_method: Optional[str] = None,
    timeout: int = 1800,
    force: bool = False,
    daemon_args: Optional[WeightCacheDaemonArgs] = None,
):
    """Launch weight cache daemon processes for this node's PP×TP ranks.

    For single-node (nnodes=1): spawns pp_size * tp_size daemons.
    For multi-node (nnodes>1): spawns this node's share of PP×TP daemons,
    mapping local gpu_id to the correct global (pp_rank, tp_rank).

    Uses ``multiprocessing`` with the ``spawn`` start method. The child starts
    in a clean interpreter and receives the complete resolved ``ServerArgs``
    through pickle, avoiding a second hand-maintained CLI configuration path.

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
    cfg = resolving_view(server_args)
    import socket as sock_mod

    daemon_args = daemon_args or WeightCacheDaemonArgs()
    weight_manifest_server, weight_manifest_registry_url = (
        _prepare_weight_heterogeneous_transfer(
            server_args,
            daemon_args,
            expected_rank_count=cfg.tp_size * cfg.pp_size,
        )
    )
    procs = []

    # Replicate _calculate_rank_ranges logic from engine.py
    pp_size_per_node = max(cfg.pp_size // cfg.nnodes, 1)
    nnodes_per_pp_rank = max(cfg.nnodes // cfg.pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (cfg.node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (cfg.node_rank // nnodes_per_pp_rank + 1),
    )
    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = cfg.tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (cfg.node_rank % nnodes_per_tp_group),
        tp_size_per_node * (cfg.node_rank % nnodes_per_tp_group + 1),
    )

    if cfg.nnodes > 1 and dist_init_method is None:
        _cleanup_weight_cache_daemon_launch(procs, weight_manifest_server)
        raise ValueError(
            "dist_init_method is required for multi-node weight cache daemons. "
            "Use --dist-init-method tcp://<node0-ip>:<port> to specify the "
            "rendezvous address accessible from all nodes."
        )

    # Auto-allocate a free port for the distributed init method
    if dist_init_method is None:
        try:
            with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                free_port = s.getsockname()[1]
        except BaseException:
            _cleanup_weight_cache_daemon_launch(procs, weight_manifest_server)
            raise
        dist_init_method = f"tcp://127.0.0.1:{free_port}"

    # Validate and clean up stale .ready/.sock files from prior runs.
    try:
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                gpu_id = compute_local_gpu_id(
                    pp_rank,
                    tp_rank,
                    pp_size_per_node,
                    tp_size_per_node,
                    base_gpu_id=cfg.base_gpu_id,
                    gpu_id_step=cfg.gpu_id_step,
                )
                device_uuid = current_platform.get_device_uuid(gpu_id)
                cleanup_stale_daemon_files(device_uuid, force=force)

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                gpu_id = compute_local_gpu_id(
                    pp_rank,
                    tp_rank,
                    pp_size_per_node,
                    tp_size_per_node,
                    base_gpu_id=cfg.base_gpu_id,
                    gpu_id_step=cfg.gpu_id_step,
                )
                child_daemon_args = dataclasses.replace(
                    daemon_args,
                    weight_heterogeneous_transfer_registry_url=(
                        weight_manifest_registry_url
                    ),
                )
                proc = spawn_weight_cache_daemon(
                    server_args,
                    gpu_id=gpu_id,
                    tp_rank=tp_rank,
                    pp_rank=pp_rank,
                    dist_init_method=dist_init_method,
                    daemon_args=child_daemon_args,
                )
                procs.append(proc)
                logger.info(
                    f"Launched weight cache daemon gpu={gpu_id} "
                    f"pp_rank={pp_rank} tp_rank={tp_rank} pid={proc.pid}"
                )
    except BaseException:
        _cleanup_weight_cache_daemon_launch(procs, weight_manifest_server)
        raise

    # Wait for all daemons on this node to become ready
    num_daemons = len(procs)
    check_interval = 2
    start_time = time.time()
    try:
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                gpu_id = compute_local_gpu_id(
                    pp_rank,
                    tp_rank,
                    pp_size_per_node,
                    tp_size_per_node,
                    base_gpu_id=cfg.base_gpu_id,
                    gpu_id_step=cfg.gpu_id_step,
                )
                device_uuid = current_platform.get_device_uuid(gpu_id)
                ready_path = get_ready_path(device_uuid)
                while not os.path.exists(ready_path):
                    time.sleep(check_interval)
                    if time.time() - start_time > timeout:
                        logger.error(
                            f"Weight cache daemon pp_rank={pp_rank} "
                            f"tp_rank={tp_rank} did not become ready within {timeout}s"
                        )
                        raise TimeoutError(
                            f"Weight cache daemon pp_rank={pp_rank} "
                            f"tp_rank={tp_rank} did not become ready within {timeout}s"
                        )
                    for proc in procs:
                        if not proc.is_alive():
                            logger.error(
                                "Weight cache daemon exited prematurely "
                                f"with code {proc.exitcode}"
                            )
                            raise RuntimeError(
                                "Weight cache daemon exited prematurely "
                                f"with code {proc.exitcode}"
                            )
                logger.info(
                    f"Weight cache daemon pp_rank={pp_rank} tp_rank={tp_rank} is ready"
                )
    except BaseException:
        _cleanup_weight_cache_daemon_launch(procs, weight_manifest_server)
        raise

    logger.info(
        f"All {num_daemons} weight cache daemons on node {cfg.node_rank} are ready "
        f"(pp_ranks={pp_rank_range.start}..{pp_rank_range.stop - 1}, "
        f"tp_ranks={tp_rank_range.start}..{tp_rank_range.stop - 1}, "
        f"dist_init_method={dist_init_method})"
    )

    # Monitor daemons and, the moment any one exits, terminate the rest and
    # raise. A serial proc.join() would not notice a
    # mid-list death (e.g. procs[1] dying while procs[0] is still alive) until
    # the earlier proc happened to exit, and it never surfaced the failure.
    exited = None
    try:
        while exited is None:
            for proc in procs:
                if not proc.is_alive():
                    exited = proc
                    break
            else:
                time.sleep(1)
                continue
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down daemons")
    finally:
        _cleanup_weight_cache_daemon_launch(procs, weight_manifest_server)
        logger.info("All weight cache daemons have been terminated")

    if exited is not None:
        raise RuntimeError(
            f"Weight cache daemon (pid={exited.pid}) exited with code "
            f"{exited.exitcode}; terminated the remaining daemons."
        )


if __name__ == "__main__":
    worker_parser = argparse.ArgumentParser(add_help=False)
    WeightCacheDaemonArgs.add_cli_args(worker_parser)
    worker_ns, server_argv = worker_parser.parse_known_args()

    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args(server_argv)
    daemon_args = WeightCacheDaemonArgs.from_cli_args(worker_ns)
    if daemon_args.gpu_id is not None or daemon_args.tp_rank is not None:
        weight_manifest_server, registry_url = (
            _prepare_weight_heterogeneous_transfer(
                server_args,
                daemon_args,
                expected_rank_count=1,
            )
        )
        gpu_id = (
            daemon_args.gpu_id
            if daemon_args.gpu_id is not None
            else daemon_args.tp_rank
        )
        tp_rank = (
            daemon_args.tp_rank
            if daemon_args.tp_rank is not None
            else daemon_args.gpu_id
        )
        device_uuid = current_platform.get_device_uuid(gpu_id)
        cleanup_stale_daemon_files(device_uuid, force=daemon_args.force)
        child_daemon_args = dataclasses.replace(
            daemon_args,
            weight_heterogeneous_transfer_registry_url=registry_url,
        )
        try:
            run_weight_cache_daemon(
                server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=daemon_args.pp_rank,
                dist_init_method=daemon_args.dist_init_method,
                daemon_args=child_daemon_args,
            )
        finally:
            if weight_manifest_server is not None:
                weight_manifest_server.close()
    else:
        launch_weight_cache_daemons(
            server_args,
            dist_init_method=daemon_args.dist_init_method,
            timeout=daemon_args.timeout,
            force=daemon_args.force,
            daemon_args=daemon_args,
        )
