# SPDX-License-Identifier: Apache-2.0
"""Weight Cache Daemon — a persistent process that holds post-quantized,
TP-sharded model weights in GPU memory and serves them via pluggable transport backends.

Each GPU runs one daemon process for its TP rank. The daemon:
1. Loads model weights from disk (full pipeline: disk → TP shard → quantize)
2. Describes the loaded model for engines (weight_cache/load_diff.py): every
   tensor it holds as a view of a storage, and every plain value that
   loading changed; and exports the storages through the selected transport
   (--weight-cache-transport): CUDA IPC handles (torch_ipc), or CUDA VMM
   allocations shared as file descriptors (vmm_fd)
3. Serves the description and transport entries over a Unix socket to
   requesting engine processes
4. Validates the protocol version and CacheConfig compatibility before serving

With speculative decoding it serves the draft model as well (draft.py): it
loads the draft after the target, as an engine does, describes it in a
manifest of its own, and exports a storage the two share once.

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
import contextlib
import dataclasses
import datetime
import gc
import logging
import multiprocessing
import os
import pickle
import signal
import socket
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    SpawnRanks,
    get_exec,
    get_parallel,
    publish,
    spawn_world_rank,
)

from .protocol import (
    DRAFT_ROLE,
    MAX_MSG_SIZE,
    PROTOCOL_VERSION,
    TARGET_ROLE,
    CacheConfig,
    ProtocolVersionError,
    check_ipc_quant_support,
    check_protocol_version,
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
from .transport import (
    TORCH_IPC_BACKEND,
    VMM_FD_BACKEND,
    create_daemon_transport_backend,
    resolve_daemon_transport,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# Per-connection timeout for the serial serve loop. A client exchange is tiny
# (a config dict + IPC handle metadata), so this generous bound never trips a
# healthy client, yet guarantees one hung/dead peer can't stall the other
# engine ranks indefinitely.
CLIENT_CONNECTION_TIMEOUT = 30.0


@dataclasses.dataclass
class ServedDraft:
    """The speculative decoding draft a daemon serves next to its target."""

    model: Any
    config: CacheConfig
    # What a client installs, as the target's (WeightCacheDaemon.manifest)
    manifest: Dict[str, Any]
    # Its storage -> the name it is exported under: its own, or the target's
    # for a storage the two share
    names: Dict[str, str]
    # draft.draft_stamp() taken in its build scope, which the client compares
    stamp: Dict[str, str]
    preloaded_weights_bytes: int = 0
    # Its storage -> transport-specific entry, filled by the export
    entries: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)


def _storage_key(tensor: torch.Tensor) -> Tuple[str, int, int]:
    """What tells one of load_diff.export's storages (a whole allocation as
    uint8) from another."""
    return (str(tensor.device), tensor.data_ptr(), tensor.nbytes)


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

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "WeightCacheDaemonArgs":
        return cls(
            gpu_id=args.gpu_id,
            tp_rank=args.tp_rank,
            pp_rank=args.pp_rank,
            dist_init_method=args.dist_init_method,
            timeout=args.timeout,
            force=args.force,
        )


class WeightCacheDaemon:
    """Persistent GPU weight cache for a single TP rank.

    Holds the complete post-quantization state_dict in GPU memory and
    serves it to engine processes over a Unix socket, through the selected
    transport: CUDA IPC handles (torch_ipc) or CUDA VMM allocations passed as
    file descriptors (vmm_fd).
    """

    # With speculative decoding: the draft model it serves too (draft.py)
    draft: Optional[ServedDraft] = None

    def __init__(
        self,
        server_args: "ServerArgs",
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dist_init_method: Optional[str] = None,
    ):
        self.server_args = server_args
        cfg = resolving_view(server_args)
        self.model_path = cfg.model_path
        self.gpu_id = gpu_id
        self.tp_size = cfg.tp_size
        self.tp_rank = tp_rank
        self.pp_size = cfg.pp_size
        self.pp_rank = pp_rank
        self.ep_size = cfg.ep_size
        self.moe_dp_size = cfg.moe_dp_size
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
        self.transport_name = resolve_daemon_transport(cfg.weight_cache_transport)

        device_uuid = current_platform.get_device_uuid(gpu_id)
        self.socket_path = get_socket_path(device_uuid)
        self.ready_path = get_ready_path(device_uuid)

        self.model = None
        self.config: Optional[CacheConfig] = None
        # storage id -> transport-specific entry of the load-diff storage
        self.state_entries: Dict[str, Dict[str, Any]] = {}
        # What a client installs (data only): the load_diff bytes, every
        # storage's size and device, and the bytes of the CPU storages, which
        # travel in the response (the transports move GPU memory).
        self.manifest: Optional[Dict[str, Any]] = None
        self.preloaded_weights_bytes = 0
        self.transport_backend = None

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

        initialize_model_parallel()

        # Initialize DP attention state (required by some models like Qwen3 MoE)
        from sglang.srt.layers.dp_attention import (
            init_dp_gathered_buffer,
            initialize_dp_attention,
        )

        initialize_dp_attention(server_args)
        init_dp_gathered_buffer(model_config)

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Distributed backend initialized (tp_size={self.tp_size}, "
            f"pp_size={self.pp_size}, "
            f"world_size={self.tp_size * self.pp_size})"
        )

    def load(self):
        """Full loading pipeline: disk → TP shard → quantize → export IPC handles."""
        # Zero-copy IPC weight sharing requires a platform whose tensors can be
        # exported through multiprocessing reduction: CUDA-alike platforms
        # (CUDA / ROCm) via _share_cuda_, and NPU via native torch_npu reduction.
        # Fail loud here instead of dying deep inside the export with an opaque
        # error on anything else.
        if not current_platform.is_cuda_alike() and not current_platform.is_npu():
            raise RuntimeError(
                f"[WeightCacheDaemon] the weight cache daemon requires an "
                f"IPC-capable platform (CUDA, ROCm, or NPU/Ascend) for zero-copy "
                f"weight sharing, but the active platform device type is "
                f"{current_platform.device_type!r}. Disable the weight cache "
                f"(--weight-cache-mode off)."
            )
        # expandable_segments makes torch's caching allocator hand out memory
        # that cannot be exported via _share_cuda_, so the IPC export below would
        # die mid-way with an opaque CUDA error. Fail fast with an actionable
        # message before touching the device. vmm_fd copies the weights into
        # allocations of its own, so it works with either allocator setting.
        if self.transport_name == TORCH_IPC_BACKEND:
            self._assert_ipc_compatible_allocator()
        current_platform.set_device(current_platform.get_device(self.gpu_id))
        # Pick the transport before loading, so an unusable one fails fast.
        self.transport_backend = create_daemon_transport_backend(
            self.transport_name, self.gpu_id
        )

        # Reduce thread contention during multi-process loading
        torch.set_num_threads(1)

        # Lazy imports to avoid circular dependencies and speed up startup
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.loader import get_model_loader

        server_args = self.server_args
        publish(
            server_args,
            role="weight_cache_daemon",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(
                    server_args, tp_rank=self.tp_rank, pp_rank=self.pp_rank
                ),
                gpu_id=self.gpu_id,
            ),
        )

        # With speculative decoding the engine's draft runner fetches its draft
        # from here too (draft.py); refuse a setup it cannot serve before loading.
        from .draft import plan_draft

        plan = plan_draft(server_args)

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

        # The MoE flags as the engine's scheduler seeds them: for a model with
        # MoE layers only. A draft built later in this process reads them, as
        # the engine's draft runner reads its scheduler's.
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.moe.utils import has_moe_layers

        if has_moe_layers(model_config.hf_text_config):
            initialize_moe_config()

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
            dp_size=get_parallel().dp_size,
            ep_size=self.ep_size,
            moe_dp_size=self.moe_dp_size,
            moe_dp_rank=moe_dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_dp_size=get_parallel().attn_dp_size,
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
            load_format=self.load_format,
            model_loader_extra_config=self.model_loader_extra_config,
            tp_rank=self.tp_rank,
        )

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Loading model from disk: {self.model_path}"
        )
        tic = time.perf_counter()

        # Load model using DefaultModelLoader (includes TP sharding + quant
        # post-process), recording each model as built to tell what loading
        # changed (load_diff).
        from sglang.srt.model_loader.loader import on_model_constructed
        from sglang.srt.weight_cache import load_diff

        loader = get_model_loader(load_config=load_config, model_config=model_config)
        records = {}
        with on_model_constructed(
            lambda m: records.__setitem__(id(m), load_diff.record_construction(m))
        ):
            self.model = loader.load_model(
                model_config=model_config,
                device_config=DeviceConfig(current_platform.device_type, self.gpu_id),
            )
        # A throwaway build in the same load is dropped.
        constructed = records.pop(id(self.model), None)
        records.clear()

        elapsed = time.perf_counter() - tic
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Model loaded from disk in {elapsed:.2f}s"
        )

        # Ensure every post-processing kernel has retired before we export the
        # memory: clients map these tensors (and must treat them as read-only)
        # and would otherwise risk observing half-written weights.
        current_platform.synchronize()
        current_platform.empty_cache()
        self.preloaded_weights_bytes = max(
            0, torch.cuda.memory_reserved(self.gpu_id) - memory_before_load
        )

        if plan is not None:
            from sglang.srt.model_executor.model_runner_components.load_model_utils import (
                reserve_rope_cache_for_long_sequences,
            )

            # The target's config as the engine's target runner holds it
            target_model_config = ModelConfig.from_server_args(server_args)
            memory_before_draft = torch.cuda.memory_reserved(self.gpu_id)
            # The engine's target runner grows the RoPE caches after loading,
            # and its draft is built after that, holding the same RoPE module
            # when it asks for the same RoPE (get_rope's process-wide cache).
            # Grow them here too, before the target is described, so that the
            # draft is built as in the engine and the grown table is the one
            # both are served with.
            reserve_rope_cache_for_long_sequences(self.model, target_model_config)

        # Describe the loaded model for clients (load_diff), then export its
        # storages through the transport backend. Nothing but the model and
        # the storage table refers to the originals by then.
        state_tensors = self._snapshot(constructed)
        del constructed
        if plan is not None:
            self._load_draft(
                plan,
                state_tensors,
                target_model_config=target_model_config,
                target_load_config=load_config,
                memory_before=memory_before_draft,
            )
        self._export_state(state_tensors)
        if self.transport_name == VMM_FD_BACKEND:
            # The packed weights live in cuMemCreate allocations, which the
            # caching allocator does not count. Add them to what it still
            # holds (originals that something else kept alive, if any). The
            # draft's runner reports its part of that, the target's the rest.
            current_platform.empty_cache()
            total = (
                max(0, torch.cuda.memory_reserved(self.gpu_id) - memory_before_load)
                + self.transport_backend.exported_bytes
            )
            if self.draft is not None:
                self.draft.preloaded_weights_bytes = min(
                    self.draft.preloaded_weights_bytes, total
                )
                total -= self.draft.preloaded_weights_bytes
            self.preloaded_weights_bytes = total

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Exported {len(self.state_entries)} storages "
            f"(transport={self.transport_name}, "
            f"preloaded_weights_bytes={self.preloaded_weights_bytes})."
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
                        f"engine itself), or use --weight-cache-transport vmm_fd."
                    )

    def _snapshot(self, constructed) -> Dict[str, Tuple[torch.Tensor, bool]]:
        """Describe the loaded model for clients (load_diff.export): every
        tensor it holds as a view of a storage, and every plain value loading
        changed. Keeps the manifest, with the bytes of the storages on the
        CPU, and returns those on this daemon's device, which the transport
        exports, by name. Raises, listing them, when loading left anything a
        client cannot get."""
        self.manifest, storages = self._describe(self.model, constructed)
        return {name: (tensor, False) for name, tensor in storages.items()}

    def _describe(
        self, model, constructed
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """(manifest, the storages on this daemon's device by name) of
        ``model``: see _snapshot."""
        from sglang.srt.weight_cache import load_diff

        if constructed is None:
            raise RuntimeError(
                f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
                f"The model loader did not build the model through "
                f"_initialize_model, so what loading changed is unknown."
            )
        storages, payload = load_diff.export(model=model, constructed=constructed)
        device = torch.device(current_platform.get_device(self.gpu_id))
        manifest = {
            "load_diff": payload,
            "storages": {
                name: [tensor.nbytes, tensor.device.type]
                for name, tensor in storages.items()
            },
            "host": {
                name: bytes(tensor.numpy())
                for name, tensor in storages.items()
                if tensor.device != device
            },
        }
        return manifest, {
            name: tensor for name, tensor in storages.items() if tensor.device == device
        }

    def _load_draft(
        self,
        plan,
        state_tensors: Dict[str, Tuple[torch.Tensor, bool]],
        *,
        target_model_config,
        target_load_config: LoadConfig,
        memory_before: int,
    ) -> None:
        """Load, describe and add to ``state_tensors`` the draft the engine's
        draft runner fetches, as its worker builds it (draft.py), after the
        target has been described. A storage the target exports already is
        not added again: the draft's transport entry for it is the target's."""
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.loader import (
            get_model_loader,
            on_model_constructed,
        )
        from sglang.srt.runtime_context import get_model, get_spec
        from sglang.srt.weight_cache import load_diff

        from .draft import draft_stamp, share_like_engine
        from .ipc_loader import WeightCacheModelLoader

        tag = f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank} draft]"
        spec = get_spec()
        # TpModelWorker._init_model_config of the draft worker
        model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=spec.speculative_draft_model_path,
            model_revision=spec.speculative_draft_model_revision,
            is_draft_model=True,
            context_length=target_model_config.context_len,
        )
        load_format = spec.speculative_draft_load_format
        layout = plan.layout(self.server_args, target_model_config)
        logger.info(
            f"{tag} Loading the draft model "
            f"{model_config.hf_config.architectures} from {model_config.model_path}"
        )
        tic = time.perf_counter()
        with (
            plan.scope(layout),
            # ModelRunner._load_format_scope of the draft runner
            get_model().override(load_format=load_format)
            if load_format is not None
            else contextlib.nullcontext(),
        ):
            # What the engine's draft runner states, before loading changes
            # the quantization config
            quant_method, quant_config = WeightCacheModelLoader._resolve_engine_quant(
                model_config
            )
            check_ipc_quant_support(quant_method, quant_config, where="daemon")
            config = WeightCacheModelLoader.cache_config(model_config)
            stamp = draft_stamp()
            loader = get_model_loader(
                load_config=LoadConfig(
                    load_format=load_format or self.load_format,
                    model_loader_extra_config=self.model_loader_extra_config,
                    # the draft runner's rank, in the draft's TP group
                    tp_rank=get_parallel().tp_rank,
                ),
                model_config=model_config,
            )
            records = {}
            with on_model_constructed(
                lambda m: records.__setitem__(id(m), load_diff.record_construction(m))
            ):
                model = loader.load_model(
                    model_config=model_config,
                    device_config=DeviceConfig(
                        current_platform.device_type, self.gpu_id
                    ),
                )
            constructed = records.pop(id(model), None)
            records.clear()
        current_platform.synchronize()
        step = plan.share_step(
            target_model=self.model,
            draft_model=model,
            target_model_config=target_model_config,
            target_load_config=target_load_config,
        )
        shared = [] if step is None else share_like_engine(self.model, model, step, tag)
        current_platform.synchronize()
        current_platform.empty_cache()
        preloaded_weights_bytes = max(
            0, torch.cuda.memory_reserved(self.gpu_id) - memory_before
        )
        manifest, storages = self._describe(model, constructed)
        del constructed
        exported = {_storage_key(t): name for name, (t, _) in state_tensors.items()}
        names = {}
        for name, tensor in storages.items():
            names[name] = exported.get(_storage_key(tensor))
            if names[name] is None:
                names[name] = exported[_storage_key(tensor)] = f"{DRAFT_ROLE}/{name}"
                state_tensors[names[name]] = (tensor, False)
        self.draft = ServedDraft(
            model=model,
            config=config,
            manifest=manifest,
            names=names,
            stamp=stamp,
            preloaded_weights_bytes=preloaded_weights_bytes,
        )
        logger.info(
            f"{tag} Draft model loaded in {time.perf_counter() - tic:.2f}s: "
            f"{preloaded_weights_bytes / (1 << 30):.2f} GiB, "
            f"{sum(not n.startswith(DRAFT_ROLE) for n in names.values())} of its "
            f"{len(names)} storages are the target's; {len(shared)} tensor(s) "
            f"hold the target's data {shared}"
        )

    def _export_state(self, state_tensors: Dict[str, Tuple[torch.Tensor, bool]]):
        """Export the snapshot's storages (the target's, and its draft's)
        through the selected transport backend, which consumes
        ``state_tensors``."""
        self.state_entries.clear()

        if self.transport_name == VMM_FD_BACKEND:
            # Packing consumes detached storage aliases. The daemon never runs
            # the model, so release its object graph before moving any bytes.
            self.model = None
            if self.draft is not None:
                self.draft.model = None
            from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
            from sglang.srt.models.transformers import _TRANSFORMERS_MOE_LAYERS

            # This process-wide cache also owns model buffers. The daemon has
            # built its models (the target, and a draft) and builds no more.
            _ROPE_DICT.clear()
            # The Transformers backend registers every MoE layer in this dict,
            # which would keep the experts' original weights alive next to
            # their packed copies.
            _TRANSFORMERS_MOE_LAYERS.clear()
            gc.collect()
        entries = self.transport_backend.prepare_export(state_tensors)
        host = self.manifest["host"]
        self.state_entries = {
            name: entries[name]
            for name in self.manifest["storages"]
            if name not in host
        }
        if self.draft is not None:
            self.draft.entries = {
                name: entries[as_name] for name, as_name in self.draft.names.items()
            }

        if self.transport_name == VMM_FD_BACKEND:
            # vmm_fd entries carry no handle blob: a client maps the packed
            # allocations, one fd each.
            exported = (
                f"{self.transport_backend.num_allocations} allocation(s), "
                f"{self.transport_backend.exported_bytes / (1 << 30):.2f} GiB"
            )
        else:
            # Approximate serialized metadata size (not payload-backed bytes).
            # Only the handle blob carries real weight, so measure it directly:
            # stringifying every entry would allocate a copy of all handles.
            total_bytes = sum(
                len(handle)
                for handle in (
                    entry.get("handle") for entry in self.state_entries.values()
                )
                if isinstance(handle, (str, bytes, bytearray))
            )
            exported = f"metadata size ~{total_bytes / 1024 / 1024:.1f} MB"
        # A response that a client would refuse to read must fail here, not
        # on every client. The rest of the response (pid, version, sizes)
        # fits in the 1 MiB left over.
        sizes = []
        for role in (TARGET_ROLE, DRAFT_ROLE) if self.draft else (TARGET_ROLE,):
            config, entries, manifest, _ = self._served(role)
            size = len(
                pickle.dumps(
                    {
                        "config": config.to_dict(),
                        "entries": entries,
                        "manifest": manifest,
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            )
            if size > MAX_MSG_SIZE - (1 << 20):
                raise RuntimeError(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] A fetch_state response "
                    f"for the {role} would take {size} bytes, more than the "
                    f"{MAX_MSG_SIZE}-byte cap."
                )
            sizes.append(f"{size / (1 << 20):.1f} MiB")
        draft = (
            ""
            if self.draft is None
            else f" and {len(self.draft.entries)} of the draft "
            f"({len(set(self.draft.names.values()) - set(self.state_entries))} its own)"
        )
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] "
            f"Exported {len(self.state_entries)} storages of the target{draft}, "
            f"transport={self.transport_backend.name}, {exported}, "
            f"response {' / '.join(sizes)}"
        )

    def _served(self, role: str):
        """(config, entries, manifest, preloaded weight bytes) of the model
        served under ``role``; None when this daemon serves no such model."""
        if role == TARGET_ROLE:
            return (
                self.config,
                self.state_entries,
                self.manifest,
                self.preloaded_weights_bytes,
            )
        if role == DRAFT_ROLE and self.draft is not None:
            draft = self.draft
            return (
                draft.config,
                draft.entries,
                draft.manifest,
                draft.preloaded_weights_bytes,
            )
        return None

    def _initialize_eplb_expert_location_metadata(self, model_config) -> None:
        """Build the same initial physical expert layout as the engine."""
        if not get_exec().moe.enable_eplb:
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

        try:
            if req.get("type") == "handshake":
                check_protocol_version(req.get("protocol_version"))
                send_msg(conn, {"status": "ok", "protocol_version": PROTOCOL_VERSION})
                req = recv_msg(conn)
            if req.get("type") == "fetch_state":
                # Before the config is read or any handle or fd exported: a
                # client of another version would misread the response.
                check_protocol_version(req.get("protocol_version"))
        except ProtocolVersionError as e:
            send_msg(
                conn,
                {
                    "status": "protocol_mismatch",
                    "protocol_version": PROTOCOL_VERSION,
                    "message": str(e),
                },
            )
            return

        # Which model: a request without a role is for the target
        role = req.get("role", TARGET_ROLE)
        served = None
        if req.get("type") in ("query_config", "fetch_state"):
            served = self._served(role)
            if served is None:
                send_msg(
                    conn,
                    {
                        "status": "error",
                        "protocol_version": PROTOCOL_VERSION,
                        "message": (
                            f"this daemon serves "
                            f"{[TARGET_ROLE, *([DRAFT_ROLE] if self.draft else [])]}, "
                            f"not {role!r}. A daemon serves the speculative "
                            f"decoding draft when it runs with the engine's "
                            f"speculative decoding settings."
                        ),
                    },
                )
                return
            config, entries, manifest, preloaded_weights_bytes = served

        if req.get("type") == "query_config":
            # Client asks for config without requesting handles
            send_msg(
                conn,
                {
                    "status": "ok",
                    "protocol_version": PROTOCOL_VERSION,
                    "config": config.to_dict(),
                    "role": role,
                },
            )

        elif req.get("type") == "fetch_state":
            # Client requests full state with IPC handles
            engine_config = CacheConfig.from_dict(req["config"])
            if not config.matches(engine_config):
                # Log detailed mismatch info for debugging
                daemon_dict = config.to_dict()
                engine_dict = engine_config.to_dict()
                mismatches = {
                    k: (daemon_dict.get(k), engine_dict.get(k))
                    for k in daemon_dict
                    if daemon_dict.get(k) != engine_dict.get(k)
                }
                logger.warning(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] "
                    f"Config mismatch ({role}): {mismatches}"
                )
                send_msg(
                    conn,
                    {
                        "status": "mismatch",
                        "protocol_version": PROTOCOL_VERSION,
                        "daemon_config": config.to_dict(),
                    },
                )
                return

            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Serving {len(entries)} storages of the {role} via "
                f"{self.transport_backend.name} transport"
            )
            self.transport_backend.send_fetch_state_response(
                conn,
                config=config.to_dict(),
                entries=entries,
                # PID so the client can watch daemon liveness: with
                # torch_ipc, if this process dies while clients hold IPC
                # mappings, their param.data (and any CUDA-graph-captured
                # addresses) dangle.
                pid=os.getpid(),
                preloaded_weights_bytes=preloaded_weights_bytes,
                extra={
                    "protocol_version": PROTOCOL_VERSION,
                    "manifest": manifest,
                    "role": role,
                    **({"draft_stamp": self.draft.stamp} if role == DRAFT_ROLE else {}),
                },
            )

        elif req.get("type") == "ping":
            send_msg(conn, {"status": "ok", "protocol_version": PROTOCOL_VERSION})

        else:
            send_msg(
                conn,
                {
                    "status": "error",
                    "protocol_version": PROTOCOL_VERSION,
                    "message": f"Unknown request type: {req.get('type')}",
                },
            )

    def shutdown(self):
        """Release GPU memory and clean up."""
        if dist.is_initialized():
            dist.destroy_process_group()
        if self.model is not None:
            del self.model
            self.model = None
        self.draft = None
        self.state_entries.clear()
        self.manifest = None
        if self.transport_backend is not None:
            # Release any model references before closing transport resources.
            self.transport_backend.close()
            self.transport_backend = None
        current_platform.empty_cache()
        self._running = False


def run_weight_cache_daemon(
    server_args: "ServerArgs",
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dist_init_method: Optional[str] = None,
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
    )

    daemon.load()
    _await_cluster_ready_to_serve(
        timeout=resolving_view(server_args).weight_cache_timeout
    )
    daemon.serve()


def _await_cluster_ready_to_serve(timeout: int) -> None:
    if timeout <= 0:
        # Gloo reads timedelta(0) as its wait-forever sentinel.
        raise ValueError(f"--weight-cache-timeout must be positive, got {timeout}")

    try:
        # Gate publication on the full PP×TP daemon world.
        dist.monitored_barrier(
            group=get_parallel().world_group.cpu_group,
            timeout=datetime.timedelta(seconds=timeout),
            wait_all_ranks=True,
        )
    except RuntimeError:
        logger.exception(
            "[weight_cache] cluster readiness barrier failed; exiting without serving."
        )
        # Process-group teardown can block after a failed collective.
        os._exit(1)


def spawn_weight_cache_daemon(
    server_args: "ServerArgs",
    *,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dist_init_method: str,
):
    """Start one daemon from the complete resolved server configuration."""
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=run_weight_cache_daemon,
        args=(server_args, gpu_id, tp_rank, pp_rank, dist_init_method),
    )
    proc.start()
    return proc


def launch_weight_cache_daemons(
    server_args: "ServerArgs",
    dist_init_method: Optional[str] = None,
    timeout: int = 1800,
    force: bool = False,
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

    # Validate and clean up stale .ready/.sock files from prior runs.
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
            cleanup_stale_daemon_files(
                current_platform.get_device_uuid(gpu_id), force=force
            )

    procs = []
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
            proc = spawn_weight_cache_daemon(
                server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                dist_init_method=dist_init_method,
            )
            procs.append(proc)
            logger.info(
                f"Launched weight cache daemon gpu={gpu_id} "
                f"pp_rank={pp_rank} tp_rank={tp_rank} pid={proc.pid}"
            )

    # Wait for all daemons on this node to become ready
    num_daemons = len(procs)
    check_interval = 2
    start_time = time.time()
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
            ready_path = get_ready_path(current_platform.get_device_uuid(gpu_id))
            while not os.path.exists(ready_path):
                time.sleep(check_interval)
                if time.time() - start_time > timeout:
                    logger.error(
                        f"Weight cache daemon pp_rank={pp_rank} tp_rank={tp_rank} "
                        f"did not become ready within {timeout}s"
                    )
                    for p in procs:
                        p.terminate()
                    raise TimeoutError(
                        f"Weight cache daemon pp_rank={pp_rank} tp_rank={tp_rank} "
                        f"did not become ready within {timeout}s"
                    )
                # Check if any daemon exited prematurely
                for p in procs:
                    if not p.is_alive():
                        logger.error(
                            f"Weight cache daemon exited prematurely "
                            f"with code {p.exitcode}"
                        )
                        for other in procs:
                            if other.is_alive():
                                other.terminate()
                        raise RuntimeError(
                            f"Weight cache daemon exited prematurely "
                            f"with code {p.exitcode}"
                        )
            logger.info(
                f"Weight cache daemon pp_rank={pp_rank} tp_rank={tp_rank} is ready"
            )

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
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
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
        cleanup_stale_daemon_files(
            current_platform.get_device_uuid(gpu_id),
            force=daemon_args.force,
        )
        run_weight_cache_daemon(
            server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            pp_rank=daemon_args.pp_rank,
            dist_init_method=daemon_args.dist_init_method,
        )
    else:
        launch_weight_cache_daemons(
            server_args,
            dist_init_method=daemon_args.dist_init_method,
            timeout=daemon_args.timeout,
            force=daemon_args.force,
        )
