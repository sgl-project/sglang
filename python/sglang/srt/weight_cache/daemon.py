# SPDX-License-Identifier: Apache-2.0
"""Weight Cache Daemon — a persistent process that holds post-quantized,
TP-sharded model weights in GPU memory and serves them via pluggable transport backends.

Each GPU runs one daemon process for its TP rank. The daemon:
1. Loads model weights from disk (full pipeline: disk → TP shard → quantize)
2. Exports every parameter/buffer as a CUDA IPC handle
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
import hmac
import logging
import multiprocessing
import os
import select
import signal
import socket
import stat
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, publish

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
from .seed import (
    PEER_IPC_SEED_SOURCE,
    RDMA_SEED_SOURCE,
    build_manifest,
    get_seed_source,
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

# CacheConfig fields a mirror daemon cannot derive on its own: they are read off
# initialized process groups, which a mirror never creates. Everything else is
# verified against the source before its config is adopted.
_MIRROR_UNVERIFIABLE_FIELDS = frozenset({"moe_dp_rank", "moe_ep_rank"})


def _split_addresses(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [addr.strip() for addr in value.split(",") if addr.strip()]


def _resolve_seed_address(server_args: "ServerArgs", global_rank: int) -> Optional[str]:
    """Address of the source daemon this rank mirrors from, if any.

    The list is indexed by global rank because rank i's shard is byte-identical
    only to the source replica's rank i (see compute_global_rank). Indexing that
    way also means the identical flag value can be handed to every node.
    """
    addresses = _split_addresses(server_args.weight_cache_seed)
    if not addresses:
        return None
    if global_rank >= len(addresses):
        raise ValueError(
            f"[WeightCacheDaemon] --weight-cache-seed lists "
            f"{len(addresses)} address(es) but this daemon is global rank "
            f"{global_rank}. Pass one address per rank of the source replica."
        )
    return addresses[global_rank]


def _resolve_listen_address(
    server_args: "ServerArgs", global_rank: int
) -> Optional[Tuple[str, int]]:
    """TCP control-plane endpoint for this rank, if cross-node seeding is on.

    The configured port is a base: rank i listens on base + i, mirroring how the
    seed list is indexed, so one flag value works for a whole replica.
    """
    if not server_args.weight_cache_listen_addr:
        return None
    host, _, port = server_args.weight_cache_listen_addr.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(
            f"[WeightCacheDaemon] --weight-cache-listen-addr must be "
            f"host:base_port, got {server_args.weight_cache_listen_addr!r}"
        )
    return host, int(port) + global_rank


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
    serves CUDA IPC handles to engine processes via Unix socket.
    """

    def __init__(
        self,
        server_args: "ServerArgs",
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dist_init_method: Optional[str] = None,
    ):
        self.server_args = server_args
        self.model_path = server_args.model_path
        self.gpu_id = gpu_id
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.pp_size = server_args.pp_size
        self.pp_rank = pp_rank
        self.dp_size = server_args.dp_size
        self.ep_size = server_args.ep_size
        self.moe_dp_size = server_args.moe_dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        self.enable_dp_lm_head = server_args.enable_dp_lm_head
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dense_tp_size = server_args.moe_dense_tp_size
        self.moe_a2a_backend = server_args.moe_a2a_backend
        self.deepep_mode = server_args.deepep_mode
        self.load_format = server_args.load_format
        self.dtype = server_args.dtype
        self.quantization = server_args.quantization
        self.model_loader_extra_config = server_args.model_loader_extra_config
        self.trust_remote_code = server_args.trust_remote_code
        self.revision = server_args.revision
        self.dist_init_method = dist_init_method

        self.socket_path = get_socket_path(gpu_id)
        self.ready_path = get_ready_path(gpu_id)

        self.global_rank = compute_global_rank(self.tp_size, pp_rank, tp_rank)
        # When set, this daemon is a *mirror*: it copies an existing daemon's
        # already-post-processed weights instead of loading from disk.
        self.seed_addr = _resolve_seed_address(server_args, self.global_rank)
        self.seed_backend = server_args.weight_cache_seed_backend
        self.seed_token = server_args.weight_cache_seed_token
        self.listen_address = _resolve_listen_address(server_args, self.global_rank)

        self.model = None
        self.config: Optional[CacheConfig] = None
        # name -> transport-specific tensor entry metadata (shape/dtype/is_param + payload metadata)
        self.state_entries: Dict[str, Dict[str, Any]] = {}
        # The tensors behind state_entries. On the disk path self.model owns them
        # too; on the mirror path this dict is their ONLY owner, so dropping it
        # would free memory that clients have already mapped.
        self.state_tensors: Dict[str, Tuple[torch.Tensor, bool]] = {}
        self.transport_backend = None
        # Built on first fetch_manifest, not at load time: a seed source can be
        # expensive to stand up (RDMA memory registration takes seconds) and
        # most daemons are never used as a seed.
        self._manifest: Optional[Dict[str, Dict[str, Any]]] = None
        self._seed_sources: Dict[str, Any] = {}

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
                moe_a2a_backend=server_args.moe_a2a_backend,
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
        """Populate this daemon's weights, then export them for clients.

        Two sources: disk (full pipeline disk -> TP shard -> quantize) or, when
        ``--weight-cache-seed`` is set, a peer daemon that already did all of
        that (see _load_from_seed).
        """
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
        # message before touching the device. A mirror re-exports its own copy,
        # so this applies to it just the same.
        self._assert_ipc_compatible_allocator()
        current_platform.set_device(current_platform.get_device(self.gpu_id))

        # Reduce thread contention during multi-process loading
        torch.set_num_threads(1)

        # Lazy imports to avoid circular dependencies and speed up startup
        from sglang.srt.configs.model_config import ModelConfig

        server_args = self.server_args
        publish(server_args, role="weight_cache_daemon")

        if self.seed_addr is None:
            # Only the disk path instantiates layers, so only it needs the MoE
            # config resolved.
            from sglang.srt.layers.moe import initialize_moe_config

            initialize_moe_config(server_args)

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

        # Refuse unsupported quant methods before creating distributed groups
        # or touching model weights. A mirror re-exports over IPC too, so the
        # allowlist is not relaxed for it.
        check_ipc_quant_support(quant_method, quant_config, where="daemon")

        if self.seed_addr is not None:
            self._load_from_seed(model_config, quant_method, quant_config)
        else:
            self._load_from_disk(model_config, quant_method, quant_config)

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Exported {len(self.state_entries)} tensors as IPC handles. "
            f"Ready to serve."
        )

    def _load_from_disk(self, model_config, quant_method, quant_config):
        """Full loading pipeline: disk -> TP shard -> quantize -> export."""
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.model_loader.loader import get_model_loader

        server_args = self.server_args

        # The initialized groups are the authority for rank identity. This
        # avoids maintaining a second copy of the model-parallel hierarchy.
        self._init_distributed(server_args, model_config)
        moe_dp_rank = get_parallel().moe_dp_rank
        moe_ep_rank = get_parallel().moe_ep_rank
        self.config = CacheConfig(
            **self._fingerprint_fields(model_config, quant_method, quant_config),
            moe_dp_rank=moe_dp_rank,
            moe_ep_rank=moe_ep_rank,
        )

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

        # Load model using DefaultModelLoader (includes TP sharding + quant post-process)
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

    def _fingerprint_fields(
        self, model_config, quant_method, quant_config
    ) -> Dict[str, Any]:
        """Every CacheConfig field derivable without initialized process groups.

        Shared by the disk path (which adds the two group-derived MoE ranks) and
        by a mirror's verification of the source's config, so the two can never
        compute the fingerprint differently.
        """
        return dict(
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

    def _load_from_seed(self, model_config, quant_method, quant_config):
        """Mirror a peer daemon's already post-processed weights.

        The peer has done the expensive part -- disk read, TP sharding and
        ``process_weights_after_loading`` -- and its shard is byte-identical to
        ours as long as the fingerprint matches, so loading collapses into a
        bandwidth copy. Consequently this path never builds an ``nn.Module`` and
        never creates a process group: _init_distributed, initialize_moe_config,
        initialize_dp_attention and the model loader are all skipped.

        The copy leaves us owning private memory, so once it returns the source
        daemon may die freely -- unlike an engine client, which maps the daemon's
        memory and must watch its liveness.
        """
        source = get_seed_source(
            self.seed_backend, **self._seed_source_kwargs_for(self.seed_backend)
        )

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Mirroring weights from {self.seed_addr} "
            f"via seed backend {self.seed_backend}"
        )
        tic = time.perf_counter()

        conn = self._connect_seed()
        try:
            request = {"type": "fetch_manifest", "seed_backend": self.seed_backend}
            if self.seed_token:
                request["token"] = self.seed_token
            send_msg(conn, request)
            # The source may have to stand up an RDMA registration covering the
            # whole model before it can answer, which outlasts the handshake-sized
            # default timeout.
            conn.settimeout(
                max(self.server_args.weight_cache_timeout, CLIENT_CONNECTION_TIMEOUT)
            )
            response = recv_msg(conn)
        finally:
            conn.close()

        if response.get("status") != "ok":
            raise RuntimeError(
                f"[WeightCacheDaemon gpu={self.gpu_id}] source daemon at "
                f"{self.seed_addr} refused to seed: "
                f"{response.get('message', response)}"
            )

        source_config = response["config"]
        self._verify_seed_config(
            source_config, model_config, quant_method, quant_config
        )
        # moe_dp_rank / moe_ep_rank are the only two fields left. They are
        # deterministic functions of the parallel *size* fields plus
        # tp_rank/pp_rank -- every one of which _verify_seed_config just proved
        # equal -- so adopting the source's values is sound, not an unchecked
        # hole. Recomputing them would require the process groups this path
        # exists to avoid.
        self.config = CacheConfig.from_dict(source_config)

        manifest = response["manifest"]
        seed_pid = response.get("pid")
        self._assert_seed_alive(seed_pid, when="before starting the copy")

        self.state_tensors = {}
        tensors = source.fill(
            manifest,
            response["seed"],
            current_platform.get_device(self.gpu_id),
        )
        # The source's memory had to stay valid for the whole copy; if it exited
        # partway we may have read freed memory, and there is no way to tell
        # good bytes from garbage after the fact. Refuse to serve.
        self._assert_seed_alive(seed_pid, when="after the copy completed")

        missing = set(manifest) - set(tensors)
        if missing:
            raise RuntimeError(
                f"[WeightCacheDaemon gpu={self.gpu_id}] seed backend "
                f"{self.seed_backend} did not fill {len(missing)} manifest "
                f"entry(ies) (e.g. {sorted(missing)[:3]})."
            )

        elapsed = time.perf_counter() - tic
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id} tp_rank={self.tp_rank}] "
            f"Mirrored {len(tensors)} tensors from {self.seed_addr} "
            f"in {elapsed:.2f}s (no disk read, no TP sharding, no post-load "
            f"processing)"
        )

        self._export_tensors(
            {name: (tensors[name], manifest[name]["is_param"]) for name in manifest}
        )

    def _connect_seed(self) -> socket.socket:
        """Connect to the source daemon named by ``--weight-cache-seed``.

        A path-like address is the node-local Unix socket (same-node seeding);
        anything else is host:port on the source's TCP control plane.
        """
        addr = self.seed_addr
        if addr.startswith("/") or addr.startswith("./"):
            # Same ownership check the engine client applies: only talk to a real
            # socket node owned by us, never a symlink or another user's socket
            # planted at a /tmp path.
            try:
                st = os.lstat(addr)
            except FileNotFoundError:
                raise RuntimeError(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] no seed socket at "
                    f"{addr}. Start the source daemon first and point "
                    f"--weight-cache-seed at its socket."
                )
            if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
                raise RuntimeError(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] refusing to seed "
                    f"from {addr}: not a socket owned by this user."
                )
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            target: Any = addr
        else:
            host, _, port = addr.rpartition(":")
            if not host or not port.isdigit():
                raise RuntimeError(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] "
                    f"--weight-cache-seed entry {addr!r} is neither an absolute "
                    f"socket path nor host:port."
                )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target = (host, int(port))

        sock.settimeout(CLIENT_CONNECTION_TIMEOUT)
        try:
            sock.connect(target)
        except Exception as e:
            sock.close()
            raise RuntimeError(
                f"[WeightCacheDaemon gpu={self.gpu_id}] failed to connect to "
                f"seed daemon at {addr}: {e}"
            ) from e
        return sock

    def _verify_seed_config(
        self, source_config: Dict[str, Any], model_config, quant_method, quant_config
    ) -> None:
        """Reject a source whose weights would not be byte-compatible with ours.

        Only same-shape replicas can seed each other: an identical shard is
        produced only by an identical (model, revision, dtype, quantization,
        parallel layout) tuple. Every such field is computable here without
        process groups, so compare them all and fail loud on any difference.
        """
        expected = self._fingerprint_fields(model_config, quant_method, quant_config)
        mismatches = {
            key: (source_config.get(key), value)
            for key, value in expected.items()
            if key not in _MIRROR_UNVERIFIABLE_FIELDS
            and source_config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"[WeightCacheDaemon gpu={self.gpu_id}] cannot mirror "
                f"{self.seed_addr}: its weights were built with a different "
                f"configuration, so its shard is not our shard. Differing "
                f"fields (source, ours): {mismatches}"
            )

        unknown = set(CacheConfig.__struct_fields__) - set(source_config)
        if unknown:
            raise RuntimeError(
                f"[WeightCacheDaemon gpu={self.gpu_id}] source daemon at "
                f"{self.seed_addr} sent a config missing {sorted(unknown)}; it "
                f"runs an incompatible SGLang version."
            )

    def _assert_seed_alive(self, seed_pid: Optional[int], *, when: str) -> None:
        """The source process must exist for its memory to be readable.

        Deliberately NOT the engine client's watchdog: that one SIGKILLs itself
        when the daemon dies because it *maps* the daemon's memory forever. A
        mirror only needs the source alive until the copy retires.
        """
        if not seed_pid or seed_pid <= 0:
            logger.warning(
                "[WeightCacheDaemon gpu=%s] seed daemon at %s reported no PID; "
                "cannot verify it stayed alive %s.",
                self.gpu_id,
                self.seed_addr,
                when,
            )
            return
        try:
            os.kill(seed_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(
                f"[WeightCacheDaemon gpu={self.gpu_id}] seed daemon "
                f"(pid={seed_pid}) at {self.seed_addr} is gone {when}. Its "
                f"weight memory was freed, so the bytes we read cannot be "
                f"trusted. Refusing to serve them."
            )
        except PermissionError:
            pass  # exists, owned by another user

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
        """Export the loaded model's state entries through the transport backend."""
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

        self._export_tensors(state_tensors, non_persistent_count=non_persistent_count)

    def _export_tensors(
        self,
        state_tensors: Dict[str, Tuple[torch.Tensor, bool]],
        *,
        non_persistent_count: Optional[int] = None,
    ):
        """Export an explicit tensor set, whether it came from disk or a peer."""
        self.state_entries.clear()
        self.state_tensors = state_tensors

        self.transport_backend = choose_daemon_transport_backend(state_tensors)
        self.state_entries = self.transport_backend.prepare_export(state_tensors)

        # Log approximate serialized metadata size (not payload-backed bytes).
        # Only the handle blob carries real weight, so measure it directly:
        # stringifying every entry would allocate a copy of all handles.
        total_bytes = sum(
            len(handle)
            for handle in (entry.get("handle") for entry in self.state_entries.values())
            if isinstance(handle, (str, bytes, bytearray))
        )
        buffers_note = (
            f"({non_persistent_count} non-persistent buffers), "
            if non_persistent_count is not None
            else ""
        )
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] "
            f"Exported {len(self.state_entries)} tensors "
            f"{buffers_note}"
            f"transport={self.transport_backend.name}, "
            f"metadata size ~{total_bytes / 1024 / 1024:.1f} MB"
        )

    def serve(self):
        """Block and serve transport entries over Unix socket (and optionally TCP)."""
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

        tcp_sock = self._maybe_listen_tcp()
        listeners = [sock] + ([tcp_sock] if tcp_sock is not None else [])

        # Write ready file
        with open(self.ready_path, "w") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"config={self.config.to_dict()}\n")

        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] " f"Listening on {self.socket_path}"
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
                # select rather than a per-socket accept timeout: with two
                # listeners, polling them in turn would add a full timeout of
                # latency to whichever one is checked second.
                try:
                    readable, _, _ = select.select(listeners, [], [], 1.0)
                except InterruptedError:
                    continue
                for listener in readable:
                    conn, peer = listener.accept()
                    # The select timeout above only bounds accept(); the accepted
                    # connection is blocking by default. Since we serve
                    # connections serially, a client that connects but never sends
                    # (or dies mid-send) would block recv_msg forever and stall
                    # every other engine rank. Bound each exchange instead.
                    conn.settimeout(CLIENT_CONNECTION_TIMEOUT)
                    try:
                        self._handle_connection(conn, remote=listener is tcp_sock)
                    except Exception as e:
                        logger.error(
                            f"[WeightCacheDaemon gpu={self.gpu_id}] "
                            f"Error handling connection from {peer}: {e}",
                            exc_info=True,
                        )
                    finally:
                        conn.close()
        finally:
            sock.close()
            if tcp_sock is not None:
                tcp_sock.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if os.path.exists(self.ready_path):
                os.unlink(self.ready_path)
            logger.info(f"[WeightCacheDaemon gpu={self.gpu_id}] Shutdown complete")

    def _maybe_listen_tcp(self) -> Optional[socket.socket]:
        """Open the cross-node control plane, if configured.

        A daemon must be able to act as a seed without any engine attached, so
        this is its own listener rather than a route on the engine's bootstrap
        server. It is deliberately narrower than the Unix socket: the peer-uid
        check that protects /tmp has no cross-machine equivalent, so a shared
        token is mandatory and only seeding requests are answered.
        """
        if self.listen_address is None:
            return None

        host, port = self.listen_address
        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_sock.bind((host, port))
        tcp_sock.listen(8)
        logger.info(
            f"[WeightCacheDaemon gpu={self.gpu_id}] Cross-node seed control "
            f"plane listening on {host}:{port}"
        )
        return tcp_sock

    def _authorize_remote(self, req: Dict[str, Any]) -> Optional[str]:
        """Return an error message if a TCP request must be rejected."""
        if not self.seed_token:
            return (
                "this daemon exposes a TCP control plane but has no "
                "--weight-cache-seed-token configured, so it cannot "
                "authenticate remote peers"
            )
        presented = req.get("token")
        if not isinstance(presented, str) or not hmac.compare_digest(
            presented, self.seed_token
        ):
            return "invalid or missing seed token"
        if req.get("type") != "fetch_manifest":
            # CUDA IPC handles are meaningful only within a node, and
            # query_config leaks the model layout; the remote plane exists purely
            # so another node can mirror our weights.
            return (
                f"request type {req.get('type')!r} is not served over the "
                f"cross-node control plane; only fetch_manifest is"
            )
        return None

    def _get_seed_source(self, backend: str):
        """Seed sources are built once, on first use, and cached.

        Standing one up can be expensive (the RDMA source registers every weight
        buffer with the NIC), so a daemon nobody seeds from never pays for it.
        """
        source = self._seed_sources.get(backend)
        if source is None:
            source = get_seed_source(backend, **self._seed_source_kwargs_for(backend))
            self._seed_sources[backend] = source
        return source

    def _seed_source_kwargs_for(self, backend: str) -> Dict[str, Any]:
        if backend == RDMA_SEED_SOURCE:
            # Reuse the HCA selection the rest of SGLang already exposes rather
            # than adding a weight-cache-only copy of the same knob.
            return {
                "ib_device": (
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                )
            }
        return {}

    def _handle_connection(self, conn: socket.socket, *, remote: bool = False):
        """Handle a single client connection.

        ``remote`` marks a connection from the cross-node TCP plane, which is
        authenticated and restricted to seeding (see _authorize_remote).
        """
        req = recv_msg(conn)

        if remote:
            denial = self._authorize_remote(req)
            if denial is not None:
                logger.warning(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] rejected remote "
                    f"request: {denial}"
                )
                send_msg(conn, {"status": "error", "message": denial})
                return

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
            )

        elif req.get("type") == "fetch_manifest":
            # A peer daemon wants to mirror our weights. Unlike fetch_state this
            # has copy semantics: we describe every tensor we hold and publish
            # whatever the requested mover needs, and the peer ends up owning its
            # own memory. Note the manifest covers the full export set, including
            # post-quant params and non-persistent buffers -- that is exactly what
            # lets the mirror skip building a model.
            backend = req.get("seed_backend") or PEER_IPC_SEED_SOURCE
            try:
                source = self._get_seed_source(backend)
                if self._manifest is None:
                    self._manifest = build_manifest(self.state_tensors)
                seed_meta = source.prepare_seed(
                    self.state_tensors,
                    transport_entries=self.state_entries,
                    gpu_id=self.gpu_id,
                )
            except Exception as e:
                logger.error(
                    f"[WeightCacheDaemon gpu={self.gpu_id}] Failed to prepare "
                    f"seed via {backend}: {e}",
                    exc_info=True,
                )
                send_msg(conn, {"status": "error", "message": str(e)})
                return

            logger.info(
                f"[WeightCacheDaemon gpu={self.gpu_id}] "
                f"Seeding {len(self._manifest)} tensors to a mirror daemon via "
                f"{backend}"
            )
            send_msg(
                conn,
                {
                    "status": "ok",
                    "config": self.config.to_dict(),
                    # The mirror checks this PID stayed alive across the copy:
                    # our memory has to exist for the whole transfer.
                    "pid": os.getpid(),
                    "manifest": self._manifest,
                    "seed": seed_meta,
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
        """Release GPU memory and clean up."""
        if dist.is_initialized():
            dist.destroy_process_group()
        for source in self._seed_sources.values():
            source.close()
        self._seed_sources.clear()
        if self.model is not None:
            del self.model
            self.model = None
        self.state_entries.clear()
        # On the mirror path this is the only reference to the weights, so it
        # must be dropped for the memory to actually be released.
        self.state_tensors.clear()
        self._manifest = None
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
    daemon.serve()


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
    import socket as sock_mod

    # Replicate _calculate_rank_ranges logic from engine.py
    pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
    nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
    )
    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
        tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
    )

    if server_args.nnodes > 1 and dist_init_method is None:
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
            cleanup_stale_daemon_files(
                compute_local_gpu_id(
                    pp_rank,
                    tp_rank,
                    pp_size_per_node,
                    tp_size_per_node,
                    base_gpu_id=server_args.base_gpu_id,
                    gpu_id_step=server_args.gpu_id_step,
                ),
                force=force,
            )

    procs = []
    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            gpu_id = compute_local_gpu_id(
                pp_rank,
                tp_rank,
                pp_size_per_node,
                tp_size_per_node,
                base_gpu_id=server_args.base_gpu_id,
                gpu_id_step=server_args.gpu_id_step,
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
                base_gpu_id=server_args.base_gpu_id,
                gpu_id_step=server_args.gpu_id_step,
            )
            ready_path = get_ready_path(gpu_id)
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
        f"All {num_daemons} weight cache daemons on node {server_args.node_rank} are ready "
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
        cleanup_stale_daemon_files(gpu_id, force=daemon_args.force)
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
