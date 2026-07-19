from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress, get_free_port, get_local_ip_auto

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Module-level shared engine instance, set by init_mooncake_transfer_engine().
_mooncake_transfer_engine: Optional[MooncakeTransferEngine] = None


def parse_ib_device_config(
    ib_device_str: Optional[str],
) -> Optional[Union[str, Dict[int, str]]]:
    """Parse IB device config from a shared string, JSON mapping, or JSON file."""
    if ib_device_str is None or not ib_device_str.strip():
        return None

    normalized_input = ib_device_str.strip()
    if not normalized_input.endswith(".json") and not normalized_input.startswith("{"):
        return normalized_input

    if normalized_input.endswith(".json"):
        if not os.path.isfile(normalized_input):
            raise RuntimeError(f"File {normalized_input} does not exist.")
        try:
            with open(normalized_input, "r", encoding="utf-8") as file:
                mapping = json.load(file)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse JSON content from file {normalized_input}"
            ) from exc
        except (IOError, OSError) as exc:
            raise RuntimeError(
                f"Failed to read JSON file {normalized_input}: {exc}"
            ) from exc
    else:
        try:
            mapping = json.loads(normalized_input)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON mapping: {normalized_input}") from exc

    if not isinstance(mapping, dict):
        raise ValueError(
            "Invalid format: expected a mapping from GPU id to IB device string"
        )

    normalized_mapping: Dict[int, str] = {}
    for gpu_key, ib_devices in mapping.items():
        normalized_key = int(gpu_key) if str(gpu_key).isdigit() else None
        if normalized_key is None or not isinstance(ib_devices, str):
            raise ValueError(
                "Invalid format: keys must be integers (or string "
                "representations of integers) and values must be strings"
            )
        normalized_mapping[normalized_key] = ib_devices.strip()

    if not normalized_mapping:
        raise ValueError("No valid GPU mappings found in JSON")

    return normalized_mapping


def get_ib_devices_for_gpu(ib_device_str: Optional[str], gpu_id: int) -> Optional[str]:
    """
    Parse IB device string and get IB devices for a specific GPU ID.

    Supports all the following formats:
    1. Old format: "ib0, ib1, ib2"
    2. New format: {0: "ib0, ib1", 1: "ib2, ib3", 2: "ib4"}
    3. JSON file: path to a JSON file containing the mapping

    Args:
        ib_device_str: The original IB device string or path to JSON file
        gpu_id: The GPU ID to get devices for

    Returns:
        IB devices string for the GPU, or None if not available
    """
    parsed_config = parse_ib_device_config(ib_device_str)
    if parsed_config is None:
        return None

    if isinstance(parsed_config, str):
        return parsed_config

    if gpu_id in parsed_config:
        return parsed_config[gpu_id]

    raise ValueError(
        f"No IB devices configured for GPU {gpu_id}. "
        f"Available GPUs: {list(parsed_config.keys())}"
    )


class MooncakeTransferEngine:
    """Shared Mooncake transfer engine for RDMA/transfer operations."""

    def __init__(
        self,
        hostname: str,
        gpu_id: Optional[int] = None,
        ib_device: Optional[str] = None,
    ):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id if gpu_id is not None else 0
        # MC_FORCE_TCP=1 makes mooncake install TcpTransport instead of RDMA,
        # in which case RDMA HCA selection is irrelevant; pass empty device.
        if os.environ.get("MC_FORCE_TCP") == "1":
            self.ib_device = ""
        else:
            self.ib_device = get_ib_devices_for_gpu(ib_device, self.gpu_id)

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        self.session_id = NetworkAddress(
            self.hostname, self.engine.get_rpc_port()
        ).to_host_port_str()

    def register(self, ptr, length):
        try:
            ret_value = self.engine.register_memory(ptr, length)
        except Exception:
            # Mark register as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory registration %s failed.", ptr)

    def deregister(self, ptr):
        try:
            ret_value = self.engine.unregister_memory(ptr)
        except Exception:
            # Mark deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory deregistration %s failed.", ptr)

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        """Batch register multiple memory regions."""
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark batch register as failed
            ret_value = -1
            if not hasattr(self.engine, "batch_register_memory"):
                raise RuntimeError(
                    "Mooncake's batch register requires a newer version of "
                    "mooncake-transfer-engine. Please upgrade Mooncake."
                )

        if ret_value != 0:
            logger.debug("Mooncake batch memory registration failed.")
        return ret_value

    def batch_deregister(self, ptrs: List[int]) -> int:
        """Batch deregister multiple memory regions."""
        try:
            ret_value = self.engine.batch_unregister_memory(ptrs)
        except Exception:
            # Mark batch deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake batch memory deregistration failed.")
        return ret_value

    def initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        if envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE.get():
            npu_phy_id = envs.ASCEND_NPU_PHY_ID.get()
            suffix = self.gpu_id if npu_phy_id == -1 else npu_phy_id
            hostname += f":{get_free_port()}:npu_{suffix}"
            protocol = "ascend"
        else:
            # MOONCAKE_PROTOCOL selects the transport (rdma | efa | tcp | ...).
            # Default is "rdma"; set MOONCAKE_PROTOCOL=efa on AWS EFA hardware.
            protocol = envs.MOONCAKE_PROTOCOL.get()

        ret_value = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            protocol,
            device_name if device_name is not None else "",
        )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        try:
            ret = self.engine.transfer_sync_write(
                session_id, buffer, peer_buffer_address, length
            )
        except Exception:
            ret = -1

        if ret < 0:
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                buffer,
                session_id,
                peer_buffer_address,
            )

        return ret

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        """Synchronously transfer data to the specified addresses in batches."""
        try:
            ret = self.engine.batch_transfer_sync_write(
                session_id, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            ret = -1
            if not hasattr(self.engine, "batch_transfer_sync_write"):
                raise RuntimeError(
                    "Mooncake's batch transfer requires mooncake-transfer-engine "
                    ">= 0.3.4.post2. Please upgrade Mooncake by "
                    "'pip install mooncake-transfer-engine --upgrade'"
                )

        if ret < 0:
            logger.debug(
                "Failed to batch transfer data. Buffers: %s, Session: %s, "
                "Peer addresses: %s",
                buffers,
                session_id,
                peer_buffer_addresses,
            )
        return ret

    def get_session_id(self):
        return self.session_id

    def send_probe(self, peer_session_id: str) -> int:
        return self.engine.send_probe(peer_session_id)

    def get_engine(self):
        return self.engine.get_engine()

    def get_ib_device(self):
        return self.ib_device


def init_mooncake_transfer_engine(
    hostname: str,
    gpu_id: Optional[int] = None,
    ib_device: Optional[str] = None,
) -> MooncakeTransferEngine:
    """
    Initialize the shared MooncakeTransferEngine. Note: if already
    initialized with the same (hostname, gpu_id, ib_device), returns existing
    instance. Call from parallel_state when model parallel is set up and
    mooncake transfer is needed.
    """
    global _mooncake_transfer_engine
    if _mooncake_transfer_engine is not None:
        return _mooncake_transfer_engine
    _mooncake_transfer_engine = MooncakeTransferEngine(
        hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
    )
    return _mooncake_transfer_engine


def get_mooncake_transfer_engine() -> Optional[MooncakeTransferEngine]:
    """Return the shared MooncakeTransferEngine if initialized, else None."""
    return _mooncake_transfer_engine


def maybe_init_shared_mooncake_transfer_engine(
    *, server_args: ServerArgs, gpu_id: int
) -> None:
    """
    Need MooncakeTransferEngine when:
    1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
    2) HiCache uses mooncake storage backend
    3) Encoder disaggregation uses mooncake
    """
    use_mooncake_te = (
        (
            server_args.disaggregation_mode != "null"
            and server_args.disaggregation_transfer_backend == "mooncake"
        )
        or (
            server_args.enable_hierarchical_cache
            and server_args.hicache_storage_backend == "mooncake"
            and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
        )
        or (
            server_args.encoder_only
            and server_args.encoder_transfer_backend == "mooncake"
        )
        or (
            server_args.language_only
            and server_args.encoder_transfer_backend == "mooncake"
        )
        or (
            server_args.enable_elastic_expert_backup
            and server_args.elastic_ep_backend is not None
        )
        or server_args.elastic_ep_backend == "mooncake"
    )

    if use_mooncake_te:
        init_mooncake_transfer_engine(
            hostname=get_local_ip_auto(),
            gpu_id=gpu_id,
            ib_device=(
                server_args.disaggregation_ib_device or server_args.mooncake_ib_device
            ),
        )

        if server_args.elastic_ep_backend == "mooncake":
            from mooncake.pg import set_transfer_engine
            set_transfer_engine(_mooncake_transfer_engine.engine)
