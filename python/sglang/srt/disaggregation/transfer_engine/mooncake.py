import json
import logging
import os
import uuid
from dataclasses import dataclass

from sglang.srt.utils import get_local_ip_by_remote
from sglang.srt.disaggregation.rdma_device_utils import find_best_rdma_device_for_gpu

logger = logging.getLogger(__name__)


@dataclass
class MooncakeTransferEngineConfig:
    local_hostname: str
    metadata_server: str
    protocol: str
    device_name: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeTransferEngineConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            local_hostname=config.get("local_hostname", None),
            metadata_server=config.get("metadata_server"),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
        )

    @staticmethod
    def load_config(gpu_id=None) -> "MooncakeTransferEngineConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
           logger.info("No config set for 'MOONCAKE_CONFIG_PATH', specified env is preferred")
           return MooncakeTransferEngineConfig.auto_config(gpu_id)
        return MooncakeTransferEngineConfig.from_file(config_file_path)

    @staticmethod
    def load_auto_config(gpu_id) -> "MooncakeTransferEngineConfig":
        """Load config from a file specified in the environment variable."""
        metadata_server = os.getenv("MOONCAKE_METADATA_SERVER", None)
        if metadata_server is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_METADATA_SERVER' is not set."
            )
        local_hostname = os.getenv("MOONCAKE_LOCAL_HOSTNAME", default=get_local_ip_by_remote())
        protocol = os.getenv("MOONCAKE_PROTOCOL", default="rdma")
        default_ib_device, _ = find_best_rdma_device_for_gpu(gpu_id)
        device_name = os.getenv("MOONCAKE_RDMA_DEVICE_NAME", default=default_ib_device)
        return MooncakeTransferEngineConfig(
            local_hostname=local_hostname,
            metadata_server=metadata_server,
            protocol=protocol,
            device_name=device_name,
        )

class MooncakeTransferEngine:

    def __init__(self, gpu_id=0):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()

        try:
            self.config = MooncakeTransferEngineConfig.load_auto_config(gpu_id)
            logger.info("Mooncake Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        session_suffix = "_" + str(uuid.uuid4())
        self.session_id = self.config.local_hostname + session_suffix
        self.initialize(
            self.session_id,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
        )

    def register(self, ptr, length):
        self.engine.register_memory(ptr, length)

    def deregister(self, ptr):
        self.engine.unregister_memory(ptr)

    def initialize(
        self,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> None:
        """Initialize the mooncake instance."""
        self.engine.initialize(local_hostname, metadata_server, protocol, device_name)

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""

        ret = self.engine.transfer_sync_write(
            session_id, buffer, peer_buffer_address, length
        )
        if ret < 0:
            logger.error("Transfer Return Error")
            raise Exception("Transfer Return Error")
        return ret

    def get_localhost(self):
        return self.config.local_hostname

    def get_session_id(self):
        return self.session_id
