import json
import logging
import os
import uuid
from dataclasses import dataclass

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
    def load_from_env() -> "MooncakeTransferEngineConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeTransferEngineConfig.from_file(config_file_path)


class MooncakeTransferEngine:

    def __init__(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        try:
            self.engine = TransferEngine()
        except RuntimeError as e:
            raise RuntimeError(f"Runtime: {e}") from e
        except ValueError as e:
            logger.error("Value error occurred for mooncake: %s", e)
            raise
        except Exception as e:
            logger.error("An unknow error occurred for mooncake: %s", e)
            raise

        try:
            self.config = MooncakeTransferEngineConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        self.config = MooncakeTransferEngineConfig.load_from_env()

        session_suffix = "_" + str(uuid.uuid4())
        self.session_id = self.config.local_hostname + session_suffix
        self.initialize(
            self.session_id,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
        )

    def register(self, ptr, length):
        ret_value = self.engine.register_memory(ptr, length)
        if ret_value != 0:
            logger.error("Mooncake memory registration failed.")
            raise RuntimeError("Mooncake memory registration failed.")

    def deregister(self, ptr):
        ret_value = self.engine.unregister_memory(ptr)
        if ret_value != 0:
            logger.error("Mooncake memory deregistration failed.")
            raise RuntimeError("Mooncake memory deregistration failed.")

    def initialize(
        self,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> None:
        """Initialize the mooncake instance."""
        ret_value = self.engine.initialize(local_hostname, metadata_server, protocol, device_name)
        if ret_value != 0:
            logger.error("Mooncake initialization failed.")
            raise RuntimeError("Mooncake initialization failed.")

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
