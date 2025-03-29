import json
import logging
import os
from dataclasses import dataclass
from typing import Union

logger = logging.getLogger(__name__)


@dataclass
class MooncakeTransferEngineConfig:
    localhost_name: str
    metadata_backend: Union[str, None]
    metadata_server: str
    protocol: str
    device_name: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeTransferEngineConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            localhost_name=config.get("localhost_name", None),
            metadata_backend=config.get("metadata_backend", None),
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
            import mooncake_sglang_adaptor as msa
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = msa.TransferEngine()

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

        self.initialize(
            self.config.localhost_name,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
            self.config.metadata_backend,
        )

    def register(self, ptr, length):
        self.engine.expRegisterMemory(ptr, length)

    def deregister(self, ptr):
        self.engine.expUnregisterMemory(ptr)

    def initialize(
        self,
        localhost_name: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
        metadata_backend: Union[str, None],
    ) -> None:
        """Initialize the mooncake instance."""
        if metadata_backend is None:
            self.engine.initialize(
                localhost_name, metadata_server, protocol, device_name
            )
        else:
            supported_backend = ["etcd", "redis"]
            metadata_backend = metadata_backend.lower()
            if metadata_backend not in supported_backend:
                raise ValueError(
                    "Mooncake Configuration error. `metadata_backend`"
                    f"should be one of {supported_backend}."
                )

            self.engine.initializeExt(
                localhost_name, metadata_server, protocol, device_name, metadata_backend
            )

    def transfer_sync(
        self, remote_url: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""

        write_op = self.engine.TransferOpcode.WRITE
        ret = self.engine.transferSyncExt(
            remote_url, buffer, peer_buffer_address, length, write_op
        )
        if ret < 0:
            logger.error("Transfer Return Error")
            raise Exception("Transfer Return Error")
        return ret

    def get_localhost(self):
        return self.config.localhost_name
