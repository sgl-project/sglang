import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, translate tensor object access for different TP ranks
    # potentially pass model and TP configs into storage backend
    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    @abstractmethod
    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    @abstractmethod
    def batch_get(
        self, keys: List[str], target_locations: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor | None]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(self, key, value) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass


class HiCacheFile(HiCacheStorage):

    def __init__(self, file_path: str = "/tmp/hicache"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            # todo: fixing the target_location logic to enable in-place loading
            loaded_tensor = torch.load(tensor_path)
            if isinstance(loaded_tensor, torch.Tensor):
                return loaded_tensor
            else:
                logger.error(f"Loaded data for key {key} is not a tensor.")
                return None
        except FileNotFoundError:
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(self, key: str, value: torch.Tensor) -> bool:
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True
        try:
            torch.save(value, tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def delete(self, key: str) -> None:
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            os.remove(tensor_path)
        except FileNotFoundError:
            logger.warning(f"Key {key} does not exist. Cannot delete.")
            return

    def clear(self) -> None:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")


class HiCacheNixl(HiCacheFile):
    def __init__(self, file_path: str = "/mnt/gds/hicache", buf_size: int = 128):
        super().__init__(file_path)

        self.buf_size = buf_size

        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError:
            logger.warning(
                "NIXL library not available – operating in plain file mode without Mooncake acceleration."
            )
            self.agent = None
            return

        # Create an agent and load Mooncake plugin.
        self.agent_config = nixl_agent_config(backends=[])
        self.agent = nixl_agent("HiCacheNixl", self.agent_config)

        plugin_list = self.agent.get_plugin_list()
        if not any(p.lower() == "mooncake" for p in plugin_list):
            logger.warning(
                "Mooncake plugin not found in NIXL plugin list: %s. Proceeding with fallback mode.",
                plugin_list,
            )
            self.agent = None
            return

        # Activate the Mooncake backend.
        try:
            self.agent.create_backend("MOONCAKE")
        except Exception as e:
            logger.error("Failed to create Mooncake backend via NIXL: %s", e)
            self.agent = None

    def _tensor_nixl_obj(self, tensor: torch.Tensor):
        """Register a (CPU) tensor with NIXL and return (xfer_descs, reg_descs)."""
        tensor_reg_descs = self.agent.register_memory(tensor)
        if tensor_reg_descs is None:
            raise RuntimeError("Tensor registration with NIXL failed")
        xfer_descs = self.agent.get_xfer_descs(tensor)
        return xfer_descs, tensor_reg_descs

    def _file_nixl_obj(self, fd: int, num_bytes: int):
        """Register a file descriptor with NIXL and return the reg descriptor."""
        agent_file_list = [(0, num_bytes, fd, "b")]
        file_descs = self.agent.register_memory(agent_file_list, "FILE")
        if file_descs is None:
            raise RuntimeError("File registration with NIXL failed")
        return file_descs

    def _nixl_transfer(self, mode: str, tensor: torch.Tensor, file_path: str):

        assert mode in {"WRITE", "READ"}, "Mode must be READ or WRITE"

        # Prepare dirs / file.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # NOTE: Do **NOT** truncate existing file in READ mode.
        flags = os.O_RDWR | os.O_CREAT
        if mode == "WRITE":
            flags |= os.O_TRUNC
        fd = os.open(file_path, flags, 0o600)

        num_bytes = tensor.numel() * tensor.element_size()

        try:
            # Register memory objects.
            tensor_xfer, tensor_regs = self._tensor_nixl_obj(tensor)
            file_regs = self._file_nixl_obj(fd, num_bytes)
            file_xfer = file_regs.trim()

            # Init & kick off transfer.
            xfer_hdl = self.agent.initialize_xfer(
                mode, tensor_xfer, file_xfer, "HiCacheNixl"
            )
            if not xfer_hdl:
                raise RuntimeError("Failed to create NIXL transfer handle")

            state = self.agent.transfer(xfer_hdl)
            if state == "ERR":
                raise RuntimeError("NIXL transfer entered ERR state")

            while True:
                state = self.agent.check_xfer_state(xfer_hdl)
                if state == "DONE":
                    break
                if state == "ERR":
                    raise RuntimeError("NIXL transfer entered ERR state")

            # Release resources.
            self.agent.release_xfer_handle(xfer_hdl)
            self.agent.deregister_memory(tensor_regs)
            self.agent.deregister_memory(file_regs)
        finally:
            os.close(fd)

    def set(self, key: str, value: torch.Tensor) -> bool:  # type: ignore[override]

        # Fast-path: same semantics as parent if no agent.
        if getattr(self, "agent", None) is None:
            return super().set(key, value)

        tensor_path = os.path.join(self.file_path, f"{key}.bin")

        # Skip if already cached.
        if self.exists(key):
            logger.debug("[HiCacheNixl] Key %s already exists – skipping transfer", key)
            return True

        try:
            # First attempt high-speed DMA.
            self._nixl_transfer("WRITE", value.cpu(), tensor_path)
            logger.debug("[HiCacheNixl] Mooncake WRITE succeeded for key %s", key)
        except Exception as e:
            logger.warning(
                "[HiCacheNixl] Mooncake WRITE failed for key %s – falling back to torch.save (%s)",
                key,
                e,
            )

        # Ensure compatibility with the vanilla file backend.
        return super().set(key, value)

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:  # type: ignore[override]

        if not self.exists(key):
            return None

        tensor_path = os.path.join(self.file_path, f"{key}.bin")

        if getattr(self, "agent", None) is not None and target_location is not None:
            try:
                self._nixl_transfer("READ", target_location, tensor_path)
                logger.debug("[HiCacheNixl] Mooncake READ succeeded for key %s", key)
                return target_location
            except Exception as e:
                logger.warning(
                    "[HiCacheNixl] Mooncake READ failed for key %s – falling back to torch.load (%s)",
                    key,
                    e,
                )

        # Fallback: defer to standard torch.load mechanism.
        return super().get(key, target_location)
