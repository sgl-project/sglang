"""Strict registered-buffer adapter for immutable training objects.

``put_from`` returns a status; ``get_into`` returns a byte count. Failed or
ambiguous transfers retain their registrations and buffers until client close.
Hard pinning is mandatory, without a compatibility downgrade.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math

import torch
from sglang.srt.training_capture.protocol import (
    CaptureError,
    ContractError,
    digest_bytes,
    tensor_bytes,
)

logger = logging.getLogger(__name__)


class TransportError(CaptureError):
    pass


class MooncakeSnapshotStore:
    def __init__(self, client, replicate_config, *, max_receive_bytes: int = 2 << 30):
        try:
            # Capability checks are intentional at this external SDK boundary.
            required = (
                client.register_buffer,
                client.unregister_buffer,
                client.put_from,
                client.get_into,
                client.is_exist,
                client.remove,
                client.close,
            )
            if not all(callable(method) for method in required):
                raise AttributeError("non-callable Store method")
            _ = replicate_config.with_hard_pin
            replicate_config.with_hard_pin = True
            if not replicate_config.with_hard_pin:
                raise AttributeError("hard pin not enabled")
        except AttributeError as error:
            raise ContractError(
                "Mooncake SDK must support registered buffers and hard pinning"
            ) from error
        if replicate_config.replica_num < 1 or max_receive_bytes <= 0:
            raise ValueError("positive replica count and receive budget are required")
        self.client = client
        self.replicate_config = replicate_config
        self.max_receive_bytes = max_receive_bytes
        self.registered: dict[int, torch.Tensor] = {}
        self.quarantined: set[int] = set()
        self.closed = False

    @classmethod
    def connect(
        cls, setup: dict, *, replica_num: int = 1, max_receive_bytes: int = 2 << 30
    ):
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        config = ReplicateConfig()
        config.replica_num = replica_num
        client = MooncakeDistributedStore()
        adapter = cls(client, config, max_receive_bytes=max_receive_bytes)
        rc = client.setup(**setup)
        if rc != 0:
            client.close()
            raise TransportError(f"Mooncake setup failed: status={rc}")
        versions = {}
        for distribution in (
            "mooncake-transfer-engine",
            "mooncake-transfer-engine-cuda13",
        ):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
        logger.info(
            "Training snapshot Store connected: sdk=%s protocol=%s replicas=%d hard_pin=true",
            versions,
            setup["protocol"],
            replica_num,
        )
        return adapter

    def register(self, tensor: torch.Tensor) -> None:
        tensor_bytes(tensor)
        if self.closed or not tensor.numel() or tensor.data_ptr() in self.registered:
            raise ContractError("invalid or duplicate buffer registration")
        pointer = tensor.data_ptr()
        self.registered[pointer] = tensor
        try:
            rc = self.client.register_buffer(
                pointer, tensor.numel() * tensor.element_size()
            )
            if rc != 0:
                raise TransportError(f"Mooncake register_buffer failed: status={rc}")
        except Exception:
            self.quarantined.add(pointer)
            raise

    def unregister(self, tensor: torch.Tensor) -> None:
        pointer = tensor.data_ptr()
        if pointer not in self.registered or pointer in self.quarantined:
            raise TransportError("cannot unregister an unknown or quarantined buffer")
        try:
            rc = self.client.unregister_buffer(pointer)
            if rc != 0:
                raise TransportError(f"Mooncake unregister_buffer failed: status={rc}")
        except Exception:
            self.quarantined.add(pointer)
            raise
        del self.registered[pointer]

    def _registration(self, tensor: torch.Tensor) -> int:
        tensor_bytes(tensor)
        begin = tensor.data_ptr()
        end = begin + tensor.numel() * tensor.element_size()
        for pointer, storage in self.registered.items():
            if (
                pointer
                <= begin
                < end
                <= pointer + storage.numel() * storage.element_size()
            ):
                if pointer in self.quarantined:
                    raise TransportError("buffer has an uncertain outstanding transfer")
                return pointer
        raise ContractError("tensor is outside all registered buffers")

    def put_registered(
        self, key: str, tensor: torch.Tensor, expected_digest: str
    ) -> None:
        pointer = self._registration(tensor)
        if digest_bytes(tensor_bytes(tensor)) != expected_digest:
            raise ContractError("immutable source changed before Store write")
        exists = self.client.is_exist(key)
        if exists not in (0, 1):
            raise TransportError(f"Mooncake is_exist failed: status={exists}")
        if exists:
            existing = self.get_tensor(
                key, list(tensor.shape), tensor.dtype, expected_digest
            )
            del existing
            return
        try:
            rc = self.client.put_from(
                key,
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                self.replicate_config,
            )
            if rc != 0:
                raise TransportError(f"Mooncake put_from failed: status={rc}")
        except Exception as error:
            self.quarantined.add(pointer)
            raise TransportError(
                "Mooncake write completion is uncertain; source quarantined"
            ) from error

    def get_tensor(
        self, key: str, shape: list[int], dtype: torch.dtype, expected_digest: str
    ) -> torch.Tensor:
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ContractError("invalid receive shape")
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        retained = sum(
            self.registered[p].numel() * self.registered[p].element_size()
            for p in self.quarantined
        )
        if nbytes + retained > self.max_receive_bytes:
            raise ContractError("receive/quarantine budget exceeded")
        if self.client.is_exist(key) != 1:
            raise TransportError("Mooncake object is missing or unavailable")
        out = torch.empty(shape, dtype=dtype)
        self.register(out)
        pointer = out.data_ptr()
        try:
            count = self.client.get_into(key, pointer, nbytes)
            if type(count) is not int or count < 0:
                raise TransportError(f"Mooncake get_into failed: status={count}")
        except Exception as error:
            self.quarantined.add(pointer)
            raise TransportError(
                "Mooncake read completion is uncertain; destination quarantined"
            ) from error
        self.unregister(out)
        if count != nbytes:
            raise ContractError(
                f"short Mooncake read: expected={nbytes}, actual={count}"
            )
        if digest_bytes(tensor_bytes(out)) != expected_digest:
            raise ContractError("Mooncake object digest mismatch")
        return out

    def remove(self, key: str) -> None:
        """Only the retention authority may call this, after lease/checkpoint checks."""
        rc = self.client.remove(key, force=False)
        if rc != 0:
            raise TransportError(f"Mooncake remove failed: status={rc}")

    def close(self) -> None:
        """Caller first stops all threads and synchronizes all outstanding CUDA copies."""
        if self.closed:
            return
        rc = self.client.close()
        if rc not in (None, 0):
            raise TransportError(f"Mooncake close failed: status={rc}")
        self.closed = True
        self.registered.clear()
        self.quarantined.clear()
