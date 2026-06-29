# SPDX-License-Identifier: Apache-2.0
"""Shared-memory helpers for diffusion transfer buffers."""

from __future__ import annotations

import ctypes
import logging
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.pinned_memory import (
    PinnedHostMemoryRegistration,
    register_pinned_host_memory,
)


@dataclass
class SharedMemoryAttachment:
    shm: shared_memory.SharedMemory
    np_array: np.ndarray

    @property
    def data_ptr(self) -> int:
        return int(self.np_array.ctypes.data)

    def close(self, logger: logging.Logger | None = None) -> None:
        try:
            self.shm.close()
        except FileNotFoundError:
            if logger is not None:
                logger.debug("Shared memory attachment already closed")


class SharedMemoryRegion:
    """Process-local mapping that may also be CUDA host-registered."""

    def __init__(
        self,
        role_name: str,
        tag: str,
        size: int,
        *,
        pin_memory: bool = False,
        pin_memory_strict: bool = False,
    ):
        self._name = f"sgl_diff_{role_name}_{tag}_{uuid.uuid4().hex}"
        self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=size)
        self._np_array = np.ndarray((size,), dtype=np.uint8, buffer=self._shm.buf)
        self._tensor = torch.from_numpy(self._np_array)
        self._pin_registration: PinnedHostMemoryRegistration | None = None
        if pin_memory:
            try:
                self._pin_registration = register_pinned_host_memory(
                    int(self._np_array.ctypes.data),
                    size,
                    enabled=True,
                    strict=pin_memory_strict,
                )
            except Exception:
                self.cleanup()
                raise

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_pinned(self) -> bool:
        return bool(self._pin_registration and self._pin_registration.registered)

    @property
    def pin_memory_status(self) -> str:
        if self._pin_registration is None:
            return "disabled"
        return self._pin_registration.status

    @property
    def pin_memory_error(self) -> str | None:
        if self._pin_registration is None:
            return None
        return self._pin_registration.error

    def cleanup(self, logger: logging.Logger | None = None) -> None:
        if self._pin_registration is not None:
            self._pin_registration.unregister()
            self._pin_registration = None
        try:
            self._shm.close()
        except FileNotFoundError:
            if logger is not None:
                logger.debug("Shared memory region %s was already closed", self._name)
        try:
            self._shm.unlink()
        except FileNotFoundError:
            if logger is not None:
                logger.debug("Shared memory region %s was already unlinked", self._name)


def get_shared_memory_attachment(
    name: str,
    attachments: dict[str, SharedMemoryAttachment],
) -> SharedMemoryAttachment:
    attachment = attachments.get(name)
    if attachment is not None:
        return attachment

    shm = shared_memory.SharedMemory(name=name)
    np_array = np.ndarray((shm.size,), dtype=np.uint8, buffer=shm.buf)
    attachment = SharedMemoryAttachment(shm=shm, np_array=np_array)
    attachments[name] = attachment
    return attachment


def copy_to_shared_memory(
    *,
    src_addr: int,
    dst_shm_name: str | None,
    dst_shm_offset: int,
    length: int,
    attachments: dict[str, SharedMemoryAttachment],
    logger: logging.Logger,
) -> bool:
    """Copy into a peer's same-host shared-memory mapping when available."""
    if not dst_shm_name or length <= 0:
        return True
    try:
        attachment = get_shared_memory_attachment(dst_shm_name, attachments)
        ctypes.memmove(attachment.data_ptr + int(dst_shm_offset), src_addr, int(length))
        return True
    except Exception:
        logger.exception(
            "TransferManager local shared-memory copy failed: shm=%s offset=%s length=%s",
            dst_shm_name,
            dst_shm_offset,
            length,
        )
        return False
