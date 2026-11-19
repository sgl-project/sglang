# SPDX-License-Identifier: Apache-2.0
"""How one rank's storages move from an owner to a reader."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from sglang.srt.model_loader.snapshot.schema import SnapshotManifest


class WeightTransport(ABC):
    """Moves one rank's storages from an owner to a reader, and withdraws them.

    It knows nothing about models or other ranks. A reader's transport is made by
    the loader's lookup for one load, so it may hold that load's session, such as
    the daemon connection that vmm_fd sends its file descriptors over.
    """

    @abstractmethod
    def publish(self, *, storages: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Owner: make ``storages`` readable; returns the manifest's ``source``.

        It may consume ``storages`` (vmm_fd repacks them), so nothing reads them after.
        """

    @abstractmethod
    def fetch(
        self, *, manifest: SnapshotManifest, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Reader: every storage of ``manifest`` as uint8 bytes, all or nothing.

        Returns once the bytes are visible on the current stream; raises otherwise.
        """

    @abstractmethod
    def close(self) -> None:
        """Withdraw what publish or fetch set up. Idempotent."""
