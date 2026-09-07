# SPDX-License-Identifier: Apache-2.0
"""What a weight source has to provide, and what distinguishes one from another.

Reading a checkpoint used to be a boolean: Run:ai streamer, or `safe_open`. The
two differ in more than speed, and the differences decide correctness and memory
behaviour rather than taste, so they are stated here as capabilities:

``supports_key_filter``
    Whether the reader can skip keys while reading. The streamer materializes
    every tensor before handing any of them over, so a caller that only wants
    part of a checkpoint cannot save anything by asking it.

``retains_file_mapping``
    Whether the tensors it yields are views into the checkpoint file. Those
    pages are file-backed, so the kernel can drop them under pressure without
    swap; a reader that copies into anonymous memory gives the kernel nothing
    to reclaim.
"""

from typing import Callable, ClassVar, Iterator, Protocol, runtime_checkable

import torch


@runtime_checkable
class WeightReader(Protocol):
    """Yields ``(name, tensor)`` for every weight in a set of checkpoint files."""

    name: ClassVar[str]
    supports_key_filter: ClassVar[bool]
    retains_file_mapping: ClassVar[bool]

    @classmethod
    def is_available(cls) -> bool:
        """Whether this reader can run at all in this install."""

    def iter_weights(
        self,
        files: list[str],
        *,
        device: str,
        to_cpu: bool,
        key_filter: Callable[[str], bool] | None = None,
        clone_tensors: bool = True,
        show_progress: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate the weights, in whatever order the reader finds them."""
