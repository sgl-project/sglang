# SPDX-License-Identifier: Apache-2.0
"""`safe_open`: slower to read, and the only source whose pages stay reclaimable.

`safe_open` maps the file, so a CPU tensor it yields is a view into the
checkpoint rather than a copy. Those pages are file-backed, which is what lets
the kernel drop them under memory pressure even on a host with no swap.
"""

from typing import Callable, ClassVar, Iterator

import torch
from safetensors.torch import safe_open
from tqdm.auto import tqdm

_BAR_FORMAT = "{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt}"


class SafetensorsMmapReader:
    name: ClassVar[str] = "safetensors"
    supports_key_filter: ClassVar[bool] = True
    retains_file_mapping: ClassVar[bool] = True

    @classmethod
    def is_available(cls) -> bool:
        return True

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
        for path in tqdm(
            files,
            desc="Loading safetensors checkpoint shards",
            disable=not show_progress,
            bar_format=_BAR_FORMAT,
        ):
            with safe_open(path, framework="pt", device=device) as handle:
                for name in handle.keys():  # noqa: SIM118
                    if key_filter is not None and not key_filter(name):
                        continue
                    yield name, handle.get_tensor(name)
