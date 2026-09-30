# SPDX-License-Identifier: Apache-2.0
"""Run:ai Model Streamer: fastest to read, but it copies into anonymous memory."""

from typing import Callable, ClassVar, Iterator

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from runai_model_streamer import SafetensorsStreamer

    HAS_RUNAI_MODEL_STREAMER = True
except ImportError:
    SafetensorsStreamer = None
    HAS_RUNAI_MODEL_STREAMER = False


class RunaiStreamerReader:
    name: ClassVar[str] = "runai_streamer"
    # it materializes every tensor before yielding, so filtering saves nothing
    supports_key_filter: ClassVar[bool] = False
    retains_file_mapping: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        return HAS_RUNAI_MODEL_STREAMER

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
        logger.info(
            "Loading safetensors with Run:ai Model Streamer to %s",
            "cpu" if to_cpu else device,
        )
        with SafetensorsStreamer() as streamer:
            if to_cpu:
                streamer.stream_files(files)
            else:
                streamer.stream_files(files, device=device)
            for name, tensor in streamer.get_tensors():
                if key_filter is not None and not key_filter(name):
                    continue
                if to_cpu or clone_tensors:
                    yield name, tensor.clone().detach()
                else:
                    yield name, tensor
