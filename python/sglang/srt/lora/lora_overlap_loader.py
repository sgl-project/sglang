import logging
from enum import Enum, auto
from typing import Dict, Optional

import torch
from torch.cuda import Event as CudaEvent
from torch.cuda import Stream as CudaStream
from torch.cuda import StreamContext as CudaStreamContext

from sglang.srt.lora.lora_manager import LoRAManager

logger = logging.getLogger(__name__)


class LoRAOverlapLoadStatus(Enum):
    LOADED = auto()
    LOADING = auto()
    NOT_LOADED = auto()


class LoRAOverlapLoader:
    def __init__(self, lora_manager):
        self.lora_manager: LoRAManager = lora_manager
        self.device_module = torch.get_device_module(self.lora_manager.device)
        self.load_stream: CudaStream = self.device_module.Stream()
        self.load_stream_context: CudaStreamContext = self.device_module.stream(
            self.load_stream
        )
        self.lora_to_overlap_load_event: Dict[Optional[str], CudaEvent] = {}

    def try_overlap_load_lora(
        self, lora_id: Optional[str], running_loras: set[Optional[str]]
    ) -> bool:
        """
        Check a LoRA adapter's asynchronous load status, and try to load it if there's capacity
        in the memory pool. Returns whether or not the adapter has been loaded.
        """
        lora_pipeline_load_status = self._check_overlap_load_status(lora_id)
        if lora_pipeline_load_status == LoRAOverlapLoadStatus.LOADING:
            return False
        elif lora_pipeline_load_status == LoRAOverlapLoadStatus.NOT_LOADED:
            res = self._try_start_overlap_load(lora_id, running_loras)
            if res:
                logger.debug(f"Loading LoRA adapter {lora_id} asynchronously")

            return False
        else:
            assert lora_pipeline_load_status == LoRAOverlapLoadStatus.LOADED
            return True

    def _check_overlap_load_status(
        self, lora_id: Optional[str]
    ) -> LoRAOverlapLoadStatus:
        if lora_id not in self.lora_to_overlap_load_event:
            return LoRAOverlapLoadStatus.NOT_LOADED

        event = self.lora_to_overlap_load_event[lora_id]

        if not event.query():
            return LoRAOverlapLoadStatus.LOADING

        torch.cuda.current_stream().wait_event(event)
        del self.lora_to_overlap_load_event[lora_id]

        return LoRAOverlapLoadStatus.LOADED

    def _try_start_overlap_load(
        self, lora_id: Optional[str], running_loras: set[Optional[str]]
    ) -> bool:
        loras_to_be_loaded = running_loras | self.lora_to_overlap_load_event.keys()

        new_lora_set = {lora_id} | loras_to_be_loaded
        if not self.lora_manager.validate_lora_batch(new_lora_set):
            return False

        with self.load_stream_context:
            self.lora_manager.fetch_new_loras({lora_id}, loras_to_be_loaded)
            event = self.device_module.Event()
            event.record(self.load_stream)

        self.lora_to_overlap_load_event[lora_id] = event
        return True
