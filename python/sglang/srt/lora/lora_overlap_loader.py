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
        # Drain completed async loads before status/capacity checks so finished
        # adapters no longer count as in-flight.
        self._drain_completed_overlap_loads()

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
        if lora_id in self.lora_to_overlap_load_event:
            return LoRAOverlapLoadStatus.LOADING

        # After completed events have been drained, a memory-pool entry with no
        # pending event is safe to use on the current stream.
        if lora_id in self.lora_manager.memory_pool.uid_to_buffer_id:
            return LoRAOverlapLoadStatus.LOADED

        return LoRAOverlapLoadStatus.NOT_LOADED

    def _drain_completed_overlap_loads(self) -> None:
        completed_loads = [
            (lora_id, event)
            for lora_id, event in self.lora_to_overlap_load_event.items()
            if event.query()
        ]
        for lora_id, event in completed_loads:
            self.device_module.current_stream().wait_event(event)
            del self.lora_to_overlap_load_event[lora_id]

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
