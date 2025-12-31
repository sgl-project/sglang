from enum import Enum, auto
from typing import Dict, Optional

import torch
from torch.cuda import Event as CudaEvent
from torch.cuda import Stream as CudaStream
from torch.cuda import StreamContext as CudaStreamContext

from sglang.srt.lora.lora_manager import LoRAManager


class LoRAPrefetchStatus(Enum):
    LOADED = auto()
    PREFETCHING = auto()
    NOT_PREFETCHED = auto()


class LoRAPrefetcher:
    def __init__(self, lora_manager):
        self.lora_manager: LoRAManager = lora_manager
        self.device_module = torch.get_device_module(self.lora_manager.device)
        self.load_stream: CudaStream = self.device_module.Stream()
        self.load_stream_context: CudaStreamContext = self.device_module.stream(
            self.load_stream
        )
        self.lora_to_prefetch_event: Dict[Optional[str], CudaEvent] = {}

    def check_prefetch_status(self, lora_id: Optional[str]) -> LoRAPrefetchStatus:
        if lora_id not in self.lora_to_prefetch_event:
            return LoRAPrefetchStatus.NOT_PREFETCHED

        event = self.lora_to_prefetch_event[lora_id]

        if not event.query():
            return LoRAPrefetchStatus.PREFETCHING

        torch.cuda.current_stream().wait_event(event)
        del self.lora_to_prefetch_event[lora_id]

        return LoRAPrefetchStatus.LOADED

    def try_start_prefetch(
        self, lora_id: Optional[str], running_loras: set[Optional[str]]
    ) -> bool:
        loras_to_be_loaded = running_loras | self.lora_to_prefetch_event.keys()

        new_lora_set = loras_to_be_loaded | {lora_id}
        if not self.lora_manager.validate_lora_batch(new_lora_set):
            return False

        with self.load_stream_context:
            self.lora_manager.fetch_new_lora(lora_id, loras_to_be_loaded)
            event = self.device_module.Event()
            event.record(self.load_stream)

        self.lora_to_prefetch_event[lora_id] = event
        return True
