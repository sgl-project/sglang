from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.server_args import ServerArgs


@dataclass(slots=True, kw_only=True)
class LoraController:
    server_args: ServerArgs
    auto_create_handle_loop: Callable[[], None]
    update_lora_adapter_communicator: Any = None  # set after facade.init_communicators
    lora_registry: LoRARegistry = field(init=False)
    # Lock to serialize LoRA update operations.
    # Please note that, unlike `model_update_lock`, this does not block inference, allowing
    # LoRA updates and inference to overlap.
    lora_update_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lora_ref_cache: Dict[str, LoRARef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # LoRA
        # Initialize the `LoRARegistry` with initial LoRA adapter paths provided in `server_args`.
        # The registry dynamically updates as adapters are loaded / unloaded during runtime. It
        # serves as the source of truth for available adapters and maps user-friendly LoRA names
        # to internally used unique LoRA IDs.
        self.lora_registry = LoRARegistry(self.server_args.lora_paths)
        # A cache for mapping the lora_name for LoRA adapters that have been loaded at any
        # point to their latest LoRARef objects, so that they can be
        # dynamically loaded if needed for inference
        if self.server_args.lora_paths is not None:
            for lora_ref in self.server_args.lora_paths:
                self.lora_ref_cache[lora_ref.lora_name] = lora_ref
