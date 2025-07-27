# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import asyncio
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union
from uuid import uuid4


@dataclass(frozen=True)
class LoRARef:
    """
    Reference record for a LoRA model.

    This object guarantees a unique ``lora_id`` and may include ``lora_name`` and ``lora_path``. The ID
    eliminates conflicts from reused LoRA names or paths and can be used to generate deterministic cache
    keys (e.g., radix cache).
    """

    lora_id: str = field(default_factory=lambda: uuid4().hex)
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None

    def __post_init__(self):
        if self.lora_id is None:
            raise ValueError("lora_id cannot be None")

    def __str__(self) -> str:
        parts = [
            f"{f.name}={value}"
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class LoRARegistry:
    """
    The central registry to keep track of available LoRA adapters.

    TODO (lifuhuang): This registry is intended as the foundation for overlapped lora update. We decided
    to keep it in a separate PR to keep code review simple and to unblock the radix cache work.
    """

    def __init__(self, lora_paths: Optional[Dict[str, LoRARef]] = None):
        assert lora_paths is None or all(
            isinstance(lora, LoRARef) for lora in lora_paths.values()
        ), (
            "server_args.lora_paths should have been normalized to LoRARef objects during server initialization. "
            "Please file an issue if you see this error."
        )

        # A dictionary to hold LoRARef objects, mapping from LoRA name to LoRARef.
        self._registry: Dict[str, LoRARef] = dict(lora_paths or {})

    async def register(self, lora_ref: LoRARef):
        """
        Register a new LoRARef object in the registry.

        Args:
            lora_ref (LoRARef): The LoRARef object to register.
        """
        if lora_ref.lora_name in self._registry:
            raise ValueError(
                f"LoRA with name {lora_ref.lora_name} already exists. Loaded LoRAs: {self._registry.keys()}"
            )
        self._registry[lora_ref.lora_name] = lora_ref

    async def unregister(self, lora_name: str) -> str:
        """
        Unregister a LoRARef object from the registry and returns the removed LoRA ID.

        Args:
            lora_name (str): The name of the LoRA model to unregister.
        """
        lora_ref = self._registry.get(lora_name, None)
        if lora_ref is None:
            raise ValueError(
                f"LoRA with name {lora_name} does not exist. Loaded LoRAs: {self._registry.keys()}"
            )
        del self._registry[lora_name]

        return lora_ref.lora_id

    async def acquire(self, lora_name: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Queries registry for LoRA IDs based on LoRA names and start tracking the usage of the corresponding LoRA adapters
        by incrementing its counter.

        TODO (lifuhuang): currently it only queries the registry and does not track the usage of LoRA adapters.
        """

        async def _acquire_single(name: str) -> str:
            lora_ref = self._registry.get(name, None)
            if lora_ref is None:
                raise ValueError(
                    f"The following requested LoRA adapters are not loaded: {name}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            # await self._counters[lora_ref.lora_id].increment()
            return lora_ref.lora_id

        if isinstance(lora_name, str):
            lora_id = await _acquire_single(lora_name)
            return lora_id
        elif isinstance(lora_name, list):
            lora_ids = await asyncio.gather(
                *[_acquire_single(name) for name in lora_name]
            )
            return lora_ids
        else:
            raise TypeError("lora_name must be either a string or a list of strings.")
