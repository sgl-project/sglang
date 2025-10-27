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

from sglang.srt.utils import ConcurrentCounter
from sglang.srt.utils.aio_rwlock import RWLock


@dataclass(frozen=True)
class LoRARef:
    """
    Reference record for a LoRA model.

    This object guarantees a unique ``lora_id`` and may include ``lora_name``, ``lora_path``, and ``pinned``.
    The ID eliminates conflicts from reused LoRA names or paths and can be used to generate deterministic cache
    keys (e.g., radix cache).
    """

    lora_id: str = field(default_factory=lambda: uuid4().hex)
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None
    pinned: Optional[bool] = None

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
    The central registry to keep track of available LoRA adapters and ongoing LoRA requests.

    The `LoRARegistry` resides in the tokenizer manager process and acts as the single source of truth for all
    available LoRA adapters. It supports concurrent inference and dynamic adapter updates through a two-phase
    update / eventual consistency model between the tokenizer manager process and the scheduler processes.
    """

    def __init__(self, lora_paths: Optional[List[LoRARef]] = None):
        assert lora_paths is None or all(
            isinstance(lora, LoRARef) for lora in lora_paths
        ), (
            "server_args.lora_paths should have been normalized to LoRARef objects during server initialization. "
            "Please file an issue if you see this error."
        )

        # A read-write lock to ensure adapters loading / unloading operations are exclusive.
        # Please note that the counter increment/decrement operations are not synchronized through this
        # lock, as they are designed to be non-blocking and can be performed concurrently.
        self._registry_lock = RWLock()
        # A dictionary to hold LoRARef objects, mapping from LoRA name to LoRARef.
        self._registry: Dict[str, LoRARef] = {}
        # Counters for ongoing requests, mapping from LoRA ID to ConcurrentCounter.
        self._counters: Dict[str, ConcurrentCounter] = {}

        # Initialize the registry with provided LoRA paths, if present.
        if lora_paths:
            for lora_ref in lora_paths:
                self._register_adapter(lora_ref)

    async def register(self, lora_ref: LoRARef):
        """
        Register a new LoRARef object in the registry.

        Args:
            lora_ref (LoRARef): The LoRARef object to register.
        """
        async with self._registry_lock.writer_lock:
            self._register_adapter(lora_ref)

    async def unregister(self, lora_name: str) -> str:
        """
        Unregister a LoRARef object from the registry and returns the removed LoRA ID.

        Args:
            lora_name (str): The name of the LoRA model to unregister.
        """
        async with self._registry_lock.writer_lock:
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
        """

        def _lookup(name: str) -> str:
            if name is None:
                return None

            lora_ref = self._registry.get(name, None)
            if lora_ref is None:
                raise ValueError(
                    f"The following requested LoRA adapters are not loaded: {name}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            return lora_ref.lora_id

        async with self._registry_lock.reader_lock:
            if isinstance(lora_name, str):
                lora_id = _lookup(lora_name)
                await self._counters[lora_id].increment(notify_all=False)
                return lora_id
            elif isinstance(lora_name, list):
                lora_ids = [_lookup(name) for name in lora_name]

                # Increment the counters only after all IDs are looked up.
                await asyncio.gather(
                    *[
                        self._counters[id].increment(notify_all=False)
                        for id in lora_ids
                        if id is not None
                    ]
                )
                return lora_ids
            else:
                raise TypeError(
                    "lora_name must be either a string or a list of strings."
                )

    async def release(self, lora_id: Union[str, List[str]]):
        """
        Decrements the usage counter for a LoRA adapter, indicating that it is no longer in use.
        """

        async with self._registry_lock.reader_lock:
            if isinstance(lora_id, str):
                await self._counters[lora_id].decrement()
            elif isinstance(lora_id, list):
                await asyncio.gather(
                    *[
                        self._counters[id].decrement()
                        for id in lora_id
                        if id is not None
                    ]
                )
            else:
                raise TypeError("lora_id must be either a string or a list of strings.")

    async def wait_for_unload(self, lora_id: str):
        """
        Waits until the usage counter for a LoRA adapter reaches zero, indicating that it is no longer in use.
        This is useful for ensuring that a LoRA adapter can be safely unloaded.

        This method itself is not synchronized, which is safe because it should only be called during LoRA unloading,
        which itself is guaranteed to be sequential.
        """
        assert (
            lora_id not in self._registry
        ), "wait_for_unload should only be called after the LoRA adapter has been unregistered. "
        assert (
            lora_id in self._counters
        ), "The LoRA ID should still have a counter if it has been registered before."

        # Wait until no requests are using this LoRA adapter.
        await self._counters[lora_id].wait_for_zero()
        del self._counters[lora_id]

    def _register_adapter(self, lora_ref: LoRARef):
        """
        Internal helper method to register a LoRA adapter.
        """

        if lora_ref.lora_name in self._registry:
            raise ValueError(
                f"LoRA with name {lora_ref.lora_name} already exists. Loaded LoRAs: {self._registry.keys()}"
            )
        self._registry[lora_ref.lora_name] = lora_ref
        self._counters[lora_ref.lora_id] = ConcurrentCounter()
        return lora_ref

    @property
    def num_registered_loras(self) -> int:
        """
        Returns the total number of LoRA adapters currently registered.
        """
        return len(self._registry)
