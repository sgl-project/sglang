from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import fastapi

from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
)

logger = logging.getLogger(__name__)
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

    async def load_lora_adapter(
        self,
        obj: LoadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start load Lora adapter. Lora name=%s, path=%s",
                obj.lora_name,
                obj.lora_path,
            )

            async with self.lora_update_lock:
                # Generate new uniquely identifiable LoRARef object.
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path=obj.lora_path,
                    pinned=obj.pinned,
                )

                # Trigger the actual loading operation at the backend processes.
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                # Register the LoRA adapter only after loading is successful.
                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter

                if self.server_args.max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > self.server_args.max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {self.server_args.max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterReqOutput(
                success=False,
                error_message=str(e),
            )

    async def load_lora_adapter_from_tensors(
        self,
        obj: LoadLoRAAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start load Lora adapter from tensors. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path="__tensor__",
                    pinned=obj.pinned,
                )
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter
                if self.server_args.max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > self.server_args.max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {self.server_args.max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterFromTensorsReqOutput(
                success=False,
                error_message=str(e),
            )

    async def unload_lora_adapter(
        self,
        obj: UnloadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert (
                obj.lora_name is not None
            ), "lora_name must be provided to unload LoRA adapter"

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start unload Lora adapter. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                return await self._unload_lora_adapter_locked(obj)
        except ValueError as e:
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))

    async def _unload_lora_adapter_locked(
        self,
        obj: UnloadLoRAAdapterReqInput,
    ) -> UnloadLoRAAdapterReqOutput:
        assert (
            self.lora_update_lock.locked()
        ), "self.lora_update_lock must be locked in order for self._unload_lora_adapter_locked() to be called"

        # Unregister the LoRA adapter from the registry to stop new requests for this adapter
        # from being started.
        lora_id = await self.lora_registry.unregister(obj.lora_name)
        obj.lora_id = lora_id

        # Initiate the actual unloading operation at the backend processes only after all
        # ongoing requests using this LoRA adapter are finished.
        await self.lora_registry.wait_for_unload(lora_id)
        result = (await self.update_lora_adapter_communicator(obj))[0]

        return result

    async def _validate_and_resolve_lora(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> None:
        if not obj.lora_path:
            return

        if not self.server_args.enable_lora:
            first_adapter = (
                obj.lora_path
                if isinstance(obj.lora_path, str)
                else next((a for a in obj.lora_path if a), None)
            )

            raise ValueError(
                f"LoRA adapter '{first_adapter}' was requested, but LoRA is not enabled. "
                "Please launch the server with --enable-lora flag and preload adapters "
                "using --lora-paths or /load_lora_adapter endpoint."
            )

        await self._resolve_lora_path(obj)

    async def _resolve_lora_path(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        if isinstance(obj.lora_path, str):
            unique_lora_paths = set([obj.lora_path])
        else:
            unique_lora_paths = set(obj.lora_path)

        if (
            self.server_args.max_loaded_loras is not None
            and len(unique_lora_paths) > self.server_args.max_loaded_loras
        ):
            raise ValueError(
                f"Received request with {len(unique_lora_paths)} unique loras requested "
                f"but max loaded loras is {self.server_args.max_loaded_loras}"
            )

        # Reload all existing LoRA adapters that have been dynamically unloaded
        unregistered_loras = await self.lora_registry.get_unregistered_loras(
            unique_lora_paths
        )
        for lora_path in unregistered_loras:
            if lora_path is None:
                continue

            if lora_path not in self.lora_ref_cache:
                raise ValueError(
                    f"Got LoRA adapter that has never been loaded: {lora_path}\n"
                    f"All loaded adapters: {self.lora_ref_cache.keys()}."
                )

            logger.info(f"Reloading evicted adapter: {lora_path}")
            new_lora_ref = self.lora_ref_cache[lora_path]
            load_result = await self.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name=new_lora_ref.lora_name,
                    lora_path=new_lora_ref.lora_path,
                    pinned=new_lora_ref.pinned,
                ),
            )
            if (
                not load_result.success
                and "already loaded" not in load_result.error_message
            ):
                raise ValueError(
                    f"Failed to implicitly load LoRA adapter {lora_path}: {load_result.error_message}"
                )

        # Look up the LoRA ID from the registry and start tracking ongoing LoRA requests.
        obj.lora_id = await self.lora_registry.acquire(obj.lora_path)
        # Propagate lora_id to any sub-objects already cached by __getitem__.
        for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():
            sub_obj.lora_id = (
                obj.lora_id[i] if isinstance(obj.lora_id, list) else obj.lora_id
            )
