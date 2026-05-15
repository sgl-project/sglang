import logging
import time
import uuid
from typing import Any, List, Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from .nixl_utils import (
    NixlBackendConfig,
    NixlBackendSelection,
    NixlFileManager,
    NixlRegistration,
)

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as e:
    raise ImportError(
        "Please install NIXL by following the instructions at "
        "https://github.com/ai-dynamo/nixl/blob/main/README.md "
        "to use HiCacheNixl storage backend."
    ) from e

logger = logging.getLogger(__name__)


class HiCacheNixl(HiCacheStorage):
    """HiCacheNixl provides high-performance storage using NIXL plugins."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache_storage",
    ):
        """Initialize NIXL storage connector."""

        # create nixlconfig from the --hicache-storage-backend-extra-config
        nixlconfig = NixlBackendConfig(storage_config.extra_config)

        # select the NIXL backend plugin from extra_config or environment variable
        plugin = nixlconfig.get_specified_plugin()

        # Might be better to be unified across HiCache backends and moved to HiCacheController
        file_path = envs.SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR.get() or file_path
        self.file_manager = (
            NixlFileManager(file_path)
            if plugin not in NixlBackendSelection.OBJ_PLUGINS
            else None
        )

        # Initialize suffix based on storage config
        tp_rank, tp_size, model_name = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
        )

        self.is_mla_model = storage_config.is_mla_model

        model_name = "-".join(model_name.split("/")) if model_name else ""

        if self.is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        agent_config = nixl_agent_config(backends=[])
        self.agent_name = f"hicache_nixl_{str(uuid.uuid4())}"
        self.agent = nixl_agent(self.agent_name, agent_config)

        self.backend_selector = NixlBackendSelection(plugin, nixlconfig)
        if not self.backend_selector.create_backend(self.agent):
            raise RuntimeError("Failed to create NIXL backend")

        self.registration = NixlRegistration(self.agent)
        self.is_zero_copy = False
        self.registered_pools = {}

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(
        self, key: str, component_name: Optional[PoolName] = None
    ) -> str:
        if component_name in (None, PoolName.KV):
            return self._get_suffixed_key(key)
        return f"{self._get_suffixed_key(key)}.{component_name}"

    def register_buffers(
        self, buffers: Union[torch.Tensor, List[torch.Tensor], List[tuple]]
    ) -> Optional[Any]:
        """Register tensor(s) or target locations in host memory (list of addr,len tuples) with NIXL."""
        if isinstance(buffers[0], tuple):
            tuples = [(x[0], x[1], 0, "") for x in buffers]
            return self.registration._register_memory(tuples, "DRAM")
        else:
            return self.registration._register_memory(buffers)

    def register_files(
        self, file_paths: List[str], open_file: Optional[bool] = True
    ) -> Optional[Any]:
        """Register files with NIXL."""
        tuples = self.file_manager.files_to_nixl_tuples(file_paths)
        return self.registration._register_memory(tuples, "FILE")

    def register_objects(
        self, keys: List[str], sizes: Optional[List[int]] = None
    ) -> Optional[Any]:
        """Register objects with NIXL."""
        if not keys:
            return None
        tuples = [(0, 0, key, "") for key in keys]
        return self.registration._register_memory(tuples, "OBJ")

    def _execute_transfer(
        self,
        buffers: Optional[List[torch.Tensor | tuple]],
        keys: List[str],
        direction: str,
    ) -> bool:
        if len(buffers) != len(keys):
            logger.error("Mismatch between number of tensors/buffers and files/objects")
            return False

        # Registering file and object keys per transfer, to be updated when
        # pre-registration for file and object is added to HiCache.
        file_fds = []
        try:
            if self.backend_selector.mem_type == "FILE":
                tuples = self.file_manager.files_to_nixl_tuples(keys)
                file_fds = [t[2] for t in tuples]
                if not tuples or not self.registration._register_memory(tuples, "FILE"):
                    logger.error("Failed to prepare files for transfer")
                    return False
            else:  # mem_type == "OBJ"
                tuples = [(0, 0, key, "") for key in keys]
                if not tuples or not self.registration._register_memory(tuples, "OBJ"):
                    logger.error("Failed to register objects")
                    return False

            # Prepare transfer descriptors
            if isinstance(buffers[0], torch.Tensor):
                tensor_sizes = [
                    tensor.element_size() * tensor.numel() for tensor in buffers
                ]
                storage_tuples = [(x[0], s, x[2]) for x, s in zip(tuples, tensor_sizes)]
                host_descs = self.agent.get_xfer_descs(buffers)

                if direction in ("READ", "WRITE"):
                    # register buffer to avoid calling initialize_xfer twice due to missing registration
                    self.register_buffers(buffers)

            elif isinstance(buffers[0], tuple):
                storage_tuples = [(x[0], y[1], x[2]) for x, y in zip(tuples, buffers)]
                host_descs = self.agent.get_xfer_descs(
                    [(x[0], x[1], 0) for x in buffers], "DRAM"
                )

                if direction in ("READ", "WRITE"):
                    # register buffer to avoid calling initialize_xfer twice due to missing registration
                    self.register_buffers(buffers)

            else:
                return False

            storage_descs = self.agent.get_xfer_descs(
                storage_tuples, self.backend_selector.mem_type
            )

            if (host_descs is None) or (storage_descs is None):
                logger.error("Failed to get transfer descriptors")
                return False

            # Initialize transfer, default assumption that tensor was registered

            try:
                xfer_req = self.agent.initialize_xfer(
                    direction, host_descs, storage_descs, self.agent_name
                )
            except Exception:
                # Check if it was due to missing pre-registration
                if not self.register_buffers(buffers):
                    logger.error("Failed to register tensors/buffers")
                    return False

                try:
                    xfer_req = self.agent.initialize_xfer(
                        direction, host_descs, storage_descs, self.agent_name
                    )
                except Exception as e:
                    logger.error(f"Failed to create transfer request: {e}")
                    return False

            # Execute transfer and wait for its completion
            try:
                state = self.agent.transfer(xfer_req)
                while state != "DONE":
                    state = self.agent.check_xfer_state(xfer_req)
                    if state == "ERR":
                        self.agent.release_xfer_handle(xfer_req)
                        logger.error("Transfer failed")
                        return False
                    time.sleep(
                        0.0001
                    )  # Can be changed to os.sched_yield() or parametrized

                self.agent.release_xfer_handle(xfer_req)
                return True

            except Exception as e:
                logger.error(f"Failed to execute transfer: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                return False

        finally:
            for fd in file_fds:
                self.file_manager.close_file(fd)

    def _transfer_by_keys(
        self,
        buffers: List[torch.Tensor | tuple],
        keys: List[str],
        direction: str,
    ) -> bool:
        if self.backend_selector.mem_type == "FILE":
            file_paths = []
            for key in keys:
                file_path = self.file_manager.get_file_path(key)
                if direction == "WRITE" and not self.file_manager.create_file(file_path):
                    logger.error(f"Failed to create file {file_path}")
                    return False
                file_paths.append(file_path)
            return self._execute_transfer(buffers, file_paths, direction)
        return self._execute_transfer(buffers, keys, direction)

    def _query_keys_exist(self, keys: List[str]) -> List[bool]:
        if not keys:
            return []
        tuples = []
        for key in keys:
            tuples += self.registration.create_query_tuples(
                key,
                self.backend_selector.mem_type,
                self.file_manager if self.backend_selector.mem_type == "FILE" else None,
            )
        query_res = self.agent.query_memory(
            tuples,
            self.backend_selector.backend_name,
            mem_type=self.backend_selector.mem_type,
        )
        return [res is not None for res in query_res]

    def _get_component_keys(
        self, keys: List[str], pool_name: Optional[PoolName] = None
    ) -> List[str]:
        return [self._get_component_key(key, pool_name) for key in keys]

    def _prepare_pool_transfer(
        self, transfer: PoolTransfer, for_write: bool
    ) -> tuple[Optional[HostKVCache], List[str], List[torch.Tensor], List[int]]:
        host_pool = self.registered_pools.get(transfer.name)
        if host_pool is None:
            logger.error("Host pool %s is not registered in HiCacheNixl", transfer.name)
            return None, [], [], []

        keys = transfer.keys or []
        host_indices = transfer.host_indices
        page_size = getattr(host_pool, "page_size", 1) or 1
        expected = len(keys) * page_size
        if host_indices is None or host_indices.numel() != expected:
            logger.error(
                "Pool %s indices length mismatch: expected %s, got %s",
                transfer.name,
                expected,
                host_indices.numel() if host_indices is not None else 0,
            )
            return host_pool, [], [], []

        page_offsets = [
            host_indices[i * page_size].item() for i in range(len(keys))
        ]
        if for_write:
            buffers = [
                host_pool.get_data_page(page_offset, flat=True).contiguous()
                for page_offset in page_offsets
            ]
        else:
            buffers = [host_pool.get_dummy_flat_data_page() for _ in page_offsets]

        return host_pool, self._get_component_keys(keys, transfer.name), buffers, page_offsets

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor | int] = None,
        target_sizes: Optional[int] = None,
    ) -> torch.Tensor | None:
        # To be removed, being compatible with the current API
        if target_location is None:
            return None
        if target_sizes:
            result = self.batch_get([key], [target_location], [target_sizes])
        else:
            result = self.batch_get([key], [target_location])
        return result[0] if result else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor | int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> List[torch.Tensor | None]:
        if not keys:
            return []

        # To be removed, being compatible with the current API
        if not target_locations:
            return [None] * len(keys)

        if target_sizes and (len(target_sizes) != len(target_locations)):
            logger.error("Mismatch between number of target_locations and target_sizes")
            return [None] * len(keys)

        if target_sizes:
            dest = list(zip(target_locations, target_sizes))
        else:
            dest = target_locations

        # Add suffix to keys
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        success = self._transfer_by_keys(dest, suffixed_keys, "READ")
        return target_locations if success and not target_sizes else [None] * len(keys)

    def set(
        self,
        key: str,
        value: Optional[torch.Tensor] = None,
        target_location: Optional[int] = None,
        target_sizes: Optional[int] = None,
    ) -> bool:
        if target_location and target_sizes:
            return self.batch_set([key], None, [target_location], [target_sizes])
        else:
            return self.batch_set([key], [value])

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        if not keys or (not values and (not target_locations or not target_sizes)):
            logger.error("Keys or values were not passed")
            return False

        if not values:
            values = list(zip(target_locations, target_sizes))

        # Add suffix to keys
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        return self._transfer_by_keys(values, suffixed_keys, "WRITE")

    ############################################################################
    # batch_*_v1 functions
    # zero copy + non-zero-copy version for get, set, exists, batch_exists
    ############################################################################

    def clear(self) -> None:
        if self.file_manager is not None:
            self.file_manager.clear()

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)

        # enable zero-copy automatically if mem layout is page_first or page_first_direct
        self.is_zero_copy = self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ]

        logger.info(
            f"HiCacheNixl: Registered mem_pool_host with layout {self.mem_pool_host.layout}, zero_copy set to {self.is_zero_copy}"
        )

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)

    def exists(self, key: str) -> bool:
        results = self.batch_exists([key])
        return results > 0

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        # Add suffix to key

        if self.is_zero_copy:
            key_list = self._get_key_list_from_meta(keys)
            key_denominator = (
                1 if self.is_mla_model else 2
            )  # MLA has one interleaved buffer; non-MLA has separate k/v buffers
        else:
            key_list = [self._get_suffixed_key(key) for key in keys]
            key_denominator = 1

        exists_results = self._query_keys_exist(key_list)

        for i, exists in enumerate(exists_results):
            if not exists:
                return i // key_denominator
        return len(exists_results) // key_denominator

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)
        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            if transfer.name not in self.registered_pools:
                final_pages = 0
                break

            component_keys = self._get_component_keys(keys[:kv_pages], transfer.name)
            exists_results = self._query_keys_exist(component_keys)

            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                try:
                    boundary = exists_results.index(False)
                except ValueError:
                    boundary = kv_pages
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        exists_results[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break

            if boundary:
                hit_count[transfer.name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _get_key_list_from_meta(self, keys: List[str]) -> List[str]:
        # construct the key list for NIXL transfer based on the keys and the suffix, for each key, we will have one suffixed key for k buffer and one suffixed key for v buffer if it's not an MLA model, and only one suffixed key for k buffer if it's an MLA model, since MLA model only has k/v interleaved buffer
        key_list = []

        for key_ in keys:
            suffixed_key = self._get_suffixed_key(key_)
            if self.is_mla_model:
                key_list.append(f"{suffixed_key}_k")
            else:
                key_list.append(f"{suffixed_key}_k")
                key_list.append(f"{suffixed_key}_v")

        return key_list

    def _get_location_and_size_list_from_meta(
        self, keys: List[str], host_indices: torch.Tensor
    ):
        # zero copy: mem_pool_host.get_data_page() does not work due to non-contiguous tensors, causing issues for NIXL transfer
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(
            host_indices
        )
        key_list = self._get_key_list_from_meta(keys)

        if len(key_list) != len(ptr_list):
            logger.error(
                f"HiCacheNixl: mismatch between number of keys and number of buffer meta entries, keys: {len(keys)}, key_list: {len(key_list)}, buffer meta entries: {len(ptr_list)}"
            )
            return [], [], [], []

        return key_list, [], ptr_list, element_size_list

    def _batch_get_preprocess(self, keys: List[str], host_indices: torch.Tensor):
        page_num = len(host_indices) // self.mem_pool_host.page_size

        if len(keys) == 0 or len(keys) != page_num:
            logger.warning(
                f"HiCacheNixl: empty keys or mismatch in keys and host_indices lengths. keys: {len(keys)}, host_indices: {len(host_indices)}, page_size: {self.mem_pool_host.page_size}"
            )
            return [], [], [], []

        if self.is_zero_copy:
            key_list, _, ptr_list, element_size_list = (
                self._get_location_and_size_list_from_meta(keys, host_indices)
            )
            return key_list, [], ptr_list, element_size_list
        else:
            # non zero copy: create contiguous, temporary tensors
            target_tensors = [
                self.mem_pool_host.get_dummy_flat_data_page() for i in range(page_num)
            ]

            key_list = [self._get_suffixed_key(key) for key in keys]
            ptr_list = [tensor.data_ptr() for tensor in target_tensors]
            element_size_list = [
                tensor.numel() * tensor.element_size() for tensor in target_tensors
            ]

            return key_list, target_tensors, ptr_list, element_size_list

    def _batch_get_zero_copy_impl(
        self,
        keys: List[str],
        key_strs: List[str],
        target_tensors: List[torch.Tensor],
        target_locations: List[int],
        target_sizes: List[int],
    ) -> List[int]:

        if not key_strs or not target_locations or not target_sizes:
            return [False] * len(keys)

        if (len(key_strs) != len(target_locations)) or (
            len(target_sizes) != len(target_locations)
        ):
            logger.error(
                "Mismatch between number of key_strs, target_locations and target_sizes"
            )
            return [False] * len(keys)

        if self.is_zero_copy:
            dest = list(zip(target_locations, target_sizes))
        else:
            dest = target_tensors

        success = self._transfer_by_keys(dest, key_strs, "READ")

        return [True] * len(key_strs) if success else [False] * len(key_strs)

    def _batch_get_postprocess(
        self,
        host_indices: torch.Tensor,
        target_tensors: List[torch.Tensor],
        results: List[bool],
    ) -> List[bool]:

        page_num = len(host_indices) // self.mem_pool_host.page_size

        if self.is_zero_copy:
            # zero copy: update final results based on the boolean results from NIXL transfer
            if self.is_mla_model:
                return results
            else:
                results = [
                    (results[2 * i] and results[2 * i + 1]) for i in range(page_num)
                ]
                return results
        else:
            # non zero copy: copy data from temporary tensors to mem_pool_host page by page
            for i in range(page_num):
                if not results[i]:
                    break
                self.mem_pool_host.set_from_flat_data_page(
                    host_indices[i * self.mem_pool_host.page_size], target_tensors[i]
                )

            return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:

        key_strs, target_tensors, buffer_ptrs, buffer_sizes = (
            self._batch_get_preprocess(keys, host_indices)
        )

        if not key_strs or not buffer_ptrs or not buffer_sizes:
            logger.error(
                "HiCacheNixl batch_get_v1: preprocessing failed, empty key_strs, buffer_ptrs or buffer_sizes"
            )
            return [False] * len(keys)

        start_time = time.perf_counter()

        results_get = self._batch_get_zero_copy_impl(
            keys, key_strs, target_tensors, buffer_ptrs, buffer_sizes
        )

        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        total_bytes = sum(s for s in buffer_sizes if s is not None)

        logger.debug(
            f"HiCacheNixl batch_get_v1 transferred: {len(keys)} keys (pages), {host_indices.numel()} host_indices, {total_bytes} bytes, total time: {elapsed_time_ms:.3f} ms, effective bandwidth: {total_bytes / (elapsed_time_ms / 1000) / (1024 * 1024):.2f} MB/s"
        )

        return self._batch_get_postprocess(host_indices, target_tensors, results_get)

    def _batch_set_preprocess(self, keys: List[str], host_indices: torch.Tensor):

        page_num = len(host_indices) // self.mem_pool_host.page_size

        if len(keys) == 0 or len(keys) != page_num:
            logger.warning(
                f"HiCacheNixl: empty keys or mismatch in keys and host_indices lengths. keys: {len(keys)}, host_indices: {len(host_indices)}, page_size: {self.mem_pool_host.page_size}"
            )
            return [], [], [], []

        if self.is_zero_copy:
            key_list, _, ptr_list, element_size_list = (
                self._get_location_and_size_list_from_meta(keys, host_indices)
            )
            return key_list, [], ptr_list, element_size_list
        else:
            # non zero copy: NIXL still requires contiguous tensors for transfer
            target_tensors = [
                self.mem_pool_host.get_data_page(
                    host_indices[i * self.mem_pool_host.page_size], flat=False
                ).contiguous()
                for i in range(page_num)
            ]

            key_list = [self._get_suffixed_key(key) for key in keys]
            ptr_list = [tensor.data_ptr() for tensor in target_tensors]
            element_size_list = [
                tensor.numel() * tensor.element_size() for tensor in target_tensors
            ]

            return key_list, target_tensors, ptr_list, element_size_list

    def _batch_set_zero_copy_impl(
        self,
        keys: List[str],
        key_strs: List[str],
        target_tensors: List[torch.Tensor],
        target_locations: List[int],
        target_sizes: List[int],
    ) -> List[bool]:

        if not key_strs or not target_locations or not target_sizes:
            return [False] * len(keys)

        if (len(key_strs) != len(target_locations)) or (
            len(target_sizes) != len(target_locations)
        ):
            logger.error(
                "Mismatch between number of key_strs, target_locations and target_sizes"
            )
            return [False] * len(keys)

        if self.is_zero_copy:
            src = list(zip(target_locations, target_sizes))
        else:
            src = target_tensors

        success = self._transfer_by_keys(src, key_strs, "WRITE")

        return [True] * len(keys) if success else [False] * len(keys)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:

        if len(keys) == 0:
            return []

        key_strs, target_tensors, buffer_ptrs, buffer_sizes = (
            self._batch_set_preprocess(keys, host_indices)
        )

        if not key_strs or not buffer_ptrs or not buffer_sizes:
            logger.error(
                "HiCacheNixl batch_set_v1: preprocessing failed, empty key_strs, buffer_ptrs or buffer_sizes"
            )
            return [False] * len(keys)

        start_time = time.perf_counter()

        results_set = self._batch_set_zero_copy_impl(
            keys, key_strs, target_tensors, buffer_ptrs, buffer_sizes
        )

        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        total_bytes = sum(s for s in buffer_sizes if s is not None)
        logger.debug(
            f"HiCacheNixl batch_set_v1 transferred: {len(keys)} keys (pages), {host_indices.numel()} host_indices, {total_bytes} bytes, total time: {elapsed_time_ms:.3f} ms, effective bandwidth: {total_bytes / (elapsed_time_ms / 1000) / (1024 * 1024):.2f} MB/s"
        )

        return results_set

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool, key_strs, buffers, page_offsets = self._prepare_pool_transfer(
                transfer, for_write=False
            )
            if host_pool is None or not key_strs:
                results[transfer.name] = [False] * len(transfer.keys or [])
                continue

            success = self._transfer_by_keys(buffers, key_strs, "READ")
            if not success:
                results[transfer.name] = [False] * len(key_strs)
                continue

            for page_offset, data_page in zip(page_offsets, buffers):
                host_pool.set_from_flat_data_page(page_offset, data_page)
            results[transfer.name] = [True] * len(key_strs)
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            _, key_strs, buffers, _ = self._prepare_pool_transfer(
                transfer, for_write=True
            )
            if not key_strs:
                results[transfer.name] = [False] * len(transfer.keys or [])
                continue

            success = self._transfer_by_keys(buffers, key_strs, "WRITE")
            results[transfer.name] = [True] * len(key_strs) if success else [False] * len(
                key_strs
            )
        return results
