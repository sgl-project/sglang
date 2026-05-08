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

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    # ------------------------------------------------------------------
    # Memory registration helpers
    # ------------------------------------------------------------------

    def _register_buffers(
        self, buffers: Union[torch.Tensor, List[torch.Tensor], List[tuple]]
    ) -> Optional[Any]:
        """Register tensor(s) or target locations in host memory (list of addr,len tuples) with NIXL."""
        if isinstance(buffers[0], tuple):
            tuples = [(x[0], x[1], 0, "") for x in buffers]
            return self.registration._register_memory(tuples, "DRAM")
        else:
            return self.registration._register_memory(buffers)

    def _register_files(
        self, file_paths: List[str], sizes: List[int]
    ) -> Optional[Any]:
        """Open and register files with NIXL.

        Returns (reg_descs, fds).  The fds must stay open until the registration
        is no longer needed; call _deregister_files() to release in the correct
        order (deregister -> close fd).  Returns None on any failure.
        """
        fds: List[int] = []
        tuples = []
        for i, path in enumerate(file_paths):
            fd = self.file_manager.open_file(path)
            if fd is None:
                for f in fds:
                    self.file_manager.close_file(f)
                return None
            fds.append(fd)
            tuples.append((0, sizes[i], fd, path))
        reg_descs = self.registration._register_memory(tuples, "FILE")
        if not reg_descs:
            for fd in fds:
                self.file_manager.close_file(fd)
            return None
        return reg_descs, fds

    def _deregister_files(self, reg_descs: Any, fds: List[int]) -> None:
        """Deregister files and close their file descriptors.

        Must be called after _register_files().  Correct teardown order:
        deregister_memory -> close fd.
        """
        try:
            self.agent.deregister_memory(reg_descs)
        except Exception as e:
            logger.debug("deregister_memory skipped: %s", e)
        for fd in fds:
            self.file_manager.close_file(fd)

    def _register_objects(self, keys: List[str]) -> Optional[Any]:
        """Register objects with NIXL."""
        tuples = [(0, 0, key, "") for key in keys]
        return self.registration._register_memory(tuples, "OBJ")

    def _create_files_for_keys(
        self, suffixed_keys: List[str]
    ) -> Optional[List[str]]:
        """Create empty files for each key; return their paths or None on first failure."""
        # New file per set, to be updated when partial writes is added to HiCache
        file_paths = []
        for key in suffixed_keys:
            file_path = self.file_manager.get_file_path(key)
            if not self.file_manager.create_file(file_path):
                logger.error(f"Failed to create file {file_path}")
                return None
            file_paths.append(file_path)
        return file_paths

    # ------------------------------------------------------------------
    # Transfer pipeline
    # ------------------------------------------------------------------

    def _get_storage_descs(
        self, buffers: List[torch.Tensor | tuple], keys: List[str]
    ) -> tuple:
        """Register storage (FILE or OBJ) and return (storage_descs, reg_descs, fds).

        For FILE: opens fds, registers with actual sizes, returns xfer descs via
        reg_descs.trim() per NIXL API contract.  Caller must call
        _deregister_files(reg_descs, fds) when done.
        For OBJ: builds xfer descs from buffer sizes; reg_descs and fds are None/[].
        Returns (None, None, []) on failure.
        """
        if isinstance(buffers[0], torch.Tensor):
            sizes = [b.element_size() * b.numel() for b in buffers]
        elif isinstance(buffers[0], tuple):
            sizes = [b[1] for b in buffers]
        else:
            return None, None, []

        if self.backend_selector.mem_type == "FILE":
            result = self._register_files(keys, sizes)
            if result is None:
                logger.error("Failed to register files for transfer")
                return None, None, []
            reg_descs, fds = result
            storage_descs = self.agent.get_xfer_descs(
                [(0, sizes[i], fds[i]) for i in range(len(fds))],
                "FILE",
            )
            return storage_descs, reg_descs, fds
        else:  # OBJ
            if not self._register_objects(keys):
                logger.error("Failed to register objects")
                return None, None, []
            storage_descs = self.agent.get_xfer_descs(
                [(0, size, key) for size, key in zip(sizes, keys)],
                self.backend_selector.mem_type,
            )
            if storage_descs is None:
                logger.error("Failed to get storage xfer descs")
                return None, None, []
            return storage_descs, None, []

    def _get_host_descs(
        self, buffers: List[torch.Tensor | tuple]
    ) -> Optional[Any]:
        """Build host xfer descs and register buffers for the transfer.

        Returns host_descs or None on failure.
        """
        if isinstance(buffers[0], torch.Tensor):
            host_descs = self.agent.get_xfer_descs(buffers)
            self._register_buffers(buffers)
        elif isinstance(buffers[0], tuple):
            host_descs = self.agent.get_xfer_descs(
                [(x[0], x[1], 0) for x in buffers], "DRAM"
            )
            self._register_buffers(buffers)
        else:
            return None

        if host_descs is None:
            logger.error("Failed to get host transfer descriptors")
            return None
        return host_descs

    def _run_xfer(
        self,
        host_descs: Any,
        storage_descs: Any,
        direction: str,
        buffers: List[torch.Tensor | tuple],
    ) -> bool:
        """Initialize and poll a NIXL transfer to completion."""
        try:
            xfer_req = self.agent.initialize_xfer(
                direction, host_descs, storage_descs, self.agent_name
            )
        except Exception:
            # Retry once after ensuring buffers are registered
            if not self._register_buffers(buffers):
                logger.error("Failed to register tensors/buffers")
                return False
            try:
                xfer_req = self.agent.initialize_xfer(
                    direction, host_descs, storage_descs, self.agent_name
                )
            except Exception as e:
                logger.error(f"Failed to create transfer request: {e}")
                return False

        try:
            state = self.agent.transfer(xfer_req)
            while state != "DONE":
                state = self.agent.check_xfer_state(xfer_req)
                if state == "ERR":
                    self.agent.release_xfer_handle(xfer_req)
                    logger.error("Transfer failed")
                    return False
                time.sleep(0.0001)  # Can be changed to os.sched_yield() or parametrized
            self.agent.release_xfer_handle(xfer_req)
            return True
        except Exception as e:
            logger.error(f"Failed to execute transfer: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _execute_transfer(
        self,
        buffers: Optional[List[torch.Tensor | tuple]],
        keys: List[str],
        direction: str,
    ) -> bool:
        if len(buffers) != len(keys):
            logger.error("Mismatch between number of tensors/buffers and files/objects")
            return False

        reg_descs, fds = None, []
        try:
            storage_descs, reg_descs, fds = self._get_storage_descs(buffers, keys)
            if storage_descs is None:
                return False
            host_descs = self._get_host_descs(buffers)
            if host_descs is None:
                return False
            return self._run_xfer(host_descs, storage_descs, direction, buffers)
        finally:
            if reg_descs is not None:
                self._deregister_files(reg_descs, fds)

    # ------------------------------------------------------------------
    # Legacy single-item / batch API (required by HiCacheStorage ABC)
    # ------------------------------------------------------------------

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor | int] = None,
        target_sizes: Optional[int] = None,
    ) -> torch.Tensor | None:
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
        if not target_locations:
            return [None] * len(keys)

        if target_sizes and (len(target_sizes) != len(target_locations)):
            logger.error("Mismatch between number of target_locations and target_sizes")
            return [None] * len(keys)

        if target_sizes:
            dest = list(zip(target_locations, target_sizes))
        else:
            dest = target_locations

        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        if self.backend_selector.mem_type == "FILE":
            file_paths = [self.file_manager.get_file_path(key) for key in suffixed_keys]
            success = self._execute_transfer(dest, file_paths, "READ")
        else:
            success = self._execute_transfer(dest, suffixed_keys, "READ")
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

        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        if self.backend_selector.mem_type == "FILE":
            file_paths = self._create_files_for_keys(suffixed_keys)
            if file_paths is None:
                return False
            return self._execute_transfer(values, file_paths, "WRITE")
        else:  # mem_type == "OBJ"
            return self._execute_transfer(values, suffixed_keys, "WRITE")

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

    def exists(self, key: str) -> bool:
        results = self.batch_exists([key])
        return results > 0

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        if self.is_zero_copy:
            key_list = self._get_key_list_from_meta(keys)
            # MLA model only has k buffer, no separate v buffer
            key_denominator = 1 if not self.is_mla_model else 2
        else:
            key_list = [self._get_suffixed_key(key) for key in keys]
            key_denominator = 1

        tuples = []
        for key in key_list:
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

        for i in range(len(query_res)):
            if query_res[i] is None:
                return i // key_denominator
        return len(query_res) // key_denominator

    # ------------------------------------------------------------------
    # batch_*_v1 implementation (zero-copy + non-zero-copy)
    # ------------------------------------------------------------------

    def _get_key_list_from_meta(self, keys: List[str]) -> List[str]:
        # Each key maps to a `_k` entry, plus a `_v` entry on non-MLA models
        # (MLA stores k/v interleaved in a single buffer).
        key_list = []
        for key in keys:
            suffixed_key = self._get_suffixed_key(key)
            key_list.append(f"{suffixed_key}_k")
            if not self.is_mla_model:
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
            return [], [], []

        return key_list, ptr_list, element_size_list

    def _batch_preprocess(
        self, keys: List[str], host_indices: torch.Tensor, op: str
    ):
        """Build (key_list, target_tensors, ptr_list, size_list) for the v1 path.

        ``op`` is "get" (allocate dummy flat target tensors) or
        "set" (snapshot current host pages as contiguous source tensors).
        Returns ([], [], [], []) on validation failure.
        """
        page_size = self.mem_pool_host.page_size
        page_num = len(host_indices) // page_size

        if len(keys) == 0 or len(keys) != page_num:
            logger.warning(
                f"HiCacheNixl: empty keys or mismatch in keys and host_indices lengths. keys: {len(keys)}, host_indices: {len(host_indices)}, page_size: {page_size}"
            )
            return [], [], [], []

        if self.is_zero_copy:
            key_list, ptr_list, size_list = self._get_location_and_size_list_from_meta(
                keys, host_indices
            )
            return key_list, [], ptr_list, size_list

        # non zero copy: build contiguous tensors NIXL can transfer
        if op == "get":
            target_tensors = [
                self.mem_pool_host.get_dummy_flat_data_page() for _ in range(page_num)
            ]
        else:  # "set"
            target_tensors = [
                self.mem_pool_host.get_data_page(
                    host_indices[i * page_size], flat=False
                ).contiguous()
                for i in range(page_num)
            ]

        key_list = [self._get_suffixed_key(key) for key in keys]
        ptr_list = [t.data_ptr() for t in target_tensors]
        size_list = [t.numel() * t.element_size() for t in target_tensors]
        return key_list, target_tensors, ptr_list, size_list

    def _batch_xfer(
        self,
        keys: List[str],
        key_strs: List[str],
        target_tensors: List[torch.Tensor],
        target_locations: List[int],
        target_sizes: List[int],
        direction: str,
    ) -> List[bool]:
        """Run a batch READ or WRITE for the v1 path."""
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
            buffers = list(zip(target_locations, target_sizes))
        else:
            buffers = target_tensors

        if self.backend_selector.mem_type == "FILE":
            if direction == "WRITE":
                file_paths = self._create_files_for_keys(key_strs)
                if file_paths is None:
                    return [False] * len(keys)
            else:
                file_paths = [
                    self.file_manager.get_file_path(key) for key in key_strs
                ]
            success = self._execute_transfer(buffers, file_paths, direction)
        else:  # mem_type == "OBJ"
            success = self._execute_transfer(buffers, key_strs, direction)

        return [success] * len(keys)

    def _batch_get_postprocess(
        self,
        host_indices: torch.Tensor,
        target_tensors: List[torch.Tensor],
        results: List[bool],
    ) -> List[bool]:
        page_size = self.mem_pool_host.page_size
        page_num = len(host_indices) // page_size

        if self.is_zero_copy:
            # zero copy: update final results based on the boolean results from NIXL transfer
            if self.is_mla_model:
                return results
            return [(results[2 * i] and results[2 * i + 1]) for i in range(page_num)]

        # non zero copy: copy data from temporary tensors to mem_pool_host page by page
        for i in range(page_num):
            if not results[i]:
                break
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * page_size], target_tensors[i]
            )
        return results

    def _log_xfer_stats(
        self,
        op_name: str,
        num_keys: int,
        host_indices: torch.Tensor,
        buffer_sizes: List[int],
        elapsed_ms: float,
    ) -> None:
        total_bytes = sum(s for s in buffer_sizes if s is not None)
        bw = total_bytes / (elapsed_ms / 1000) / (1024 * 1024) if elapsed_ms else 0.0
        logger.debug(
            f"HiCacheNixl {op_name} transferred: {num_keys} keys (pages), "
            f"{host_indices.numel()} host_indices, {total_bytes} bytes, "
            f"total time: {elapsed_ms:.3f} ms, effective bandwidth: {bw:.2f} MB/s"
        )

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        key_strs, target_tensors, buffer_ptrs, buffer_sizes = self._batch_preprocess(
            keys, host_indices, "get"
        )

        if not key_strs or not buffer_ptrs or not buffer_sizes:
            logger.error(
                "HiCacheNixl batch_get_v1: preprocessing failed, empty key_strs, buffer_ptrs or buffer_sizes"
            )
            return [False] * len(keys)

        start_time = time.perf_counter()
        results = self._batch_xfer(
            keys, key_strs, target_tensors, buffer_ptrs, buffer_sizes, "READ"
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._log_xfer_stats(
            "batch_get_v1", len(keys), host_indices, buffer_sizes, elapsed_ms
        )

        return self._batch_get_postprocess(host_indices, target_tensors, results)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if len(keys) == 0:
            return []

        key_strs, target_tensors, buffer_ptrs, buffer_sizes = self._batch_preprocess(
            keys, host_indices, "set"
        )

        if not key_strs or not buffer_ptrs or not buffer_sizes:
            logger.error(
                "HiCacheNixl batch_set_v1: preprocessing failed, empty key_strs, buffer_ptrs or buffer_sizes"
            )
            return [False] * len(keys)

        start_time = time.perf_counter()
        results = self._batch_xfer(
            keys, key_strs, target_tensors, buffer_ptrs, buffer_sizes, "WRITE"
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._log_xfer_stats(
            "batch_set_v1", len(keys), host_indices, buffer_sizes, elapsed_ms
        )

        return results
