from queue import Empty, Queue
from typing import Optional

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager


class C_HiCacheController(BaseHook):
    HOOK_CLASS_NAME = "HiCacheController"
    HOOK_MODULE_NAME = "sglang.srt.managers.cache_controller"
    REQUIRED = False

    KV_CACHE_BYTES: Optional[int] = None
    DISK_READ_BANDWIDTH_BYTES: Optional[float] = None
    DISK_WRITE_BANDWIDTH_BYTES: Optional[float] = None

    @staticmethod
    def calc_prefetch_pages(
        required_pages: int, page_size_byte: int, max_dur: float, bandwidth: float
    ) -> tuple[float, float]:
        _prefetch_dur = required_pages * page_size_byte / bandwidth
        if _prefetch_dur > max_dur:
            _completed_pages = max(max_dur * bandwidth / page_size_byte, 1)
            return _completed_pages, max_dur
        else:
            return required_pages, _prefetch_dur

    @classmethod
    def hook(cls, target):

        original_terminate_prefetch = target.terminate_prefetch
        original_storage_hit_query = target._storage_hit_query
        original_init = target.__init__
        original_append_host_mem_release = target.append_host_mem_release

        def wrapped_init(self, *args, **kwargs):
            self.sim_prefetch_buffer = Queue()
            result = original_init(self, *args, **kwargs)
            # The real IO thread normally creates this queue. The simulator
            # replaces that thread, so initialize the handoff queue here.
            if hasattr(self, "prefetch_hit_queue"):
                self.prefetch_buffer = Queue()
            return result

        def wrapped_append_host_mem_release(self, host_indices):
            # A terminated prefetch may not have allocated host memory yet.
            if host_indices is None:
                return
            return original_append_host_mem_release(self, host_indices)

        def override_backup_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_backup_operation`
            pass

        def override_prefetch_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_prefetch_operation`
            pass

        def handle_backup_operation(self):
            if not self.enable_storage:
                return
            while True:
                try:
                    operation = self.backup_queue.get(block=False)
                    if operation is None:
                        return

                    if not self.backup_skip:
                        self._page_backup(operation)
                    # TODO: Track the backup operation according to the global clock
                    self.ack_backup_queue.put(operation)

                except Empty:
                    return

        def handle_prefetch_operation(self):
            if not self.enable_storage:
                return

            if C_HiCacheController.KV_CACHE_BYTES is None:
                C_HiCacheController.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()
            if C_HiCacheController.DISK_READ_BANDWIDTH_BYTES is None:
                C_HiCacheController.DISK_READ_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().disk_read_bandwidth
                )

            # TODO: Overlap schedule
            remain_dur = StateManager.get_current_inference_dur()

            # Process all operations in the prefetch_queue: place those meeting
            # the prefetch criteria into the sim_prefetch_buffer, and release the
            # remaining operations along with any excess memory they have allocated.
            while not self.prefetch_queue.empty():
                try:
                    operation = self.prefetch_queue.get(block=False)
                    if operation is None:
                        break

                    # Ignore terminated operation
                    if operation._terminated_flag:
                        if hasattr(self, "prefetch_revoke_queue"):
                            self.prefetch_revoke_queue.put(operation.request_id)
                        else:
                            self.append_host_mem_release(operation.host_indices)
                        continue

                    hash_value, storage_hit_count = self._storage_hit_query(operation)
                    # not to prefetch if not enough benefits
                    if (
                        self.prefetch_threshold is not None
                        and storage_hit_count < self.prefetch_threshold
                    ):
                        if hasattr(self, "prefetch_revoke_queue"):
                            self.prefetch_revoke_queue.put(operation.request_id)
                            continue
                        operation.mark_terminate()
                        self.append_host_mem_release(operation.host_indices)
                        continue
                    else:
                        operation.hash_value = hash_value[
                            : (storage_hit_count // self.page_size)
                        ]
                        if hasattr(self, "prefetch_hit_queue"):
                            # Allocate only the storage-hit range on the scheduler.
                            operation.storage_hit_count = storage_hit_count
                            self.prefetch_hit_queue.put(operation)
                            continue

                        storage_hit_count = (
                            storage_hit_count // self.page_size * self.page_size
                        )
                        # free the pre-allocated memory for pages that are not hit
                        self.append_host_mem_release(
                            operation.host_indices[storage_hit_count:]
                        )
                        operation.host_indices = operation.host_indices[
                            :storage_hit_count
                        ]
                        self.sim_prefetch_buffer.put(operation)
                except Empty:
                    break

            # handle operation which not yet fully prefetched
            chunked_prefetch_operation = getattr(
                self, "chunked_prefetch_operation", None
            )
            if chunked_prefetch_operation is not None:
                operation = chunked_prefetch_operation["operation"]
                if operation._terminated_flag:
                    setattr(self, "chunked_prefetch_operation", None)
                    self.append_host_mem_release(
                        operation.host_indices[int(operation.completed_tokens) :]
                    )
                else:
                    storage_hit_count = chunked_prefetch_operation["storage_hit_count"]
                    completed_tokens, prefetch_dur = (
                        C_HiCacheController.calc_prefetch_pages(
                            (storage_hit_count - operation.completed_tokens),
                            C_HiCacheController.KV_CACHE_BYTES,
                            remain_dur,
                            C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                        )
                    )
                    if (
                        completed_tokens
                        < storage_hit_count - operation.completed_tokens
                    ):
                        operation.completed_tokens += completed_tokens
                        remain_dur = 0
                    else:
                        operation.completed_tokens = int(storage_hit_count)
                        operation.mark_terminate()
                        remain_dur -= prefetch_dur
                        setattr(self, "chunked_prefetch_operation", None)
                        # Release host memory after current operation is finished
                        self.append_host_mem_release(
                            operation.host_indices[storage_hit_count:]
                        )

            # Feed operations whose host pages were allocated by the scheduler
            # into the virtual-time transfer loop.
            prefetch_buffer = getattr(self, "prefetch_buffer", None)
            if prefetch_buffer is not None:
                while not prefetch_buffer.empty():
                    try:
                        operation = prefetch_buffer.get(block=False)
                        if operation is not None:
                            self.sim_prefetch_buffer.put(operation)
                    except Empty:
                        break

            # handle operation in sim_prefetch_buffer
            while remain_dur > 0:
                try:
                    operation = self.sim_prefetch_buffer.get(block=False)
                    if operation is None:
                        return

                    # Ignore terminated operation
                    if operation._terminated_flag:
                        self.append_host_mem_release(
                            operation.host_indices[int(operation.completed_tokens) :]
                        )
                        continue

                    storage_hit_count = len(operation.host_indices)
                    completed_tokens, prefetch_dur = (
                        C_HiCacheController.calc_prefetch_pages(
                            storage_hit_count,
                            C_HiCacheController.KV_CACHE_BYTES,
                            remain_dur,
                            C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                        )
                    )
                    if completed_tokens < storage_hit_count:
                        # Continue to prefetch data next time.
                        operation.completed_tokens = completed_tokens
                        setattr(
                            self,
                            "chunked_prefetch_operation",
                            {
                                "operation": operation,
                                "storage_hit_count": storage_hit_count,
                            },
                        )
                        remain_dur = 0
                    else:
                        operation.completed_tokens = int(
                            storage_hit_count // self.page_size * self.page_size
                        )
                        # TODO: Track the prefetch operation according to the global clock
                        operation.mark_terminate()
                        remain_dur -= prefetch_dur

                except Empty:
                    return

        def override_generic_page_set(
            self, hash_values, host_indices, extra_info=None
        ) -> bool:
            host_pool = getattr(self, "storage_host_pool", self.mem_pool_host)
            # Always pass extra_info to storage_backend.
            data = [
                host_pool.get_data_page(host_indices[i * self.page_size])
                for i in range(len(hash_values))
            ]
            return self.storage_backend.batch_set(hash_values, data, extra_info)

        def wrapped_terminate_prefetch(self, operator):
            result = original_terminate_prefetch(self, operator)
            # This value may be a float if prefetch progress is interrupted by HiRadixCache.check_prefetch_progress.
            result = (int(result[0]), result[1])
            # operation.completed_tokens, operation.hash_value = result
            req_stats = request_stats_manager.get_req_stats(operator.request_id)
            req_stats.final_storage_hit_len = result[0]
            return result

        def wrapped_storage_hit_query(self, operator):
            result = original_storage_hit_query(self, operator)
            # hash_value, storage_query_count = result
            req_stats = request_stats_manager.get_req_stats(operator.request_id)
            req_stats.recv_storage_hit_len = result[1]
            return result

        target.__init__ = wrapped_init
        target.prefetch_thread_func = override_prefetch_thread_func
        target.backup_thread_func = override_backup_thread_func
        target.handle_backup_operation = handle_backup_operation
        target.handle_prefetch_operation = handle_prefetch_operation
        target.append_host_mem_release = wrapped_append_host_mem_release
        target._generic_page_set = override_generic_page_set
        target.terminate_prefetch = wrapped_terminate_prefetch
        target.storage_hit_query = wrapped_storage_hit_query
        if hasattr(target, "_storage_hit_query"):
            target._storage_hit_query = wrapped_storage_hit_query


class C_HybridCacheController(BaseHook):
    """Adapt UnifiedRadixCache's controller without duplicating legacy logic."""

    HOOK_CLASS_NAME = "HybridCacheController"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        # HybridCacheController inherits the deterministic thread replacements and
        # handle_* methods installed on HiCacheController. Its own initialization
        # creates Unified's control queues after the base initializer returns.
        original_init = target.__init__
        original_storage_hit_query = target._storage_hit_query

        def wrapped_init(self, *args, **kwargs):
            result = original_init(self, *args, **kwargs)
            if hasattr(self, "prefetch_hit_queue") and not hasattr(
                self, "prefetch_buffer"
            ):
                self.prefetch_buffer = Queue()
            return result

        def wrapped_storage_hit_query(self, operator):
            result = original_storage_hit_query(self, operator)
            req_stats = request_stats_manager.get_req_stats(operator.request_id)
            req_stats.recv_storage_hit_len = result[1]
            return result

        target.__init__ = wrapped_init
        target._storage_hit_query = wrapped_storage_hit_query
