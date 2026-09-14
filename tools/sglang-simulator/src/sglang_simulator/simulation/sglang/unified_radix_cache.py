from sglang_simulator.hook import BaseHook


class C_UnifiedRadixCacheHook(BaseHook):
    """Drive Unified HiCache storage work from the simulator's logical clock."""

    HOOK_CLASS_NAME = "UnifiedRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.unified_radix_cache"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events

        def handle_pending_operations(controller):
            if controller is None:
                return
            backup_handler = getattr(controller, "handle_backup_operation", None)
            prefetch_handler = getattr(controller, "handle_prefetch_operation", None)
            if backup_handler is not None:
                backup_handler()
            if prefetch_handler is not None:
                prefetch_handler()

        def wrapped_check_hicache_events(self, *args, **kwargs):
            controller = getattr(self, "cache_controller", None)
            handle_pending_operations(controller)
            result = original_check_hicache_events(self, *args, **kwargs)
            # Unified allocates host pages while draining its scheduler-side
            # control queues. Process those newly admitted reads immediately.
            handle_pending_operations(controller)
            return result

        target.check_hicache_events = wrapped_check_hicache_events
