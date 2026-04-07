from sglang_simulator.hook import BaseHook


class C_HiRadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HiRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hiradix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events

        def wrapped_check_hicache_events(self, *args, **kwargs):
            # The async thread for prefetching and backup in `HiCacheController` has been deprecated.
            # So we have to handle the backup or prefetch operation manually.
            self.cache_controller.handle_backup_operation()
            self.cache_controller.handle_prefetch_operation()
            return original_check_hicache_events(self, *args, **kwargs)

        target.check_hicache_events = wrapped_check_hicache_events
