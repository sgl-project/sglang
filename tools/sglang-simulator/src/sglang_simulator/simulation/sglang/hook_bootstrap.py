"""Install SGLang Simulator hooks in the parent and spawned SGLang worker processes."""

import sglang_simulator.hook as sglang_simulator_hook
from sglang_simulator.simulation.sglang import (
    cache_controller,
    hicache_storage,
    hiradix_cache,
    mem_cache_allocator,
    mem_pool_host,
    model_runner,
    scheduler,
    sgl_kernel_hook,
    unified_radix_cache,
)

# A spawned worker imports this module while unpickling its target. ModelConfig
# can import GPU kernels while later arguments are still being unpickled, before
# the target wrapper executes, so the loader stub must already be present here.
sgl_kernel_hook.install_load_utils_stub()

_HOOKS_INSTALLED = False


def install_simulator_hooks() -> None:
    """Install hooks once in the current Python interpreter."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    # The package __init__ loads GPU ops before a child-module import hook can
    # run reliably under spawn. Seed the loader module before importing SGLang.
    sgl_kernel_hook.install_load_utils_stub()

    sglang_simulator_hook.install_class_hooks(
        [
            scheduler.C_SchedulerHook,
            scheduler.C_SglangPrefillAdderHook,
            scheduler.C_SchedulerRequestReceiver,
            model_runner.C_ModelRunnerHook,
            model_runner.C_KVCacheConfiguratorHook,
            hicache_storage.C_StorageBackendFactory,
            cache_controller.C_HiCacheController,
            cache_controller.C_HybridCacheController,
            hiradix_cache.C_HiRadixCacheHook,
            unified_radix_cache.C_UnifiedRadixCacheHook,
            mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
            mem_pool_host.C_MHATokenToKVPoolHostHook,
            mem_pool_host.C_HostKVCacheHook,
            mem_pool_host.C_PackedSingleKVPoolHook,
            mem_pool_host.C_GenericHostKVCacheSubclassHook,
        ]
    )
    _HOOKS_INSTALLED = True


def run_simulator_scheduler_process(*args, **kwargs):
    """Spawn-safe scheduler entry point which installs SGLang Simulator before SGLang imports."""
    install_simulator_hooks()

    # Import only after hook installation. SGLang v0.5.16 forces the spawn start
    # method, so parent-process monkey patches are not inherited by the scheduler.
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def run_simulator_detokenizer_process(*args, **kwargs):
    """Spawn-safe detokenizer entry point which installs SGLang Simulator before imports."""
    install_simulator_hooks()

    # Detokenizer imports schedule_batch and memory_pool on v0.5.16. Installing
    # hooks first keeps this CPU-only process from loading real GPU kernels.
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    return run_detokenizer_process(*args, **kwargs)
