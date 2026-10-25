"""Install SGLang Simulator hooks in the parent and spawned SGLang worker processes."""

import os

# Spawned interpreters inherit this marker before usercustomize runs.
os.environ["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"

from usercustomize import apply_cpu_simulation_compat

apply_cpu_simulation_compat()

from sglang_simulator.simulation.sglang import sgl_kernel_hook

sgl_kernel_hook.install_load_utils_stub()

import sglang_simulator.hook as sglang_simulator_hook
from sglang_simulator.simulation.sglang import (
    cache_controller,
    dspark_worker,
    hicache_storage,
    mem_cache_allocator,
    mem_pool_host,
    model_runner,
    scheduler,
    unified_radix_cache,
)

# A spawned worker imports this module while unpickling its target. ModelConfig
# can import GPU kernels while later arguments are still being unpickled, before
# the target wrapper executes, so the loader stub must already be present here.
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
            dspark_worker.C_DSparkWorkerHook,
            dspark_worker.C_DSparkDraftInputHook,
            model_runner.C_ModelRunnerHook,
            model_runner.C_KVCacheConfiguratorHook,
            model_runner.C_DSV4PayloadHook,
            hicache_storage.C_StorageBackendFactory,
            cache_controller.C_HiCacheController,
            cache_controller.C_HybridCacheController,
            unified_radix_cache.C_UnifiedRadixCacheHook,
            mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
            mem_pool_host.C_MHATokenToKVPoolHostHook,
            mem_pool_host.C_HostKVCacheHook,
            mem_pool_host.C_GenericHostKVCacheSubclassHook,
        ]
    )
    if os.getenv("SGLANG_SIMULATOR_COORDINATOR"):
        from sglang_simulator.simulation.sglang.dp_coordinated import (
            C_DPCoordinatedHook,
        )
        from sglang_simulator.simulation.sglang.timing_metadata import (
            C_TimingChatHook,
            C_TimingOutputStreamerHook,
            C_TimingTokenizerHook,
        )

        sglang_simulator_hook.install_class_hooks(
            [
                C_TimingOutputStreamerHook,
                C_TimingTokenizerHook,
                C_TimingChatHook,
                C_DPCoordinatedHook,
            ]
        )
    _HOOKS_INSTALLED = True


def run_simulator_scheduler_process(*args, **kwargs):
    """Spawn-safe scheduler entry point which installs SGLang Simulator before SGLang imports."""
    install_simulator_hooks()

    # Spawned workers do not inherit parent-process monkey patches, so import
    # the scheduler only after installing hooks in this process.
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def run_simulator_detokenizer_process(*args, **kwargs):
    """Spawn-safe detokenizer entry point which installs SGLang Simulator before imports."""
    install_simulator_hooks()

    # Install hooks before transitive schedule_batch and memory_pool imports
    # so this CPU-only process does not load real GPU kernels.
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    return run_detokenizer_process(*args, **kwargs)


def run_simulator_data_parallel_controller_process(*args, **kwargs):
    """Prepare a spawned controller before it constructs or launches workers."""
    install_simulator_hooks()

    from sglang.srt.managers.data_parallel_controller import (
        run_data_parallel_controller_process,
    )

    return run_data_parallel_controller_process(*args, **kwargs)
