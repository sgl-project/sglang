"""Host compatibility for single-worker CPU simulation on macOS."""

import os
import sys


def _host_cpu_ids_by_node():
    # macOS has neither lscpu nor Linux CPU-affinity APIs. This describes the
    # host for runtime bookkeeping, not the accelerator in the simulation.
    import psutil

    count = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    return {0: set(range(count))}


def _skip_thread_binding(**kwargs):
    # macOS does not support the Linux NUMA/OpenMP affinity setup.
    return None


def _skip_native_cpu_threads_env(**kwargs):
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if (parallel.tp_size, parallel.pp_size, parallel.dp_size) != (1, 1, 1):
        raise RuntimeError("macOS simulation requires a single runtime worker.")
    # The runtime groups still use Gloo. Native CPU kernel thread setup and
    # shared-memory all-reduce are unnecessary for a single simulated worker.


def install_macos_cpu_compat():
    if sys.platform != "darwin" or os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
        return

    from sglang.srt.distributed import bootstrap
    from sglang.srt.utils import common, numa_utils

    common.get_physical_cpus_by_numa = _host_cpu_ids_by_node
    common.get_cpu_ids_by_node.cache_clear()
    numa_utils.init_threads_binding = _skip_thread_binding
    bootstrap._init_cpu_threads_env = _skip_native_cpu_threads_env
