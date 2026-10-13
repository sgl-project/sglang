"""Early CPU-simulation compatibility for spawned SGLang workers."""

import os


def apply_cpu_simulation_compat() -> None:
    if not (
        os.environ.get("SGLANG_SIMULATOR_BOOTSTRAP") == "1"
        and os.environ.get("SGLANG_USE_CPU_ENGINE") == "1"
    ):
        return

    import torch

    # Some model-specific import-time checks probe the target GPU even though
    # SGLang Simulator never executes a real model forward. Spawned workers reach those
    # imports before the simulator target wrapper can run. CPU simulation is
    # explicit, so physical GPU visibility must not affect this shim.
    torch.cuda.is_available = lambda: False
    torch.cuda.get_device_capability = lambda *_args, **_kwargs: (10, 0)
    torch.version.hip = None

    # SGLang's helper first gates on ``is_available`` and otherwise returns
    # ``(None, None)``. Some quantization modules compare that tuple during
    # import even for a CPU dummy load, so patch both the defining module and
    # the compatibility re-export before those modules are imported.
    from sglang.srt import utils as sglang_utils
    from sglang.srt.utils import common as sglang_common

    def simulated_capability(*_args, **_kwargs):
        return (10, 0)

    sglang_common.get_device_capability = simulated_capability
    sglang_utils.get_device_capability = simulated_capability

    # Dummy execution needs no accelerator library. Install only the loader
    # shim; preserve SGLang's quantization registry and configuration checks.
    from sglang_simulator.simulation.sglang import sgl_kernel_hook

    sgl_kernel_hook.install_load_utils_stub()


apply_cpu_simulation_compat()
