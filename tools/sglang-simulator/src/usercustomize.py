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

    # SGLang's serving benchmark imports model configuration helpers even
    # though the simulator client never loads a model.  On ROCm images those
    # helpers can eagerly import AITER and probe the host with ``rocminfo``.
    # Install the same lightweight registries used by the simulator server
    # before any SGLang module is imported.
    from sglang_simulator.simulation.sglang import sgl_kernel_hook

    sgl_kernel_hook.install_load_utils_stub()
    sgl_kernel_hook.install_quantization_stub()


apply_cpu_simulation_compat()
