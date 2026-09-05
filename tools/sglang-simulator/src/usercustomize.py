"""Early CPU-simulation compatibility for spawned SGLang workers."""

import os

if (
    os.environ.get("SGLANG_SIMULATOR_BOOTSTRAP") == "1"
    and os.environ.get("SGLANG_USE_CPU_ENGINE") == "1"
):
    import torch

    # Some model-specific import-time checks probe the target GPU even though
    # SGLang Simulator never executes a real model forward. Spawned workers reach those
    # imports before the simulator target wrapper can run. CPU simulation is
    # explicit, so physical GPU visibility must not affect this shim.
    torch.cuda.get_device_capability = lambda *_args, **_kwargs: (10, 0)
