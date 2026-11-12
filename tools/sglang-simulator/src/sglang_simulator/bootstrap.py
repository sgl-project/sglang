"""Activate SGLang Simulator hooks before application imports."""

import os
import sys
import traceback

_BOOTSTRAP_ENV = "SGLANG_SIMULATOR_BOOTSTRAP"


def activate() -> None:
    try:
        os.environ[_BOOTSTRAP_ENV] = "1"
        if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
            import torch

            # Some model-specific import-time checks probe the target GPU even
            # though the simulator never executes a real model forward.
            torch.cuda.get_device_capability = lambda *_args, **_kwargs: (10, 0)

        from sglang_simulator.simulation.sglang.hook_bootstrap import (
            install_simulator_hooks,
        )

        install_simulator_hooks()
    # Startup must fail closed even if an imported hook raises SystemExit.
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        sys.stderr.write(
            "SGLang Simulator activation failed; refusing to start without hooks.\n"
        )
        sys.stderr.flush()
        raise SystemExit(1)


def activate_from_env() -> None:
    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        activate()
