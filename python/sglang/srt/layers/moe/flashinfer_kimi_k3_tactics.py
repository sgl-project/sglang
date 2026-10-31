"""Versioned serving-derived FlashInfer tactics for Kimi-K3 prefill."""

from pathlib import Path

import torch


def load_kimi_k3_prefill_tactics(
    tuner, *, model_type, tp_size, pp_size, ep_size, chunk_size, dtype, device
):
    """Seed the measured 16K bucket before loading the user's autotune cache.

    This is a startup default, not a process-wide AutoTuner monkey patch.
    FlashInfer's cache key includes the activation/weight formats, expert
    geometry and routing representation. Its metadata check additionally
    rejects other library builds and devices. Explicit user caches loaded
    afterwards retain precedence. Do not reuse tactic IDs across versions.
    """
    import flashinfer

    if (
        model_type != "kimi_k3"
        or (tp_size, pp_size, ep_size) != (1, 8, 1)
        or chunk_size != 16384
        or dtype != torch.bfloat16
        or flashinfer.__version__ != "0.7.2"
        or torch.cuda.get_device_capability(device) != (10, 3)
    ):
        return False
    path = Path(__file__).with_name("configs") / "kimi_k3_sm103_fi072_prefill.json"
    return tuner.load_configs(str(path))
