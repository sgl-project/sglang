"""Config fields of the ``device`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``device`` bag, which is what ``get_device()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import dataclasses
from typing import (
    Callable,
    List,
    Optional,
)

from sglang.srt.arg_groups.arg_utils import A


@dataclasses.dataclass
class Device:
    """Namespace ``device``."""

    _NS_PATH = "device"

    # -------------------------------------------------------------------------
    # Device info and server timeout
    # -------------------------------------------------------------------------
    device: A[
        Optional[str],
        "The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu', 'musa'). Defaults to auto-detection if not specified.",
    ] = None
    base_gpu_id: A[
        int,
        "The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
    ] = 0
    gpu_id_step: A[
        int,
        "The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
    ] = 1
    random_seed: A[Optional[int], "The random seed."] = None
    mlx_enable_sampling: A[
        bool,
        (
            "MLX backend only: sample decode tokens (temperature / top-k / "
            "top-p / min-p) instead of greedy argmax. Sampling runs inside "
            "the lazy MLX graph, so it works with the overlap scheduler; "
            "first tokens from prefill/extend are sampled too. Greedy "
            "requests keep exact argmax behavior. Also enables on the MLX "
            "path: grammar vocab masks and custom logit processors (these "
            "break decode chaining per step; custom processors run on "
            "pure-decode steps only), logit_bias, output logprobs (sampled "
            "token / top-k / token_ids; prompt input logprobs are not "
            "computed), NaN sanitization (SGLANG_SANITIZE_NAN_LOGITS), and "
            "per-request sampling_seed under "
            "--enable-deterministic-inference (deterministic within MLX "
            "only). Penalties are not applied."
        ),
    ] = False
    watchdog_timeout: A[
        float,
        "Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
    ] = 300
    soft_watchdog_timeout: A[
        Optional[float],
        "Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging.",
    ] = None
    sleep_on_idle: A[bool, "Reduce CPU usage when sglang is idle."] = False
    use_ray: A[
        bool,
        "Use Ray actors for scheduler process management.",
    ] = False
    custom_sigquit_handler: Optional[Callable] = None
    numa_node: A[
        Optional[List[int]],
        "Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. If unset, will be automatically detected on NUMA systems.",
    ] = None
    gc_threshold: A[
        Optional[List[int]],
        "Set the garbage collection thresholds (the collection frequency). Accepts 1 to 3 integers.",
    ] = None
