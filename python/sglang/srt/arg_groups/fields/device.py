"""Config fields of the ``device`` namespace."""

from __future__ import annotations

from typing import (
    Callable,
    List,
    Optional,
)

import msgspec

from sglang.srt.arg_groups.arg_utils import A, Derived


class Device(msgspec.Struct):
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
    gpu_id = Derived(
        doc=(
            "Which device this process runs on. Nobody types it and nothing "
            "computes it from the configuration: the parent decides -- "
            "reindexing narrows the visible devices before the spawn, and Ray "
            "allocates from its own pool -- so the entry states it in the "
            "bundle it hands `publish`. `None` is an answer rather than an "
            "absence: most roles run on no device at all."
        ),
        default=None,
    )
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
    gc_freeze_on_idle_secs: A[
        float,
        "After the scheduler has been idle this many seconds, run a full garbage collection "
        "and freeze the surviving objects, so later collections do not scan them. Runs once per "
        "idle period. A cache flush or PD role switch unfreezes the heap until the next idle "
        "check. Any request, including a /health or /health_generate probe, restarts the idle "
        "timer. At most every 10 minutes, an idle freeze also unfreezes first, so frozen "
        "garbage is freed and logged. Set to 0 to disable.",
    ] = 0.0
    gc_freeze_after_batches: A[
        int,
        "Once the scheduler has launched this many forward batches, collect the young "
        "garbage-collection generations and freeze the surviving objects, so full collections "
        "stop scanning the long-lived heap. Unlike --gc-freeze-on-idle-secs it needs no idle "
        "time, so it also works on a scheduler that is never idle. Frozen objects are never "
        "cycle-collected, so every this many batches after the first freeze, a full pass "
        "unfreezes the heap and runs one full collection before freezing again: this bounds "
        "the cyclic garbage kept frozen, costs one full-GC pause over the whole long-lived "
        "heap, and can trigger the --gc-warning-threshold-secs warning. This value is "
        "therefore also the full-pass period: choose one past warm-up whose batches span "
        "minutes, since a small value means frequent full-GC pauses. A cache flush or PD role "
        "switch unfreezes the heap, runs one full collection and freezes it again while the "
        "queues are empty. Each rank counts its own launched batches. Set to 0 to disable.",
    ] = 0
    gc_freeze_every_batches: A[
        int,
        "With --gc-freeze-after-batches, also freeze every this many batches between the full "
        "passes, so objects that became long-lived since are frozen too; a few tens to a few "
        "hundred batches keeps each freeze short. Set to 0 to freeze only at the full passes.",
    ] = 0
