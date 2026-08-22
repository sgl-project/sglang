# SPDX-License-Identifier: Apache-2.0
"""How much host memory this process may still commit, and to what.

Offloaded weights live in pinned host memory, which the kernel can neither swap
nor drop. That is what makes the asynchronous host-to-device copies work, and
also what turns "using a lot of RAM" into "the container gets OOM-killed":
ordinary pages would have been reclaimed instead.

`psutil.virtual_memory()` cannot be the whole answer here, because it reads
/proc/meminfo, which is host-wide and blind to a container's limit. Measured on
a rented 4-GPU box: psutil reports 2015.7 GiB total while the cgroup caps the
container at 1117.2 GiB, a 900 GiB over-report. Serving runs in containers, so
the cap is read directly from whichever cgroup version is mounted.
"""

import os

import psutil

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

GIB_BYTES = 1024**3

# (mount root, limit file, usage file), v2 before v1.
_CGROUP_MOUNTS = (
    ("/sys/fs/cgroup", "memory.max", "memory.current"),
    ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes"),
)
_PROC_SELF_CGROUP = "/proc/self/cgroup"

# An unlimited v1 cgroup reports a sentinel near 2**63 rather than omitting the
# file, so treat anything implausibly large as "no cap".
_UNLIMITED_ABOVE = 1 << 62

# Left unpinned so the process can still allocate activations, staging buffers
# and whatever the allocator needs mid-request. A share of the cap rather than a
# flat number, because the same absolute headroom is generous on a desktop and
# nothing on a serving host.
HOST_RESERVE_FRACTION = 0.05
MIN_HOST_RESERVE_BYTES = 2 * GIB_BYTES

# Left free when weighing a checkpoint against host memory: activations, staging
# buffers and allocator slack are none of them in the weight total.
HOST_COPY_RESERVE_BYTES = 4 * GIB_BYTES


def _read_int(path: str) -> int | None:
    try:
        with open(path) as handle:
            text = handle.read().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _own_cgroup_path() -> str:
    """The cgroup path /proc reports for this process, or "" if it reports none."""
    try:
        with open(_PROC_SELF_CGROUP) as handle:
            lines = handle.read().splitlines()
    except OSError:
        return ""
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        # v2 leaves the controller field empty; v1 lists memory among its own
        if not fields[1] or "memory" in fields[1].split(","):
            return fields[2]
    return ""


def _cgroup_dirs(mount: str) -> list[str]:
    """This process's cgroup directory and its ancestors up to `mount`.

    The path /proc reports is relative to the host's cgroup root, while the
    mount seen inside a container is already the container's own cgroup -- so
    the two do not simply concatenate. Measured in a Docker container: /proc
    says /docker/8e10..., and the deepest directory that exists under the mount
    is the mount itself. Trying progressively shorter suffixes finds the leaf
    whichever way the container was set up.
    """
    if not os.path.isdir(mount):
        return []
    parts = [part for part in _own_cgroup_path().split("/") if part]
    leaf = mount
    for start in range(len(parts)):
        candidate = os.path.join(mount, *parts[start:])
        if os.path.isdir(candidate):
            leaf = candidate
            break
    dirs = [leaf]
    while dirs[-1] != mount:
        dirs.append(os.path.dirname(dirs[-1]))
    return dirs


def cgroup_memory_limit_bytes() -> tuple[int, int] | None:
    """This process's (cap, usage) under its cgroup, or None when uncapped.

    The tightest cap in the chain wins. A nested cgroup -- a systemd scope with
    MemoryMax, a container started with --cgroup-parent -- holds this process
    below whatever the mount root allows, and planning against the root would
    commit memory the process cannot have.
    """
    for mount, limit_name, usage_name in _CGROUP_MOUNTS:
        tightest = None
        for directory in _cgroup_dirs(mount):
            limit = _read_int(os.path.join(directory, limit_name))
            if limit is None or limit >= _UNLIMITED_ABOVE:
                continue
            if tightest is not None and limit >= tightest[0]:
                continue
            tightest = (limit, _read_int(os.path.join(directory, usage_name)) or 0)
        if tightest is not None:
            return tightest
    return None


def host_memory_available_bytes() -> int:
    """Bytes this process can still commit without hitting a wall.

    The smaller of what the kernel reports free and what the cgroup still
    allows, so a container does not plan against the whole machine.
    """
    forced_gib = envs.SGLANG_DIFFUSION_TEST_FORCE_HOST_AVAILABLE_GIB
    if forced_gib is not None:
        # Behave like a machine of that size: what such a host would still
        # have free is the pretend total minus what this process has already
        # taken in anonymous memory.
        own_anonymous = 0
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith("RssAnon:"):
                        own_anonymous = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
        return max(0, int(forced_gib * GIB_BYTES) - own_anonymous)

    available = int(psutil.virtual_memory().available)
    capped = cgroup_memory_limit_bytes()
    if capped is None:
        return available
    limit, usage = capped
    return min(available, max(0, limit - usage))


def host_copies_would_not_fit(weight_bytes: int) -> bool:
    """Whether copying `weight_bytes` into host memory would run the host out.

    The alternative to a copy is leaving the weights on their file mapping,
    which the kernel may drop under pressure and re-read from disk. That is
    slower per byte but bounded, so it is the right answer exactly when the
    copies do not fit -- and the wrong one when they do.
    """
    if weight_bytes <= 0:
        return False
    return weight_bytes >= host_memory_available_bytes() - HOST_COPY_RESERVE_BYTES


class HostPinBudget:
    """Hands out pinned-host-memory allowances until the headroom runs out.

    Pinning is not all-or-nothing per process: a component whose weights stream
    once per request gains far less from pinning than one re-streamed on every
    denoise step. So the hot components are offered the budget first, and a cold
    component that no longer fits falls back to pageable host memory.

    That fallback is a last resort, not a cheap safety net. Measured on an
    RTX 4090 with Wan2.1-1.3B, dropping the text encoder to pageable left the
    denoise loop untouched but doubled its own stage (3.04 s -> 6.26 s at best,
    and up to 8x when the host's memory bandwidth was contended). It is still
    the right trade against exhausting host memory -- slower is not dead -- but
    it only fires when the bytes genuinely do not fit.
    """

    def __init__(self, available_bytes: int | None = None) -> None:
        if available_bytes is None:
            available_bytes = host_memory_available_bytes()
        self.available_bytes = available_bytes
        self.reserve_bytes = max(
            int(available_bytes * HOST_RESERVE_FRACTION), MIN_HOST_RESERVE_BYTES
        )
        self.committed_bytes = 0

    @property
    def spendable_bytes(self) -> int:
        return max(0, self.available_bytes - self.reserve_bytes - self.committed_bytes)

    def request(self, *, component_name: str, weight_bytes: int) -> bool:
        """Whether `component_name` may pin `weight_bytes`, and book it if so.

        The cap is hard even for the hot components. Granting past it does not
        buy a smaller footprint, it just moves the failure: the pinned
        allocation itself starts failing, or the box begins swapping. Priority
        is expressed by asking in hot-first order, not by overrunning.
        """
        if weight_bytes <= 0:
            return True
        if weight_bytes <= self.spendable_bytes:
            self.committed_bytes += weight_bytes
            return True
        logger.info(
            "Host pin budget: %s stays pageable (%.2f GB of weights, %.2f GB "
            "spendable of %.2f GB available). Its host-to-device copies fall "
            "back to staged transfers -- measured at roughly 2x the time for "
            "the stage that uses it, and more under memory-bandwidth "
            "contention. Nothing is re-read from disk.",
            component_name,
            weight_bytes / GIB_BYTES,
            self.spendable_bytes / GIB_BYTES,
            self.available_bytes / GIB_BYTES,
        )
        return False


def pin_benefit_bytes(*, weight_bytes: int, uses_per_request: int) -> int:
    """Host-to-device bytes a pin would cover for one request.

    Ranking on this product rather than on "is it the DiT" matters for few-step
    models: a 20 GB text encoder used once moves more per request than a 1 GB
    DiT stepped four times, so it is the one that should claim the budget.
    """
    return max(0, weight_bytes) * max(1, uses_per_request)


def module_weight_bytes(module) -> int:
    """Bytes of parameters and buffers a module would hand to the host."""
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        try:
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
            storage_bytes = storage.nbytes()
        except RuntimeError:
            continue
        if pointer == 0 or pointer in seen:
            continue
        seen.add(pointer)
        total += storage_bytes
    return total


def describe_host_memory() -> str:
    """One line for startup logs: what the cap is and where it came from."""
    capped = cgroup_memory_limit_bytes()
    available = host_memory_available_bytes()
    if capped is None:
        return f"host memory available: {available / GIB_BYTES:.1f} GiB (no cgroup cap)"
    limit, usage = capped
    return (
        f"host memory available: {available / GIB_BYTES:.1f} GiB "
        f"(cgroup cap {limit / GIB_BYTES:.1f} GiB, in use {usage / GIB_BYTES:.1f} GiB)"
    )
