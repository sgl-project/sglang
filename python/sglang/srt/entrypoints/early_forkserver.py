"""Start the multiprocessing forkserver early, with the worker modules preloaded
(``SGLANG_EARLY_FORKSERVER=1``).

Under the default ``spawn`` start method every worker process (scheduler, DP
controller, detokenizer) re-imports sglang and torch, 7-9 s each on a cold
start, and the DP controller pays it again for the schedulers it spawns. With
a forkserver whose preload list already holds those modules, a worker is a
``fork()`` of the server and starts in well under a second.

``start_early()`` is called from the CLI entry before the launcher's own heavy
imports, so the server's import overlaps the launcher's. ``python -m
sglang.srt.entrypoints.early_forkserver`` runs a *daemon* forkserver that
outlives individual ``sglang serve`` launches and publishes its address in
``SGLANG_FORKSERVER_FILE``; a launcher attaches to it when present, so the
preload cost is paid once per machine instead of once per launch.

This module is first in the preload list: importing it inside the forkserver
installs the child-side patches. Forkserver children inherit the *server's*
environment and stdio, not the launcher's, so the launcher snapshots
``os.environ`` at ``Process.start()`` and the child re-applies it in ``run()``,
along with the launcher's stdout/stderr and its identity as the logical
parent process.
"""

import multiprocessing as mp
import multiprocessing.process as mpp
import os
from typing import Dict, Mapping, Optional

from sglang.srt.environ import envs

PRELOAD = [
    "sglang.srt.entrypoints.early_forkserver",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.detokenizer_manager",
    "sglang.srt.managers.data_parallel_controller",
]

_ENV_ATTR = "_sglang_env_snapshot"
_STDIO_ATTR = "_sglang_stdio"
_PARENT_ATTR = "_sglang_logical_parent"
# Environment the forkserver process started with. This module is first in the
# preload list, so os.environ writes made by later preload imports are not in
# here (see merge_launcher_env).
_SERVER_INITIAL_ENV = dict(os.environ)


def enabled() -> bool:
    return envs.SGLANG_EARLY_FORKSERVER.get()


# ---------------------------------------------------------------------------
# Launcher side
# ---------------------------------------------------------------------------


def start_early() -> None:
    """Call once at CLI entry, before heavy imports. No-op unless enabled."""
    if not enabled():
        return
    import multiprocessing.forkserver as fs

    _configure_forkserver_env()
    mp.set_start_method("forkserver", force=True)
    mp.set_forkserver_preload(PRELOAD)
    _install_process_patches()
    if _try_attach_daemon(fs=fs):
        return
    fs.ensure_running()
    # Children forked from the server (e.g. the DP controller) would otherwise
    # start a *second* forkserver with the default preload when they call
    # Process.start(); publish this one so they reuse it (see _reuse_forkserver).
    _publish_forkserver(
        address=fs._forkserver._forkserver_address, pid=fs._forkserver._forkserver_pid
    )


def _configure_forkserver_env() -> None:
    # _set_envs_and_config re-applies the start method from this variable; the
    # NUMA numactl wrapper (SGLANG_NUMA_BIND_V2) execs in front of a spawned
    # interpreter and cannot apply to a forked child, so binding falls back to
    # the in-process implementation.
    envs.SGLANG_MP_START_METHOD.set("forkserver")
    if not envs.SGLANG_NUMA_BIND_V2.is_set():
        envs.SGLANG_NUMA_BIND_V2.set(False)
    # torch.cuda.is_available() goes through cudaGetDeviceCount and marks the
    # process as unsafe to fork even without a context; the NVML-based check
    # does not. sglang calls is_available() at import time.
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


def _publish_forkserver(*, address: str, pid: int) -> None:
    envs.SGLANG_FORKSERVER_ADDRESS.set(address)
    envs.SGLANG_FORKSERVER_PID.set(pid)


def _try_attach_daemon(*, fs) -> bool:
    """Reuse a forkserver started by ``python -m sglang.srt.entrypoints.early_forkserver``.

    The daemon has already imported the worker stack, so workers fork
    immediately instead of waiting for a fresh preload. The launcher keeps one
    end of a pipe as the "alive fd" the forkserver protocol hands to each
    forked child.
    """
    path = envs.SGLANG_FORKSERVER_FILE.get()
    if not path or not os.path.exists(path):
        return False
    try:
        info = read_daemon_info(path)
    except Exception as e:  # stale file, dead daemon, permission
        import logging

        logging.getLogger(__name__).warning(
            "[early_forkserver] daemon at %s unusable (%s); starting a private one",
            path,
            e,
        )
        return False
    inst = fs._forkserver
    inst._forkserver_address = info["address"]
    inst._forkserver_pid = info["pid"]
    r, w = os.pipe()
    inst._forkserver_alive_fd = w
    inst._keep_r = r  # never closed on purpose: it is the children's alive fd
    inst.ensure_running = lambda: None
    _publish_forkserver(address=info["address"], pid=info["pid"])
    return True


def read_daemon_info(path: str) -> Dict[str, object]:
    """Load and validate a daemon file: both the forkserver and the daemon
    process must be alive (and not zombies) and the listener socket present.
    Raises on anything else."""
    import json

    with open(path) as f:
        info = json.load(f)
    addr, pid = info["address"], int(info["pid"])
    for check_pid in (pid, int(info.get("daemon_pid", pid))):
        os.kill(check_pid, 0)  # raises ProcessLookupError if gone
        if _is_zombie(check_pid):
            raise ProcessLookupError(f"pid {check_pid} is a zombie")
    # Do NOT probe-connect: the forkserver main loop treats a connection that
    # sends nothing as a fatal EOF and exits.
    if not os.path.exists(addr):
        raise FileNotFoundError(addr)
    return {"address": addr, "pid": pid}


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status") as st:
            return any(line.startswith("State:") and "Z" in line for line in st)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Process patches (installed in the launcher and inside the forkserver)
# ---------------------------------------------------------------------------


class _FdHandle:
    """Pickles as a duplicated fd via multiprocessing.reduction (sent over the
    forkserver socket), so the child can dup2() the launcher's stdout/stderr."""

    def __init__(self, fd):
        self.fd = fd
        self._dup = None

    def __reduce__(self):
        from multiprocessing import reduction

        return (_FdHandle._rebuild, (reduction.DupFd(self.fd),))

    @staticmethod
    def _rebuild(dup):
        h = _FdHandle(None)
        h._dup = dup
        return h

    def detach(self):
        return self._dup.detach() if self._dup is not None else None


def _install_process_patches() -> None:
    if getattr(mpp.BaseProcess, "_sglang_forkserver_patched", False):
        return
    orig_start = mpp.BaseProcess.start
    orig_run = mpp.BaseProcess.run

    def start(self):
        setattr(self, _ENV_ATTR, dict(os.environ))
        # The OS parent of a forkserver child is the forkserver, not the
        # process that called start(); sglang's workers signal / watch their
        # psutil parent, so carry the logical parent along (see _adopt_parent).
        setattr(self, _PARENT_ATTR, os.getpid())
        # Forkserver children inherit the *server's* stdio. With a shared daemon
        # that is the daemon's log, so hand the child our own stdout/stderr.
        try:
            setattr(self, _STDIO_ATTR, (_FdHandle(1), _FdHandle(2)))
        except Exception:
            pass
        return orig_start(self)

    def run(self):
        env = getattr(self, _ENV_ATTR, None)
        if env:
            merged = merge_launcher_env(
                current=os.environ,
                launcher=env,
                server_initial=_SERVER_INITIAL_ENV,
            )
            os.environ.clear()
            os.environ.update(merged)
        _adopt_stdio(getattr(self, _STDIO_ATTR, None))
        _restore_torch_cuda()
        _reuse_forkserver()
        _sync_offline_flags()
        _adopt_parent(getattr(self, _PARENT_ATTR, None))
        return orig_run(self)

    mpp.BaseProcess.start = start
    mpp.BaseProcess.run = run
    mpp.BaseProcess._sglang_forkserver_patched = True


def merge_launcher_env(
    *,
    current: Mapping[str, str],
    launcher: Mapping[str, str],
    server_initial: Mapping[str, str],
) -> Dict[str, str]:
    """Environment a forked child runs with.

    The launcher's environment wins. Variables the (daemon) forkserver was
    *started* with but the launcher does not have are dropped: the daemon's
    environment belongs to whoever started it, not to this launch. Variables
    that worker modules set at import time inside the server (e.g. kernel cache
    directories) are kept, since the launcher never imports those modules.
    """
    merged = {
        k: v for k, v in current.items() if k in launcher or k not in server_initial
    }
    merged.update(launcher)
    return merged


def _adopt_stdio(stdio) -> None:
    if not stdio:
        return
    try:
        for target, handle in zip((1, 2), stdio):
            fd = handle.detach()
            if fd is not None:
                os.dup2(fd, target)
                os.close(fd)
    except Exception:
        pass


def _adopt_parent(parent_pid: Optional[int]) -> None:
    """Child side. sglang's scheduler / DP controller / detokenizer do
    ``psutil.Process().parent()`` and (a) SIGQUIT it when they crash, (b) rely
    on it to kill_process_tree() them at shutdown. Under a forkserver the OS
    parent is the forkserver -- or PID 1 once the forkserver is gone, and a
    SIGQUIT to PID 1 stops a container. Make psutil report the process that
    called Process.start() instead, and exit when it dies, since its
    kill_process_tree() cannot see us among its children."""
    if not parent_pid or parent_pid == os.getpid():
        return
    import signal
    import threading

    try:
        import psutil
    except ImportError:
        return
    orig_parent = psutil.Process.parent
    me = os.getpid()

    def parent(self):
        if self.pid == me:
            try:
                return psutil.Process(parent_pid)
            except psutil.NoSuchProcess:
                pass
        p = orig_parent(self)
        if p is not None and p.pid == 1:
            return None  # never signal the container's init
        return p

    psutil.Process.parent = parent

    def watch():
        while _pid_alive(parent_pid):
            threading.Event().wait(1.0)
        import sys

        sys.stderr.write(
            f"[early_forkserver] pid {me}: launching process {parent_pid} is gone; exiting\n"
        )
        sys.stderr.flush()
        os.kill(me, signal.SIGKILL)

    threading.Thread(target=watch, name="sglang-parent-watch", daemon=True).start()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return os.path.exists(f"/proc/{pid}") and not _is_zombie(pid)


def sync_offline_flags(environ: Mapping[str, str], modules: Mapping[str, object]):
    """huggingface_hub / transformers read HF_HUB_OFFLINE into module constants
    at import time -- in the (daemon) forkserver's environment. Re-derive them
    from the launcher's environment, so an offline launcher does not make its
    workers hit the Hub."""
    val = environ.get("HF_HUB_OFFLINE", "")
    offline = val.lower() in ("1", "true", "yes", "on")
    hub = modules.get("huggingface_hub.constants")
    if hub is not None and hasattr(hub, "HF_HUB_OFFLINE"):
        hub.HF_HUB_OFFLINE = offline
    for name in ("transformers.utils.hub", "transformers.utils.import_utils"):
        mod = modules.get(name)
        if mod is None:
            continue
        for attr in ("_is_offline_mode", "HF_HUB_OFFLINE"):
            if isinstance(getattr(mod, attr, None), bool):
                setattr(mod, attr, offline)


def _sync_offline_flags() -> None:
    import sys

    sync_offline_flags(environ=os.environ, modules=sys.modules)


def _reuse_forkserver() -> None:
    """Child side: point multiprocessing at the launcher's forkserver instead of
    starting a nested one. A child forked by the server already holds a valid
    alive fd (handed over by the server), only the address/pid are missing.
    ensure_running() would waitpid() the server, which is our parent, so it is
    disabled on this instance."""
    addr = envs.SGLANG_FORKSERVER_ADDRESS.get()
    if not addr:
        return
    import multiprocessing.forkserver as fs

    inst = fs._forkserver
    # forkserver.main() already stores the listener address in children it
    # forks, but not the pid, and ensure_running() keys on the pid.
    if inst._forkserver_pid is None:
        inst._forkserver_address = addr
        inst._forkserver_pid = envs.SGLANG_FORKSERVER_PID.get()
        inst.ensure_running = lambda: None


# ---------------------------------------------------------------------------
# Forkserver side: keep the preload imports from initializing CUDA
# ---------------------------------------------------------------------------

_TORCH_CUDA_ORIG: Dict[str, object] = {}


def _nvml_device(index: int = 0):
    import pynvml
    import torch

    idx = index
    getter = getattr(torch.cuda, "_get_nvml_device_index", None)
    if getter is not None:
        try:
            idx = getter(index)
        except Exception:
            idx = index
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        name = pynvml.nvmlDeviceGetName(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h).total
        try:
            cores = pynvml.nvmlDeviceGetNumGpuCores(h)
        except Exception:
            cores = 0
    finally:
        pynvml.nvmlShutdown()
    return major, minor, name, mem, cores


def _install_import_time_cuda_shim() -> None:
    """Third-party kernel packages (sgl_kernel, flashinfer) probe the GPU at
    import via torch.cuda.get_device_capability / get_device_properties /
    current_device, which initializes a CUDA context and makes fork() unusable.
    Answer those probes from NVML while the preload imports run; children
    restore the real functions in run()."""
    import torch

    if _TORCH_CUDA_ORIG:
        return
    for name in ("get_device_capability", "get_device_properties", "current_device"):
        _TORCH_CUDA_ORIG[name] = getattr(torch.cuda, name)

    def get_device_capability(device=None):
        if torch.cuda.is_initialized():
            return _TORCH_CUDA_ORIG["get_device_capability"](device)
        idx = device if isinstance(device, int) else 0
        major, minor, *_ = _nvml_device(idx)
        return (major, minor)

    def get_device_properties(device=None):
        if torch.cuda.is_initialized():
            return _TORCH_CUDA_ORIG["get_device_properties"](device)
        idx = device if isinstance(device, int) else 0
        return _NvmlDeviceProperties(idx, _TORCH_CUDA_ORIG["get_device_properties"])

    def current_device():
        if torch.cuda.is_initialized():
            return _TORCH_CUDA_ORIG["current_device"]()
        return 0

    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties
    torch.cuda.current_device = current_device


class _NvmlDeviceProperties:
    """torch.cuda.get_device_properties() answered from NVML during the preload.
    An attribute NVML cannot supply falls back to the real query, which
    initializes CUDA in this process."""

    def __init__(self, index: int, real_query):
        major, minor, name, total_memory, cores = _nvml_device(index)
        self._index = index
        self._real_query = real_query
        self.major = major
        self.minor = minor
        self.name = name
        self.total_memory = total_memory
        # FP32 cores per SM: 64 on Volta/Turing and GA100, 128 on GA10x, Ada,
        # Hopper and Blackwell.
        per_sm = 64 if major < 8 or (major == 8 and minor < 6) else 128
        if cores:
            self.multi_processor_count = cores // per_sm

    def __getattr__(self, attr):
        return getattr(self._real_query(self._index), attr)


def _restore_torch_cuda() -> None:
    if not _TORCH_CUDA_ORIG:
        return
    import torch

    for name, fn in _TORCH_CUDA_ORIG.items():
        setattr(torch.cuda, name, fn)
    _TORCH_CUDA_ORIG.clear()


def _set_default_flashinfer_arch() -> None:
    """flashinfer enumerates device capabilities at import unless told the
    arch; derive it from NVML (same format as utils.common.set_cuda_arch)."""
    if "FLASHINFER_CUDA_ARCH_LIST" in os.environ:
        return
    try:
        major, minor, *_ = _nvml_device(0)
    except Exception:
        return
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = (
        f"{major}.{minor}{'a' if major >= 9 else ''}"
    )


def _in_forkserver_process() -> bool:
    """True inside the forkserver (spawned as ``python -c 'from
    multiprocessing.forkserver import main; ...'``) and hence in the workers it
    forks (module state is inherited). False in the launcher, which configures
    itself explicitly in start_early()."""
    try:
        with open("/proc/self/cmdline", "rb") as f:
            return b"multiprocessing.forkserver" in f.read()
    except OSError:
        import sys

        return any("multiprocessing.forkserver" in a for a in sys.argv[:3])


def _configure_forkserver_process() -> None:
    """Forkserver / child side (module preloaded): make nested Process() calls
    (the DP controller launching schedulers) reuse the same forkserver and
    carry the launcher's environment; keep the preload imports fork-safe."""
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass
    _install_process_patches()
    mp.set_forkserver_preload(PRELOAD)  # fallback if a nested server is ever started
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    _set_default_flashinfer_arch()
    _install_import_time_cuda_shim()


if enabled() and _in_forkserver_process():
    _configure_forkserver_process()


# ---------------------------------------------------------------------------
# Daemon: `python -m sglang.srt.entrypoints.early_forkserver`
# ---------------------------------------------------------------------------


def run_daemon() -> None:
    """Start a preloaded forkserver that outlives individual ``sglang serve``
    launches and publish its address in SGLANG_FORKSERVER_FILE. Must be started
    from the same environment and code as the servers that will use it."""
    import json
    import multiprocessing.forkserver as fs
    import signal
    import time

    envs.SGLANG_EARLY_FORKSERVER.set(True)
    _configure_forkserver_env()
    mp.set_start_method("forkserver", force=True)
    mp.set_forkserver_preload(PRELOAD)
    _install_process_patches()
    t0 = time.time()
    fs.ensure_running()
    # Force the preload now (the server imports lazily on first connect otherwise).
    p = mp.Process(target=_noop)
    p.start()
    p.join()
    inst = fs._forkserver
    path = envs.SGLANG_FORKSERVER_FILE.get()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "address": inst._forkserver_address,
                "pid": inst._forkserver_pid,
                "daemon_pid": os.getpid(),
            },
            f,
        )
    os.replace(tmp, path)
    print(
        f"[early_forkserver] daemon ready in {time.time() - t0:.1f} s: "
        f"address={inst._forkserver_address} pid={inst._forkserver_pid} file={path}",
        flush=True,
    )

    def _cleanup(*_):
        try:
            os.unlink(path)
        except OSError:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)
    while True:
        time.sleep(3600)


def _noop():
    pass


if __name__ == "__main__":
    run_daemon()
