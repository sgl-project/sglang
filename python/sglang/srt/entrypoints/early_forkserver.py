"""Fork the worker processes from a preloaded forkserver instead of spawning
them (``SGLANG_ENABLE_EARLY_FORKSERVER=1``).

Under ``spawn`` every worker (scheduler, DP controller, detokenizer) is a fresh
interpreter that imports sglang, torch and the kernel packages: 7-9 s per
worker, paid once more by the DP controller for the schedulers it starts. With
a forkserver the launcher starts one helper process at CLI entry that imports
the worker modules once (the preload list) and forks every worker from itself,
so a worker starts in well under a second.

A forkserver child differs from a spawned one in three ways, each handled in
one place here:

* it inherits the *forkserver's* process state, not the launcher's: the start
  method snapshots the launcher's ``os.environ`` and rlimits at ``start()`` and
  the child applies, before ``run()``, the changes the launcher made since CLI
  entry (``merge_env``; rlimits are set to the launcher's);
* its OS parent is the forkserver, so the child stops holding the server's
  alive pipe: the server then exits with the launcher and, through the
  ``PR_SET_PDEATHSIG`` the workers already arm, so do they. Workers that need
  the launcher's pid ask ``multiprocessing`` for who called ``start()``
  (``utils.common.get_parent_process``);
* the modules it inherits were imported in a process that must never
  initialize CUDA, because a forked child cannot use an inherited context. The
  GPU probes the kernel packages make at import are answered from NVML while
  the preload runs (``configure_forkserver_process``), and the child restores
  the real ``torch.cuda`` functions before ``run()``.
"""

import logging
import multiprocessing as mp
import multiprocessing.context as mp_context
import multiprocessing.forkserver as mp_forkserver
import multiprocessing.util as mp_util
import os
import resource
import signal
import sys
import traceback
from collections.abc import Mapping
from typing import NamedTuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Registered as a multiprocessing start method; the name a child sees from
# mp.get_start_method().
START_METHOD = "sglang_forkserver"

# This module comes first: imported by the forkserver, it configures the
# process before the worker modules load (see the end of the file).
PRELOAD = (
    "sglang.srt.entrypoints.early_forkserver",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.detokenizer_manager",
    "sglang.srt.managers.data_parallel_controller",
)

# The environment this process was started with. Inside the forkserver that is
# the launcher's environment at CLI entry, the base of merge_env.
_INITIAL_ENV: dict[str, str] = dict(os.environ)

# True in the launcher once start_early() has a forkserver running.
_started = False

# The real torch.cuda functions, while the NVML stand-ins are installed.
_TORCH_CUDA_ORIG: dict[str, object] = {}

# Inherited at exec like the environment; set_ulimit() raises the file and stack
# limits after the forkserver has started.
_RLIMITS = tuple(getattr(resource, n) for n in dir(resource) if n.startswith("RLIMIT_"))


# ---------------------------------------------------------------------------
# Launcher side
# ---------------------------------------------------------------------------


def enabled() -> bool:
    if not envs.SGLANG_ENABLE_EARLY_FORKSERVER.get():
        return False
    if sys.platform != "linux":
        logger.warning(
            "SGLANG_ENABLE_EARLY_FORKSERVER needs Linux; workers are spawned"
        )
        return False
    return not ({"-h", "--help"} & set(sys.argv))  # help text starts no workers


def start_early() -> None:
    """Start the preloaded forkserver. Called once at CLI entry, before the
    launcher's own heavy imports, so the preload overlaps them. No-op unless
    enabled()."""
    global _started
    if not enabled():
        return
    _register_context()
    mp.set_start_method(START_METHOD, force=True)
    mp.set_forkserver_preload(list(PRELOAD))
    # torch.cuda.is_available(), called at import, marks the process unsafe to
    # fork through cudaGetDeviceCount; torch's NVML-based check does not. Set
    # for the forkserver's exec only; the workers get the launcher's value.
    user_value = os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK")
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    try:
        mp_forkserver.ensure_running()
    finally:
        if user_value is None:
            del os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"]
        else:
            os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = user_value
    _started = True


def start_method(*, enable_memory_saver: bool) -> str:
    """Start method for this launch: the forkserver start_early() started in
    this process, else spawn."""
    if not _started:
        return "spawn"
    if enable_memory_saver:
        # torch_memory_saver LD_PRELOADs its hook around Process.start(), which
        # reaches a spawned worker at exec but never a forked one.
        logger.warning(
            "--enable-memory-saver needs spawned workers; "
            "SGLANG_ENABLE_EARLY_FORKSERVER is ignored for this launch"
        )
        stop()
        return "spawn"
    # NUMA binding v2 execs numactl in front of a spawned interpreter; a forked
    # worker binds in-process (v1) instead.
    if envs.SGLANG_NUMA_BIND_V2.get():
        if envs.SGLANG_NUMA_BIND_V2.is_set():
            logger.info(
                "SGLANG_NUMA_BIND_V2 is ignored with the forkserver; binding in-process"
            )
        envs.SGLANG_NUMA_BIND_V2.set(False)
    return START_METHOD


def stop() -> None:
    """Stop the forkserver started by start_early(), for a launch that turns
    out not to use it, so it does not stay resident holding the worker modules.
    Only before any worker was forked from it."""
    global _started
    if not _started:
        return
    server = mp_forkserver._forkserver
    os.close(server._forkserver_alive_fd)
    os.kill(server._forkserver_pid, signal.SIGKILL)
    try:
        os.waitpid(server._forkserver_pid, 0)
    except ChildProcessError:
        pass  # already reaped
    server._forkserver_alive_fd = None
    server._forkserver_address = None
    server._forkserver_pid = None
    mp.set_start_method("spawn", force=True)
    _started = False


# ---------------------------------------------------------------------------
# The start method: a forkserver whose children run as the launcher's
# ---------------------------------------------------------------------------


class _Process(mp_context.ForkServerProcess):
    _start_method = START_METHOD  # for Process objects made from this context

    @staticmethod
    def _Popen(process_obj):
        # mp.Process() builds a generic process object and only delegates here,
        # at start(): attach what a spawned child would inherit at exec. Pickled
        # with the object, read back by _configure_forked_child.
        process_obj._launcher_env = dict(os.environ)
        process_obj._launcher_rlimits = {r: resource.getrlimit(r) for r in _RLIMITS}
        return mp_context.ForkServerProcess._Popen(process_obj)


class _Context(mp_context.ForkServerContext):
    _name = START_METHOD
    Process = _Process


def _register_context() -> None:
    mp_context._concrete_contexts.setdefault(START_METHOD, _Context())


def merge_env(
    *,
    current: Mapping[str, str],
    launcher: Mapping[str, str],
    base: Mapping[str, str],
) -> dict[str, str]:
    """The environment a forked child runs with: the forkserver's (``current``)
    with every change the launcher made since CLI entry (``launcher`` against
    ``base``) applied on top. What the preload imports set stays; where both
    sides changed a variable, the launcher wins."""
    merged = dict(current)
    for key in base.keys() | launcher.keys():
        if base.get(key) == launcher.get(key):
            continue
        if key in launcher:
            merged[key] = launcher[key]
        else:
            merged.pop(key, None)
    return merged


def _configure_forked_child(_context) -> None:
    """Runs in every child the server forks, before run(); registered with
    multiprocessing's after-fork hook inside the server. Raises SystemExit
    rather than an Exception, which the hook runner would log and swallow."""
    try:
        _apply_launcher_state(mp.current_process())
        _restore_torch_cuda()
        _release_server(mp_forkserver._forkserver)
    except Exception:
        raise SystemExit(traceback.format_exc())
    if PRELOAD[1] not in sys.modules:
        # forkserver.main() swallows ImportError in the preload; the workers
        # still work, they just import everything themselves.
        logger.warning("the forkserver preload did not import %s", PRELOAD[1])


def _release_server(server) -> None:
    # The server exits once the last holder of its alive pipe closes it. Leave
    # that to the launcher alone: the server, and through PR_SET_PDEATHSIG the
    # workers, then go down with the launcher as spawned workers do. A nested
    # start still sends an fd in this slot; nothing reads it.
    os.close(server._forkserver_alive_fd)
    server._forkserver_alive_fd = os.open(os.devnull, os.O_RDONLY)
    # The child holds the server's address but not its pid, so ensure_running()
    # would exec a second, empty forkserver instead of reusing this one.
    server.ensure_running = lambda: None


def _apply_launcher_state(process) -> None:
    # Not defensive: a process started through the stdlib forkserver context
    # shares this server and carries no launcher state; it keeps the server's.
    launcher_env = vars(process).get("_launcher_env")
    if launcher_env is None:
        return
    for limit, values in process._launcher_rlimits.items():
        if resource.getrlimit(limit) != values:
            try:
                resource.setrlimit(limit, values)
            except ValueError as exc:  # hard limit above ours, as in set_ulimit
                logger.warning("cannot set rlimit %s to %s: %s", limit, values, exc)
    merged = merge_env(current=os.environ, launcher=launcher_env, base=_INITIAL_ENV)
    # Written as a delta: clear() would relocate the C environ block under any
    # library holding a getenv() pointer.
    for key in [k for k in os.environ if k not in merged]:
        del os.environ[key]
    for key, value in merged.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Forkserver side: import the worker modules without initializing CUDA
# ---------------------------------------------------------------------------


def _importing_in_forkserver() -> bool:
    """True when this import is the forkserver's preload, i.e. forkserver.main()
    is on the stack; the launcher imports this module from the CLI instead."""
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code is mp_forkserver.main.__code__:
            return True
        frame = frame.f_back
    return False


def configure_forkserver_process() -> None:
    """Runs inside the forkserver before the worker modules are imported."""
    _title_forkserver()
    _register_context()
    mp_util.register_after_fork(
        mp_context._concrete_contexts[START_METHOD], _configure_forked_child
    )
    import torch

    if torch.version.cuda is not None:
        _install_import_time_cuda_shim()


def _title_forkserver() -> None:
    """Title the server, and do it first: setproctitle locates argv and environ
    on its first call and gives up once an environ write has moved them, so a
    later first call would leave the forked workers with the server's title."""
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle("sglang::forkserver")


class _NvmlDevice(NamedTuple):
    major: int
    minor: int
    multi_processor_count: int


def _nvml_device(index: int) -> _NvmlDevice:
    """What the preload's GPU probes ask for, read from NVML."""
    import pynvml
    import torch

    # torch ordinal -> NVML index: they differ under CUDA_VISIBLE_DEVICES and MIG.
    nvml_index = torch.cuda._get_nvml_device_index(index)
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
    finally:
        pynvml.nvmlShutdown()
    # FP32 cores per SM: 64 up to GA100, 128 from GA10x (8.6) on.
    per_sm = 64 if (major, minor) < (8, 6) else 128
    return _NvmlDevice(major, minor, cores // per_sm)


class _NvmlDeviceProperties:
    """torch.cuda.get_device_properties() during the preload. Only what NVML
    supplies; the real query would initialize CUDA and break fork()."""

    def __init__(self, index: int):
        device = _nvml_device(index)
        self.major = device.major
        self.minor = device.minor
        self.multi_processor_count = device.multi_processor_count

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)  # hasattr / copy / pickle probes
        raise AttributeError(
            f"torch.cuda.get_device_properties().{attr} was read while the "
            "forkserver imports the worker modules; only major, minor and "
            "multi_processor_count are available from NVML there, and the real "
            "query would initialize CUDA and break fork(). Read it after import, "
            f"or add it to {__name__}._NvmlDevice."
        )


def _install_import_time_cuda_shim() -> None:
    """Answer the GPU probes sgl_kernel and flashinfer make at import from NVML,
    which leaves fork() usable; the children restore the real functions."""
    import torch

    for name in ("get_device_capability", "get_device_properties", "current_device"):
        _TORCH_CUDA_ORIG[name] = getattr(torch.cuda, name)

    def get_device_capability(device=None):
        d = _nvml_device(torch.cuda._get_device_index(device, optional=True))
        return d.major, d.minor

    def get_device_properties(device=None):
        return _NvmlDeviceProperties(
            torch.cuda._get_device_index(device, optional=True)
        )

    def current_device():
        return 0

    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties
    torch.cuda.current_device = current_device


def _restore_torch_cuda() -> None:
    import torch

    if torch.cuda._is_in_bad_fork():
        raise SystemExit(
            "the forkserver initialized the GPU runtime while importing the worker "
            "modules, so no worker forked from it can use the GPU; an import-time "
            f"GPU query bypassed the stand-ins in {__name__} (CUDA only)"
        )
    for name, fn in _TORCH_CUDA_ORIG.items():
        setattr(torch.cuda, name, fn)
    _TORCH_CUDA_ORIG.clear()


if _importing_in_forkserver():
    configure_forkserver_process()
