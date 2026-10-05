"""Adapter for Foundry CUDA graph persistence (``--cuda-graph-persistence``).

Foundry (https://github.com/foundry-org/foundry, PyPI ``foundry-core``,
optional dependency ``sglang[foundry]``) saves the captured CUDA graphs and
the memory layout they reference on a first start (``save``) and rebuilds
them from that archive on later starts (``load``) instead of capturing. It runs a CUDA driver hook in
the scheduler processes (``LD_PRELOAD``, set by ``configure_subprocess()``
around their spawn) and is called from the call sites below; the methods of
the no-op adapter do nothing, so the call sites stay unconditional.

Call sites: the resolution steps in ``arg_groups/cuda_graph_hook.py``,
``_set_envs_and_config`` and the scheduler spawns (``entrypoints/engine.py``,
``managers/data_parallel_controller.py``), ``run_scheduler_process``,
``bootstrap.init_parallel_runtime``, ``ModelRunner.init_torch_distributed`` and
``alloc_memory_pool``, ``KVCacheConfigurator._resolve_memory_pool_config``, the
decode / prefill runners' ``capture`` and ``FullCudaGraphBackend.capture_one``.
Restored decode graphs carry no in-graph shared-read marker; the decode
runner's own fallback for that case (POST_REPLAY) is the fence they need.

Foundry is imported only when the flag is set: importing it loads its CUDA
extension, which a server without the flag must not pay for.

Process contract: a process is persistence-enabled from its first
``activate_foundry`` or never. Activation preloads Foundry's CUDA driver hook
into the children, reserves the allocation region and pins environment
variables, none of which can be undone in a live process (resetting the
adapter would not clear them), so a later record without
``--cuda-graph-persistence`` in an enabled process is an error, not a
fallback to the no-op adapter.
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Version of foundry.integration.sglang.api this adapter calls: same major,
# at least this minor.
FOUNDRY_INTEGRATION_API_MAJOR = 1
FOUNDRY_INTEGRATION_API_MIN_MINOR = 0
# Distribution name (import name ``foundry``) and minimum version, checked
# when the flag is set.
FOUNDRY_PACKAGE = "foundry-core"
FOUNDRY_MIN_VERSION = "0.1.0rc2"
FOUNDRY_INSTALL_HINT = (
    'Install it with `pip install "sglang[foundry]"` (package foundry-core; see '
    "docs/advanced_features/cuda_graph_persistence)."
)


class FoundryAdapter:
    """No-op base: what every process uses unless ``--cuda-graph-persistence``
    is set."""

    enabled = False
    mode: Optional[str] = None

    @staticmethod
    def create(
        enable: bool, mode: Optional[str] = None, config_path: Optional[str] = None
    ) -> "FoundryAdapter":
        if not enable:
            return FoundryAdapter()
        return _FoundryAdapterReal(_import_foundry_api(), mode, config_path)

    # Resolution (launcher).
    def pin_server_args(self, server_args: Any) -> None:
        pass

    def validate_graph_config(self, server_args: Any) -> None:
        pass

    def validate_resolved_server_args(self, server_args: Any) -> None:
        pass

    def apply_env_pins(self, server_args: Any) -> None:
        pass

    # Process spawn.
    @contextmanager
    def configure_subprocess(self, server_args: Any = None):
        yield

    # Scheduler process.
    def before_parallel_init(self, device: str) -> None:
        pass

    def after_parallel_init(self) -> None:
        pass

    def after_runner_distributed_init(self, model_runner: Any) -> None:
        pass

    def replay_saved_memory_pool_config(self):
        return None

    def record_memory_pool_overrides(self) -> None:
        pass

    def before_alloc_memory_pool(self, model_runner: Any) -> None:
        pass

    def after_alloc_memory_pool(self, model_runner: Any) -> None:
        pass

    @contextmanager
    def capture_scope(self, runner: Any):
        yield

    def capture_one(
        self,
        shape_key: Any,
        forward_fn,
        *,
        pool: Any,
        stream: Any,
        prefill_req_slots: Optional[int] = None,
    ):
        """Returns ``(graph, output)`` for the backend to store."""
        raise RuntimeError("capture_one is only called when Foundry is enabled")


class _FoundryAdapterReal(FoundryAdapter):
    enabled = True

    def __init__(self, api, mode: str, config_path: Optional[str]):
        self._api = api
        self.mode = mode
        api.activate(mode=mode, config_path=config_path)

    def pin_server_args(self, server_args):
        self._api.pin_server_args(server_args)

    def validate_graph_config(self, server_args):
        self._api.validate_graph_config(server_args)

    def validate_resolved_server_args(self, server_args):
        self._api.validate_resolved_server_args(server_args)

    def apply_env_pins(self, server_args):
        self._api.apply_env_pins(server_args)

    def configure_subprocess(self, server_args=None):
        return self._api.configure_subprocess(server_args)

    def before_parallel_init(self, device):
        self._api.before_parallel_init(device)

    def after_parallel_init(self):
        self._api.after_parallel_init()

    def after_runner_distributed_init(self, model_runner):
        self._api.after_runner_distributed_init(model_runner)

    def replay_saved_memory_pool_config(self):
        return self._api.replay_saved_memory_pool_config()

    def record_memory_pool_overrides(self):
        self._api.record_memory_pool_overrides()

    def before_alloc_memory_pool(self, model_runner):
        self._api.before_alloc_memory_pool(model_runner)

    def after_alloc_memory_pool(self, model_runner):
        self._api.after_alloc_memory_pool(model_runner)

    def capture_scope(self, runner):
        return self._api.capture_scope(runner)

    def capture_one(
        self, shape_key, forward_fn, *, pool, stream, prefill_req_slots=None
    ):
        return self._api.capture_one(
            shape_key,
            forward_fn,
            pool=pool,
            stream=stream,
            prefill_req_slots=prefill_req_slots,
        )


def _import_foundry_api():
    try:
        from foundry.integration.sglang import api
    except ImportError as e:
        logger.warning(
            "--cuda-graph-persistence is set, but Foundry is not installed. %s",
            FOUNDRY_INSTALL_HINT,
        )
        raise e
    from sglang.srt.utils.common import assert_pkg_version

    assert_pkg_version(FOUNDRY_PACKAGE, FOUNDRY_MIN_VERSION, FOUNDRY_INSTALL_HINT)
    found = getattr(api, "INTEGRATION_API_VERSION", None)
    if (
        found is None
        or found[0] != FOUNDRY_INTEGRATION_API_MAJOR
        or found[1] < FOUNDRY_INTEGRATION_API_MIN_MINOR
    ):
        raise RuntimeError(
            f"Foundry integration API {found} is not supported by this SGLang, which "
            f"needs {FOUNDRY_INTEGRATION_API_MAJOR}.x with x >= "
            f"{FOUNDRY_INTEGRATION_API_MIN_MINOR}. {FOUNDRY_INSTALL_HINT}"
        )
    return api


_NOOP = FoundryAdapter()
_active: FoundryAdapter = _NOOP


def get_foundry_adapter() -> FoundryAdapter:
    """This process's adapter: the one ``activate_foundry`` created, else the
    no-op one."""
    return _active


def activate_foundry(server_args: Any) -> FoundryAdapter:
    """This process's adapter for ``server_args``, from its raw
    ``--cuda-graph-persistence`` and ``--cuda-graph-persistence-config``.

    Without the flag: the no-op adapter, and an error if this process already
    activated Foundry (see the process contract in the module docstring).
    With it: activates Foundry on the first call; later calls must name the
    same mode and config. Run by the first resolution step in the launcher and
    at the top of the scheduler and data-parallel-controller processes."""
    global _active
    mode = server_args.cuda_graph_persistence
    if mode is None:
        if _active.enabled:
            raise RuntimeError(
                "Foundry CUDA graph persistence is active in this process "
                f"(--cuda-graph-persistence {_active.mode}), and a server_args "
                "without it cannot run here: Foundry's CUDA hook, allocation region "
                "and environment pins cannot be undone in a live process. Use a "
                "separate process for engines without --cuda-graph-persistence."
            )
        return _NOOP
    config_path = server_args.cuda_graph_persistence_config
    if _active.enabled:
        # A second record in the same process must name the same archive.
        _active._api.activate(mode=mode, config_path=config_path)
        return _active
    _active = FoundryAdapter.create(True, mode=mode, config_path=config_path)
    return _active
