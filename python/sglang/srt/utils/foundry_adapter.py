"""Adapter for Foundry CUDA graph persistence (``--cuda-graph-persistence``).

Foundry (https://github.com/foundry-org/foundry, optional dependency
``sglang[foundry]``) saves the captured CUDA graphs and the memory layout they
reference on a first start (``save``) and rebuilds them from that archive on
later starts (``load``) instead of capturing. It runs a CUDA driver hook in
the scheduler processes (``LD_PRELOAD``, set by ``configure_subprocess()``
around their spawn) and is called from the call sites below; the methods of
the no-op adapter do nothing, so the call sites stay unconditional.

Call sites: the resolution steps in ``arg_groups/cuda_graph_hook.py``,
``_set_envs_and_config`` and the scheduler spawns (``entrypoints/engine.py``,
``managers/data_parallel_controller.py``), ``run_scheduler_process``,
``bootstrap.init_parallel_runtime``, ``ModelRunner.init_torch_distributed`` and
``alloc_memory_pool``, ``KVCacheConfigurator._resolve_memory_pool_config``, the
decode / prefill runners' ``capture``, ``FullCudaGraphBackend.capture_one`` and
``DecodeCudaGraphRunner._resolve_shared_read_ends``.

Foundry is imported only when the flag is set: importing it loads its CUDA
extension, which a server without the flag must not pay for.
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Major version of foundry.integration.sglang.api this adapter calls.
FOUNDRY_INTEGRATION_API_MAJOR = 1
# Distribution name and minimum version, checked when the flag is set.
FOUNDRY_PACKAGE = "foundry"
FOUNDRY_MIN_VERSION = "0.0.3"
FOUNDRY_INSTALL_HINT = (
    'Install it with `pip install --no-build-isolation "sglang[foundry]"` (its CUDA '
    "extension is built against the installed torch; see "
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

    def begin_memory_pool_resolution(self):
        return None

    def end_memory_pool_resolution(self) -> None:
        pass

    def before_alloc_memory_pool(self, model_runner: Any) -> None:
        pass

    def after_alloc_memory_pool(self, model_runner: Any) -> None:
        pass

    @contextmanager
    def capture_scope(self, runner: Any):
        yield

    def capture_one(self, backend: Any, shape_key: Any, forward_fn) -> None:
        raise RuntimeError("capture_one is only called when Foundry is enabled")

    def shared_read_ends_override(self, runner, attn_backend, forward_mode):
        return None


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

    def begin_memory_pool_resolution(self):
        return self._api.begin_memory_pool_resolution()

    def end_memory_pool_resolution(self):
        self._api.end_memory_pool_resolution()

    def before_alloc_memory_pool(self, model_runner):
        self._api.before_alloc_memory_pool(model_runner)

    def after_alloc_memory_pool(self, model_runner):
        self._api.after_alloc_memory_pool(model_runner)

    def capture_scope(self, runner):
        return self._api.capture_scope(runner)

    def capture_one(self, backend, shape_key, forward_fn):
        self._api.capture_one(backend, shape_key, forward_fn)

    def shared_read_ends_override(self, runner, attn_backend, forward_mode):
        return self._api.shared_read_ends_override(runner, attn_backend, forward_mode)


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
    if found is None or found[0] != FOUNDRY_INTEGRATION_API_MAJOR:
        raise RuntimeError(
            f"Foundry integration API {found} is not supported by this SGLang, which "
            f"needs major version {FOUNDRY_INTEGRATION_API_MAJOR}. {FOUNDRY_INSTALL_HINT}"
        )
    return api


_NOOP = FoundryAdapter()
_active: FoundryAdapter = _NOOP


def get_foundry_adapter() -> FoundryAdapter:
    """This process's adapter: the one ``activate_foundry`` created, else the
    no-op one."""
    return _active


def activate_foundry(server_args: Any) -> FoundryAdapter:
    """Create this process's adapter from the raw ``--cuda-graph-persistence``
    and ``--cuda-graph-persistence-config`` of ``server_args``. Idempotent;
    run by the first resolution step in the launcher and at the top of the
    scheduler and data-parallel-controller processes."""
    global _active
    mode = getattr(server_args, "cuda_graph_persistence", None)
    if mode is None:
        return _active
    config_path = getattr(server_args, "cuda_graph_persistence_config", None)
    if _active.enabled:
        # A second record in the same process must name the same archive.
        _active._api.activate(mode=mode, config_path=config_path)
        return _active
    _active = FoundryAdapter.create(True, mode=mode, config_path=config_path)
    return _active
