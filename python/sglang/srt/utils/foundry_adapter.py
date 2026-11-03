"""Adapter for Foundry CUDA graph persistence (``--cuda-graph-persistence``).

Foundry (https://github.com/foundry-org/foundry, PyPI ``foundry-core``,
optional dependency ``sglang[foundry]``) saves the captured CUDA graphs and
the memory layout they reference on a first start (``save``) and rebuilds
them from that archive on later starts (``load``) instead of capturing. It
runs a CUDA driver hook in the scheduler processes (``LD_PRELOAD``, set by
``configure_subprocess()`` around their spawn). The methods of the no-op
adapter do nothing, so the call sites stay unconditional. Call sites pass
plain values (model, flags, pool config), never a runner or backend.

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

from contextlib import contextmanager
from typing import Any, Optional

# Version of foundry.integration.sglang.api this adapter calls: same major,
# at least this minor.
FOUNDRY_INTEGRATION_API_MAJOR = 2
FOUNDRY_INTEGRATION_API_MIN_MINOR = 3
FOUNDRY_INSTALL_HINT = (
    'Install it with `pip install "sglang[foundry]"` (package foundry-core; see '
    "docs/advanced_features/cuda_graph_persistence)."
)


class FoundryAdapter:
    """No-op base: what every process uses unless ``--cuda-graph-persistence``
    is set."""

    enabled = False
    mode: Optional[str] = None

    # Resolution (launcher).
    def pin_server_args(self, server_args: Any) -> None:
        pass

    def validate_resolved_server_args(self, server_args: Any) -> None:
        pass

    # Process spawn.
    @contextmanager
    def configure_subprocess(self, server_args: Any = None):
        yield

    # Scheduler process.
    def before_parallel_init(self, device: str) -> None:
        pass

    def replay_saved_memory_pool_config(self):
        return None

    def record_memory_pool_config(self, memory_pool_config: Any) -> None:
        pass

    @contextmanager
    def capture_scope(
        self,
        *,
        phase: str,
        model: Any,
        req_slots: Optional[int] = None,
        kind: Optional[str] = None,
        flashinfer_tactics: Optional[str] = None,
    ):
        """phase "decode" passes kind; phase "prefill" passes req_slots."""
        yield

    def capture_one(self, *, shape_key: Any, forward_fn, pool: Any, stream: Any):
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

    def validate_resolved_server_args(self, server_args):
        self._api.validate_resolved_server_args(server_args)

    def configure_subprocess(self, server_args=None):
        return self._api.configure_subprocess(server_args)

    def before_parallel_init(self, device):
        self._api.before_parallel_init(device)

    def replay_saved_memory_pool_config(self):
        return self._api.replay_saved_memory_pool_config()

    def record_memory_pool_config(self, memory_pool_config):
        self._api.record_memory_pool_config(memory_pool_config)

    def capture_scope(
        self, *, phase, model, req_slots=None, kind=None, flashinfer_tactics=None
    ):
        return self._api.capture_scope(
            phase=phase,
            model=model,
            req_slots=req_slots,
            kind=kind,
            flashinfer_tactics=flashinfer_tactics,
        )

    def capture_one(self, *, shape_key, forward_fn, pool, stream):
        return self._api.capture_one(
            shape_key=shape_key, forward_fn=forward_fn, pool=pool, stream=stream
        )


def _import_foundry_api():
    try:
        from foundry.integration.sglang import api
    except ImportError as e:
        raise ImportError(
            f"--cuda-graph-persistence needs Foundry. {FOUNDRY_INSTALL_HINT}"
        ) from e
    found = api.INTEGRATION_API_VERSION
    if (
        found[0] != FOUNDRY_INTEGRATION_API_MAJOR
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
    """This process's adapter; with the flag, activates Foundry once per
    process (see the process contract)."""
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
    _active = _FoundryAdapterReal(
        _import_foundry_api(), mode=mode, config_path=config_path
    )
    return _active
