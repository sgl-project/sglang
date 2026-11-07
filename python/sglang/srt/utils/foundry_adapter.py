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

Fallback contract: outside the configurations Foundry supports, Foundry
raises ``api.Unsupported`` at a detection point before any graph is restored
(resolution, LOAD of an archive SAVE marked invalid, the first capture scope,
a SAVE capture). The adapter catches exactly that type, logs the step and the
reason, tells Foundry (``api.deactivate``) and from then on every method is a
no-op: ``capture_scope`` yields, ``capture_one`` returns None (capture
natively), ``replay_saved_memory_pool_config`` returns None (resolve
natively). Any other exception from Foundry propagates.
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Version of foundry.integration.sglang.api this adapter calls: same major,
# at least this minor.
FOUNDRY_INTEGRATION_API_MAJOR = 2
FOUNDRY_INTEGRATION_API_MIN_MINOR = 4
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
        """Returns ``(graph, output)`` for the backend to store, or None: the
        backend captures the shape natively."""
        return None

    # End of the scheduler's init.
    def log_init_summary(self) -> None:
        pass


class _FoundryAdapterReal(FoundryAdapter):
    def __init__(self, api, mode: str, config_path: Optional[str]):
        self._api = api
        self.mode = mode
        # (step, reason) once Foundry reported this configuration unsupported.
        self._unsupported: Optional[tuple[str, str]] = None
        self._quiet = False
        self._guarded(api.activate, mode=mode, config_path=config_path)

    @property
    def enabled(self) -> bool:
        return self._unsupported is None

    def _guarded(self, fn, **kwargs):
        """``fn(**kwargs)``, or None once unsupported; an ``Unsupported`` it
        raises deactivates the adapter."""
        if self._unsupported is not None:
            return None
        try:
            return fn(**kwargs)
        except self._api.Unsupported as unsupported:
            self._deactivate(unsupported)
            return None

    def _deactivate(self, unsupported) -> None:
        self._unsupported = (unsupported.step, unsupported.reason)
        self._quiet = unsupported.quiet
        self._api.deactivate(
            step=unsupported.step, reason=unsupported.reason, quiet=unsupported.quiet
        )
        if unsupported.quiet:
            logger.info(
                f"[Foundry] cuda-graph persistence inactive: {unsupported.reason}"
            )
            return
        if unsupported.inherited:
            logger.info(
                f"[Foundry] {unsupported}; cuda-graph persistence disabled by the "
                "launcher, continuing natively"
            )
            return
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.warning(
                f"[Foundry] {unsupported}; cuda-graph persistence disabled, "
                "continuing natively"
            )
        else:
            logger.warning(
                f"[Foundry] rank {rank}: unsupported at {unsupported.step} (see rank "
                "0's log); continuing natively"
            )

    def pin_server_args(self, server_args):
        self._guarded(self._api.pin_server_args, server_args=server_args)

    def validate_resolved_server_args(self, server_args):
        self._guarded(self._api.validate_resolved_server_args, server_args=server_args)

    def configure_subprocess(self, server_args=None):
        # Also after a fallback: Foundry then passes the decision to the child.
        return self._api.configure_subprocess(server_args)

    def before_parallel_init(self, device):
        self._guarded(self._api.before_parallel_init, device=device)

    def replay_saved_memory_pool_config(self):
        return self._guarded(self._api.replay_saved_memory_pool_config)

    def record_memory_pool_config(self, memory_pool_config):
        self._guarded(
            self._api.record_memory_pool_config, memory_pool_config=memory_pool_config
        )

    @contextmanager
    def capture_scope(
        self, *, phase, model, req_slots=None, kind=None, flashinfer_tactics=None
    ):
        if self._unsupported is not None:
            yield
            return
        entered = False
        try:
            with self._api.capture_scope(
                phase=phase,
                model=model,
                req_slots=req_slots,
                kind=kind,
                flashinfer_tactics=flashinfer_tactics,
            ):
                entered = True
                yield
        except self._api.Unsupported as unsupported:
            # At the scope's entry the capture loop has not run yet: run it natively.
            self._deactivate(unsupported)
            if not entered:
                yield

    def capture_one(self, *, shape_key, forward_fn, pool, stream):
        return self._guarded(
            self._api.capture_one,
            shape_key=shape_key,
            forward_fn=forward_fn,
            pool=pool,
            stream=stream,
        )

    def log_init_summary(self) -> None:
        if self._unsupported is None:
            self._api.log_init_summary()
            return
        if self._quiet:
            return
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            step, reason = self._unsupported
            logger.warning(
                f"[Foundry] init summary: cuda-graph persistence {self.mode} disabled "
                f"at {step} ({reason}); graphs captured natively"
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
        if isinstance(_active, _FoundryAdapterReal):
            raise RuntimeError(
                "Foundry CUDA graph persistence is active in this process "
                f"(--cuda-graph-persistence {_active.mode}), and a server_args "
                "without it cannot run here: Foundry's CUDA hook, allocation region "
                "and environment pins cannot be undone in a live process. Use a "
                "separate process for engines without --cuda-graph-persistence."
            )
        return _NOOP
    config_path = server_args.cuda_graph_persistence_config
    if isinstance(_active, _FoundryAdapterReal):
        # A second record in the same process must name the same archive.
        _active._guarded(_active._api.activate, mode=mode, config_path=config_path)
        return _active
    _active = _FoundryAdapterReal(
        _import_foundry_api(), mode=mode, config_path=config_path
    )
    return _active
