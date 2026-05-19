"""PseudoEngine — test harness wrapping sglang ``Engine`` for pseudo-mode tests.

Drives an in-subprocess sglang scheduler via ``--enable-pseudo-mode``,
with single-step / inspect / preempt RPCs added by the pseudo-mode
install hook (``_pseudo_*`` methods on the scheduler).

Wiring summary:

* ``launch(...)`` constructs a normal :class:`sglang.srt.entrypoints.engine.Engine`
  with pseudo-mode + dummy load format flags. The scheduler subprocess
  comes up with admit / sampler / oracle hooks already wired by the
  block-3 install glue.
* ``admit(...)`` tokenizes locally (the prompt is already a token list
  in this test environment) and fire-and-forgets a
  :class:`TokenizedGenerateReqInput` via the tokenizer_manager queue.
  The request sits in the scheduler's waiting queue until ``step()``.
* ``step()`` issues a ``_pseudo_step`` RPC: scheduler runs exactly one
  outer event-loop iteration and reports back which reqs ran and what
  forward mode fired (or ``idle`` if no batch was produced).
* ``canary_violations()`` issues ``_pseudo_pull_violations`` and
  decodes each row through :class:`CanaryViolationView`.

The harness pauses the scheduler at launch so no forwards fire between
RPCs, and resumes only inside ``shutdown`` to let any in-flight
plumbing drain cleanly.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import zmq

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import (
    RpcReqInput,
    RpcReqOutput,
    TokenizedGenerateReqInput,
)
from sglang.srt.pseudo_mode.install import decode_harness_ipc_payload
from sglang.srt.sampling.sampling_params import SamplingParams

from test.registered.pseudo_mode._violation_view import CanaryViolationView

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class PseudoEngineConfig:
    """User-facing knobs for :func:`PseudoEngine.launch`."""

    model: str = "Qwen/Qwen3-0.6B"
    num_hidden_layers: int = 1
    enable_overlap: bool = True
    speculative_algorithm: Optional[str] = None
    radix_cache: bool = False
    cuda_graph: bool = True
    oracle_seed: int = 0xC0FFEE
    mem_fraction_static: float = 0.65
    extra_server_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class PseudoReqHandle:
    """Handle returned by :meth:`PseudoEngine.admit`."""

    req_id: str
    rid: str
    prompt: List[int]
    max_new_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PseudoStepResult:
    """One step's metadata, returned from :meth:`PseudoEngine.step`."""

    forward_mode: str
    active_rids: List[str]
    canary_violations_this_step: List[CanaryViolationView]


@dataclass(frozen=True, slots=True, kw_only=True)
class AllocatorStats:
    """KV-pool occupancy snapshot."""

    free: int
    used: int
    total: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveReqInfo:
    """One entry from :meth:`PseudoEngine.active_reqs`."""

    rid: str
    state: str
    output_len: int


class PseudoEngine:
    """Test harness for pseudo-mode scheduler control.

    Construct via :meth:`launch`. Always use as a context manager (or
    call :meth:`shutdown` in a ``finally``) — the wrapped Engine spawns
    subprocesses that must be reaped.
    """

    def __init__(self, *, config: PseudoEngineConfig) -> None:
        self._config = config
        self._engine: Optional[Engine] = None
        self._shutdown_called = False

    @classmethod
    def launch(cls, **kwargs: Any) -> "PseudoEngine":
        """Launch a pseudo-mode sglang Engine subprocess + harness.

        Keyword arguments map to :class:`PseudoEngineConfig` fields.
        ``extra_server_kwargs`` is forwarded to the sglang
        :class:`ServerArgs` for any knob this dataclass does not name
        explicitly.
        """
        engine_kwargs = dict(kwargs)
        extra = engine_kwargs.pop("extra_server_kwargs", {})
        config = PseudoEngineConfig(**engine_kwargs, extra_server_kwargs=extra)
        instance = cls(config=config)
        instance._start()
        return instance

    def _start(self) -> None:
        config = self._config
        server_kwargs: Dict[str, Any] = {
            "model_path": config.model,
            "enable_pseudo_mode": True,
            "disable_radix_cache": not config.radix_cache,
            "disable_cuda_graph": not config.cuda_graph,
            "disable_overlap_schedule": not config.enable_overlap,
            "mem_fraction_static": config.mem_fraction_static,
            "json_model_override_args": (
                f'{{"num_hidden_layers": {config.num_hidden_layers}}}'
            ),
            "log_level": "warning",
            "skip_tokenizer_init": True,
        }
        if config.speculative_algorithm is not None:
            server_kwargs["speculative_algorithm"] = config.speculative_algorithm
        server_kwargs.update(config.extra_server_kwargs)
        logger.info("PseudoEngine launching with kwargs=%r", server_kwargs)
        self._engine = Engine(**server_kwargs)
        # Pause the scheduler immediately so subsequent admits queue
        # without firing a forward until ``step()`` resumes it.
        self._rpc("_pseudo_pause")

    def admit(
        self,
        *,
        prompt: List[int],
        max_new_tokens: int = 32,
        eos_at: Optional[int] = None,
        req_id: Optional[str] = None,
    ) -> PseudoReqHandle:
        """Submit a request to the scheduler queue without stepping it.

        ``eos_at`` is forwarded to the oracle via the standard admit
        hook (scheduler-side install glue reads ``max_new_tokens`` /
        oracle picks the eos token deterministically). ``req_id`` is
        used both as the sglang ``rid`` and the oracle's ``req_id``.
        """
        if self._engine is None:
            raise RuntimeError("PseudoEngine: not started")
        if eos_at is not None:
            # v1: the oracle decides EOS placement from max_new_tokens.
            # Passing an explicit ``eos_at`` is a future extension.
            logger.warning(
                "PseudoEngine.admit: eos_at=%r ignored in v1 (oracle picks)",
                eos_at,
            )
        rid = req_id if req_id is not None else f"pseudo-{uuid.uuid4().hex[:12]}"
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens, temperature=0.0
        )
        sampling_params.normalize(None)
        sampling_params.verify(None)
        tokenized = TokenizedGenerateReqInput(
            rid=rid,
            input_text=None,
            input_ids=list(prompt),
            mm_inputs=None,
            sampling_params=sampling_params,
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
        )
        tm = self._engine.tokenizer_manager
        tm.send_to_scheduler.send_pyobj(tokenized)
        return PseudoReqHandle(
            req_id=rid,
            rid=rid,
            prompt=list(prompt),
            max_new_tokens=max_new_tokens,
        )

    def step(self) -> PseudoStepResult:
        """Drive scheduler one event-loop iteration and report what fired.

        Returns a sentinel ``forward_mode='idle'`` result if the
        scheduler had nothing to run this iteration.
        """
        # Resume → step → re-pause keeps the event-loop body inside one
        # RPC: scheduler returns to the recv-only state before the next
        # admit / inspect call.
        payload = self._rpc("_pseudo_step")
        violations = self.canary_violations()
        return PseudoStepResult(
            forward_mode=str(payload.get("forward_mode", "idle")),
            active_rids=list(payload.get("active_rids", [])),
            canary_violations_this_step=violations,
        )

    def step_until(
        self, req: PseudoReqHandle, *, n: int, max_steps: int = 256
    ) -> List[PseudoStepResult]:
        """Step until ``req`` has emitted >= n output tokens or finishes.

        Raises if ``max_steps`` is exhausted without reaching ``n``.
        """
        results: List[PseudoStepResult] = []
        for _ in range(max_steps):
            results.append(self.step())
            for entry in self.active_reqs():
                if entry.rid == req.rid and entry.output_len >= n:
                    return results
            # If the req disappeared from active_reqs it either finished
            # or was aborted — either way, stop.
            if not any(e.rid == req.rid for e in self.active_reqs()):
                return results
        raise RuntimeError(
            f"PseudoEngine.step_until: req={req.rid} did not reach "
            f"n={n} output tokens within max_steps={max_steps}"
        )

    def step_until_idle(self, *, max_steps: int = 100) -> List[PseudoStepResult]:
        """Drain steps until no active reqs remain. Hard cap on iterations."""
        results: List[PseudoStepResult] = []
        for _ in range(max_steps):
            if not self.active_reqs():
                return results
            results.append(self.step())
        return results

    def force_preempt(self, req: PseudoReqHandle) -> None:
        """Best-effort: drop ``req`` from the running batch on next step.

        See class docstring caveats: production sglang only retracts on
        OOM. This harness directly releases the req from the running
        batch via :meth:`ScheduleBatch.release_req`.
        """
        self._rpc("_pseudo_force_preempt", rid=req.rid)

    def abort(self, req: PseudoReqHandle) -> None:
        """Abort ``req`` via the standard sglang abort path."""
        if self._engine is None:
            raise RuntimeError("PseudoEngine: not started")
        self._engine.abort_request(rid=req.rid)

    def canary_violations(self) -> List[CanaryViolationView]:
        """Pull current first-violation rows from every canary slot.

        Empty list = clean run. Each entry is one (runner, slot-kind)
        whose stored fail_reason is non-NONE.
        """
        payload = self._rpc("_pseudo_pull_violations")
        out: List[CanaryViolationView] = []
        for entry in payload:
            out.append(
                CanaryViolationView.from_row(
                    row=entry["row"],
                    shadow_kind=entry["kind"],
                    write_index=entry["write_index"],
                )
            )
        return out

    def assert_no_canary_violations(self) -> None:
        """Fail-fast: raise if any violation row is non-NONE."""
        violations = [v for v in self.canary_violations() if v.is_real()]
        if violations:
            joined = "\n".join(str(v) for v in violations)
            raise AssertionError(f"Canary violations recorded:\n{joined}")

    def allocator_stats(self) -> AllocatorStats:
        """Return KV-pool free / used / total slots."""
        payload = self._rpc("_pseudo_allocator_stats")
        return AllocatorStats(
            free=int(payload["free"]),
            used=int(payload["used"]),
            total=int(payload["total"]),
        )

    def active_reqs(self) -> List[ActiveReqInfo]:
        """Return one entry per req currently in waiting + running."""
        payload = self._rpc("_pseudo_active_reqs")
        return [
            ActiveReqInfo(
                rid=str(entry["rid"]),
                state=str(entry["state"]),
                output_len=int(entry["output_len"]),
            )
            for entry in payload
        ]

    @property
    def oracle(self) -> Any:
        """Access the scheduler-side oracle.

        v1 limitation: the oracle lives in the scheduler subprocess
        and cannot be returned as a Python reference. Reading it from
        the harness requires extending the IPC. Tests that need
        oracle-state inspection should add a new ``_pseudo_*`` RPC
        handler.
        """
        raise NotImplementedError(
            "PseudoEngine.oracle: cross-process oracle access is a v2 "
            "feature; add a dedicated _pseudo_* RPC handler instead"
        )

    def shutdown(self) -> None:
        """Stop the underlying Engine + its subprocess tree.

        Safe to call multiple times.
        """
        if self._shutdown_called or self._engine is None:
            return
        self._shutdown_called = True
        try:
            self._rpc("_pseudo_resume")
        except Exception as exc:  # noqa: BLE001
            logger.warning("PseudoEngine: resume on shutdown failed: %s", exc)
        try:
            self._engine.shutdown()
        finally:
            self._engine = None

    def __enter__(self) -> "PseudoEngine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def _rpc(self, method: str, **parameters: Any) -> Any:
        """Send one ``RpcReqInput`` and decode the harness payload."""
        if self._engine is None:
            raise RuntimeError("PseudoEngine: not started")
        socket = self._engine.send_to_rpc
        if socket is None:
            raise RuntimeError(
                "PseudoEngine: engine.send_to_rpc is None — multi-node "
                "rank > 0 layout is not supported by the harness"
            )
        obj = RpcReqInput(method=method, parameters=parameters or None)
        socket.send_pyobj(obj)
        reply = socket.recv_pyobj(zmq.BLOCKY)
        if not isinstance(reply, RpcReqOutput):
            raise RuntimeError(
                f"PseudoEngine: unexpected RPC reply type {type(reply)!r}"
            )
        if not reply.success:
            raise RuntimeError(
                f"PseudoEngine: RPC {method} failed: {reply.message}"
            )
        return decode_harness_ipc_payload(reply.message)
