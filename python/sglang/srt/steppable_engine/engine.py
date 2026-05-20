from __future__ import annotations

import logging
import pickle
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import zmq

if TYPE_CHECKING:
    from sglang.srt.entrypoints.engine import Engine

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.steppable_engine.messages import (
    ActiveReqsReq,
    ActiveReqsResp,
    AllocatorStatsReq,
    AllocatorStatsResp,
    BlockTableReq,
    BlockTableResp,
    CanaryOverheadPctReq,
    CanaryOverheadPctResp,
    CanaryViolationsReq,
    CanaryViolationsResp,
    EnterSteppingModeReq,
    InjectPerturbationReq,
    IsActiveReq,
    IsActiveResp,
    LastWritePlanReq,
    LastWritePlanResp,
    OutputHistoryReq,
    OutputHistoryResp,
    StepReq,
    _ApplyPrFixTogglesReq,
)
from sglang.srt.steppable_engine.views import (
    AllocatorStats,
    CanaryViolationView,
    CanaryWritePlanView,
    SteppableReqHandle,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SteppableEngineConfig:
    apply_pr_25015_fix: Optional[bool] = None


class SteppableEngine:

    def __init__(self, *, config: SteppableEngineConfig, engine: Engine) -> None:
        self._config = config
        self._engine = engine
        self._shutdown_done = False

    @classmethod
    def launch(
        cls,
        *,
        apply_pr_25015_fix: Optional[bool] = None,
        **engine_kwargs: Any,
    ) -> "SteppableEngine":
        engine = Engine(**engine_kwargs)
        cls._validate_server_args(engine)

        config = SteppableEngineConfig(apply_pr_25015_fix=apply_pr_25015_fix)
        instance = cls(config=config, engine=engine)
        instance._enter_stepping_mode()
        return instance

    @staticmethod
    def _validate_server_args(engine: Engine) -> None:
        server_args = engine.server_args
        if server_args.tp_size != 1 or server_args.pp_size != 1:
            raise NotImplementedError(
                "tp_size>1 / pp_size>1 is not supported (single-GPU only)"
            )
        if server_args.attn_cp_size != 1:
            raise NotImplementedError(
                "attn_cp_size>1 is not supported (single-GPU only)"
            )
        if server_args.enable_dp_attention:
            raise NotImplementedError("enable_dp_attention=True is not supported")

    def _enter_stepping_mode(self) -> None:
        self._apply_pr_fix_toggles()
        self._rpc(EnterSteppingModeReq())

    def _apply_pr_fix_toggles(self) -> None:
        if self._config.apply_pr_25015_fix is None:
            return
        choices: Dict[int, Optional[bool]] = {25015: self._config.apply_pr_25015_fix}
        self._rpc(_ApplyPrFixTogglesReq(choices_pickled=pickle.dumps(choices)))

    def __enter__(self) -> "SteppableEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        try:
            self._engine.shutdown()
        finally:
            self._shutdown_done = True

    def _check_alive(self) -> None:
        if self._shutdown_done:
            raise RuntimeError("SteppableEngine is shut down")

    def admit(self, prompt: List[int], max_new_tokens: int) -> SteppableReqHandle:
        self._check_alive()
        rid = uuid.uuid4().hex
        sampling = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
        req = TokenizedGenerateReqInput(
            rid=rid,
            input_text="",
            input_ids=list(prompt),
            mm_inputs=None,
            sampling_params=sampling,
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=[],
            stream=False,
        )
        self._engine.tokenizer_manager.send_to_scheduler.send_pyobj(req)
        return SteppableReqHandle(
            rid=rid,
            prompt_len=len(prompt),
            max_new_tokens=max_new_tokens,
        )

    def step(self) -> None:
        self._check_alive()
        self._rpc(StepReq())

    def step_until(self, handle: SteppableReqHandle, n: int) -> None:
        self._check_alive()
        cap = handle.prompt_len + n + 64
        for _ in range(cap):
            self.step()
            if len(self.output_history(handle)) >= n or not self.is_active(handle):
                return
        raise RuntimeError(f"step_until: exhausted {cap} steps")

    def step_until_idle(self, max_steps: int) -> None:
        self._check_alive()
        for _ in range(max_steps):
            self.step()
            if not self.active_reqs():
                return
        raise RuntimeError(f"step_until_idle: exhausted {max_steps} steps")

    def is_active(self, handle: SteppableReqHandle) -> bool:
        self._check_alive()
        resp: IsActiveResp = self._rpc(IsActiveReq(rid=handle.rid))
        return resp.active

    def active_reqs(self) -> List[SteppableReqHandle]:
        self._check_alive()
        resp: ActiveReqsResp = self._rpc(ActiveReqsReq())
        return pickle.loads(resp.handles_pickled)

    def output_history(self, handle: SteppableReqHandle) -> List[int]:
        self._check_alive()
        resp: OutputHistoryResp = self._rpc(OutputHistoryReq(rid=handle.rid))
        return list(resp.tokens)

    def canary_violations(self) -> List[CanaryViolationView]:
        self._check_alive()
        resp: CanaryViolationsResp = self._rpc(CanaryViolationsReq())
        return pickle.loads(resp.violations_pickled)

    def assert_no_canary_violations(self) -> None:
        violations = self.canary_violations()
        if violations:
            details = "\n  ".join(repr(v) for v in violations)
            raise AssertionError(f"canary violations:\n  {details}")

    def allocator_stats(self) -> AllocatorStats:
        self._check_alive()
        resp: AllocatorStatsResp = self._rpc(AllocatorStatsReq())
        return AllocatorStats(
            free=resp.free,
            used=resp.used,
            held=resp.held,
            total=resp.total,
            held_slots=pickle.loads(resp.held_slots_pickled),
        )

    def block_table(self, handle: SteppableReqHandle) -> List[int]:
        self._check_alive()
        resp: BlockTableResp = self._rpc(BlockTableReq(rid=handle.rid))
        return list(resp.slot_indices)

    def last_write_plan(self) -> Optional[CanaryWritePlanView]:
        self._check_alive()
        resp: LastWritePlanResp = self._rpc(LastWritePlanReq())
        if resp.plan_pickled is None:
            return None
        return pickle.loads(resp.plan_pickled)

    def canary_overhead_pct(self) -> float:
        self._check_alive()
        resp: CanaryOverheadPctResp = self._rpc(CanaryOverheadPctReq())
        return resp.pct

    def inject_perturbation(
        self,
        *,
        channel: str = "default",
        kind: str,
        rank: Optional[int] = None,
    ) -> None:
        self._check_alive()
        from sglang.srt.steppable_engine.perturb import validate_channel_kind

        validate_channel_kind(channel=channel, kind=kind)
        self._rpc(InjectPerturbationReq(channel=channel, kind=kind, rank=rank))

    def _rpc(self, msg: Any) -> Any:
        sock = self._engine.send_to_rpc
        sock.send_pyobj(msg)
        return sock.recv_pyobj(zmq.BLOCKY)
