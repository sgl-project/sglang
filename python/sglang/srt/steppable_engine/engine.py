from __future__ import annotations

import asyncio
import logging
import pickle
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

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
from sglang.utils import TypeBasedDispatcher

logger = logging.getLogger(__name__)


_MSG_TO_RESP: Dict[Type, Type] = {
    CanaryViolationsReq: CanaryViolationsResp,
    AllocatorStatsReq: AllocatorStatsResp,
    BlockTableReq: BlockTableResp,
    OutputHistoryReq: OutputHistoryResp,
    IsActiveReq: IsActiveResp,
    ActiveReqsReq: ActiveReqsResp,
    LastWritePlanReq: LastWritePlanResp,
    CanaryOverheadPctReq: CanaryOverheadPctResp,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class SteppableEngineConfig:
    model: str = "Qwen/Qwen3-0.6B"
    num_hidden_layers: int = 1
    tp_size: int = 1
    pp_size: int = 1
    mem_fraction_static: float = 0.65
    cuda_graph: bool = True
    enable_overlap: bool = True
    multimodal: bool = False
    radix_cache: bool = False
    speculative_algorithm: Optional[str] = None
    disagg_prefill_decode: bool = False

    mock_model: bool = True
    oracle_seed: int = 0xC0FFEE

    canary_full: bool = False
    canary_real_data: str = "off"
    sweep_every_n: int = 0

    apply_pr_25015_fix: Optional[bool] = None
    apply_pr_22819_fix: Optional[bool] = None
    apply_pr_24230_fix: Optional[bool] = None
    apply_pr_24401_fix: Optional[bool] = None
    apply_pr_20711_fix: Optional[bool] = None

    scripted_pr_scenario: Optional[int] = None
    extra_server_kwargs: Dict[str, Any] = field(default_factory=dict)


class SteppableEngine:

    def __init__(self, *, config: SteppableEngineConfig, engine: Engine) -> None:
        self._config = config
        self._engine = engine
        self._shutdown_done = False
        self._pending_resps: Dict[Type, asyncio.Future] = {}
        self._install_response_dispatchers()

    @classmethod
    def launch(
        cls,
        config: Optional[SteppableEngineConfig] = None,
        /,
        **kwargs: Any,
    ) -> "SteppableEngine":
        if config is not None and kwargs:
            raise ValueError("launch: pass either config OR kwargs, not both")
        if config is None:
            config = SteppableEngineConfig(**kwargs)

        cls._validate_config(config)

        server_kwargs = cls._build_server_kwargs(config)
        engine = Engine(**server_kwargs)
        cls._validate_server_args(engine)

        instance = cls(config=config, engine=engine)
        instance._enter_stepping_mode()
        return instance

    @staticmethod
    def _validate_config(config: SteppableEngineConfig) -> None:
        if config.tp_size != 1 or config.pp_size != 1:
            raise NotImplementedError(
                "tp_size>1 / pp_size>1 is not supported (single-GPU only)"
            )
        if config.multimodal:
            raise NotImplementedError("multimodal=True is not supported")
        if config.disagg_prefill_decode:
            raise NotImplementedError("disagg_prefill_decode=True is not supported")

    @staticmethod
    def _validate_server_args(engine: Engine) -> None:
        server_args = engine.server_args
        if server_args.attn_cp_size != 1:
            raise NotImplementedError(
                "attn_cp_size>1 is not supported (single-GPU only)"
            )
        if server_args.enable_dp_attention:
            raise NotImplementedError("enable_dp_attention=True is not supported")

    @staticmethod
    def _build_server_kwargs(config: SteppableEngineConfig) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_path": config.model,
            "mem_fraction_static": config.mem_fraction_static,
            "tp_size": config.tp_size,
            "pp_size": config.pp_size,
            "disable_cuda_graph": not config.cuda_graph,
            "disable_radix_cache": not config.radix_cache,
            "disable_overlap_schedule": not config.enable_overlap,
            "skip_tokenizer_init": True,
        }
        if config.speculative_algorithm is not None:
            kwargs["speculative_algorithm"] = config.speculative_algorithm
        if config.mock_model:
            kwargs["mock_model_enabled"] = True
            kwargs["num_hidden_layers_override"] = config.num_hidden_layers
        if config.canary_full:
            kwargs["kv_canary"] = "raise"
        kwargs["kv_canary_real_data"] = config.canary_real_data
        kwargs["kv_canary_real_data_sweep_every_n_steps"] = config.sweep_every_n
        kwargs.update(config.extra_server_kwargs)
        return kwargs

    def _install_response_dispatchers(self) -> None:
        tm = self._engine.tokenizer_manager
        pairs = [
            (resp_type, self._make_resp_handler(resp_type))
            for resp_type in _MSG_TO_RESP.values()
        ]
        tm._result_dispatcher += TypeBasedDispatcher(pairs)

    def _make_resp_handler(self, resp_type: Type):
        def handler(resp: Any) -> None:
            future = self._pending_resps.pop(resp_type, None)
            if future is None:
                logger.warning(
                    "received unexpected %s with no pending future", resp_type.__name__
                )
                return
            if not future.done():
                future.set_result(resp)

        return handler

    def _enter_stepping_mode(self) -> None:
        self._apply_pr_fix_toggles()
        self._send_to_scheduler(EnterSteppingModeReq())

    def _apply_pr_fix_toggles(self) -> None:
        choices: Dict[int, Optional[bool]] = {
            25015: self._config.apply_pr_25015_fix,
            22819: self._config.apply_pr_22819_fix,
            24230: self._config.apply_pr_24230_fix,
            24401: self._config.apply_pr_24401_fix,
            20711: self._config.apply_pr_20711_fix,
        }
        if all(c is None for c in choices.values()):
            return
        for pr_num, choice in choices.items():
            if choice is not None and pr_num != 25015:
                raise NotImplementedError(
                    f"apply_pr_{pr_num}_fix toggle is not supported yet"
                )
        self._send_to_scheduler(
            _ApplyPrFixTogglesReq(choices_pickled=pickle.dumps(choices))
        )

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
            for future in list(self._pending_resps.values()):
                if not future.done():
                    future.cancel()
            self._pending_resps.clear()

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
        self._send_to_scheduler(req)
        return SteppableReqHandle(
            rid=rid,
            prompt_len=len(prompt),
            max_new_tokens=max_new_tokens,
        )

    def step(self) -> None:
        self._check_alive()
        self._send_to_scheduler(StepReq())

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
        self._send_to_scheduler(
            InjectPerturbationReq(channel=channel, kind=kind, rank=rank)
        )

    def _send_to_scheduler(self, msg: Any) -> None:
        self._engine.tokenizer_manager.send_to_scheduler.send_pyobj(msg)

    def _rpc(self, msg: Any, timeout_s: float = 30.0) -> Any:
        resp_type = _MSG_TO_RESP[type(msg)]
        return self._engine.loop.run_until_complete(
            self._rpc_async(msg=msg, resp_type=resp_type, timeout_s=timeout_s)
        )

    async def _rpc_async(self, *, msg: Any, resp_type: Type, timeout_s: float) -> Any:
        tm = self._engine.tokenizer_manager
        tm.auto_create_handle_loop()
        if resp_type in self._pending_resps:
            raise RuntimeError(
                f"_rpc: a previous {resp_type.__name__} is still pending"
            )

        future: asyncio.Future = self._engine.loop.create_future()
        self._pending_resps[resp_type] = future

        self._send_to_scheduler(msg)

        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            self._pending_resps.pop(resp_type, None)
            raise TimeoutError(
                f"_rpc: timed out waiting for {resp_type.__name__} after {timeout_s}s"
            ) from exc
