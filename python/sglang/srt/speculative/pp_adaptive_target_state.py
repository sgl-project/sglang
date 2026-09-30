"""Adaptive target runtime states for PP stages without a draft model."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.runtime_context import get_context, get_exec, get_spec
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class PPAdaptiveTargetStateManager:
    """Own target-only adaptive states on non-last pipeline stages.

    The last stage uses ``EAGLEWorkerV2`` because it also owns draft resources.
    Other stages only need matching target attention backends and graph runners;
    keeping that construction here avoids teaching ``TpModelWorker`` speculative
    resource details.
    """

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self._pending_transition: Optional[int] = None
        self._controller = AdaptiveController(
            self,
            AdaptiveSpeculativeParams(
                initial_steps=self.speculative_num_steps,
                cfg_path=get_spec().speculative_adaptive_config,
            ),
        )
        self._controller.register(self._current_state())
        self._controller.init_states(
            cuda_graph_bs=get_exec().graph.cuda_graph_bs_decode
        )

    def _current_state(self) -> SpecRuntimeState:
        return SpecRuntimeState(
            speculative_num_steps=self.speculative_num_steps,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            draft_attn_backend=None,
            cuda_graph_runner=None,
            target_attn_backend=self.model_runner.attn_backend,
            target_graph_runner=self.model_runner.decode_cuda_graph_runner,
            draft_extend_attn_backend=None,
            cuda_graph_runner_for_draft_extend=None,
        )

    def activate(self, speculative_num_steps: int) -> None:
        self._controller.activate_step(speculative_num_steps)

    def select_for_batch(self, batch_size: int) -> int:
        return self._controller.activate_step_by_batch(batch_size)

    def observe(
        self,
        num_correct_drafts_per_req: list[int],
        *,
        batch_size: int,
        executed_steps: Optional[int],
    ) -> None:
        new_step = self._controller.on_verify_complete(
            num_correct_drafts_per_req,
            batch_size=batch_size,
            executed_steps=executed_steps,
        )
        if new_step is not None:
            self._pending_transition = new_step

    def pop_transition(self) -> Optional[int]:
        step = self._pending_transition
        self._pending_transition = None
        return step

    # AdaptiveSpecWorker protocol used by AdaptiveController.
    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: Optional[list[int]] = None,
    ) -> SpecRuntimeState:
        with self._capture_config(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs,
        ):
            runner = self.model_runner
            backup_init = runner.init_new_workspace
            try:
                target_attn_backend = runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                runner.init_new_workspace = backup_init
            target_graph_runner = self._build_graph_runner(
                target_attn_backend,
                speculative_num_steps,
                speculative_num_draft_tokens,
            )

        return SpecRuntimeState(
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            draft_attn_backend=None,
            cuda_graph_runner=None,
            target_attn_backend=target_attn_backend,
            target_graph_runner=target_graph_runner,
            draft_extend_attn_backend=None,
            cuda_graph_runner_for_draft_extend=None,
        )

    def _build_graph_runner(
        self,
        target_attn_backend,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
    ):
        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return None

        runner = self.model_runner
        if runner.device == "npu":
            from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
                NPUGraphRunner,
            )

            graph_runner_cls = NPUGraphRunner
        elif runner.device == "xpu":
            from sglang.srt.hardware_backend.xpu.graph_runner.xpu_graph_runner import (
                XPUGraphRunner,
            )

            graph_runner_cls = XPUGraphRunner
        elif runner.device == "cpu":
            from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner

            graph_runner_cls = CPUGraphRunner
        else:
            graph_runner_cls = runner._decode_cuda_graph_runner_cls()

        if runner.device not in ("xpu", "cpu"):
            return graph_runner_cls(
                runner,
                attn_backend=target_attn_backend,
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            )

        # XPU and CPU constructors read the active backend from model_runner.
        old_attn_backend = runner.attn_backend
        runner.attn_backend = target_attn_backend
        try:
            return graph_runner_cls(runner)
        finally:
            runner.attn_backend = old_attn_backend

    @contextmanager
    def _capture_config(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: Optional[list[int]],
    ):
        old_values = (
            get_spec().speculative_num_steps,
            get_spec().speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_bs_decode,
            get_exec().graph.disable_cuda_graph,
        )
        get_context().override(
            "pp_adaptive_spec.capture",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        if cuda_graph_bs is not None:
            get_context().override(
                "pp_adaptive_spec.capture",
                cuda_graph_bs_decode=cuda_graph_bs,
                **({"disable_cuda_graph": True} if not cuda_graph_bs else {}),
            )
        try:
            yield
        finally:
            get_context().override(
                "pp_adaptive_spec.capture_restore",
                speculative_num_steps=old_values[0],
                speculative_num_draft_tokens=old_values[1],
                cuda_graph_bs_decode=old_values[2],
                disable_cuda_graph=old_values[3],
            )

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        if self.speculative_num_steps == state.speculative_num_steps:
            return
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        self.model_runner.attn_backend = state.target_attn_backend
        self.model_runner.decode_cuda_graph_runner = state.target_graph_runner
        get_context().override(
            "pp_adaptive_spec.activate",
            speculative_num_steps=state.speculative_num_steps,
            speculative_num_draft_tokens=state.speculative_num_draft_tokens,
        )
