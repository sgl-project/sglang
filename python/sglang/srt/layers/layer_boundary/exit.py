# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Producer output choices, reduction scopes, and completion."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import msgspec
import torch

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    can_use_dp_reduce_scatter,
    is_dp_attention_enabled,
    is_enable_moe_cp_allgather,
)
from sglang.srt.layers.layer_boundary.contracts import (
    BatchVariant,
    ExitFacts,
    StageKind,
)
from sglang.srt.layers.layer_boundary.layout import (
    SumGroup,
    _ffn_has_tokens,
    _sum_group,
)
from sglang.srt.layers.layer_boundary.ops import (
    _dp_scatter_step,
    all_reduce_to_dp_local,
    dp_reduce_scatter,
    dp_reduce_scatterv,
    sum_output,
    to_dp_local,
)
from sglang.srt.layers.layer_boundary.output import (
    UnreducedOutput,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    aiter_ar_fusion_applies,
    flashinfer_ar_fusion_applies,
)
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.moe import (
    can_merge_post_experts_all_reduce,
    post_experts_sum_is_one_all_reduce,
    should_use_dp_reduce_scatterv,
)
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    post_experts_reduction_group,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_exec,
    get_forward,
    get_lora,
    get_parallel,
)


def _select_dp_reduce_scatter(
    forward_batch: ForwardBatch,
    *,
    may_reduce_scatter: bool,
    reduce_scatterv: bool,
) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
    """The reduce-scatter that brings an FFN output gathered over attention
    DP back to this rank's tokens when the FFN leaves its sum to it (see
    OutputContract); None when the FFN reduces the output and only a scatter
    remains. Whether the reduce-scatter is usable follows the DP group, which
    an elastic scale-up changes, so it is read for each batch."""
    if reduce_scatterv:
        return dp_reduce_scatterv
    if (
        may_reduce_scatter
        and forward_batch.dp_padding_mode.is_max_len()
        and can_use_dp_reduce_scatter()
    ):
        return dp_reduce_scatter
    return None


def exit_facts(kind, plan, variant, path) -> ExitFacts:
    """What ``path``'s exit decides from fixed facts on ``variant``: the
    parallel configuration, the declarations and the bound path. A stage
    without an exit (an attention that always leaves its sum) decides
    nothing. A condition is read only where the exit used to read it."""
    if plan.finishes_directly:
        return ExitFacts()
    output = path.output
    scattered = variant is BatchVariant.INPUT_SCATTERED
    tp = get_parallel().tp_size > 1
    if kind is StageKind.ATTENTION:
        return ExitFacts(
            defers_mixer_sum=output.may_defer_to_next
            and tp
            and not is_dp_attention_enabled()
            and _sum_deferral_allowed(plan, output, scattered=scattered)
        )
    sum_in_reduce_scatter = output.may_reduce_scatter and (
        path.output_move_completes_sum or (scattered and not plan.terminal)
    )
    may_defer_sum = (
        tp
        and not plan.terminal
        and _sum_deferral_allowed(plan, output, scattered=scattered)
        # Under attention DP the next layer must also run postprocess's
        # scatter back to this rank's tokens, and nothing more.
        and (
            not is_dp_attention_enabled()
            or (path.returns_over_dp and not output.update.applied_at_exit)
        )
    )
    single_sum = may_defer_sum and post_experts_sum_is_one_all_reduce()
    return ExitFacts(
        may_defer_sum=may_defer_sum,
        sum_in_reduce_scatter=sum_in_reduce_scatter,
        # On an input-scattered batch the next input's TP reduce-scatter
        # completes the sum; the other reduce-scatters are this exit's move.
        sum_left_to_next_input=(
            SumGroup.TP
            if sum_in_reduce_scatter
            and not path.returns_over_dp
            and not path.output_move_completes_sum
            else None
        ),
        reduce_scatterv=path.returns_over_dp
        and should_use_dp_reduce_scatterv()
        and output.may_reduce_scatterv,
        single_sum=single_sum,
        fused_consumer_sum=may_defer_sum and not single_sum and _fused_consumer_sum(),
    )


def _sum_deferral_allowed(plan, output, *, scattered: bool) -> bool:
    # Under the MoE-CP all-gather the fusion path would skip complete_now
    # and its MoE-CP scatter, leaving hidden_states longer than the residual.
    if is_enable_moe_cp_allgather() or output.group is None:
        return False

    # The fused residual+LN reduces over a single group. Hybrid EP+TP spans
    # two disjoint groups; post_experts_all_reduce() merges them into one
    # _TP reduction when moe_dp_size == 1, which the fused kernel can absorb.
    # When merging is blocked, no single group covers both, so fusion stays off.
    parallel = get_parallel()
    if (
        parallel.moe_ep_size > 1
        and parallel.moe_tp_size > 1
        and not can_merge_post_experts_all_reduce()
    ):
        return False

    if (
        is_dp_attention_enabled()
        and plan._speculative_algo is not None
        and plan._speculative_algo.is_eagle()
    ):
        return False

    return not scattered


def _fused_consumer_sum() -> bool:
    # LoRA-B is replicated and linear. TP1 shared experts add on rank zero
    # when the sum is deferred. Preserve these fused paths, but keep their
    # producer-side reduction order when no fused consumer is enabled.
    return (
        (get_lora().enable_lora or envs.SGLANG_SHARED_EXPERT_TP1.get())
        and get_moe_a2a_backend().is_none()
        and not get_exec().comm.enable_quant_communications
        and post_experts_reduction_group() is get_parallel().tp_group
    )


class ExitPolicy:
    """Choose and finish one producer's output; the batch owns its residual."""

    def __init__(self, plan):
        self.plan = plan

    def complete_now(
        self,
        hidden_states: torch.Tensor,
        stream: ResidualStream,
        forward_batch: ForwardBatch,
        *,
        already_reduced: bool = False,
    ):
        """Complete the output of the operation-scheduled producer path, which
        never defers: the sum it owes, then its move and write-back."""
        steps = self.plan.path_for(forward_batch)
        if already_reduced and steps.output_move_completes_sum:
            raise ValueError("already-reduced output cannot use a reduce-scatter move")
        hidden_states, residual = self._complete_now(
            hidden_states,
            stream.residual,
            forward_batch=forward_batch,
            dp_step=None,
            steps=steps,
            output_move=steps.output_move,
            owes_sum=not already_reduced
            and steps.output.group is not None
            and not steps.output_move_completes_sum,
        )
        return self._record_output(
            hidden_states,
            residual,
            stream,
            steps.output.update,
        )

    @staticmethod
    def _record_output(hidden_states, residual, stream, update, declared_sum=None):
        stream.residual = residual
        if update.applied_at_exit:
            return stream.write(hidden_states)
        return stream.record(hidden_states, update, declared_sum=declared_sum)

    def _next_input_can_scatter(self, steps) -> bool:
        """Whether the next layer's input can run this layer's move of its FFN
        output back to this rank's tokens: the base postprocess scatter, when
        the next layer's input also writes the output into the residual."""
        return steps.returns_over_dp and not steps.output.update.applied_at_exit

    def _dp_reduce_scatter_step(
        self, forward_batch: ForwardBatch, steps
    ) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
        """The reduce-scatter that brings this layer's FFN output back to this
        rank's tokens under attention DP; None when the base postprocess would
        only scatter, or does not move tokens."""
        if not steps.returns_over_dp:
            return None
        return _select_dp_reduce_scatter(
            forward_batch,
            may_reduce_scatter=steps.output.may_reduce_scatter,
            reduce_scatterv=steps.exit.reduce_scatterv,
        )

    def _sum_in_reduce_scatter(self, steps, dp_step: Optional[Callable]) -> bool:
        """Whether a reduce-scatter completes the FFN output's sum: the
        attention-DP one ``dp_step`` names, or the CP / input-scattered one."""
        return dp_step is not None or steps.exit.sum_in_reduce_scatter

    def ffn_reduction_group(self, steps) -> GroupCoordinator:
        """The group this layer's FFN output owes its sum over: the MoE output's
        group on a sparse layer, the TP group a dense MLP reduces over."""
        return _sum_group(steps.output.group)

    def _decide(self, forward_batch: ForwardBatch, steps) -> ExitDecision:
        """Decide once, before the FFN runs, what completes its output's sum:
        the next layer's input, or this layer's postprocess step."""
        dp_step = self._dp_reduce_scatter_step(forward_batch, steps)
        sum_in_reduce_scatter = self._sum_in_reduce_scatter(steps, dp_step)
        complete_now = partial(
            self._complete_now,
            forward_batch=forward_batch,
            dp_step=dp_step,
            steps=steps,
            output_move=steps.output_move,
            # A reduce-scatter completes the sum on its way back.
            owes_sum=steps.output.group is not None and not sum_in_reduce_scatter,
        )
        defer_moe_finalize = (
            self.plan.fusions is not None
            and self.plan.fusions.can_defer_finalize(self.plan, forward_batch)
        )
        # An FFN that writes its output at a pipeline handoff completes it here.
        if steps.writes_at_handoff or (
            not steps.output.may_defer_to_next
            and not (self.plan.terminal and defer_moe_finalize)
        ):
            return ExitDecision(
                defer_moe_finalize=False,
                sum_in_reduce_scatter=sum_in_reduce_scatter,
                complete=complete_now,
            )
        # Producers declare remaining work independently of the kernel chosen
        # by the consumer. Every handoff also carries an unfused completion.
        defers = defer_moe_finalize or self._defers_sum(
            forward_batch,
            steps,
            sum_in_reduce_scatter=sum_in_reduce_scatter,
            dp_step=dp_step,
        )
        if defers:
            group = self.ffn_reduction_group(steps)
            if steps.returns_over_dp:
                # Under attention DP the next layer also brings the sum back to
                # this rank's tokens.
                wrap = partial(
                    UnreducedOutput,
                    reduce_to_dp_local=partial(
                        all_reduce_to_dp_local, group, forward_batch
                    ),
                )
            else:
                wrap = partial(UnreducedOutput, group=group)
            complete = partial(_defer, wrap)
        elif (
            dp_step is not None
            and not self.plan.terminal
            and self._next_input_can_scatter(steps)
        ):
            complete = partial(
                _defer,
                partial(
                    UnreducedOutput,
                    reduce_to_dp_local=partial(to_dp_local, dp_step, forward_batch),
                ),
            )
        else:
            complete = complete_now
        return ExitDecision(
            defer_moe_finalize=defer_moe_finalize,
            sum_in_reduce_scatter=sum_in_reduce_scatter,
            complete=complete,
        )

    def _complete_now(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        forward_batch: ForwardBatch,
        dp_step: Optional[Callable],
        steps,
        output_move: Optional[Callable],
        owes_sum: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This layer's postprocess, run with the attention-DP step already
        chosen: the sum the output owes, the move back to where the next layer
        reads it, then the write-back into the residual for a layer that does
        it itself."""
        if owes_sum:
            hidden_states = sum_output(
                hidden_states,
                steps.output.group,
                forward_batch,
                may_quantize=steps.output.update.quantized_sum,
            )
        if steps.output.transform is not None:
            hidden_states = steps.output.transform.apply(hidden_states)
        if steps.returns_over_dp:
            hidden_states = to_dp_local(
                dp_step or _dp_scatter_step, forward_batch, hidden_states
            )
        else:
            hidden_states, residual = output_move(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        update = steps.output.update
        if residual is not None and (update.applied_at_exit or steps.writes_at_handoff):
            hidden_states = update.update(hidden_states, residual)
            residual = None
        return hidden_states, residual

    def mixer_exit(
        self, forward_batch: ForwardBatch, *, stream: ResidualStream
    ) -> MixerExit:
        """Decide once whether this stage's mixer (an attention-like stage)
        output's sum is completed here or carried on. Call ``finish`` on the
        result with the mixer's output."""
        return MixerExit(self, forward_batch, stream=stream)

    def ffn_exit(
        self, forward_batch: ForwardBatch, *, stream: ResidualStream
    ) -> FfnExit:
        """Decide once how this layer's FFN output reduction completes. Use the
        result as a context manager around the FFN call when the FFN may hand
        off its MoE finalize, then call ``finish``."""
        return FfnExit(self, forward_batch, stream=stream)

    def _defers_sum(
        self,
        forward_batch: ForwardBatch,
        steps,
        *,
        sum_in_reduce_scatter: bool,
        dp_step: Optional[Callable],
    ) -> bool:
        """Whether the FFN leaves its output's all-reduce to the next layer's
        input when no fused kernel takes it: when the next layer would run the
        same all-reduce the FFN itself would have. A reduce-scatter completing
        the sum rules it out, and under attention DP so does any move back but
        the scatter the next input runs."""
        return (
            steps.exit.may_defer_sum
            and _batch_allows_deferred_sum(forward_batch, steps.exit, self.plan)
            and not sum_in_reduce_scatter
            and dp_step is None
        )


def _batch_allows_deferred_sum(
    forward_batch: ForwardBatch, facts: ExitFacts, boundary=None
) -> bool:
    """Admit ordinary single-sum outputs, plus LoRA and TP1 shared-expert
    outputs when a fused consumer (the backend's can_defer_all_reduce,
    FlashInfer or aiter) can take them.

    A fused-kernel fallback completes the partial-output contract selected
    before the producer ran, without recomputing the producer's LoRA path.
    """
    if not _ffn_has_tokens(forward_batch):
        return False
    if facts.single_sum:
        return True
    if not facts.fused_consumer_sum:
        return False
    if (
        boundary is not None
        and boundary.fusions is not None
        and boundary.fusions.can_defer_all_reduce(boundary, forward_batch)
    ):
        return True
    if flashinfer_ar_fusion_applies(forward_batch.input_ids.shape[0]):
        return True
    # Aiter also checks width and bytes, so use the actual residual storage.
    residual = forward_batch.residual_stream.residual
    return residual is not None and aiter_ar_fusion_applies(residual, forward_batch)


class ExitDecision(msgspec.Struct, frozen=True):
    """One decision for an FFN output, made before compute runs.

    Fields:
        defer_moe_finalize: Compute may return a producer-specific finalize handoff.
        sum_in_reduce_scatter: A reduce-scatter on the way back completes the sum.
        complete: Callable(output, residual) returning the completed or wrapped
            output and its corresponding residual rows.

    Do not independently reselect completion after compute has run.
    """

    defer_moe_finalize: bool
    sum_in_reduce_scatter: bool
    complete: Callable[[torch.Tensor, torch.Tensor], Tuple]


def _defer(
    wrap: Callable[[torch.Tensor], UnreducedOutput],
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> Tuple[UnreducedOutput, torch.Tensor]:
    return wrap(hidden_states), residual


class MixerExit:
    """A mixer's (an attention-like stage's) output decision. The mixer leaves
    its output's sum; the stage either carries it to the next attention stage
    (when the declaration permits deferring and the batch allows it: TP > 1,
    no attention DP, and no input-scattered or MoE-CP all-gather layout),
    hands it on as the declared sum an FFN stage completes in its input, or
    completes it here. The consumer decides whether to fuse its completion
    with the input norm."""

    __slots__ = (
        "_defers",
        "_group",
        "_update",
        "_declared_sum",
        "_stream",
        "_forward_batch",
    )

    def __init__(
        self,
        boundary: ExitPolicy,
        forward_batch: ForwardBatch,
        *,
        stream: ResidualStream,
    ):
        steps = boundary.plan.path_for(forward_batch)
        produced = steps.output
        self._stream = stream
        self._forward_batch = forward_batch
        self._update = produced.update
        self._group = produced.group
        self._declared_sum = produced.group if produced.always_partial else None
        self._defers = steps.exit.defers_mixer_sum

    def __enter__(self) -> MixerExit:
        return self

    def __exit__(self, *exc_info):
        return None

    def finish(self, hidden_states: torch.Tensor):
        """Hand the actual mixer result to the next boundary through the stream."""
        if self._defers:
            hidden_states = UnreducedOutput(
                hidden_states, group=get_parallel().tp_group
            )
        elif self._declared_sum is None and self._group is not None:
            hidden_states = sum_output(
                hidden_states,
                self._group,
                self._forward_batch,
                may_quantize=self._update.quantized_sum,
            )
        return self._stream.record(
            hidden_states, self._update, declared_sum=self._declared_sum
        )


class FfnExit:
    """Scope a selected FFN output decision and retain its completion action.

    Args:
        boundary: ExitPolicy that owns the producer's bound paths.
        forward_batch: Batch selecting reduction/finalize eligibility.
        stream: This invocation's residual stream, updated by finish().

    Fields:
        boundary: The owning output boundary.
        defer_moe_finalize: Whether compute may return a finalize handoff.

    The FFN leaves its output's sum; finish(output) completes it or carries
    it on with the action selected before compute. ``defer_moe_finalize`` is
    published on get_forward() only inside the context.
    """

    __slots__ = (
        "boundary",
        "defer_moe_finalize",
        "_complete",
        "_update",
        "_declared_sum",
        "_scope",
        "_stream",
    )

    def __init__(
        self,
        boundary: ExitPolicy,
        forward_batch: ForwardBatch,
        *,
        stream: ResidualStream,
    ):
        self._stream = stream
        self.boundary = boundary
        # The same path the entry took: the flags that select it hold for the
        # whole call, and a stage keeps no state between its entry and exit.
        steps = boundary.plan.path_for(forward_batch)
        completion = boundary._decide(forward_batch, steps)
        self.defer_moe_finalize = completion.defer_moe_finalize
        self._complete = completion.complete
        self._update = steps.output.update
        # A reduce-scatter that completes the sum on the way back is the
        # next input's only on an input-scattered batch.
        self._declared_sum = steps.exit.sum_left_to_next_input
        self._scope = get_forward().scoped(defer_moe_finalize=self.defer_moe_finalize)

    def __enter__(self) -> FfnExit:
        self._scope.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._scope.__exit__(*exc_info)

    def finish(self, hidden_states: torch.Tensor):
        """Complete or carry this output, preserving its producer update."""
        residual = self._stream.residual
        if not isinstance(hidden_states, torch.Tensor):
            # A deferred MoE finalize handoff, consumed by the next attention input.
            assert self.defer_moe_finalize, "unrequested deferred MoE handoff"
        else:
            hidden_states, residual = self._complete(hidden_states, residual)
        return self.boundary._record_output(
            hidden_states, residual, self._stream, self._update, self._declared_sum
        )
