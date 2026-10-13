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
from sglang.srt.layers.layer_boundary.adapters.attention import get_attn_tp_context
from sglang.srt.layers.layer_boundary.layout import (
    SumGroup,
    _ffn_has_tokens,
    _sum_group,
)
from sglang.srt.layers.layer_boundary.ops import (
    _all_reduce_then_to_local_tokens,
    _redistribute_output,
    _reduce_and_redistribute_output_max_len,
    _reduce_and_redistribute_output_varlen,
    _to_local_tokens,
)
from sglang.srt.layers.layer_boundary.output import (
    UnreducedOutput,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    apply_aiter_all_reduce_fusion,
    apply_flashinfer_allreduce_fusion,
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
from sglang.srt.true_on_policy import (
    should_disable_mlp_allreduce_fusion_for_on_policy,
)


def _reduce_and_redistribute_output_step(
    forward_batch: ForwardBatch,
    *,
    leaves_for_reduce_scatter: bool,
    leaves_for_reduce_scatterv: bool,
) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
    """The reduce-scatter that brings an FFN output gathered over attention
    DP back to this rank's tokens when the FFN leaves its sum to it (see
    StageOutput); None when the FFN reduces the output and only a scatter
    remains."""
    if should_use_dp_reduce_scatterv() and leaves_for_reduce_scatterv:
        return _reduce_and_redistribute_output_varlen
    if (
        leaves_for_reduce_scatter
        and forward_batch.dp_padding_mode.is_max_len()
        and can_use_dp_reduce_scatter()
    ):
        return _reduce_and_redistribute_output_max_len
    return None


class OutputBoundary:
    """Choose and finish one producer's output; the batch owns its residual."""

    def __init__(self, plan):
        self.plan = plan

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        stream: ResidualStream,
        forward_batch: ForwardBatch,
    ):
        """Move a complete output from the operation-scheduled producer path."""
        steps = self.plan._batch_steps(forward_batch)
        hidden_states, residual = self._complete_ffn_output_now(
            hidden_states,
            stream.residual,
            forward_batch=forward_batch,
            dp_step=None,
            steps=steps,
        )
        return self._leave_ffn_output(
            hidden_states,
            residual,
            stream,
            steps.output.update,
        )

    @staticmethod
    def _leave_ffn_output(hidden_states, residual, stream, update, declared_sum=None):
        stream.residual = residual
        if update.at_producer:
            return stream.write(hidden_states)
        return stream.leave(hidden_states, update, declared_sum=declared_sum)

    @staticmethod
    def _declared_ffn_sum(steps, skipped_reduction):
        # The input-scattered path leaves the sum for the next input's TP
        # reduce-scatter. Other skip paths complete it in the output move.
        if (
            skipped_reduction
            and not steps.returns_over_dp
            and not steps.output_move_completes_sum
        ):
            return SumGroup.TP
        return None

    def _local_token_move_can_go_to_next_layer(self, steps) -> bool:
        """Whether the next layer's input can run this layer's move of its FFN
        output back to this rank's tokens: the base postprocess scatter, when
        the next layer's input also writes the output into the residual."""
        return steps.returns_over_dp and not steps.output.update.at_producer

    def _postprocess_dp_step(
        self, forward_batch: ForwardBatch, steps
    ) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
        """The reduce-scatter that brings this layer's FFN output back to this
        rank's tokens under attention DP; None when the base postprocess would
        only scatter, or does not move tokens."""
        if not steps.returns_over_dp:
            return None
        return _reduce_and_redistribute_output_step(
            forward_batch,
            leaves_for_reduce_scatter=steps.output.leaves_for_reduce_scatter,
            leaves_for_reduce_scatterv=steps.output.leaves_for_reduce_scatterv,
        )

    def _ffn_leaves_sum_to_reduce_scatter(
        self, steps, dp_step: Optional[Callable]
    ) -> bool:
        """Whether the FFN leaves its sum out because a reduce-scatter completes
        it: the attention-DP one ``dp_step`` names, or the CP / input-scattered
        one."""
        if dp_step is not None:
            return True
        if not steps.output.leaves_for_reduce_scatter:
            return False
        if steps.output_move_completes_sum:
            return True
        return get_attn_tp_context().input_scattered and not self.plan.terminal

    def ffn_reduction_group(self, steps) -> GroupCoordinator:
        """The group this layer's FFN output owes its sum over: the MoE output's
        group on a sparse layer, the TP group a dense MLP reduces over."""
        return _sum_group(steps.output.group)

    def _select_ffn_completion(
        self, forward_batch: ForwardBatch, steps
    ) -> FfnCompletion:
        """Decide once, before the FFN runs, what it skips and what completes its
        output: the next layer's input, or this layer's postprocess step."""
        dp_step = self._postprocess_dp_step(forward_batch, steps)
        mlp_reduce_scatter = self._ffn_leaves_sum_to_reduce_scatter(steps, dp_step)
        complete_now = partial(
            self._complete_ffn_output_now,
            forward_batch=forward_batch,
            dp_step=dp_step,
            steps=steps,
        )
        defer_moe_finalize = (
            not should_disable_mlp_allreduce_fusion_for_on_policy()
            and self.plan.fusions is not None
            and self.plan.fusions.can_defer_finalize(self.plan, forward_batch)
        )
        if not steps.output.leaves_for_next_layer and not (
            self.plan.terminal and defer_moe_finalize
        ):
            return FfnCompletion(
                defer_moe_finalize=False,
                fuse_mlp_allreduce=False,
                mlp_reduce_scatter=mlp_reduce_scatter,
                complete=complete_now,
            )
        # Producers declare remaining work independently of the kernel chosen
        # by the consumer. Every handoff also carries an unfused completion.
        fuse_mlp_allreduce = defer_moe_finalize or self._ffn_sum_moves_to_next_layer(
            forward_batch, steps, mlp_reduce_scatter=mlp_reduce_scatter, dp_step=dp_step
        )
        if fuse_mlp_allreduce:
            group = self.ffn_reduction_group(steps)
            if steps.returns_over_dp:
                # Under attention DP the next layer also brings the sum back to
                # this rank's tokens.
                wrap = partial(
                    UnreducedOutput,
                    reduce_and_redistribute=partial(
                        _all_reduce_then_to_local_tokens, group, forward_batch
                    ),
                )
            else:
                wrap = partial(UnreducedOutput, group=group)
            complete = partial(_leave_to_next_layer, wrap)
        elif (
            dp_step is not None
            and not self.plan.terminal
            and self._local_token_move_can_go_to_next_layer(steps)
        ):
            complete = partial(
                _leave_to_next_layer,
                partial(
                    UnreducedOutput,
                    reduce_and_redistribute=partial(
                        _to_local_tokens, dp_step, forward_batch
                    ),
                ),
            )
        else:
            complete = complete_now
        return FfnCompletion(
            defer_moe_finalize=defer_moe_finalize,
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
            complete=complete,
        )

    def _complete_ffn_output_now(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        forward_batch: ForwardBatch,
        dp_step: Optional[Callable],
        steps,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This layer's postprocess, run with the attention-DP step already
        chosen: the move back to where the next layer reads the FFN output, then
        the write-back into the residual for a layer that does it itself."""
        if steps.output.transform is not None:
            hidden_states = steps.output.transform.apply(hidden_states)
        if steps.returns_over_dp:
            hidden_states = _to_local_tokens(
                dp_step or _redistribute_output, forward_batch, hidden_states
            )
        else:
            hidden_states, residual = steps.output_move(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        update = steps.output.update
        if residual is not None and update.at_producer:
            hidden_states = update.update(hidden_states, residual)
            residual = None
        return hidden_states, residual

    def mixer_exit(
        self, forward_batch: ForwardBatch, *, stream: ResidualStream
    ) -> MixerExit:
        """Decide once whether this stage's mixer (an attention-like stage)
        skips its output all-reduce. Use the result as a context manager around
        the mixer, then call ``finish``."""
        return MixerExit(self, forward_batch, stream=stream)

    def ffn_exit(
        self, forward_batch: ForwardBatch, *, stream: ResidualStream
    ) -> FfnExit:
        """Decide once how this layer's FFN output reduction completes. Use the
        result as a context manager around the FFN call, then call ``finish``."""
        return FfnExit(self, forward_batch, stream=stream)

    def _ffn_sum_can_move_to_next_layer(self, steps) -> bool:
        # Under the MoE-CP all-gather the fusion path would skip postprocess_layer
        # and its MoE-CP scatter, leaving hidden_states longer than the residual.
        if is_enable_moe_cp_allgather() or steps.output.group is None:
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
            and self.plan._speculative_algo is not None
            and self.plan._speculative_algo.is_eagle()
        ):
            return False

        return not get_attn_tp_context().input_scattered

    def _ffn_sum_moves_to_next_layer(
        self,
        forward_batch: ForwardBatch,
        steps,
        *,
        mlp_reduce_scatter: bool,
        dp_step: Optional[Callable],
    ) -> bool:
        """Whether the FFN leaves its output's all-reduce to the next layer's
        input when no fused kernel takes it: when the next layer would run the
        same all-reduce the FFN itself would have."""
        return (
            get_parallel().tp_size > 1
            and not self.plan.terminal
            and self._ffn_sum_can_move_to_next_layer(steps)
            and _can_defer_ffn_reduction(forward_batch, self.plan)
            and not mlp_reduce_scatter
            # Under attention DP the next layer must also run postprocess's
            # scatter back to this rank's tokens, and nothing more.
            and (
                not is_dp_attention_enabled()
                or (
                    self._local_token_move_can_go_to_next_layer(steps)
                    and dp_step is None
                )
            )
        )


def _can_defer_ffn_reduction(forward_batch: ForwardBatch, boundary=None) -> bool:
    """Admit ordinary single-sum outputs, plus LoRA and TP1 shared-expert
    outputs when a fused consumer (the backend's can_defer_all_reduce,
    FlashInfer or aiter) can take them.

    A fused-kernel fallback completes the partial-output contract selected
    before the producer ran, without recomputing the producer's LoRA path.
    """
    if not _ffn_has_tokens(forward_batch):
        return False
    if post_experts_sum_is_one_all_reduce():
        return True
    # LoRA-B is replicated and linear. TP1 shared experts add on rank zero
    # when the sum is deferred. Preserve these fused paths, but keep their
    # producer-side reduction order when no fused consumer is enabled.
    if not (
        (get_lora().enable_lora or envs.SGLANG_SHARED_EXPERT_TP1.get())
        and get_moe_a2a_backend().is_none()
        and not get_exec().comm.enable_quant_communications
        and post_experts_reduction_group() is get_parallel().tp_group
    ):
        return False
    if (
        boundary is not None
        and boundary.fusions is not None
        and boundary.fusions.can_defer_all_reduce(boundary, forward_batch)
    ):
        return True
    if apply_flashinfer_allreduce_fusion(forward_batch.input_ids.shape[0]):
        return True
    # Aiter also checks width and bytes, so use the actual residual storage.
    residual = forward_batch.residual_stream.residual
    return residual is not None and apply_aiter_all_reduce_fusion(
        residual, forward_batch
    )


class FfnCompletion(msgspec.Struct, frozen=True):
    """One decision shared by compute flags and the matching output completion.

    Fields:
        defer_moe_finalize: Compute may return a producer-specific finalize handoff.
        fuse_mlp_allreduce: Compute skips its all-reduce; completion runs it later
            or carries it to the next consumer, whether fused there or unfused.
        mlp_reduce_scatter: Compute leaves reduction to the selected scatter path.
        complete: Callable(output, residual) returning the completed or wrapped
            output and its corresponding residual rows.

    Do not independently reselect completion after compute has used these flags.
    """

    defer_moe_finalize: bool
    fuse_mlp_allreduce: bool
    mlp_reduce_scatter: bool
    complete: Callable[[torch.Tensor, torch.Tensor], Tuple]


def _leave_to_next_layer(
    wrap: Callable[[torch.Tensor], UnreducedOutput],
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> Tuple[UnreducedOutput, torch.Tensor]:
    return wrap(hidden_states), residual


class MixerExit:
    """The scope that publishes a mixer's decision while it runs: inside the
    ``with`` block ``fuse_mlp_allreduce`` on ``get_forward()`` tells its
    row-parallel output projection to skip the all-reduce. It skips when the
    stage's output always leaves its sum (to an FFN stage, which completes it in
    its input), or when its declaration permits deferring the sum to the next
    attention stage and the batch allows it: TP > 1, no attention DP, and no
    input-scattered or MoE-CP all-gather layout. The consumer decides whether
    to fuse its completion with the input norm."""

    __slots__ = (
        "skips_reduction",
        "_hands_on",
        "_scope",
        "_update",
        "_declared_sum",
        "_stream",
    )

    def __init__(
        self,
        boundary: OutputBoundary,
        forward_batch: ForwardBatch,
        *,
        stream: ResidualStream,
    ):
        steps = boundary.plan._batch_steps(forward_batch)
        produced = steps.output
        self._stream = stream
        self._update = produced.update
        self._declared_sum = produced.group if produced.always_leaves else None
        self._hands_on = (
            produced.leaves_for_next_layer
            and get_parallel().tp_size > 1
            and not is_dp_attention_enabled()
            and boundary._ffn_sum_can_move_to_next_layer(steps)
        )
        self.skips_reduction = produced.always_leaves or self._hands_on
        self._scope = get_forward().scoped(fuse_mlp_allreduce=self.skips_reduction)

    def __enter__(self) -> MixerExit:
        self._scope.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._scope.__exit__(*exc_info)

    def finish(self, hidden_states: torch.Tensor):
        """Hand the actual mixer result to the next boundary through the stream."""
        if self._hands_on:
            hidden_states = UnreducedOutput(
                hidden_states, group=get_parallel().tp_group
            )
        return self._stream.leave(
            hidden_states, self._update, declared_sum=self._declared_sum
        )


class FfnExit:
    """Scope a selected FFN output decision and retain its completion action.

    Args:
        boundary: OutputBoundary that owns the producer's bound paths.
        forward_batch: Batch selecting reduction/finalize eligibility.
        stream: This invocation's residual stream, updated by finish().

    Fields:
        boundary: The owning output boundary.
        defer_moe_finalize: Whether compute may return a finalize handoff.
        fuse_mlp_allreduce: Whether compute leaves its all-reduce to completion.
        mlp_reduce_scatter: Whether compute leaves reduction to a scatter path.

    The three flags are published on get_forward() only inside the context.
    finish(output) uses the same selected action after successful compute.
    """

    __slots__ = (
        "boundary",
        "defer_moe_finalize",
        "fuse_mlp_allreduce",
        "mlp_reduce_scatter",
        "_complete",
        "_update",
        "_declared_sum",
        "_scope",
        "_stream",
    )

    def __init__(
        self,
        boundary: OutputBoundary,
        forward_batch: ForwardBatch,
        *,
        stream: ResidualStream,
    ):
        self._stream = stream
        self.boundary = boundary
        steps = boundary.plan._batch_steps(forward_batch)
        completion = boundary._select_ffn_completion(forward_batch, steps)
        self.defer_moe_finalize = completion.defer_moe_finalize
        self.fuse_mlp_allreduce = completion.fuse_mlp_allreduce
        self.mlp_reduce_scatter = completion.mlp_reduce_scatter
        self._complete = completion.complete
        self._update = steps.output.update
        self._declared_sum = boundary._declared_ffn_sum(
            steps, completion.mlp_reduce_scatter
        )
        self._scope = get_forward().scoped(
            fuse_mlp_allreduce=self.fuse_mlp_allreduce,
            mlp_reduce_scatter=self.mlp_reduce_scatter,
            defer_moe_finalize=self.defer_moe_finalize,
        )

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
        return self.boundary._leave_ffn_output(
            hidden_states, residual, self._stream, self._update, self._declared_sum
        )
