"""Family-agnostic decoder adapter: stage split, local compute, stage join.

Everything here depends only on SGLang's ForwardBatch and on the model-side AFD
contract (``forward_attention_for_afd`` / ``compute_ffn_output`` /
``layer_communicator.postprocess_layer``). A model family contributes an
identity check and a guard class; it does not reimplement orchestration.
"""

from __future__ import annotations

import copy
from typing import Any

import msgspec

from sglang.srt.model_executor.forward_batch_view import slice_batch_field
from sglang.srt.runtime_context import get_parallel

from ..contracts import (
    AFDError,
    AFDRole,
    MetadataContract,
    validate_non_speculative_batch,
)


class AFDStage(msgspec.Struct, kw_only=True):
    index: int
    request_start: int
    request_stop: int
    token_start: int
    token_stop: int
    hidden_states: Any
    residual: Any
    positions: Any
    forward_batch: Any
    attention_metadata: Any = None
    merge_sizes: tuple[int, ...] = ()
    merge_lane: int = 0
    # Padded width of each attention lane's block inside this FFN rank's own
    # contribution to the merge; empty on the attention role, one entry per lane
    # the rank serves on the FFN role.
    group_widths: tuple[int, ...] = ()
    # Empty-ingress F ranks still own a device/dtype template and compute experts.
    collective_input: Any = None
    graph_output: Any = None

    @property
    def rows(self) -> int:
        return self.token_stop - self.token_start


def _slice_positions(value: Any, token_slice: slice) -> Any:
    if value is None:
        return None
    return value[..., token_slice] if value.ndim >= 2 else value[token_slice]


def _request_stage_sizes(*, batch_size: int, stages: int) -> tuple[int, ...]:
    if stages != 2:
        raise AFDError("AFD_GRAPH_STAGE_COUNT_UNSUPPORTED", f"stages={stages}")
    return ((batch_size + 1) // 2, batch_size // 2)


def _token_lengths(forward_batch: Any) -> tuple[int, ...]:
    validate_non_speculative_batch(forward_batch)
    if forward_batch.forward_mode.is_decode_or_idle():
        # An idle rank is not an empty lane. DP attention idles a rank with no
        # requests to keep its peers' collectives symmetric, and MAX_LEN padding
        # -- which decode always uses -- then sets its batch size to the global
        # max and pads its inputs to match, so it carries one token per padded
        # row like any decode rank. Only an unpadded extend step leaves it at
        # zero rows, which this same formula yields.
        return (1,) * forward_batch.batch_size
    if forward_batch.extend_seq_lens_cpu is None:
        raise AFDError(
            "AFD_FORWARD_MODE_UNSUPPORTED",
            f"mode={forward_batch.forward_mode!r}",
        )
    return tuple(int(value) for value in forward_batch.extend_seq_lens_cpu)


def _stage_ranges(
    *,
    batch_size: int,
    token_lengths: tuple[int, ...],
    stages: int,
) -> tuple[tuple[int, int, int, int], ...]:
    request_sizes = _request_stage_sizes(batch_size=batch_size, stages=stages)
    request_start = 0
    token_start = 0
    result = []
    for request_size in request_sizes:
        request_stop = request_start + request_size
        token_stop = token_start + sum(token_lengths[request_start:request_stop])
        result.append((request_start, request_stop, token_start, token_stop))
        request_start = request_stop
        token_start = token_stop
    return tuple(result)


def _slice_forward_batch(
    *,
    forward_batch: Any,
    request_slice: slice,
    token_slice: slice,
) -> Any:
    child = copy.copy(forward_batch)
    child.input_ids = slice_batch_field(forward_batch.input_ids, token_slice)
    child.positions = _slice_positions(forward_batch.positions, token_slice)
    child.out_cache_loc = slice_batch_field(forward_batch.out_cache_loc, token_slice)
    child.input_embeds = slice_batch_field(forward_batch.input_embeds, token_slice)
    child.token_type_ids = slice_batch_field(forward_batch.token_type_ids, token_slice)
    child.req_pool_indices = slice_batch_field(
        forward_batch.req_pool_indices, request_slice
    )
    child.req_pool_indices_cpu = slice_batch_field(
        forward_batch.req_pool_indices_cpu, request_slice
    )
    child.seq_lens = slice_batch_field(forward_batch.seq_lens, request_slice)
    child.seq_lens_cpu = slice_batch_field(forward_batch.seq_lens_cpu, request_slice)
    child.extend_seq_lens = slice_batch_field(
        forward_batch.extend_seq_lens, request_slice
    )
    child.extend_prefix_lens = slice_batch_field(
        forward_batch.extend_prefix_lens, request_slice
    )
    child.extend_seq_lens_cpu = slice_batch_field(
        forward_batch.extend_seq_lens_cpu, request_slice
    )
    child.extend_prefix_lens_cpu = slice_batch_field(
        forward_batch.extend_prefix_lens_cpu, request_slice
    )
    child.extend_logprob_start_lens_cpu = slice_batch_field(
        forward_batch.extend_logprob_start_lens_cpu, request_slice
    )
    child.lora_ids = slice_batch_field(forward_batch.lora_ids, request_slice)
    child.rids = slice_batch_field(forward_batch.rids, request_slice)
    child.batch_size = request_slice.stop - request_slice.start
    # Sum on the host copy. `.item()` on the device tensor blocks this host until
    # a freshly launched reduce comes back, and that reduce queues behind whatever
    # is already on the stream -- under CPU-overlap scheduling that is the previous
    # step's whole graph, so the sync would re-serialize the pipeline it is meant
    # to overlap. seq_lens_cpu is sliced just above.
    if child.seq_lens_cpu is not None:
        child.seq_lens_sum = int(child.seq_lens_cpu.sum())
    elif child.seq_lens is not None:
        child.seq_lens_sum = int(child.seq_lens.sum().item())
    else:
        child.seq_lens_sum = 0
    child.extend_num_tokens = token_slice.stop - token_slice.start
    if child.extend_seq_lens is not None:
        starts = [0]
        lengths = [int(value) for value in child.extend_seq_lens_cpu]
        for length in lengths[:-1]:
            starts.append(starts[-1] + length)
        child.extend_start_loc = child.extend_seq_lens.new_tensor(starts)
    child.forward_metadata_ready = False
    child.forward_metadata_planned_bs = None
    child.forward_metadata_planned_num_tokens = None
    return child


class AFDDecoderAdapter:
    """Shared stage orchestration for one decoder-layer model family."""

    guard_class: type
    family_error: str

    def __init__(
        self,
        *,
        role: AFDRole,
        model: Any,
        attention_backend: Any | None,
    ) -> None:
        self.role = role
        self.model = model
        self.inner = model.model
        self.attention_backend = attention_backend
        self.num_layers = len(self.inner.layers)
        self.hidden_size = self.inner.config.hidden_size

    @property
    def metadata_contract(self) -> MetadataContract:
        return self.guard_class.metadata_contract

    def matches_family(self) -> bool:
        raise NotImplementedError

    def validate_context_parallel(self) -> None:
        """v0.5.20 owns CP settings in the published parallel configuration."""

        cfg = get_parallel()
        if cfg.attn_cp_size != 1 or cfg.attn_dcp_size != 1 or cfg.enable_prefill_cp:
            raise AFDError(
                "AFD_CONTEXT_PARALLEL_UNSUPPORTED",
                f"attn_cp_size={cfg.attn_cp_size!r} "
                f"attn_dcp_size={cfg.attn_dcp_size!r} "
                f"enable_prefill_cp={cfg.enable_prefill_cp!r}",
            )

    def validate_model(self) -> None:
        if not self.matches_family():
            raise AFDError(
                self.family_error,
                f"class={type(self.model).__name__}",
            )
        if self.role == AFDRole.ATTENTION:
            if self.attention_backend is None:
                raise AFDError("AFD_ATTENTION_BACKEND_REQUIRED")
            self.guard_class.validate_backend(self.attention_backend)
        if self.inner.start_layer != 0 or self.inner.end_layer != self.num_layers:
            raise AFDError("AFD_PIPELINE_PARALLEL_UNSUPPORTED")
        self.validate_context_parallel()

    def validate_merge_reduce_scatter(self) -> None:
        """Refuse the reduce-scatter merge when shared experts sit outside it.

        Replicated shared experts are added *after* the post-experts all-reduce,
        so they are not part of the partial sum. A reduce-scatter over the whole
        output would therefore sum the same shared contribution once per rank.
        """

        for index, layer in enumerate(self.inner.layers):
            if getattr(
                getattr(layer, "mlp", None), "_shared_expert_tp1", False
            ) and not (getattr(layer, "afd_shared_expert_partial", False)):
                raise AFDError(
                    "AFD_FFN_MERGE_REDUCE_SCATTER_SHARED_TP1",
                    f"layer={index}",
                )

    def attention_capability(
        self,
        *,
        configured_backend: str,
    ) -> dict[str, str]:
        """Return the exact shared backend contract, checking the live role."""

        guard_class = self.guard_class
        if configured_backend not in guard_class.backends:
            raise AFDError(
                "AFD_ATTENTION_BACKEND_CONFIG_UNSUPPORTED",
                f"configured={configured_backend!r} supported={guard_class.backends!r}",
            )
        if self.role == AFDRole.ATTENTION:
            identity = guard_class.validate_backend(self.attention_backend)
            if identity[2:] != (configured_backend, configured_backend):
                raise AFDError(
                    "AFD_ATTENTION_BACKEND_CONFIG_RUNTIME_DRIFT",
                    f"configured={configured_backend!r} actual={identity[2:]!r}",
                )
        return {
            "backend": configured_backend,
            "backend_class": guard_class.backend_class,
            "metadata_contract": self.metadata_contract.value,
        }

    def initialize_graph_metadata(self, *, max_rows: int) -> int:
        if self.role != AFDRole.ATTENTION or max_rows < 1:
            raise AFDError(
                "AFD_METADATA_STATE_INVALID",
                f"role={self.role.value} max_rows={max_rows}",
            )
        return self.guard_class.initialize_shared_state(
            backend=self.attention_backend,
            max_rows=max_rows,
        )

    def prepare_stage(self, *, stage: AFDStage) -> None:
        if self.role == AFDRole.ATTENTION:
            if stage.attention_metadata is None:
                self.attention_backend.init_forward_metadata(stage.forward_batch)
                stage.attention_metadata = self.attention_backend.forward_metadata
            self.attention_backend.forward_metadata = stage.attention_metadata

    def make_ffn_stages(
        self,
        *,
        descriptor: Any,
        buffers: tuple[Any, ...],
        lane: int,
        shape: Any,
    ) -> tuple[list[AFDStage], list[Any]]:
        group = shape.group_lanes(ffn_ordinal=lane)
        stage_count = len(shape.lane_stage_rows[0])
        slots = max(1, len(group))
        if len(buffers) != stage_count * slots:
            raise AFDError(
                "AFD_FFN_STAGE_BUFFER_COUNT_INVALID",
                f"buffers={len(buffers)} stages={stage_count} group={len(group)}",
            )
        stages = []
        views = []
        token_start = 0
        for index in range(stage_count):
            offset = index * slots
            lane_rows = tuple(
                descriptor.stage_rows(lane=member)[index] for member in group
            )
            lane_views = tuple(
                buffer[:rows]
                for rows, buffer in zip(
                    lane_rows,
                    buffers[offset : offset + len(group)],
                )
            )
            token_stop = token_start + sum(lane_rows)
            stages.append(
                AFDStage(
                    index=index,
                    request_start=0,
                    request_stop=sum(lane_rows),
                    token_start=token_start,
                    token_stop=token_stop,
                    hidden_states=lane_views,
                    residual=None,
                    positions=None,
                    forward_batch=None,
                    merge_sizes=shape.merge_plan(stage=index),
                    merge_lane=lane,
                    collective_input=buffers[offset] if not group else None,
                    group_widths=tuple(
                        shape.lane_bucket_rows[member][index] for member in group
                    ),
                )
            )
            views.append(lane_views)
            token_start = token_stop
        return stages, views

    def lane_stage_rows(
        self,
        *,
        forward_batch: Any,
        local_rows: tuple[int, ...],
        lanes: int,
        lane: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Every attention lane's per-stage rows, which set the FFN merge widths."""

        if lanes == 1:
            return (local_rows,)
        if forward_batch.is_extend_in_batch:
            # Extend stages break on request boundaries, so a lane's per-stage
            # rows follow its own sequence lengths and no per-lane aggregate can
            # reconstruct them. The TP group spans exactly the attention lanes.
            # The flag is a max over every lane, so an idle lane picks the same
            # branch as its peers instead of posting a gather nobody joins.
            gathered_rows = get_parallel().tp_group.all_gather_object(local_rows)
            if len(gathered_rows) != lanes:
                raise AFDError(
                    "AFD_LANE_ROW_PLAN_UNAVAILABLE",
                    f"gathered={gathered_rows!r} lanes={lanes}",
                )
            return tuple(
                tuple(int(value) for value in vector) for vector in gathered_rows
            )
        # DP attention already all-gathered this vector; in decode each request
        # contributes exactly one token, so it is the per-lane batch size.
        gathered = forward_batch.global_num_tokens_cpu
        if gathered is None or len(gathered) != lanes:
            raise AFDError(
                "AFD_LANE_ROW_PLAN_UNAVAILABLE",
                f"gathered={gathered!r} lanes={lanes}",
            )
        validate_non_speculative_batch(forward_batch)
        derived = tuple(
            _request_stage_sizes(batch_size=int(value), stages=len(local_rows))
            for value in gathered
        )
        if derived[lane] != local_rows:
            raise AFDError(
                "AFD_LANE_ROW_PLAN_LOCAL_DISAGREES",
                f"lane={lane} derived={derived[lane]!r} local={local_rows!r}",
            )
        return derived

    def split_step(
        self,
        *,
        hidden_states: Any,
        residual: Any,
        positions: Any,
        forward_batch: Any,
        stages: int,
    ) -> list[AFDStage]:
        token_lengths = _token_lengths(forward_batch)
        ranges = _stage_ranges(
            batch_size=forward_batch.batch_size,
            token_lengths=token_lengths,
            stages=stages,
        )
        result = []
        for index, (req_start, req_stop, token_start, token_stop) in enumerate(ranges):
            token_slice = slice(token_start, token_stop)
            child = _slice_forward_batch(
                forward_batch=forward_batch,
                request_slice=slice(req_start, req_stop),
                token_slice=token_slice,
            )
            result.append(
                AFDStage(
                    index=index,
                    request_start=req_start,
                    request_stop=req_stop,
                    token_start=token_start,
                    token_stop=token_stop,
                    hidden_states=hidden_states[token_slice],
                    residual=None if residual is None else residual[token_slice],
                    positions=_slice_positions(positions, token_slice),
                    forward_batch=child,
                    attention_metadata=None,
                )
            )
        return result

    def local_compute(
        self,
        *,
        layer: int,
        stage: AFDStage,
        hidden_states: Any,
        residual: Any,
        positions: Any = None,
    ) -> tuple[Any, ...]:
        decoder_layer = self.inner.layers[layer]
        if self.role == AFDRole.ATTENTION:
            return decoder_layer.forward_attention_for_afd(
                positions=stage.positions if positions is None else positions,
                hidden_states=hidden_states,
                forward_batch=stage.forward_batch,
                residual=residual,
            )
        return (
            self._group_ffn_output(
                decoder_layer=decoder_layer,
                stage=stage,
                lane_hidden_states=hidden_states,
            ),
            residual,
        )

    def _group_ffn_output(
        self,
        *,
        decoder_layer: Any,
        stage: AFDStage,
        lane_hidden_states: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Run MoE over one batch for this rank's lanes, then split the rows back.

        A rank serving k attention lanes concatenates their blocks so the gate and
        the experts see one batch and the cross-rank gather stays a plain
        one-size-per-rank collective. Where a lane's rows sit inside that batch is
        the bucket's padded width whenever the gather runs, not the live count, so
        a captured merge keeps one shape while real row counts move inside the
        bucket and ranks that independently pick replay or eager still post
        identical collectives.

        Every rank's experts produce a partial sum for every merged row, and this
        rank keeps only the slice belonging to the lanes it serves, so
        all-reducing and then slicing is a reduce-scatter spelled out the long
        way -- and it moves twice the bytes. ``mlp_reduce_scatter`` is how the
        tree already asks an MLP to hand back unreduced partials: its readers are
        the dense row-parallel projection and the post-experts TP all-reduce,
        which are exactly the two reductions this path takes over.
        """

        import torch

        from sglang.srt.runtime_context import get_forward

        widths = stage.group_widths
        rows = tuple(tensor.shape[0] for tensor in lane_hidden_states)
        if len(rows) != len(widths) or any(
            row > width for row, width in zip(rows, widths)
        ):
            raise AFDError(
                "AFD_FFN_MERGE_WIDTH_EXCEEDED",
                f"rows={rows!r} widths={widths!r} stage={stage.index}",
            )
        gather = len(stage.merge_sizes) > 1
        strides = widths if gather else rows
        total = sum(strides)

        def pack(tensors):
            if len(rows) == 1 and rows[0] == total:
                return tensors[0]
            packed = torch.zeros(
                (total,) + tuple(tensors[0].shape[1:]),
                dtype=tensors[0].dtype,
                device=tensors[0].device,
            )
            offset = 0
            for tensor, stride, row in zip(tensors, strides, rows):
                packed[offset : offset + row] = tensor
                offset += stride
            return packed

        if lane_hidden_states:
            local = pack(lane_hidden_states)
        else:
            local = stage.collective_input
            if local is None or local.shape[0] != 0:
                raise AFDError("AFD_FFN_EMPTY_INGRESS_TEMPLATE_INVALID")
        merged_tensors = [local]
        tp_group = get_parallel().tp_group
        sizes = list(stage.merge_sizes)
        if gather:
            merged_tensors = tp_group.all_gatherv(local, sizes=sizes)
            # The collective returns this rank's chunk, with lane offsets local
            # to that chunk. Lane order must therefore agree with rank order.
            if stage.merge_lane != tp_group.rank_in_group:
                raise AFDError(
                    "AFD_FFN_MERGE_LANE_RANK_MISMATCH",
                    f"lane={stage.merge_lane} rank={tp_group.rank_in_group}",
                )
        merged = merged_tensors[0]
        if gather:
            with get_forward().scoped(mlp_reduce_scatter=True):
                partial = decoder_layer.compute_ffn_output(merged, stage.forward_batch)
            output = tp_group.reduce_scatterv(partial, sizes=sizes)
            if not lane_hidden_states:
                # Expose real expert work to the replay sentinel, not an empty
                # reduce-scatter result or a made-up constant.
                stage.graph_output = partial
        else:
            output = decoder_layer.compute_ffn_output(merged, stage.forward_batch)
        if len(strides) == 1 and rows[0] == output.shape[0]:
            return (output,)
        result = []
        offset = 0
        for stride, row in zip(strides, rows):
            result.append(output.narrow(0, offset, row))
            offset += stride
        return tuple(result)

    def finish_layer(
        self,
        *,
        layer: int,
        stage: AFDStage,
        ffn_output: Any,
        residual: Any,
    ) -> tuple[Any, Any]:
        if self.role != AFDRole.ATTENTION:
            return ffn_output, residual
        return self.inner.layers[layer].layer_communicator.postprocess_layer(
            ffn_output,
            residual,
            stage.forward_batch,
        )

    def join_step(self, *, stages: list[AFDStage]) -> tuple[Any, Any]:
        import torch

        for stage in stages:
            if stage.residual is None:
                continue
            try:
                residual_rows = int(stage.residual.shape[0])
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                raise AFDError(
                    "AFD_STAGE_RESIDUAL_SHAPE_DRIFT",
                    f"stage={stage.index} rows={stage.rows}",
                ) from exc
            if residual_rows != stage.rows:
                raise AFDError(
                    "AFD_STAGE_RESIDUAL_SHAPE_DRIFT",
                    (
                        f"stage={stage.index} rows={stage.rows} "
                        f"residual_rows={residual_rows}"
                    ),
                )
        live = tuple(stage for stage in stages if stage.rows > 0)
        consensus = live if live else tuple(stages)
        residuals = tuple(stage.residual for stage in consensus)
        has_none = tuple(value is None for value in residuals)
        if any(has_none) and not all(has_none):
            raise AFDError("AFD_STAGE_RESIDUAL_IDENTITY_DRIFT")
        hidden_states = torch.cat(
            [stage.hidden_states for stage in stages],
            dim=0,
        )
        if all(has_none):
            residual = None
        else:
            residual = torch.cat(
                [stage.residual for stage in stages if stage.residual is not None],
                dim=0,
            )
        return hidden_states, residual

    def metadata_guard(
        self,
        *,
        stage: AFDStage,
        bucket_rows: int,
    ) -> Any | None:
        if self.role == AFDRole.FFN:
            return None
        return self.guard_class(
            backend=self.attention_backend,
            stage=stage,
            bucket_rows=bucket_rows,
        )
