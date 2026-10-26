"""MoE operand transport and completion, preserving source residual rows."""

from contextlib import contextmanager, nullcontext

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.interleave import attn_cp_interleave_gather
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.dp_attention import (
    _tbo_event,
    attn_tp_all_gather,
    attn_tp_all_gather_into_tensor,
    dp_gather_partial,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_reduce_scatterv_async,
    dp_scatter,
    get_dp_global_num_tokens,
    get_dp_tbo_comm_stream,
    get_global_dp_buffer,
    get_global_dp_buffer_len,
    get_local_dp_buffer,
    get_local_dp_buffer_len,
    get_tbo_persistent_buffer,
    is_dp_gatherv_active,
)
from sglang.srt.layers.layer_boundary.ops import attn_cp_interleave_reduce_scatter
from sglang.srt.layers.layer_boundary.output import DeferredFinalize
from sglang.srt.layers.layer_boundary.row_transport import ComputeTransport
from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
SHARED_EXPERT_LOCAL = get_bool_env_var("SGLANG_DP_SHARED_EXPERT_LOCAL")


def shared_expert_runs_locally(moe):
    """Whether the replicated shared expert can run before the DP gather."""
    return (
        SHARED_EXPERT_LOCAL
        and getattr(moe, "shared_experts", None) is not None
        and getattr(moe, "_shared_expert_tp1", False)
    )


@contextmanager
def _every_row_routed(forward_batch, num_rows: int):
    # Under CP the real rows are not a prefix, so every row must be routed.
    saved = (
        forward_batch.num_token_non_padded,
        forward_batch.global_num_token_non_padded_cpu,
    )
    if saved[0] is not None:
        forward_batch.num_token_non_padded = torch.full_like(saved[0], num_rows)
    forward_batch.global_num_token_non_padded_cpu = num_rows
    try:
        yield
    finally:
        (
            forward_batch.num_token_non_padded,
            forward_batch.global_num_token_non_padded_cpu,
        ) = saved


def _mask_dp_pad_rows(local_hidden_states, forward_batch) -> None:
    n_real = forward_batch.num_token_non_padded
    num_rows = local_hidden_states.shape[0]
    if n_real is None or num_rows == 0:
        return
    pad = torch.arange(num_rows, device=local_hidden_states.device) >= n_real
    local_hidden_states.masked_fill_(
        pad.view(-1, *([1] * (local_hidden_states.dim() - 1))), 0
    )


def _attn_tp_all_gather_chunks(
    local: torch.Tensor, chunk_rows: list[int]
) -> torch.Tensor:
    # Tensor all_gather stays on aiter AG / pynccl under graph capture; the list
    # all_gather falls back to ProcessGroupNCCL, which invalidates HIP capture.
    max_rows = max(chunk_rows)
    padded = local
    if local.shape[0] != max_rows:
        padded = local.new_zeros((max_rows, *local.shape[1:]))
        padded[: local.shape[0]].copy_(local)
    gathered = local.new_empty((len(chunk_rows) * max_rows, *local.shape[1:]))
    attn_tp_all_gather_into_tensor(gathered, padded.contiguous())
    if all(rows == max_rows for rows in chunk_rows):
        return gathered
    return torch.cat(
        [
            gathered[i * max_rows : i * max_rows + rows]
            for i, rows in enumerate(chunk_rows)
        ]
    )


class MoeRowTransport(ComputeTransport):
    def __init__(
        self,
        moe,
        hidden_states,
        forward_batch,
        input_ids,
        input_ids_global,
        *,
        cp_return_contiguous=False,
        use_cp=None,
        moe_gather=None,
        attn_scatter=None,
    ):
        _use_cp = is_cp_active(forward_batch) if use_cp is None else use_cp
        _use_tp_moe_gather = (
            moe_gather
            if moe_gather is not None
            else (
                not _use_cp
                and get_parallel().attn_dp_size > 1
                and get_moe_a2a_backend().is_none()
            )
        )
        _use_tp_attn_a2a_scatter = (
            attn_scatter
            if attn_scatter is not None
            else (
                not _use_cp
                and get_parallel().attn_tp_size > 1
                and not get_moe_a2a_backend().is_none()
            )
        )
        # Variable-length combine sums TP-sharded expert output while returning
        # each DP rank's rows. The producer leaves that sum to the transport.
        _use_reduce_scatterv = (
            _use_tp_moe_gather
            and is_dp_gatherv_active()
            and forward_batch.dp_padding_mode is not None
            and not forward_batch.dp_padding_mode.is_max_len()
        )
        # Equal-chunk combine requires MAX_LEN padding and one attention rank
        # per DP shard; the variable-length path takes precedence.
        _use_reduce_scatter = (
            envs.SGLANG_DP_USE_REDUCE_SCATTER.get()
            and _use_tp_moe_gather
            and not _use_reduce_scatterv
            and not should_use_dp_reduce_scatterv()
            and forward_batch.dp_padding_mode is not None
            and forward_batch.dp_padding_mode.is_max_len()
            and get_parallel().tp_size == get_parallel().attn_dp_size
        )
        mlp_reduce_scatter = (
            _use_cp
            or _use_reduce_scatterv
            or _use_reduce_scatter
            or (_use_tp_moe_gather and should_use_dp_reduce_scatterv())
        )
        # A replicated shared expert can run on local rows before the gather;
        # its output is added after returning the routed experts' local slice.
        _shared_local = None
        do_shared_local = _use_tp_moe_gather and shared_expert_runs_locally(moe)
        if _use_cp:
            moe_a2a_backend = get_moe_a2a_backend()
            if moe_a2a_backend.is_none():
                hidden_states = attn_cp_interleave_gather(hidden_states)
            else:
                assert (
                    moe_a2a_backend.is_deepep()
                    or moe_a2a_backend.is_megamoe()
                    or moe_a2a_backend.is_mori()
                ), (
                    "CP requires moe_a2a_backend in ('deepep', 'megamoe', 'mori'), "
                    f"got {moe_a2a_backend.value!r}."
                )
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_parallel().tp_group),
                hidden_states,
            )
            if do_shared_local and local_hidden_states.shape[0] > 0:
                _shared_local = moe._forward_shared_experts(local_hidden_states)
            # Pad rows are not refreshed during graph replay. Stale bf16-max
            # values can overflow the MoE and contaminate the combine output.
            if _is_hip:
                _mask_dp_pad_rows(local_hidden_states, forward_batch)
            # self_attn has already reduced across attention TP, so these hidden
            # states are replicated and must not be summed by a partial gather.
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_parallel().attn_tp_size, get_parallel().attn_tp_rank
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            # DSpark next-token layers are not hash-routed and intentionally do not
            # carry token IDs. Only split IDs for callers that actually provide them.
            if input_ids is not None:
                input_ids = input_ids.tensor_split(s)[r].contiguous()
            if input_ids_global is not None:
                input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()
        gathered_rows = (
            _every_row_routed(forward_batch, hidden_states.shape[0])
            if _use_cp and get_moe_a2a_backend().is_none()
            else nullcontext()
        )
        self.input = hidden_states
        self.input_ids, self.input_ids_global = input_ids, input_ids_global
        self.batch = forward_batch
        self.cp_return_contiguous = cp_return_contiguous
        self._use_cp = _use_cp
        self._use_tp_moe_gather = _use_tp_moe_gather
        self._use_tp_attn_a2a_scatter = _use_tp_attn_a2a_scatter
        self._use_reduce_scatterv = _use_reduce_scatterv
        self._use_reduce_scatter = _use_reduce_scatter
        self.mlp_reduce_scatter = mlp_reduce_scatter
        self._shared_local = _shared_local
        self.do_shared_local = do_shared_local
        self._a2a_scatter_chunks = _a2a_scatter_chunks
        self.gathered_rows = gathered_rows

    def return_output(self, hidden_states):
        forward_batch = self.batch
        _use_cp = self._use_cp
        _use_tp_moe_gather = self._use_tp_moe_gather
        _use_tp_attn_a2a_scatter = self._use_tp_attn_a2a_scatter
        _use_reduce_scatterv = self._use_reduce_scatterv
        _use_reduce_scatter = self._use_reduce_scatter
        _shared_local = self._shared_local
        _a2a_scatter_chunks = self._a2a_scatter_chunks
        if _use_cp and get_moe_a2a_backend().is_none():
            if self.cp_return_contiguous:
                parallel = get_parallel()
                hidden_states = parallel.tp_group.all_reduce(hidden_states)
                hidden_states = hidden_states.tensor_split(parallel.attn_cp_size)[
                    parallel.attn_cp_rank
                ].contiguous()
            else:
                hidden_states = attn_cp_interleave_reduce_scatter(hidden_states)
        elif _use_tp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_parallel().tp_group),
                hidden_states,
            )
            if should_use_dp_reduce_scatterv() or _use_reduce_scatterv:
                get_parallel().tp_group.reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=get_dp_global_num_tokens(),
                )
            elif _use_reduce_scatter:
                dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
            else:
                dp_scatter(hidden_states, global_hidden_states, forward_batch)
            if _shared_local is not None:
                n = hidden_states.shape[0]
                hidden_states = hidden_states + _shared_local[:n]
        if _use_tp_attn_a2a_scatter:
            assert _a2a_scatter_chunks is not None
            if _is_hip:
                hidden_states = _attn_tp_all_gather_chunks(
                    hidden_states, chunk_rows=[t.shape[0] for t in _a2a_scatter_chunks]
                )
            else:
                gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
                attn_tp_all_gather(gathered, hidden_states.contiguous())
                hidden_states = torch.cat(gathered)
        return hidden_states

    @contextmanager
    def scope(self):
        with (
            get_forward().scoped(
                mlp_reduce_scatter=self.mlp_reduce_scatter, defer_moe_finalize=False
            ),
            self.gathered_rows,
        ):
            yield

    def release(self):
        self.input = self._shared_local = self._a2a_scatter_chunks = None


def _tbo_gather_moe_input(moe, local, batch, sub, do_shared_local):
    shared_local = (
        moe._forward_shared_experts(local)
        if do_shared_local and local.shape[0] > 0
        else None
    )
    global_hidden = get_tbo_persistent_buffer(
        ("gh", sub),
        get_global_dp_buffer_len(),
        local.shape[1],
        local.dtype,
        local.device,
    )
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        dp_gather_partial(global_hidden, local, batch)
        event = _tbo_event(("gather", sub))
        event.record(comm)
    return global_hidden, event, shared_local


def _tbo_combine_moe_output(output, sub):
    local = get_tbo_persistent_buffer(
        ("lo", sub),
        get_local_dp_buffer_len(),
        output.shape[1],
        output.dtype,
        output.device,
    )
    event = dp_reduce_scatterv_async(
        local, output, get_dp_global_num_tokens(), event_key=("combine", sub)
    )
    return local, event


def _tbo_wait_moe_output(local, event, shared_local):
    torch.cuda.current_stream().wait_event(event)
    return local + shared_local[: local.shape[0]] if shared_local is not None else local


class MoeFinalize(DeferredFinalize, frozen=True):
    pieces: object = None
    partial: object = None

    def complete(self, *, reduce=True):
        from sglang.srt.layers.moe import post_experts_all_reduce
        from sglang.srt.layers.moe.utils import should_add_replicated_moe_output

        with get_forward().scoped(mlp_reduce_scatter=not reduce):
            value = (
                self.pieces.get_merged() if self.pieces is not None else self.partial
            )
            if reduce:
                value = post_experts_all_reduce(value)
            if (
                self.pieces is not None
                and self.pieces.shared is not None
                and self.pieces.shared_is_replicated
                and should_add_replicated_moe_output()
            ):
                value += self.pieces.shared
            return value


class OverlappedMoeTransport(ComputeTransport):
    """The TBO issue/wait protocol, owned by one microbatch.

    EP ops already return source rows. The TP path gathers normalized 2-D
    input and sums/scatters expert output asynchronously; residual coefficients
    stay on the source rows throughout. Neither finish path sums again.
    """

    is_overlapped = True

    def __init__(self, moe, hidden_states, batch, subbatch_index, rows):
        self.rows = rows
        self.moe, self.input, self.batch = moe, hidden_states, batch
        self.ep = not get_moe_a2a_backend().is_none()
        self.sub = subbatch_index
        self.do_shared_local = not self.ep and shared_expert_runs_locally(moe)
        self.shared_local = None
        self.gather_event = self.combine_event = None
        self.gather_keepalive = self.combine_keepalive = None
        self.global_hidden = self.local_out = None
        self.combined = False

    def release(self):
        events = (self.gather_event, self.combine_event)
        if any(event is not None for event in events):
            current = torch.cuda.current_stream()
            for event in events:
                if event is not None:
                    current.wait_event(event)
        self.gather_event = self.combine_event = None
        self.gather_keepalive = self.combine_keepalive = None
        self.global_hidden = self.local_out = self.shared_local = self.input = None

    def issue_gather(self):
        if self.ep or self.gather_event is not None or self.global_hidden is not None:
            raise RuntimeError("TBO gather cannot be issued twice")
        local = self.input
        self.global_hidden, self.gather_event, self.shared_local = (
            _tbo_gather_moe_input(
                self.moe, local, self.batch, self.sub, self.do_shared_local
            )
        )
        self.gather_keepalive = local
        self.input = None

    def wait_gather(self):
        if self.gather_event is None:
            raise RuntimeError("TBO gather was not issued")
        torch.cuda.current_stream().wait_event(self.gather_event)
        self.gather_event = self.gather_keepalive = None
        return self.global_hidden

    def issue_combine(self, output):
        if self.ep or self.combine_event is not None or self.combined:
            raise RuntimeError("TBO combine cannot be issued twice")
        self.local_out, self.combine_event = _tbo_combine_moe_output(output, self.sub)
        self.combine_keepalive = output
        self.global_hidden = None

    def wait_combine(self):
        if self.combine_event is None:
            raise RuntimeError("TBO combine was not issued")
        value = _tbo_wait_moe_output(
            self.local_out, self.combine_event, self.shared_local
        )
        self.combine_event = self.combine_keepalive = None
        self.local_out = self.shared_local = None
        self.combined = True
        return value

    def prepare_output(self, output):
        if not self.ep and not self.combined:
            raise RuntimeError("TBO finish requires the combine wait")
        return output


def _tbo_run_moe(layer, hidden, batch, do_shared_local):
    global_ids = batch._tbo_global_input_ids
    with get_forward().scoped(mlp_reduce_scatter=True):
        return layer.mlp(
            hidden,
            batch,
            input_ids=global_ids,
            input_ids_global=global_ids,
            skip_shared_experts=do_shared_local,
        )
