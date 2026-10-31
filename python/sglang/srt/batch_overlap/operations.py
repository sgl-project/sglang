from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Union,
)

import torch

from sglang.srt.layers.dp_attention import set_dp_buffer_len
from sglang.srt.model_executor.forward_context import (
    forward_context,
    get_forward_context,
)
from sglang.srt.utils.nvtx_utils import operations_nvtx_range

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_context import ForwardContext


def execute_operations(inputs, operations):
    stages = _convert_operations_to_stages(operations)
    executor = _StageExecutor("primary", stages, inputs=inputs)
    for _ in range(executor.num_stages):
        executor.next()
    assert executor.done
    return executor.output


def execute_overlapped_operations(
    inputs_arr: Sequence,
    operations_arr: Sequence,
    delta_stages: Sequence[int],
) -> Sequence:
    # Make it explicit for clarity; if we need multi-batch overlap, this can be generalized
    inputs_a, inputs_b = inputs_arr
    operations_a, operations_b = operations_arr
    delta_stage_a, delta_stage_b = delta_stages
    assert delta_stage_a == 0
    delta_stage = delta_stage_b

    # Each TBO child sub-batch dispatches against its own per-child backend
    # (children[i] has metadata init'd for sub-batch i; the parent's primary
    # has metadata for the full pre-split batch).
    child_ctx_a, child_ctx_b = _resolve_tbo_child_contexts()

    stages_a = _convert_operations_to_stages(operations_a)
    stages_b = _convert_operations_to_stages(operations_b)
    executor_a = _StageExecutor(
        "a", stages_a, inputs=inputs_a, child_ctx=child_ctx_a, micro_batch=True
    )
    executor_b = _StageExecutor(
        "b", stages_b, inputs=inputs_b, child_ctx=child_ctx_b, micro_batch=True
    )
    side_stream = _micro_batch_b_side_stream()
    if side_stream is not None:
        main_stream = torch.cuda.current_stream()
        side_stream.wait_stream(main_stream)
        # B's inputs come from the main stream's pool: keep A from reusing their
        # blocks while B may still read them.
        _record_stream(inputs_b, side_stream)
        executor_b = _SideStreamExecutor(executor_b, side_stream)

    for _ in range(delta_stage):
        executor_a.next()

    for _ in range(executor_a.num_stages - delta_stage):
        executor_a.next()
        executor_b.next()

    for _ in range(delta_stage):
        executor_b.next()

    assert executor_a.done and executor_b.done
    if side_stream is not None:
        main_stream.wait_stream(side_stream)
        # B's outputs come from the side stream's pool; the main stream uses them.
        _record_stream(executor_b.output, main_stream)
    return [executor_a.output, executor_b.output]


_side_streams: Dict[int, torch.cuda.Stream] = {}
_side_issuers: Dict[int, ThreadPoolExecutor] = {}


def _micro_batch_b_side_stream() -> Optional[torch.cuda.Stream]:
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        mori_tbo_side_stream_enabled,
    )

    if not mori_tbo_side_stream_enabled() or torch.cuda.is_current_stream_capturing():
        return None
    device = torch.cuda.current_device()
    if device not in _side_streams:
        _side_streams[device] = torch.cuda.Stream(device=device)
    return _side_streams[device]


def _record_stream(values, stream: torch.cuda.Stream) -> None:
    """record_stream the tensors in `values` (a dict, list or tuple, one level of
    nesting), so the caching allocator waits for `stream` before reusing them."""
    items = values.values() if isinstance(values, dict) else values
    for value in items:
        tensors = value if isinstance(value, (list, tuple)) else (value,)
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                tensor.record_stream(stream)


def _side_issuer(device: int) -> ThreadPoolExecutor:
    """The device's host thread that issues micro-batch B's stages.

    torch keeps one BLAS handle per host thread, and hipBLASLt Stream-K GEMMs
    running concurrently on two streams through one handle can leave a workgroup
    spinning in its fixup forever; issuing B from its own thread gives B its own
    handle.
    """
    issuer = _side_issuers.get(device)
    if issuer is None:
        issuer = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tbo-side-issuer",
            initializer=torch.cuda.set_device,
            initargs=(device,),
        )
        issuer.submit(_create_blas_handle).result()
        _side_issuers[device] = issuer
    return issuer


def _create_blas_handle() -> None:
    # Create this thread's BLAS handle on an idle device, so its setup cannot race
    # GEMMs in flight on other streams.
    torch.cuda.synchronize()
    x = torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16)
    torch.nn.functional.linear(x, x)
    torch.cuda.synchronize()


class _SideStreamExecutor:
    """Runs a _StageExecutor's stages on `stream`, issued from the device's side
    issuer thread; the caller waits for each stage to be issued, so the host still
    issues A's and B's stages in the same order."""

    def __init__(self, executor: _StageExecutor, stream: torch.cuda.Stream):
        self._executor = executor
        self._stream = stream
        self._issuer = _side_issuer(stream.device.index)

    def next(self):
        inference_mode = torch.is_inference_mode_enabled()
        grad_enabled = torch.is_grad_enabled()

        def issue():
            with (
                torch.inference_mode(inference_mode),
                torch.set_grad_enabled(grad_enabled),
                torch.cuda.stream(self._stream),
            ):
                self._executor.next()

        # B reads the caller's context variables; its writes stay in the copy.
        self._issuer.submit(contextvars.copy_context().run, issue).result()

    def __getattr__(self, item):
        return getattr(self._executor, item)


def _resolve_tbo_child_contexts():
    """Return (child_ctx_a, child_ctx_b) derived from the active TboAttnBackend,
    or (None, None) if the active backend is not a TBO dispatcher (e.g. a
    backend that handles TBO splitting internally like DeepSeek MHA's
    resolve_attn_backend path)."""
    # Lazy import to avoid circular dependency at module load time.
    from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

    ctx = get_forward_context()
    backend = ctx.attn_backend
    if not isinstance(backend, TboAttnBackend):
        return None, None
    child_a, child_b = backend.children
    return (
        replace(ctx, attn_backend=child_a),
        replace(ctx, attn_backend=child_b),
    )


class YieldOperation:
    pass


@dataclass
class ExecutionOperation:
    debug_name: str
    fn: Callable


Operation = Union[YieldOperation, ExecutionOperation, Callable]
Stage = List[ExecutionOperation]


class _StageExecutor:
    def __init__(
        self,
        debug_name: str,
        stages: List[Stage],
        inputs: dict,
        child_ctx: Optional[ForwardContext] = None,
        micro_batch: bool = False,
    ):
        self._debug_name = debug_name
        self._stages = stages
        self._index = 0
        self._stage_state = _StateDict()
        self._stage_output = inputs
        # When set, every next() runs inside this ForwardContext so that
        # get_attn_backend() inside RadixAttention.forward resolves to the
        # per-child backend (with sub-batch metadata) instead of the TBO
        # parent's primary.
        self._child_ctx = child_ctx

        # handling DP attention
        forward_batch: ForwardBatch = inputs["forward_batch"]
        self._child_ctx = replace(
            child_ctx if child_ctx is not None else get_forward_context(),
            moe_num_token_non_padded=forward_batch.moe_num_token_non_padded,
            is_micro_batch=micro_batch,
            dp_global_num_tokens_bound=(
                forward_batch.tbo_dp_global_num_tokens_bound if micro_batch else None
            ),
        )
        self._global_dp_buffer_len = forward_batch.global_dp_buffer_len
        self._local_dp_buffer_len = forward_batch.tbo_padded_len
        self._global_num_tokens = forward_batch.global_num_tokens_cpu
        self._is_dp_max_padding = forward_batch.dp_padding_mode.is_max_len()

    def next(self):
        assert not self.done

        stage = self._stages[self._index]

        # TODO: We currently always call set_dp_buffer_len here because sub-batches
        # may have different padded lengths. It can likely be removed after TBO slice &
        # pad logic is refactored.
        set_dp_buffer_len(
            self._global_dp_buffer_len,
            self._local_dp_buffer_len,
            self._is_dp_max_padding,
            self._global_num_tokens,
        )

        ctx_mgr = (
            forward_context(self._child_ctx)
            if self._child_ctx is not None
            else nullcontext()
        )
        stage_range = operations_nvtx_range(
            debug_name=f"{self._debug_name}{self._index}",
            color="orange",
        )
        with ctx_mgr, stage_range:
            for op in stage:
                with operations_nvtx_range(
                    debug_name=op.debug_name,
                    color="yellow",
                ):
                    self._stage_output = op.fn(
                        state=self._stage_state,
                        **(
                            self._stage_output if self._stage_output is not None else {}
                        ),
                    )

        self._index += 1

    @property
    def output(self):
        assert self.done
        return self._stage_output

    @property
    def done(self):
        return self._index >= self.num_stages

    @property
    def num_stages(self):
        return len(self._stages)


class _StateDict:
    def __init__(self):
        self._data = {}

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
            return
        assert key not in self._data, (
            f"`{key}` already exist, are you sure you want to override it?"
        )
        self._data[key] = value

    def __getattr__(self, item):
        return self._data[item]

    def __delattr__(self, item):
        del self._data[item]

    def pop(self, item):
        return self._data.pop(item)

    def update(self, values: Dict[str, Any]):
        for k, v in values.items():
            setattr(self, k, v)

    def get(self, item):
        return self._data.get(item)

    def clear(self, expect_keys: Sequence[str]):
        if set(self._data.keys()) != set(expect_keys):
            raise Exception(
                f"Unexpected keys when clearing. This may indicate you do not release memory early enough but leave it until here. {list(self._data.keys())=} {expect_keys=}"
            )

        self._data.clear()


def _convert_operations_to_stages(operations: List[Operation]) -> List[Stage]:
    operations = _decorate_operations(operations)
    operation_chunks = list(
        _chunk_by_separator(operations, lambda op: isinstance(op, YieldOperation))
    )
    assert all(len(chunk) > 0 for chunk in operation_chunks)
    return operation_chunks


def _chunk_by_separator(
    items: List[Any], is_separator: Callable[[Any], bool]
) -> Generator[List[Any], None, None]:
    pending_items = []
    for item in items:
        if is_separator(item):
            yield pending_items
            pending_items = []
        else:
            pending_items.append(item)
    if len(pending_items) > 0:
        yield pending_items


def _decorate_operations(operations: List[Operation], debug_name_prefix: str = ""):
    return [_decorate_operation(op, debug_name_prefix) for op in operations]


def _decorate_operation(operation: Operation, debug_name_prefix: str):
    if isinstance(operation, YieldOperation):
        return operation
    return ExecutionOperation(
        debug_name=debug_name_prefix
        + getattr(operation, "__name__", "unknown").replace("op_", ""),
        fn=operation,
    )
