from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import torch

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

from .cuda_graph_decode_runner import (
    _init_cuda_graph_capture_metadata,
    _init_cuda_graph_replay_metadata,
)

# How padded rows of a CG replay batch are filled.
#
# "small_real" (default, legacy): padded rows look like miniature real
# requests — `seq_lens[padded] = capture_prefix_len + num_tokens_per_req`,
# `extend_seq_lens[padded] = num_tokens_per_req`. Easy for the reference
# module to compute an expected attention output for, but doesn't match
# production's padding shape.
#
# "prod_fill": mirrors `eagle_draft_extend_cuda_graph_runner.py:466-474`
# (and similar in `multi_layer_eagle_draft_extend_cuda_graph_runner.py`):
# padded rows are pure scratch — `seq_lens[padded] = seq_len_fill_value`,
# `extend_seq_lens[padded] = num_tokens_per_req`, `req_pool_indices[padded] = 0`,
# `out_cache_loc[padded] = 0`, `positions[padded] = 0`. seq_lens and
# extend_seq_lens are intentionally inconsistent for padded rows (their
# subtraction goes negative), so backends must defend against that — the
# attention output for padded rows is never used in production.
PadStyle = Literal["small_real", "prod_fill"]


@dataclass(frozen=True)
class SpeculativeCudaGraphAdapter:
    build_fixture: Callable[..., Any]
    make_capture_case: Callable[[Any, str, int, int], Any]
    make_replay_case: Callable[[Any, str, tuple[int, ...]], Any]
    make_forward_batch: Callable[..., Any]
    fixture_inputs: Callable[[Any], dict[str, Any]]
    make_capture_inputs: Callable[..., dict[str, Any]]
    make_replay_inputs: Callable[..., dict[str, Any]]
    prepare_batch: Callable[[Any, Any], None]
    prepare_inputs: Callable[..., None]
    run_forward: Callable[[Any, Any, dict[str, Any]], torch.Tensor]
    expected_output: Callable[[Any, Any, dict[str, Any], Any], torch.Tensor]
    max_num_tokens: Callable[[Any, int], int] | None = None
    clone_state: Callable[[Any], Any] = lambda _: None
    restore_state: Callable[[Any, Any], None] = lambda _fixture, _state: None
    allow_padding: bool = True
    run_graph_eager: bool = True
    compare_replay_to_graph_eager: bool = True
    atol: float = 0.0
    rtol: float = 0.0
    # Padding behavior for replay batches when raw_bs < capture_batch_size.
    # See PadStyle docstring above for semantics.
    pad_style: PadStyle = "small_real"
    # Required when pad_style == "prod_fill": draft tokens per request,
    # used to fill the padded slots of extend_seq_lens / spec_info.
    pad_num_tokens_per_req: Optional[int] = None


def _apply_prod_fill_padding(
    batch,
    *,
    real_bs: int,
    capture_bs: int,
    seq_len_fill_value: int,
    num_tokens_per_req: int,
) -> None:
    """Overwrite padded slots of `batch` to match the production CG runner.

    Production sets padded rows to: seq_lens = fill, extend_seq_lens = N,
    req_pool_indices = 0, out_cache_loc = 0, positions = 0. The subtraction
    `seq_lens - extend_seq_lens` then goes negative for padded rows — a
    well-behaved backend must clamp or otherwise handle this.
    """
    if real_bs >= capture_bs:
        return

    pad_lo, pad_hi = real_bs, capture_bs

    # Per-request length tensors.
    batch.seq_lens[pad_lo:pad_hi] = seq_len_fill_value
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu[pad_lo:pad_hi] = seq_len_fill_value
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

    if getattr(batch, "extend_seq_lens", None) is not None:
        batch.extend_seq_lens[pad_lo:pad_hi] = num_tokens_per_req
    if getattr(batch, "extend_seq_lens_cpu", None) is not None:
        ext = list(batch.extend_seq_lens_cpu)
        for i in range(pad_lo, min(pad_hi, len(ext))):
            ext[i] = num_tokens_per_req
        batch.extend_seq_lens_cpu = ext

    # Per-request slot tensors.
    batch.req_pool_indices[pad_lo:pad_hi] = 0

    # Per-token tensors: padded rows occupy slots
    # [real_bs * num_tokens_per_req, capture_bs * num_tokens_per_req).
    tok_lo = pad_lo * num_tokens_per_req
    tok_hi = pad_hi * num_tokens_per_req
    for field in ("out_cache_loc", "positions", "input_ids"):
        t = getattr(batch, field, None)
        if t is not None and t.numel() >= tok_hi:
            t[tok_lo:tok_hi] = 0

    # Mirror spec_info's per-request length tensor if present (set by the
    # V2 production runner at replay: `spec_info.extend_seq_lens_tensor =
    # buffers.extend_seq_lens[:bs]`).
    spec_info = getattr(batch, "spec_info", None)
    if spec_info is not None:
        eslt = getattr(spec_info, "extend_seq_lens_tensor", None)
        if isinstance(eslt, torch.Tensor) and eslt.numel() >= pad_hi:
            eslt[pad_lo:pad_hi] = num_tokens_per_req
        eslc = getattr(spec_info, "extend_seq_lens_cpu", None)
        if isinstance(eslc, list):
            for i in range(pad_lo, min(pad_hi, len(eslc))):
                eslc[i] = num_tokens_per_req


def _check_speculative_cuda_graph_case(
    case,
    capture_batch_size: int,
    *,
    allow_padding: bool,
) -> None:
    if allow_padding:
        if case.batch_size > capture_batch_size:
            raise ValueError("CUDA graph capture must cover replay batch size.")
    elif case.batch_size != capture_batch_size:
        raise ValueError(
            "This CUDA graph coverage uses an unpadded replay batch; choose a case "
            "whose batch size matches the capture batch size."
        )


def run_speculative_cuda_graph_case(
    testcase,
    case,
    *,
    adapter: SpeculativeCudaGraphAdapter,
    build_kwargs: dict,
    capture_batch_size: int,
    max_context_len: int,
    dtype: torch.dtype,
    device: str,
):
    _check_speculative_cuda_graph_case(
        case,
        capture_batch_size,
        allow_padding=adapter.allow_padding,
    )

    graph_fixture = adapter.build_fixture(
        testcase,
        case,
        **build_kwargs,
        disable_cuda_graph=False,
        runner_batch_size=capture_batch_size,
    )
    backend = graph_fixture.backend
    graph_inputs = adapter.fixture_inputs(graph_fixture)
    graph_initial_state = adapter.clone_state(graph_fixture)
    graph_eager_actual = None

    if adapter.run_graph_eager:
        if adapter.max_num_tokens is not None:
            backend.init_cuda_graph_state(
                max_bs=capture_batch_size,
                max_num_tokens=adapter.max_num_tokens(case, capture_batch_size),
            )
        graph_batch = graph_fixture.forward_batch
        adapter.prepare_batch(case, graph_batch)
        # Run prepare_inputs in the eager leg too so backends whose reference
        # depends on cache state / per-fixture stashes (e.g. DSV4 reads BF16
        # K from `fixture._swa_bf16_k_per_req`, populated by
        # `prepare_dsv4_runner_inputs`) work the same way as the
        # capture/replay legs. Backends whose reference is self-contained
        # (dense / MLA — they re-project from `inputs`) are unaffected;
        # `prepare_inputs` just re-writes the SWA cache.
        adapter.prepare_inputs(
            graph_fixture,
            case,
            graph_batch,
            graph_inputs,
            max_context_len=max_context_len,
        )
        graph_expected = adapter.expected_output(
            graph_fixture,
            case,
            graph_inputs,
            graph_initial_state,
        )

        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            backend.init_forward_metadata(graph_batch)
            graph_eager_actual = adapter.run_forward(
                graph_fixture,
                graph_batch,
                graph_inputs,
            )

        torch.testing.assert_close(
            graph_eager_actual,
            graph_expected,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )

    capture_prefix_len = backend.get_cuda_graph_seq_len_fill_value()
    capture_case = adapter.make_capture_case(
        case,
        f"{case.name}_cuda_graph_capture",
        capture_prefix_len,
        capture_batch_size,
    )
    capture_inputs = adapter.make_capture_inputs(
        capture_case,
        graph_fixture,
        dtype=dtype,
        device=device,
    )
    capture_batch = adapter.make_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    adapter.prepare_batch(capture_case, capture_batch)
    adapter.prepare_inputs(
        graph_fixture,
        capture_case,
        capture_batch,
        capture_inputs,
        max_context_len=max_context_len,
    )
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(backend, capture_batch_size, capture_batch)
        # Capture forward is a JIT warmup that mirrors production: the
        # captured CUDA graph records kernel launches against buffers
        # that *will* be populated by replay-init at replay. The
        # capture-time output itself is discarded in production — and
        # we discard it here too. Only the replay output is
        # contractually required to match the reference.
        adapter.run_forward(graph_fixture, capture_batch, capture_inputs)
        backend.on_after_cuda_graph_warmup()

    adapter.restore_state(graph_fixture, graph_initial_state)
    replay_pad_prefix_lens = (
        (capture_prefix_len,) * (capture_batch_size - case.batch_size)
        if adapter.allow_padding
        else ()
    )
    replay_case = adapter.make_replay_case(
        case,
        f"{case.name}_cuda_graph_replay",
        replay_pad_prefix_lens,
    )
    replay_inputs = adapter.make_replay_inputs(
        replay_case,
        graph_fixture,
        replay_pad_prefix_lens,
        graph_inputs,
        dtype=dtype,
        device=device,
    )
    replay_batch = adapter.make_forward_batch(
        replay_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    adapter.prepare_batch(replay_case, replay_batch)
    adapter.prepare_inputs(
        graph_fixture,
        replay_case,
        replay_batch,
        replay_inputs,
        max_context_len=max_context_len,
    )
    replay_expected = adapter.expected_output(
        graph_fixture,
        replay_case,
        replay_inputs,
        graph_initial_state,
    )

    # Optionally overwrite padded rows to match the production CG runner's
    # fill pattern. Done after prepare_batch + expected_output so that the
    # reference is computed against the "small_real" layout (where padded
    # rows have an interpretable attention output); the assertion below
    # then compares only the real-row slice.
    real_bs = case.batch_size
    if (
        adapter.pad_style == "prod_fill"
        and adapter.allow_padding
        and real_bs < capture_batch_size
    ):
        if adapter.pad_num_tokens_per_req is None:
            raise ValueError(
                "SpeculativeCudaGraphAdapter.pad_num_tokens_per_req must be set "
                "when pad_style='prod_fill'."
            )
        _apply_prod_fill_padding(
            replay_batch,
            real_bs=real_bs,
            capture_bs=capture_batch_size,
            seq_len_fill_value=capture_prefix_len,
            num_tokens_per_req=adapter.pad_num_tokens_per_req,
        )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_replay_metadata(backend, capture_batch_size, replay_batch)
        replay_actual = adapter.run_forward(
            graph_fixture,
            replay_batch,
            replay_inputs,
        )

    if adapter.pad_style == "prod_fill":
        # Padded rows have undefined output (their state is scratch in
        # production; the runner discards their result). Assert only on the
        # real-row slice.
        torch.testing.assert_close(
            replay_actual[: case.num_input_tokens],
            replay_expected[: case.num_input_tokens],
            atol=adapter.atol,
            rtol=adapter.rtol,
        )
    else:
        torch.testing.assert_close(
            replay_actual,
            replay_expected,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )
    if adapter.compare_replay_to_graph_eager:
        torch.testing.assert_close(
            replay_actual[: case.num_input_tokens],
            graph_eager_actual,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )
