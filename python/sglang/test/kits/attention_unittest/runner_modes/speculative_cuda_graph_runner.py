from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

from .cuda_graph_decode_runner import (
    _init_cuda_graph_capture_metadata,
    _init_cuda_graph_replay_metadata,
)


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

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_replay_metadata(backend, capture_batch_size, replay_batch)
        replay_actual = adapter.run_forward(
            graph_fixture,
            replay_batch,
            replay_inputs,
        )

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
