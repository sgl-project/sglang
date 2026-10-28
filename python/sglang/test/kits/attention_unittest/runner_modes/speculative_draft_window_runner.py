"""Check FlashInfer draft-extend attention against a window oracle, then replay it."""

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from ..attention_methods.dense_attention import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAX_CONTEXT_LEN,
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    _dense_attention_reference,
    build_dense_attention_fixture,
)
from .speculative_draft_extend_runner import (
    EagleDraftExtendCudaGraphRunnerAdapter,
    _build_eagle_draft_extend_fixture,
    _capture_eagle_draft_extend_graph_runner,
    _draft_extend_select_index,
    _make_dense_draft_extend_inputs,
    _make_dense_draft_extend_model_forward,
    _make_dense_eagle_draft_extend_v2_forward_batch,
    _prepare_dense_draft_extend_replay_state,
    _run_eagle_draft_extend_eager,
)
from .speculative_draft_runner import (
    EagleDraftRunnerSettings,
    _reset_cuda_graph_test_buffers,
)


class _AttentionRecorder:
    """Each call of an attention module: its input hidden states and output."""

    def __init__(self, module):
        self.calls = []
        forward = module.forward

        def recording_forward(hidden_states, forward_batch):
            out = forward(hidden_states, forward_batch)
            self.calls.append((hidden_states.detach().clone(), out.detach().clone()))
            return out

        module.forward = recording_forward


def _settings(
    *,
    topk,
    speculative_num_steps,
    speculative_num_draft_tokens,
    cuda_graph_capture_batch_size,
    hidden_size,
    vocab_size,
    max_context_len,
    dtype,
    device,
) -> EagleDraftRunnerSettings:
    return EagleDraftRunnerSettings(
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )


def run_dense_eagle_draft_extend_window_case(
    testcase,
    case: DenseAttentionCase,
    *,
    num_accept_tokens: tuple[int, ...],
    cuda_graph_capture_batch_size: int = 4,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    vocab_size: int = 64,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    """The eager draft extend against the window oracle (every row of the
    constant-width block), then the captured draft extend's replay against
    the eager outputs."""
    if case.sliding_window_size is None or not case.forward_mode.is_draft_extend_v2():
        raise ValueError(
            "draft-extend window coverage expects DRAFT_EXTEND_V2 cases with a window"
        )
    if len(set(case.input_lens)) != 1:
        raise ValueError("DRAFT_EXTEND_V2 runner coverage uses fixed token counts.")
    num_tokens_per_req = case.input_lens[0]
    if len(num_accept_tokens) != case.batch_size or any(
        n < 1 or n > num_tokens_per_req for n in num_accept_tokens
    ):
        raise ValueError("Acceptance counts must fit each request's fixed-width block.")
    settings = _settings(
        topk=1,
        speculative_num_steps=num_tokens_per_req - 1,
        speculative_num_draft_tokens=num_tokens_per_req,
        cuda_graph_capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    if case.batch_size > settings.capture_batch_size:
        raise ValueError("Capture batch size must cover the replay batch size.")

    def make_forward_batch(fixture, case, draft_inputs, settings):
        batch = _make_dense_eagle_draft_extend_v2_forward_batch(
            fixture, case, draft_inputs, settings
        )
        # The forward still queries the whole block when verification accepted
        # fewer tokens. Full acceptance would hide the replay's short-window bug.
        accepted = torch.tensor(num_accept_tokens, dtype=torch.int32, device=device)
        batch.spec_info.num_accept_tokens = accepted
        batch.spec_info.num_accept_tokens_cpu = list(num_accept_tokens)
        batch.spec_info.num_correct_drafts = accepted - 1
        return batch

    adapter = EagleDraftExtendCudaGraphRunnerAdapter(
        build_fixture=build_dense_attention_fixture,
        make_model_forward=_make_dense_draft_extend_model_forward,
        make_draft_inputs=_make_dense_draft_extend_inputs,
        prepare_replay_state=_prepare_dense_draft_extend_replay_state,
        make_forward_batch=make_forward_batch,
    )
    # The replay state writes the prefix's K/V at contiguous slots; a shuffled
    # prefix would read over the slots those writes land in.
    build_kwargs = dict(
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        loc_layout="contiguous",
    )
    try:
        draft_inputs = adapter.make_draft_inputs(case, settings)

        fixture, worker, _ = _build_eagle_draft_extend_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        recorder = _AttentionRecorder(fixture.actual_module)
        adapter.prepare_replay_state(fixture, case, draft_inputs, settings)
        batch = adapter.make_forward_batch(fixture, case, draft_inputs, settings)
        eager = _run_eagle_draft_extend_eager(worker, batch, settings)
        ((hidden, out),) = recorder.calls
        torch.testing.assert_close(
            out,
            _dense_attention_reference(
                fixture.reference_module, case, fixture.prefix_hidden, hidden
            ),
            atol=settings.atol,
            rtol=settings.rtol,
        )
        select_index = _draft_extend_select_index(batch, settings)
        expected = LogitsProcessorOutput(
            next_token_logits=eager.next_token_logits[select_index],
            hidden_states=eager.hidden_states[select_index],
        )

        graph_fixture, graph_worker, graph_backend = _build_eagle_draft_extend_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)
        graph_batch = adapter.make_forward_batch(
            graph_fixture, case, draft_inputs, settings
        )
        graph_runner = _capture_eagle_draft_extend_graph_runner(
            graph_worker, graph_backend, settings
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)
        testcase.assertTrue(graph_runner.can_run_graph(graph_batch))
        adapter.assert_outputs_close(
            graph_runner.execute(graph_batch, select_index), expected, settings
        )
    finally:
        _reset_cuda_graph_test_buffers()
