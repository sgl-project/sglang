"""EAGLE draft attention on a sliding-window layer, against an independent oracle.

The production draft loop, and separately the draft extend, runs eagerly with
every call of the fixture's attention module recorded. Each call is checked
against a reference that attends to the keys at positions [pos - W, pos]
(W = the layer's sliding_window_size), recomputed from the recorded inputs, so
a read of the wrong keys and a write to the wrong slot both show. Graph replay
is then checked against that eager run. A draft step's row reads its own branch
column's earlier slots, as every backend's draft index builder lays them out.
"""

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
    _expand_gqa,
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
    EagleDraftCudaGraphRunnerAdapter,
    EagleDraftRunnerSettings,
    _assert_draft_outputs_close,
    _build_eagle_draft_fixture,
    _capture_eagle_draft_graph_runner,
    _check_dense_draft_cache_layout,
    _make_dense_draft_inputs,
    _make_dense_eagle_draft_forward_batch,
    _make_dense_model_forward,
    _prepare_dense_draft_replay_state,
    _reset_cuda_graph_test_buffers,
    _run_eagle_draft_eager,
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


def _window_attention(module, query, keys, values, *, query_pos, window, num_heads):
    """One query row over keys at positions 0.., keeping [query_pos - window,
    query_pos]; [num_heads * head_dim] in float."""
    start = 0 if window is None else max(0, query_pos - window)
    k = _expand_gqa(keys[start : query_pos + 1].movedim(0, 1), num_heads).float()
    v = _expand_gqa(values[start : query_pos + 1].movedim(0, 1), num_heads).float()
    scores = torch.einsum("hd,hkd->hk", query.float(), k) * module.scaling
    return torch.einsum("hk,hkd->hd", torch.softmax(scores, dim=-1), v).reshape(-1)


def _check_draft_steps(testcase, fixture, case, calls, settings) -> None:
    module = fixture.reference_module
    heads, kv_heads, dim = case.num_heads, case.num_kv_heads, module.head_dim
    testcase.assertEqual(len(calls), settings.speculative_num_steps - 1)
    steps = [module.project_qkv(hidden) for hidden, _ in calls]
    for req, prefix in enumerate(fixture.prefix_hidden):
        _, prefix_k, prefix_v = module.project_qkv(prefix)
        for branch in range(settings.topk):
            row = req * settings.topk + branch
            for step, (_, out) in enumerate(calls):
                chain = range(step + 1)
                keys = torch.cat(
                    [prefix_k] + [steps[j][1][row : row + 1] for j in chain]
                )
                values = torch.cat(
                    [prefix_v] + [steps[j][2][row : row + 1] for j in chain]
                )
                attn = _window_attention(
                    module,
                    steps[step][0][row].view(heads, dim),
                    keys.view(-1, kv_heads, dim),
                    values.view(-1, kv_heads, dim),
                    query_pos=case.prefix_lens[req] + step,
                    window=case.sliding_window_size,
                    num_heads=heads,
                )
                expected = module.reconstruct_output(attn.to(out.dtype)[None])[0]
                torch.testing.assert_close(
                    out[row],
                    expected,
                    atol=settings.atol,
                    rtol=settings.rtol,
                    msg=lambda m: f"req {req} branch {branch} step {step}: {m}",
                )


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


def run_dense_eagle_draft_window_case(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int = 1,
    speculative_num_steps: int = 3,
    speculative_num_draft_tokens: int | None = None,
    swa_pool: bool = False,
    cuda_graph_capture_batch_size: int = 4,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    vocab_size: int = 64,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    """Every eager draft step against the window oracle, then the captured
    draft loop's replay against the eager outputs. ``swa_pool`` stores the
    layer in a static hybrid-SWA pool with shuffled full->swa slots."""
    if case.sliding_window_size is None or not case.forward_mode.is_decode():
        raise ValueError("draft window coverage expects DECODE cases with a window")
    settings = _settings(
        topk=topk,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=(
            speculative_num_steps + 1
            if speculative_num_draft_tokens is None
            else speculative_num_draft_tokens
        ),
        cuda_graph_capture_batch_size=cuda_graph_capture_batch_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    if case.batch_size > settings.capture_batch_size:
        raise ValueError("Capture batch size must cover the replay batch size.")
    adapter = EagleDraftCudaGraphRunnerAdapter(
        build_fixture=build_dense_attention_fixture,
        make_model_forward=_make_dense_model_forward,
        make_draft_inputs=_make_dense_draft_inputs,
        prepare_replay_state=_prepare_dense_draft_replay_state,
        make_forward_batch=_make_dense_eagle_draft_forward_batch,
        check_case=_check_dense_draft_cache_layout,
    )
    # The replay state writes the prefix's K/V at contiguous slots; a shuffled
    # prefix would read over the slots those writes land in.
    build_kwargs = dict(
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        swa_pool=swa_pool,
        loc_layout="contiguous",
    )
    try:
        adapter.check_case(case, settings)
        draft_inputs = adapter.make_draft_inputs(case, settings)

        fixture, worker, _ = _build_eagle_draft_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        recorder = _AttentionRecorder(fixture.actual_module)
        adapter.prepare_replay_state(fixture, case, draft_inputs, settings)
        batch = adapter.make_forward_batch(case, draft_inputs, settings)
        fixture.runner.kv_index_translator.bind_own_plan(batch)
        expected = _run_eagle_draft_eager(worker, batch, settings=settings)
        _check_draft_steps(testcase, fixture, case, recorder.calls, settings)

        graph_fixture, graph_worker, graph_backend = _build_eagle_draft_fixture(
            testcase,
            case,
            adapter=adapter,
            build_kwargs=build_kwargs,
            settings=settings,
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)
        graph_batch = adapter.make_forward_batch(case, draft_inputs, settings)
        graph_fixture.runner.kv_index_translator.bind_own_plan(graph_batch)
        graph_runner = _capture_eagle_draft_graph_runner(
            graph_worker, graph_backend, settings
        )
        adapter.prepare_replay_state(graph_fixture, case, draft_inputs, settings)
        testcase.assertTrue(graph_runner.can_run_graph(graph_batch))
        _assert_draft_outputs_close(
            graph_runner.execute(graph_batch), expected, settings
        )
    finally:
        _reset_cuda_graph_test_buffers()


def run_dense_eagle_draft_extend_window_case(
    testcase,
    case: DenseAttentionCase,
    *,
    swa_pool: bool = False,
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
    adapter = EagleDraftExtendCudaGraphRunnerAdapter(
        build_fixture=build_dense_attention_fixture,
        make_model_forward=_make_dense_draft_extend_model_forward,
        make_draft_inputs=_make_dense_draft_extend_inputs,
        prepare_replay_state=_prepare_dense_draft_extend_replay_state,
        make_forward_batch=_make_dense_eagle_draft_extend_v2_forward_batch,
    )
    # The replay state writes the prefix's K/V at contiguous slots; a shuffled
    # prefix would read over the slots those writes land in.
    build_kwargs = dict(
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        swa_pool=swa_pool,
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
