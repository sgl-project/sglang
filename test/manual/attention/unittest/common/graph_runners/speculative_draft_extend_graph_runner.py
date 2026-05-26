from typing import Literal

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftExtendInput

from ..attention_methods.dense_attention import DEFAULT_DEVICE as DENSE_DEFAULT_DEVICE
from ..attention_methods.dense_attention import DEFAULT_DTYPE as DENSE_DEFAULT_DTYPE
from ..attention_methods.dense_attention import (
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
)
from ..attention_methods.dense_attention import (
    DEFAULT_MAX_CONTEXT_LEN as DENSE_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
)
from ..attention_methods.dense_attention import (
    _make_forward_batch as _make_dense_forward_batch,
)
from ..attention_methods.dense_attention import (
    build_dense_attention_fixture,
    dense_fixture_inputs,
    expected_dense_output_from_inputs,
    make_dense_padded_replay_inputs,
    make_dense_random_inputs,
    prepare_dense_runner_inputs,
    run_dense_forward,
)
from ..attention_methods.mla_attention import DEFAULT_DEVICE as MLA_DEFAULT_DEVICE
from ..attention_methods.mla_attention import DEFAULT_DTYPE as MLA_DEFAULT_DTYPE
from ..attention_methods.mla_attention import (
    DEFAULT_HIDDEN_SIZE as MLA_DEFAULT_HIDDEN_SIZE,
)
from ..attention_methods.mla_attention import (
    DEFAULT_KV_LORA_RANK,
)
from ..attention_methods.mla_attention import (
    DEFAULT_MAX_CONTEXT_LEN as MLA_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.mla_attention import (
    DEFAULT_QK_ROPE_HEAD_DIM,
    MLA_ATOL,
    MLA_RTOL,
    MLAAttentionCase,
)
from ..attention_methods.mla_attention import (
    _make_forward_batch as _make_mla_forward_batch,
)
from ..attention_methods.mla_attention import (
    build_mla_attention_fixture,
    expected_mla_output_from_inputs,
    make_mla_case_with_lens,
    make_mla_case_with_prefix_lens,
    make_mla_padded_replay_inputs,
    make_mla_random_inputs,
    mla_fixture_inputs,
    prepare_mla_runner_inputs,
    run_mla_forward,
)
from .cuda_graph_decode_runner import (
    _init_cuda_graph_capture_metadata,
    _init_cuda_graph_replay_metadata,
)

DraftExtendKind = Literal["eagle", "frozen_kv_mtp"]


def _make_dense_spec_case_with_lens(
    case: DenseAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
    input_lens: tuple[int, ...],
) -> DenseAttentionCase:
    return DenseAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=input_lens,
        sliding_window_size=case.sliding_window_size,
    )


def _make_eagle_draft_extend_input(case, batch, *, device: str):
    num_accept_tokens = torch.tensor(
        case.input_lens,
        dtype=torch.int32,
        device=device,
    )
    return EagleDraftExtendInput(
        hidden_states=None,
        num_correct_drafts=num_accept_tokens - 1,
        num_accept_tokens=num_accept_tokens,
        num_accept_tokens_cpu=list(case.input_lens),
        input_ids=batch.input_ids,
        seq_lens=batch.seq_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        req_pool_indices=batch.req_pool_indices,
        positions=batch.positions,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=max(case.input_lens),
        num_tokens_for_logprob_per_req=1,
    )


def _make_frozen_kv_mtp_draft_extend_input(case, batch, *, device: str):
    draft_extend_input = _make_eagle_draft_extend_input(case, batch, device=device)
    return FrozenKVMTPDraftExtendInput(
        hidden_states=draft_extend_input.hidden_states,
        num_correct_drafts=draft_extend_input.num_correct_drafts,
        num_accept_tokens=draft_extend_input.num_accept_tokens,
        num_accept_tokens_cpu=draft_extend_input.num_accept_tokens_cpu,
        input_ids=draft_extend_input.input_ids,
        seq_lens=draft_extend_input.seq_lens,
        seq_lens_cpu=draft_extend_input.seq_lens_cpu,
        req_pool_indices=draft_extend_input.req_pool_indices,
        positions=draft_extend_input.positions,
        bonus_tokens=draft_extend_input.bonus_tokens,
        capture_hidden_mode=draft_extend_input.capture_hidden_mode,
        num_tokens_per_req=draft_extend_input.num_tokens_per_req,
        num_tokens_for_logprob_per_req=(
            draft_extend_input.num_tokens_for_logprob_per_req
        ),
    )


def _make_draft_extend_input(
    case,
    batch,
    *,
    device: str,
    spec_kind: DraftExtendKind,
):
    if spec_kind == "eagle":
        return _make_eagle_draft_extend_input(case, batch, device=device)
    if spec_kind == "frozen_kv_mtp":
        return _make_frozen_kv_mtp_draft_extend_input(case, batch, device=device)
    raise ValueError(f"Unsupported draft-extend spec kind: {spec_kind}")


def _make_eagle_draft_extend_v2_input(case, batch, *, device: str):
    draft_extend_input = _make_eagle_draft_extend_input(case, batch, device=device)
    draft_extend_input.extend_seq_lens_tensor = torch.tensor(
        case.input_lens,
        dtype=torch.int32,
        device=device,
    )
    draft_extend_input.extend_seq_lens_cpu = list(case.input_lens)
    return draft_extend_input


def _set_draft_extend_v2_prefix_lens(batch, case, *, device: str):
    prefix_lens = torch.tensor(case.prefix_lens, dtype=torch.int32, device=device)
    batch.seq_lens = prefix_lens
    batch.seq_lens_cpu = torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu")
    batch.seq_lens_sum = sum(case.prefix_lens)


def run_dense_eagle_draft_extend_case(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
    spec_kind: DraftExtendKind = "eagle",
):
    if not case.forward_mode.is_draft_extend():
        raise ValueError("EAGLE draft-extend coverage expects DRAFT_EXTEND cases.")
    fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    fixture.forward_batch.spec_info = _make_draft_extend_input(
        case,
        fixture.forward_batch,
        device=device,
        spec_kind=spec_kind,
    )
    inputs = dense_fixture_inputs(fixture)
    expected = expected_dense_output_from_inputs(fixture, case, inputs, None)

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        actual = run_dense_forward(fixture, fixture.forward_batch, inputs)

    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


def run_dense_draft_extend_cuda_graph_case(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
    spec_kind: DraftExtendKind = "eagle",
    cuda_graph_capture_batch_size: int = 4,
):
    if not case.forward_mode.is_draft_extend():
        raise ValueError("Draft-extend CUDA graph coverage expects DRAFT_EXTEND.")
    if case.batch_size > cuda_graph_capture_batch_size:
        raise ValueError("Draft-extend capture batch must cover replay batch size.")

    num_tokens_per_req = max(case.input_lens)
    graph_fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        disable_cuda_graph=False,
        runner_batch_size=cuda_graph_capture_batch_size,
    )
    backend = graph_fixture.backend
    backend.init_cuda_graph_state(
        max_bs=cuda_graph_capture_batch_size,
        max_num_tokens=cuda_graph_capture_batch_size * num_tokens_per_req,
    )

    graph_batch = graph_fixture.forward_batch
    graph_batch.spec_info = _make_draft_extend_input(
        case,
        graph_batch,
        device=device,
        spec_kind=spec_kind,
    )
    graph_inputs = dense_fixture_inputs(graph_fixture)
    graph_expected = expected_dense_output_from_inputs(
        graph_fixture,
        case,
        graph_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        backend.init_forward_metadata(graph_batch)
        graph_eager_actual = run_dense_forward(
            graph_fixture,
            graph_batch,
            graph_inputs,
        )

    torch.testing.assert_close(
        graph_eager_actual,
        graph_expected,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )

    capture_prefix_len = backend.get_cuda_graph_seq_len_fill_value()
    capture_case = _make_dense_spec_case_with_lens(
        case,
        f"{case.name}_cuda_graph_capture",
        (capture_prefix_len,) * cuda_graph_capture_batch_size,
        (num_tokens_per_req,) * cuda_graph_capture_batch_size,
    )
    capture_inputs = make_dense_random_inputs(
        capture_case,
        graph_fixture,
        dtype=dtype,
        device=device,
    )
    capture_batch = _make_dense_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    capture_batch.spec_info = _make_draft_extend_input(
        capture_case,
        capture_batch,
        device=device,
        spec_kind=spec_kind,
    )
    prepare_dense_runner_inputs(
        graph_fixture,
        capture_case,
        capture_batch,
        capture_inputs,
        max_context_len=max_context_len,
    )
    capture_expected = expected_dense_output_from_inputs(
        graph_fixture,
        capture_case,
        capture_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(
            backend,
            cuda_graph_capture_batch_size,
            capture_batch,
        )
        capture_actual = run_dense_forward(
            graph_fixture,
            capture_batch,
            capture_inputs,
        )
        backend.on_after_cuda_graph_warmup()

    replay_pad_prefix_lens = (capture_prefix_len,) * (
        cuda_graph_capture_batch_size - case.batch_size
    )
    replay_case = _make_dense_spec_case_with_lens(
        case,
        f"{case.name}_cuda_graph_replay",
        case.prefix_lens + replay_pad_prefix_lens,
        case.input_lens + (num_tokens_per_req,) * len(replay_pad_prefix_lens),
    )
    replay_inputs = make_dense_padded_replay_inputs(
        replay_case,
        graph_fixture,
        replay_pad_prefix_lens,
        graph_inputs,
        dtype=dtype,
        device=device,
    )
    replay_batch = _make_dense_forward_batch(
        replay_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    replay_batch.spec_info = _make_draft_extend_input(
        replay_case,
        replay_batch,
        device=device,
        spec_kind=spec_kind,
    )
    prepare_dense_runner_inputs(
        graph_fixture,
        replay_case,
        replay_batch,
        replay_inputs,
        max_context_len=max_context_len,
    )
    replay_expected = expected_dense_output_from_inputs(
        graph_fixture,
        replay_case,
        replay_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_replay_metadata(
            backend,
            cuda_graph_capture_batch_size,
            replay_batch,
        )
        replay_actual = run_dense_forward(graph_fixture, replay_batch, replay_inputs)

    torch.testing.assert_close(
        capture_actual,
        capture_expected,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )
    torch.testing.assert_close(
        replay_actual,
        replay_expected,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )
    torch.testing.assert_close(
        replay_actual[: case.num_input_tokens],
        graph_eager_actual,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )


def run_dense_draft_extend_v2_cuda_graph_case(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = 4,
):
    if not case.forward_mode.is_draft_extend_v2():
        raise ValueError("Draft-extend-v2 CUDA graph coverage expects DRAFT_EXTEND_V2.")
    if case.batch_size > cuda_graph_capture_batch_size:
        raise ValueError("Draft-extend-v2 capture batch must cover replay batch size.")
    if len(set(case.input_lens)) != 1:
        raise ValueError(
            "Draft-extend-v2 CUDA graph coverage uses a fixed token count per request."
        )

    num_tokens_per_req = case.input_lens[0]
    graph_fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        disable_cuda_graph=False,
        runner_batch_size=cuda_graph_capture_batch_size,
    )
    backend = graph_fixture.backend

    capture_prefix_len = backend.get_cuda_graph_seq_len_fill_value()
    capture_case = _make_dense_spec_case_with_lens(
        case,
        f"{case.name}_cuda_graph_capture",
        (capture_prefix_len,) * cuda_graph_capture_batch_size,
        (num_tokens_per_req,) * cuda_graph_capture_batch_size,
    )
    capture_inputs = make_dense_random_inputs(
        capture_case,
        graph_fixture,
        dtype=dtype,
        device=device,
    )
    capture_batch = _make_dense_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _set_draft_extend_v2_prefix_lens(capture_batch, capture_case, device=device)
    capture_batch.spec_info = _make_eagle_draft_extend_v2_input(
        capture_case,
        capture_batch,
        device=device,
    )
    prepare_dense_runner_inputs(
        graph_fixture,
        capture_case,
        capture_batch,
        capture_inputs,
        max_context_len=max_context_len,
    )
    capture_expected = expected_dense_output_from_inputs(
        graph_fixture,
        capture_case,
        capture_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(
            backend,
            cuda_graph_capture_batch_size,
            capture_batch,
        )
        capture_actual = run_dense_forward(
            graph_fixture,
            capture_batch,
            capture_inputs,
        )
        backend.on_after_cuda_graph_warmup()

    replay_pad_prefix_lens = (capture_prefix_len,) * (
        cuda_graph_capture_batch_size - case.batch_size
    )
    replay_case = _make_dense_spec_case_with_lens(
        case,
        f"{case.name}_cuda_graph_replay",
        case.prefix_lens + replay_pad_prefix_lens,
        case.input_lens + (num_tokens_per_req,) * len(replay_pad_prefix_lens),
    )
    graph_inputs = dense_fixture_inputs(graph_fixture)
    replay_inputs = make_dense_padded_replay_inputs(
        replay_case,
        graph_fixture,
        replay_pad_prefix_lens,
        graph_inputs,
        dtype=dtype,
        device=device,
    )
    replay_batch = _make_dense_forward_batch(
        replay_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _set_draft_extend_v2_prefix_lens(replay_batch, replay_case, device=device)
    replay_batch.spec_info = _make_eagle_draft_extend_v2_input(
        replay_case,
        replay_batch,
        device=device,
    )
    prepare_dense_runner_inputs(
        graph_fixture,
        replay_case,
        replay_batch,
        replay_inputs,
        max_context_len=max_context_len,
    )
    replay_expected = expected_dense_output_from_inputs(
        graph_fixture,
        replay_case,
        replay_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_replay_metadata(
            backend,
            cuda_graph_capture_batch_size,
            replay_batch,
        )
        replay_actual = run_dense_forward(graph_fixture, replay_batch, replay_inputs)

    torch.testing.assert_close(
        capture_actual,
        capture_expected,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )
    torch.testing.assert_close(
        replay_actual,
        replay_expected,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
    )


def run_mla_draft_extend_v2_cuda_graph_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = 4,
):
    if not case.forward_mode.is_draft_extend_v2():
        raise ValueError("Draft-extend-v2 CUDA graph coverage expects DRAFT_EXTEND_V2.")
    if case.batch_size > cuda_graph_capture_batch_size:
        raise ValueError("Draft-extend-v2 capture batch must cover replay batch size.")
    if len(set(case.input_lens)) != 1:
        raise ValueError(
            "Draft-extend-v2 CUDA graph coverage uses a fixed token count per request."
        )

    num_tokens_per_req = case.input_lens[0]
    graph_fixture = build_mla_attention_fixture(
        testcase,
        case,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        disable_cuda_graph=False,
        runner_batch_size=cuda_graph_capture_batch_size,
    )
    backend = graph_fixture.backend

    capture_prefix_len = backend.get_cuda_graph_seq_len_fill_value()
    capture_case = make_mla_case_with_prefix_lens(
        case,
        f"{case.name}_cuda_graph_capture",
        (capture_prefix_len,) * cuda_graph_capture_batch_size,
    )
    capture_inputs = make_mla_random_inputs(
        capture_case,
        graph_fixture,
        dtype=dtype,
        device=device,
    )
    capture_batch = _make_mla_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _set_draft_extend_v2_prefix_lens(capture_batch, capture_case, device=device)
    capture_batch.spec_info = _make_eagle_draft_extend_v2_input(
        capture_case,
        capture_batch,
        device=device,
    )
    prepare_mla_runner_inputs(
        graph_fixture,
        capture_case,
        capture_batch,
        capture_inputs,
        max_context_len=max_context_len,
    )
    capture_expected = expected_mla_output_from_inputs(
        graph_fixture,
        capture_case,
        capture_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(
            backend,
            cuda_graph_capture_batch_size,
            capture_batch,
        )
        capture_actual = run_mla_forward(
            graph_fixture,
            capture_batch,
            capture_inputs,
        )
        backend.on_after_cuda_graph_warmup()

    replay_pad_prefix_lens = (capture_prefix_len,) * (
        cuda_graph_capture_batch_size - case.batch_size
    )
    replay_case = make_mla_case_with_prefix_lens(
        case,
        f"{case.name}_cuda_graph_replay",
        case.prefix_lens + replay_pad_prefix_lens,
    )
    graph_inputs = mla_fixture_inputs(graph_fixture)
    replay_inputs = make_mla_padded_replay_inputs(
        replay_case,
        graph_fixture,
        replay_pad_prefix_lens,
        graph_inputs,
        dtype=dtype,
        device=device,
    )
    replay_batch = _make_mla_forward_batch(
        replay_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _set_draft_extend_v2_prefix_lens(replay_batch, replay_case, device=device)
    replay_batch.spec_info = _make_eagle_draft_extend_v2_input(
        replay_case,
        replay_batch,
        device=device,
    )
    prepare_mla_runner_inputs(
        graph_fixture,
        replay_case,
        replay_batch,
        replay_inputs,
        max_context_len=max_context_len,
    )
    replay_expected = expected_mla_output_from_inputs(
        graph_fixture,
        replay_case,
        replay_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_replay_metadata(
            backend,
            cuda_graph_capture_batch_size,
            replay_batch,
        )
        replay_actual = run_mla_forward(graph_fixture, replay_batch, replay_inputs)

    torch.testing.assert_close(
        capture_actual,
        capture_expected,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )
    torch.testing.assert_close(
        replay_actual,
        replay_expected,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )


def run_mla_draft_extend_cuda_graph_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = 4,
):
    if not case.forward_mode.is_draft_extend():
        raise ValueError("Draft-extend CUDA graph coverage expects DRAFT_EXTEND.")
    if case.batch_size > cuda_graph_capture_batch_size:
        raise ValueError("Draft-extend capture batch must cover replay batch size.")

    num_tokens_per_req = max(case.input_lens)
    graph_fixture = build_mla_attention_fixture(
        testcase,
        case,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        disable_cuda_graph=False,
        runner_batch_size=cuda_graph_capture_batch_size,
    )
    backend = graph_fixture.backend
    backend.init_cuda_graph_state(
        max_bs=cuda_graph_capture_batch_size,
        max_num_tokens=cuda_graph_capture_batch_size * num_tokens_per_req,
    )

    graph_batch = graph_fixture.forward_batch
    graph_batch.spec_info = _make_eagle_draft_extend_input(
        case,
        graph_batch,
        device=device,
    )
    graph_inputs = mla_fixture_inputs(graph_fixture)
    graph_expected = expected_mla_output_from_inputs(
        graph_fixture,
        case,
        graph_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        backend.init_forward_metadata(graph_batch)
        graph_eager_actual = run_mla_forward(
            graph_fixture,
            graph_batch,
            graph_inputs,
        )

    torch.testing.assert_close(
        graph_eager_actual,
        graph_expected,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )

    capture_prefix_len = backend.get_cuda_graph_seq_len_fill_value()
    capture_case = make_mla_case_with_lens(
        case,
        f"{case.name}_cuda_graph_capture",
        (capture_prefix_len,) * cuda_graph_capture_batch_size,
        (num_tokens_per_req,) * cuda_graph_capture_batch_size,
    )
    capture_inputs = make_mla_random_inputs(
        capture_case,
        graph_fixture,
        dtype=dtype,
        device=device,
    )
    capture_batch = _make_mla_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    capture_batch.spec_info = _make_eagle_draft_extend_input(
        capture_case,
        capture_batch,
        device=device,
    )
    prepare_mla_runner_inputs(
        graph_fixture,
        capture_case,
        capture_batch,
        capture_inputs,
        max_context_len=max_context_len,
    )
    capture_expected = expected_mla_output_from_inputs(
        graph_fixture,
        capture_case,
        capture_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(
            backend,
            cuda_graph_capture_batch_size,
            capture_batch,
        )
        capture_actual = run_mla_forward(
            graph_fixture,
            capture_batch,
            capture_inputs,
        )
        backend.on_after_cuda_graph_warmup()

    replay_pad_prefix_lens = (capture_prefix_len,) * (
        cuda_graph_capture_batch_size - case.batch_size
    )
    replay_case = make_mla_case_with_lens(
        case,
        f"{case.name}_cuda_graph_replay",
        case.prefix_lens + replay_pad_prefix_lens,
        case.input_lens + (num_tokens_per_req,) * len(replay_pad_prefix_lens),
    )
    replay_inputs = make_mla_padded_replay_inputs(
        replay_case,
        graph_fixture,
        replay_pad_prefix_lens,
        graph_inputs,
        dtype=dtype,
        device=device,
    )
    replay_batch = _make_mla_forward_batch(
        replay_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    replay_batch.spec_info = _make_eagle_draft_extend_input(
        replay_case,
        replay_batch,
        device=device,
    )
    prepare_mla_runner_inputs(
        graph_fixture,
        replay_case,
        replay_batch,
        replay_inputs,
        max_context_len=max_context_len,
    )
    replay_expected = expected_mla_output_from_inputs(
        graph_fixture,
        replay_case,
        replay_inputs,
        None,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_replay_metadata(
            backend,
            cuda_graph_capture_batch_size,
            replay_batch,
        )
        replay_actual = run_mla_forward(graph_fixture, replay_batch, replay_inputs)

    torch.testing.assert_close(
        capture_actual,
        capture_expected,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )
    torch.testing.assert_close(
        replay_actual,
        replay_expected,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )
    torch.testing.assert_close(
        replay_actual[: case.num_input_tokens],
        graph_eager_actual,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
    )


def run_mla_eagle_draft_extend_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
):
    if not case.forward_mode.is_draft_extend():
        raise ValueError("EAGLE draft-extend coverage expects DRAFT_EXTEND cases.")
    fixture = build_mla_attention_fixture(
        testcase,
        case,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    fixture.forward_batch.spec_info = _make_eagle_draft_extend_input(
        case,
        fixture.forward_batch,
        device=device,
    )
    inputs = mla_fixture_inputs(fixture)
    expected = expected_mla_output_from_inputs(fixture, case, inputs, None)

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        actual = run_mla_forward(fixture, fixture.forward_batch, inputs)

    torch.testing.assert_close(actual, expected, atol=MLA_ATOL, rtol=MLA_RTOL)
