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
from .speculative_cuda_graph_runner import (
    SpeculativeCudaGraphAdapter,
    run_speculative_cuda_graph_case,
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


def _prepare_draft_extend_batch(
    case,
    batch,
    *,
    device: str,
    spec_kind: DraftExtendKind,
) -> None:
    batch.spec_info = _make_draft_extend_input(
        case,
        batch,
        device=device,
        spec_kind=spec_kind,
    )


def _prepare_eagle_draft_extend_batch(case, batch, *, device: str) -> None:
    batch.spec_info = _make_eagle_draft_extend_input(
        case,
        batch,
        device=device,
    )


def _prepare_eagle_draft_extend_v2_batch(case, batch, *, device: str) -> None:
    _set_draft_extend_v2_prefix_lens(batch, case, device=device)
    batch.spec_info = _make_eagle_draft_extend_v2_input(
        case,
        batch,
        device=device,
    )


def _run_draft_extend_cuda_graph_case(
    testcase,
    case,
    *,
    build_fixture,
    make_capture_case,
    make_replay_case,
    make_forward_batch,
    fixture_inputs,
    make_capture_inputs,
    make_replay_inputs,
    prepare_batch,
    prepare_inputs,
    run_forward,
    expected_output,
    build_kwargs: dict,
    max_context_len: int,
    dtype: torch.dtype,
    device: str,
    capture_batch_size: int,
    atol: float,
    rtol: float,
    max_num_tokens=None,
    run_graph_eager: bool = True,
    compare_replay_to_graph_eager: bool = True,
):
    adapter = SpeculativeCudaGraphAdapter(
        build_fixture=build_fixture,
        make_capture_case=make_capture_case,
        make_replay_case=make_replay_case,
        make_forward_batch=make_forward_batch,
        fixture_inputs=fixture_inputs,
        make_capture_inputs=make_capture_inputs,
        make_replay_inputs=make_replay_inputs,
        prepare_batch=prepare_batch,
        prepare_inputs=prepare_inputs,
        run_forward=run_forward,
        expected_output=lambda fixture, draft_case, inputs, _state: expected_output(
            fixture,
            draft_case,
            inputs,
            None,
        ),
        max_num_tokens=max_num_tokens,
        run_graph_eager=run_graph_eager,
        compare_replay_to_graph_eager=compare_replay_to_graph_eager,
        atol=atol,
        rtol=rtol,
    )
    run_speculative_cuda_graph_case(
        testcase,
        case,
        adapter=adapter,
        build_kwargs=build_kwargs,
        capture_batch_size=capture_batch_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )


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

    num_tokens_per_req = max(case.input_lens)
    _run_draft_extend_cuda_graph_case(
        testcase,
        case,
        build_fixture=build_dense_attention_fixture,
        make_capture_case=lambda base, name, prefix_len, bs: (
            _make_dense_spec_case_with_lens(
                base,
                name,
                (prefix_len,) * bs,
                (num_tokens_per_req,) * bs,
            )
        ),
        make_replay_case=lambda base, name, pad_prefix_lens: (
            _make_dense_spec_case_with_lens(
                base,
                name,
                base.prefix_lens + pad_prefix_lens,
                base.input_lens + (num_tokens_per_req,) * len(pad_prefix_lens),
            )
        ),
        make_forward_batch=_make_dense_forward_batch,
        fixture_inputs=dense_fixture_inputs,
        make_capture_inputs=make_dense_random_inputs,
        make_replay_inputs=make_dense_padded_replay_inputs,
        prepare_batch=lambda draft_case, batch: _prepare_draft_extend_batch(
            draft_case,
            batch,
            device=device,
            spec_kind=spec_kind,
        ),
        prepare_inputs=prepare_dense_runner_inputs,
        run_forward=run_dense_forward,
        expected_output=expected_dense_output_from_inputs,
        build_kwargs=dict(
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        capture_batch_size=cuda_graph_capture_batch_size,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
        max_num_tokens=lambda _case, bs: bs * num_tokens_per_req,
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
    if len(set(case.input_lens)) != 1:
        raise ValueError(
            "Draft-extend-v2 CUDA graph coverage uses a fixed token count per request."
        )

    num_tokens_per_req = case.input_lens[0]
    _run_draft_extend_cuda_graph_case(
        testcase,
        case,
        build_fixture=build_dense_attention_fixture,
        make_capture_case=lambda base, name, prefix_len, bs: (
            _make_dense_spec_case_with_lens(
                base,
                name,
                (prefix_len,) * bs,
                (num_tokens_per_req,) * bs,
            )
        ),
        make_replay_case=lambda base, name, pad_prefix_lens: (
            _make_dense_spec_case_with_lens(
                base,
                name,
                base.prefix_lens + pad_prefix_lens,
                base.input_lens + (num_tokens_per_req,) * len(pad_prefix_lens),
            )
        ),
        make_forward_batch=_make_dense_forward_batch,
        fixture_inputs=dense_fixture_inputs,
        make_capture_inputs=make_dense_random_inputs,
        make_replay_inputs=make_dense_padded_replay_inputs,
        prepare_batch=lambda draft_case, batch: _prepare_eagle_draft_extend_v2_batch(
            draft_case,
            batch,
            device=device,
        ),
        prepare_inputs=prepare_dense_runner_inputs,
        run_forward=run_dense_forward,
        expected_output=expected_dense_output_from_inputs,
        build_kwargs=dict(
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        capture_batch_size=cuda_graph_capture_batch_size,
        atol=DENSE_ATOL,
        rtol=DENSE_RTOL,
        run_graph_eager=False,
        compare_replay_to_graph_eager=False,
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
    if len(set(case.input_lens)) != 1:
        raise ValueError(
            "Draft-extend-v2 CUDA graph coverage uses a fixed token count per request."
        )

    _run_draft_extend_cuda_graph_case(
        testcase,
        case,
        build_fixture=build_mla_attention_fixture,
        make_capture_case=lambda base, name, prefix_len, bs: (
            make_mla_case_with_prefix_lens(base, name, (prefix_len,) * bs)
        ),
        make_replay_case=lambda base, name, pad_prefix_lens: (
            make_mla_case_with_prefix_lens(
                base,
                name,
                base.prefix_lens + pad_prefix_lens,
            )
        ),
        make_forward_batch=_make_mla_forward_batch,
        fixture_inputs=mla_fixture_inputs,
        make_capture_inputs=make_mla_random_inputs,
        make_replay_inputs=make_mla_padded_replay_inputs,
        prepare_batch=lambda draft_case, batch: _prepare_eagle_draft_extend_v2_batch(
            draft_case,
            batch,
            device=device,
        ),
        prepare_inputs=prepare_mla_runner_inputs,
        run_forward=run_mla_forward,
        expected_output=expected_mla_output_from_inputs,
        build_kwargs=dict(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        capture_batch_size=cuda_graph_capture_batch_size,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
        run_graph_eager=False,
        compare_replay_to_graph_eager=False,
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

    num_tokens_per_req = max(case.input_lens)
    _run_draft_extend_cuda_graph_case(
        testcase,
        case,
        build_fixture=build_mla_attention_fixture,
        make_capture_case=lambda base, name, prefix_len, bs: (
            make_mla_case_with_lens(
                base,
                name,
                (prefix_len,) * bs,
                (num_tokens_per_req,) * bs,
            )
        ),
        make_replay_case=lambda base, name, pad_prefix_lens: (
            make_mla_case_with_lens(
                base,
                name,
                base.prefix_lens + pad_prefix_lens,
                base.input_lens + (num_tokens_per_req,) * len(pad_prefix_lens),
            )
        ),
        make_forward_batch=_make_mla_forward_batch,
        fixture_inputs=mla_fixture_inputs,
        make_capture_inputs=make_mla_random_inputs,
        make_replay_inputs=make_mla_padded_replay_inputs,
        prepare_batch=lambda draft_case, batch: _prepare_eagle_draft_extend_batch(
            draft_case,
            batch,
            device=device,
        ),
        prepare_inputs=prepare_mla_runner_inputs,
        run_forward=run_mla_forward,
        expected_output=expected_mla_output_from_inputs,
        build_kwargs=dict(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            hidden_size=hidden_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        capture_batch_size=cuda_graph_capture_batch_size,
        atol=MLA_ATOL,
        rtol=MLA_RTOL,
        max_num_tokens=lambda _case, bs: bs * num_tokens_per_req,
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


def run_dsv4_eagle_draft_extend_cuda_graph_case(
    testcase,
    case,
    *,
    swa_size: int = 1024,
    max_context_len: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    cuda_graph_capture_batch_size: int = 2,
):
    """DSV4 EAGLE DRAFT_EXTEND CUDA-graph capture/replay. SWA-only:
    `init_forward_metadata_draft_extend` (`deepseek_v4_backend.py:636-663`)
    hardcodes `need_compress=False`, so the C4/C128 metadata fields are
    None and a `forward(compress_ratio=4 or 128)` would crash. The runner
    asserts compress_ratio == 0 to make this explicit at the call site.
    """
    assert case.compress_ratio == 0, (
        "DSV4 DRAFT_EXTEND is SWA-only — `init_forward_metadata_draft_extend` "
        "uses `need_compress=False` so C4/C128 metadata is unpopulated. See "
        "the 'Production-Unsupported' note in dsv4/README.md."
    )
    assert case.forward_mode.is_draft_extend(include_v2=True), (
        f"run_dsv4_eagle_draft_extend_cuda_graph_case requires DRAFT_EXTEND; "
        f"got {case.forward_mode}"
    )
    from ..attention_methods.dsv4_attention import (
        DSV4_GRAPH_ATOL,
        DSV4_GRAPH_RTOL,
        build_dsv4_attention_fixture,
        dsv4_fixture_inputs,
        expected_dsv4_output_from_inputs,
        make_dsv4_case_with_lens,
        make_dsv4_padded_replay_inputs,
        make_dsv4_random_inputs,
        prepare_dsv4_runner_inputs,
        run_dsv4_forward,
    )
    from ..attention_methods.dsv4_attention import (
        _make_forward_batch as _make_dsv4_forward_batch,
    )

    # DSV4 graph contract requires uniform tokens per request: the graph-bound
    # `init_forward_metadata_draft_extend` uses
    # `num_tokens_per_bs = max_num_tokens // max_bs` and treats every request
    # as having that many extend tokens. DSV4 forward then asserts that
    # `swa_page_indices.shape[0] == q.shape[0]` via `_pad_tensor_to_size`,
    # so q must also be the uniform per-request token count. Use a single
    # `num_tokens_per_req = max(case.input_lens)` for both capture and replay
    # (this differs from the MLA twin — MLA's forward tolerates ragged q vs
    # padded metadata, DSV4 does not).
    num_tokens_per_req = max(case.input_lens)
    _run_draft_extend_cuda_graph_case(
        testcase,
        case,
        build_fixture=build_dsv4_attention_fixture,
        make_capture_case=lambda base, name, prefix_len, bs: make_dsv4_case_with_lens(
            base, name, (prefix_len,) * bs, (num_tokens_per_req,) * bs
        ),
        make_replay_case=lambda base, name, pad_prefix_lens: make_dsv4_case_with_lens(
            base,
            name,
            base.prefix_lens + pad_prefix_lens,
            (num_tokens_per_req,) * (len(base.prefix_lens) + len(pad_prefix_lens)),
        ),
        make_forward_batch=_make_dsv4_forward_batch,
        fixture_inputs=dsv4_fixture_inputs,
        make_capture_inputs=make_dsv4_random_inputs,
        make_replay_inputs=make_dsv4_padded_replay_inputs,
        prepare_batch=lambda draft_case, batch: _prepare_eagle_draft_extend_batch(
            draft_case, batch, device=device
        ),
        prepare_inputs=prepare_dsv4_runner_inputs,
        run_forward=run_dsv4_forward,
        expected_output=expected_dsv4_output_from_inputs,
        build_kwargs=dict(
            swa_size=swa_size,
            max_context_len=max_context_len,
            dtype=dtype,
            device=device,
        ),
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        capture_batch_size=cuda_graph_capture_batch_size,
        atol=DSV4_GRAPH_ATOL,
        rtol=DSV4_GRAPH_RTOL,
        max_num_tokens=lambda _case, bs: bs * num_tokens_per_req,
    )
