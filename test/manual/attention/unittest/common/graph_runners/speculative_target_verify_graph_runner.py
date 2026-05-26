from typing import Literal

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPVerifyInput
from sglang.srt.speculative.ngram_info import NgramVerifyInput

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
    dense_attention_reference_with_custom_mask,
    dense_fixture_inputs,
    make_dense_case_with_prefix_lens,
    make_dense_padded_replay_inputs,
    make_dense_random_inputs,
    prepare_dense_runner_inputs,
    run_dense_forward,
)
from ..attention_methods.gdn_attention import DEFAULT_DEVICE as GDN_DEFAULT_DEVICE
from ..attention_methods.gdn_attention import DEFAULT_DTYPE as GDN_DEFAULT_DTYPE
from ..attention_methods.gdn_attention import (
    DEFAULT_HEAD_K_DIM,
    DEFAULT_HEAD_V_DIM,
)
from ..attention_methods.gdn_attention import (
    DEFAULT_MAX_CONTEXT_LEN as GDN_DEFAULT_MAX_CONTEXT_LEN,
)
from ..attention_methods.gdn_attention import (
    GDN_ATOL,
    GDN_RTOL,
    GDN_TREE_ATOL,
    GDNAttentionCase,
    _clone_gdn_cache,
)
from ..attention_methods.gdn_attention import (
    _make_forward_batch as _make_gdn_forward_batch,
)
from ..attention_methods.gdn_attention import (
    _restore_gdn_cache,
    build_gdn_attention_fixture,
    expected_gdn_verify_output_from_inputs,
    gdn_fixture_inputs,
    make_gdn_case_with_prefix_lens,
    make_gdn_random_inputs,
    prepare_gdn_runner_inputs,
    run_gdn_forward,
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
    make_mla_case_with_prefix_lens,
    make_mla_padded_replay_inputs,
    make_mla_random_inputs,
    mla_attention_reference_with_custom_mask,
    mla_fixture_inputs,
    prepare_mla_runner_inputs,
    run_mla_forward,
)
from .cuda_graph_decode_runner import (
    _init_cuda_graph_capture_metadata,
    _init_cuda_graph_replay_metadata,
)

SpecVerifyKind = Literal["eagle", "frozen_kv_mtp", "dflash", "ngram"]


def _check_target_verify_case(case) -> int:
    if not case.forward_mode.is_target_verify():
        raise ValueError("Speculative verify coverage expects TARGET_VERIFY cases.")
    input_lens = case.input_lens
    if len(set(input_lens)) != 1:
        raise ValueError("EAGLE verify cases require one draft_token_num per batch.")
    return input_lens[0]


def _draft_tree_mask(
    *,
    draft_token_num: int,
    topk: int,
    device: str,
) -> torch.Tensor:
    if topk == 1:
        return torch.tril(
            torch.ones(
                draft_token_num,
                draft_token_num,
                dtype=torch.bool,
                device=device,
            )
        )

    if draft_token_num < 3:
        raise ValueError("Tree-draft verify coverage expects at least 3 draft tokens.")
    mask = torch.eye(draft_token_num, dtype=torch.bool, device=device)
    mask[:, 0] = True
    mask[1, :2] = True
    return mask


def _make_custom_masks(
    case,
    *,
    topk: int,
    device: str,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    draft_token_num = _check_target_verify_case(case)
    draft_mask = _draft_tree_mask(
        draft_token_num=draft_token_num,
        topk=topk,
        device=device,
    )
    masks_by_req = []
    flattened_masks = []
    for prefix_len in case.prefix_lens:
        seq_len = prefix_len + draft_token_num
        reference_mask = torch.ones(
            draft_token_num,
            seq_len,
            dtype=torch.bool,
            device=device,
        )
        reference_mask[:, prefix_len:] = draft_mask
        masks_by_req.append(reference_mask)

        backend_mask = torch.zeros_like(reference_mask)
        backend_mask[:, :seq_len] = reference_mask
        flattened_masks.append(backend_mask.reshape(-1))

    return masks_by_req, torch.cat(flattened_masks, dim=0)


def _make_retrieve_tensors(
    case,
    *,
    topk: int,
    device: str,
):
    draft_token_num = _check_target_verify_case(case)
    retrieve_index = torch.arange(
        draft_token_num,
        dtype=torch.long,
        device=device,
    ).repeat(case.batch_size, 1)
    retrieve_next_token = torch.full_like(retrieve_index, -1)
    retrieve_next_sibling = torch.full_like(retrieve_index, -1)
    if topk > 1:
        retrieve_next_token[:, 0] = 1
        retrieve_next_sibling[:, 1] = 2

    return retrieve_index, retrieve_next_token, retrieve_next_sibling


def _make_spec_verify_input(
    case,
    batch,
    *,
    topk: int,
    device: str,
    spec_kind: SpecVerifyKind,
):
    draft_token_num = _check_target_verify_case(case)
    _, custom_mask = _make_custom_masks(case, topk=topk, device=device)
    retrieve_index, retrieve_next_token, retrieve_next_sibling = _make_retrieve_tensors(
        case,
        topk=topk,
        device=device,
    )

    if spec_kind == "dflash":
        if topk != 1:
            raise ValueError("DFlash verify is linear and expects topk=1.")
        return DFlashVerifyInput(
            draft_token=batch.input_ids,
            positions=batch.positions,
            draft_token_num=draft_token_num,
            topk=1,
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

    if spec_kind == "ngram":
        return NgramVerifyInput(
            draft_token=batch.input_ids,
            tree_mask=custom_mask,
            positions=batch.positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=draft_token_num,
        )

    verify_cls = {
        "eagle": EagleVerifyInput,
        "frozen_kv_mtp": FrozenKVMTPVerifyInput,
    }[spec_kind]
    return verify_cls(
        draft_token=batch.input_ids,
        custom_mask=custom_mask,
        positions=batch.positions,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_cum_len=torch.arange(
            case.batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
        * draft_token_num,
        spec_steps=max(0, draft_token_num - 1),
        topk=topk,
        draft_token_num=draft_token_num,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=batch.seq_lens_sum,
        seq_lens_cpu=batch.seq_lens_cpu,
    )


def _make_eagle_verify_input(
    case,
    batch,
    *,
    topk: int,
    device: str,
):
    return _make_spec_verify_input(
        case,
        batch,
        topk=topk,
        device=device,
        spec_kind="eagle",
    )


def _prepare_target_verify_batch(batch, case, device: str) -> None:
    prefix_lens = torch.tensor(case.prefix_lens, dtype=torch.int32, device=device)
    batch.seq_lens = prefix_lens
    batch.seq_lens_cpu = torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu")
    batch.seq_lens_sum = sum(case.prefix_lens)


def _target_verify_expected_output(
    *,
    reference_fn,
    fixture,
    case,
    inputs,
    topk: int,
    device: str,
):
    masks_by_req, _ = _make_custom_masks(case, topk=topk, device=device)
    return reference_fn(
        fixture.reference_module,
        case,
        inputs["prefix_hidden"],
        inputs["input_hidden"],
        masks_by_req,
    )


def _prepare_spec_verify_batch(
    case,
    batch,
    *,
    topk: int,
    spec_kind: SpecVerifyKind,
    device: str,
) -> None:
    _prepare_target_verify_batch(batch, case, device)
    batch.spec_info = _make_spec_verify_input(
        case,
        batch,
        topk=topk,
        device=device,
        spec_kind=spec_kind,
    )


def _run_spec_verify_cuda_graph_case(
    testcase,
    case,
    *,
    topk: int,
    spec_kind: SpecVerifyKind,
    build_fixture,
    make_case_with_prefix_lens,
    make_forward_batch,
    fixture_inputs,
    make_capture_inputs,
    make_replay_inputs,
    prepare_inputs,
    run_forward,
    reference_fn,
    build_kwargs: dict,
    max_context_len: int,
    dtype: torch.dtype,
    device: str,
    capture_batch_size: int,
    atol: float,
    rtol: float,
):
    draft_token_num = _check_target_verify_case(case)
    if case.batch_size > capture_batch_size:
        raise ValueError("Spec verify CUDA graph capture must cover replay batch size.")

    graph_fixture = build_fixture(
        testcase,
        case,
        **build_kwargs,
        disable_cuda_graph=False,
        runner_batch_size=capture_batch_size,
    )
    backend = graph_fixture.backend
    backend.init_cuda_graph_state(
        max_bs=capture_batch_size,
        max_num_tokens=capture_batch_size * draft_token_num,
    )

    graph_batch = graph_fixture.forward_batch
    _prepare_spec_verify_batch(
        case,
        graph_batch,
        topk=topk,
        spec_kind=spec_kind,
        device=device,
    )
    graph_inputs = fixture_inputs(graph_fixture)
    graph_expected = _target_verify_expected_output(
        reference_fn=reference_fn,
        fixture=graph_fixture,
        case=case,
        inputs=graph_inputs,
        topk=topk,
        device=device,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        backend.init_forward_metadata(graph_batch)
        graph_eager_actual = run_forward(graph_fixture, graph_batch, graph_inputs)

    torch.testing.assert_close(
        graph_eager_actual,
        graph_expected,
        atol=atol,
        rtol=rtol,
    )

    capture_prefix_len = backend.get_cuda_graph_seq_len_fill_value()
    capture_case = make_case_with_prefix_lens(
        case,
        f"{case.name}_cuda_graph_capture",
        (capture_prefix_len,) * capture_batch_size,
    )
    capture_inputs = make_capture_inputs(
        capture_case,
        graph_fixture,
        dtype=dtype,
        device=device,
    )
    capture_batch = make_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _prepare_spec_verify_batch(
        capture_case,
        capture_batch,
        topk=topk,
        spec_kind=spec_kind,
        device=device,
    )
    prepare_inputs(
        graph_fixture,
        capture_case,
        capture_batch,
        capture_inputs,
        max_context_len=max_context_len,
    )
    capture_expected = _target_verify_expected_output(
        reference_fn=reference_fn,
        fixture=graph_fixture,
        case=capture_case,
        inputs=capture_inputs,
        topk=topk,
        device=device,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(
            backend,
            capture_batch_size,
            capture_batch,
        )
        capture_actual = run_forward(graph_fixture, capture_batch, capture_inputs)
        backend.on_after_cuda_graph_warmup()

    replay_pad_prefix_lens = (capture_prefix_len,) * (
        capture_batch_size - case.batch_size
    )
    replay_case = make_case_with_prefix_lens(
        case,
        f"{case.name}_cuda_graph_replay",
        case.prefix_lens + replay_pad_prefix_lens,
    )
    replay_inputs = make_replay_inputs(
        replay_case,
        graph_fixture,
        replay_pad_prefix_lens,
        graph_inputs,
        dtype=dtype,
        device=device,
    )
    replay_batch = make_forward_batch(
        replay_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _prepare_spec_verify_batch(
        replay_case,
        replay_batch,
        topk=topk,
        spec_kind=spec_kind,
        device=device,
    )
    prepare_inputs(
        graph_fixture,
        replay_case,
        replay_batch,
        replay_inputs,
        max_context_len=max_context_len,
    )
    replay_expected = _target_verify_expected_output(
        reference_fn=reference_fn,
        fixture=graph_fixture,
        case=replay_case,
        inputs=replay_inputs,
        topk=topk,
        device=device,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_replay_metadata(backend, capture_batch_size, replay_batch)
        replay_actual = run_forward(graph_fixture, replay_batch, replay_inputs)

    torch.testing.assert_close(
        capture_actual,
        capture_expected,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        replay_actual,
        replay_expected,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        replay_actual[: case.num_input_tokens],
        graph_eager_actual,
        atol=atol,
        rtol=rtol,
    )


def run_dense_spec_verify_case(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int,
    spec_kind: SpecVerifyKind = "eagle",
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
):
    fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    _prepare_target_verify_batch(fixture.forward_batch, case, device)
    masks_by_req, _ = _make_custom_masks(case, topk=topk, device=device)
    fixture.forward_batch.spec_info = _make_spec_verify_input(
        case,
        fixture.forward_batch,
        topk=topk,
        device=device,
        spec_kind=spec_kind,
    )
    inputs = dense_fixture_inputs(fixture)
    expected = dense_attention_reference_with_custom_mask(
        fixture.reference_module,
        case,
        inputs["prefix_hidden"],
        inputs["input_hidden"],
        masks_by_req,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        actual = run_dense_forward(fixture, fixture.forward_batch, inputs)

    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


def run_dense_eagle_verify_case(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
):
    run_dense_spec_verify_case(
        testcase,
        case,
        topk=topk,
        spec_kind="eagle",
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )


def run_dense_spec_verify_cuda_graph_case(
    testcase,
    case: DenseAttentionCase,
    *,
    topk: int,
    spec_kind: SpecVerifyKind = "eagle",
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = 4,
):
    _run_spec_verify_cuda_graph_case(
        testcase,
        case,
        topk=topk,
        spec_kind=spec_kind,
        build_fixture=build_dense_attention_fixture,
        make_case_with_prefix_lens=make_dense_case_with_prefix_lens,
        make_forward_batch=_make_dense_forward_batch,
        fixture_inputs=dense_fixture_inputs,
        make_capture_inputs=make_dense_random_inputs,
        make_replay_inputs=make_dense_padded_replay_inputs,
        prepare_inputs=prepare_dense_runner_inputs,
        run_forward=run_dense_forward,
        reference_fn=dense_attention_reference_with_custom_mask,
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
    )


def run_gdn_eagle_verify_case(
    testcase,
    case: GDNAttentionCase,
    *,
    topk: int,
    head_k_dim: int = DEFAULT_HEAD_K_DIM,
    head_v_dim: int = DEFAULT_HEAD_V_DIM,
    max_context_len: int = GDN_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = GDN_DEFAULT_DTYPE,
    device: str = GDN_DEFAULT_DEVICE,
):
    fixture = build_gdn_attention_fixture(
        testcase,
        case,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
    )
    _prepare_target_verify_batch(fixture.forward_batch, case, device)
    fixture.forward_batch.spec_info = _make_eagle_verify_input(
        case,
        fixture.forward_batch,
        topk=topk,
        device=device,
    )
    inputs = gdn_fixture_inputs(fixture)
    initial_state = _clone_gdn_cache(fixture)
    expected = expected_gdn_verify_output_from_inputs(
        fixture,
        case,
        inputs,
        initial_state,
        topk=topk,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        actual = run_gdn_forward(fixture, fixture.forward_batch, inputs)

    atol = GDN_TREE_ATOL if topk > 1 else GDN_ATOL
    torch.testing.assert_close(actual, expected, atol=atol, rtol=GDN_RTOL)


def _prepare_gdn_verify_batch(case, batch, *, topk: int, device: str) -> None:
    _prepare_target_verify_batch(batch, case, device)
    batch.spec_info = _make_eagle_verify_input(
        case,
        batch,
        topk=topk,
        device=device,
    )


def run_gdn_eagle_verify_cuda_graph_case(
    testcase,
    case: GDNAttentionCase,
    *,
    topk: int,
    head_k_dim: int = DEFAULT_HEAD_K_DIM,
    head_v_dim: int = DEFAULT_HEAD_V_DIM,
    max_context_len: int = GDN_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = GDN_DEFAULT_DTYPE,
    device: str = GDN_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int | None = None,
):
    cuda_graph_capture_batch_size = cuda_graph_capture_batch_size or case.batch_size
    if case.batch_size != cuda_graph_capture_batch_size:
        raise ValueError("GDN verify graph coverage currently uses unpadded replay.")

    graph_fixture = build_gdn_attention_fixture(
        testcase,
        case,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        disable_cuda_graph=False,
        runner_batch_size=cuda_graph_capture_batch_size,
    )
    backend = graph_fixture.backend
    initial_state = _clone_gdn_cache(graph_fixture)

    capture_prefix_len = backend.get_cuda_graph_seq_len_fill_value()
    capture_case = make_gdn_case_with_prefix_lens(
        case,
        f"{case.name}_cuda_graph_capture",
        (capture_prefix_len,) * cuda_graph_capture_batch_size,
    )
    capture_inputs = make_gdn_random_inputs(
        capture_case,
        graph_fixture,
        dtype=dtype,
        device=device,
    )
    capture_batch = _make_gdn_forward_batch(
        capture_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _prepare_gdn_verify_batch(capture_case, capture_batch, topk=topk, device=device)
    prepare_gdn_runner_inputs(
        graph_fixture,
        capture_case,
        capture_batch,
        capture_inputs,
        max_context_len=max_context_len,
    )
    capture_expected = expected_gdn_verify_output_from_inputs(
        graph_fixture,
        capture_case,
        capture_inputs,
        initial_state,
        topk=topk,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_capture_metadata(
            backend,
            cuda_graph_capture_batch_size,
            capture_batch,
        )
        capture_actual = run_gdn_forward(
            graph_fixture,
            capture_batch,
            capture_inputs,
        )
        backend.on_after_cuda_graph_warmup()

    _restore_gdn_cache(graph_fixture, initial_state)
    replay_case = make_gdn_case_with_prefix_lens(
        case,
        f"{case.name}_cuda_graph_replay",
        case.prefix_lens,
    )
    replay_inputs = gdn_fixture_inputs(graph_fixture)
    replay_batch = _make_gdn_forward_batch(
        replay_case,
        graph_fixture.runner,
        max_context_len=max_context_len,
        device=device,
    )
    _prepare_gdn_verify_batch(replay_case, replay_batch, topk=topk, device=device)
    prepare_gdn_runner_inputs(
        graph_fixture,
        replay_case,
        replay_batch,
        replay_inputs,
        max_context_len=max_context_len,
    )
    replay_expected = expected_gdn_verify_output_from_inputs(
        graph_fixture,
        replay_case,
        replay_inputs,
        initial_state,
        topk=topk,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
        _init_cuda_graph_replay_metadata(
            backend,
            cuda_graph_capture_batch_size,
            replay_batch,
        )
        replay_actual = run_gdn_forward(graph_fixture, replay_batch, replay_inputs)

    atol = GDN_TREE_ATOL if topk > 1 else GDN_ATOL
    torch.testing.assert_close(
        capture_actual,
        capture_expected,
        atol=atol,
        rtol=GDN_RTOL,
    )
    torch.testing.assert_close(
        replay_actual,
        replay_expected,
        atol=atol,
        rtol=GDN_RTOL,
    )


def run_mla_eagle_verify_case(
    testcase,
    case: MLAAttentionCase,
    *,
    topk: int,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
):
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
    _prepare_target_verify_batch(fixture.forward_batch, case, device)
    masks_by_req, _ = _make_custom_masks(case, topk=topk, device=device)
    fixture.forward_batch.spec_info = _make_eagle_verify_input(
        case,
        fixture.forward_batch,
        topk=topk,
        device=device,
    )
    inputs = mla_fixture_inputs(fixture)
    expected = mla_attention_reference_with_custom_mask(
        fixture.reference_module,
        case,
        inputs["prefix_hidden"],
        inputs["input_hidden"],
        masks_by_req,
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        actual = run_mla_forward(fixture, fixture.forward_batch, inputs)

    torch.testing.assert_close(actual, expected, atol=MLA_ATOL, rtol=MLA_RTOL)


def run_mla_eagle_verify_cuda_graph_case(
    testcase,
    case: MLAAttentionCase,
    *,
    topk: int,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
    cuda_graph_capture_batch_size: int = 4,
):
    _run_spec_verify_cuda_graph_case(
        testcase,
        case,
        topk=topk,
        spec_kind="eagle",
        build_fixture=build_mla_attention_fixture,
        make_case_with_prefix_lens=make_mla_case_with_prefix_lens,
        make_forward_batch=_make_mla_forward_batch,
        fixture_inputs=mla_fixture_inputs,
        make_capture_inputs=make_mla_random_inputs,
        make_replay_inputs=make_mla_padded_replay_inputs,
        prepare_inputs=prepare_mla_runner_inputs,
        run_forward=run_mla_forward,
        reference_fn=mla_attention_reference_with_custom_mask,
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
    )
