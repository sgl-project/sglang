import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput, EagleVerifyInput

from .dense_attention import DEFAULT_DEVICE as DENSE_DEFAULT_DEVICE
from .dense_attention import DEFAULT_DTYPE as DENSE_DEFAULT_DTYPE
from .dense_attention import (
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
)
from .dense_attention import DEFAULT_MAX_CONTEXT_LEN as DENSE_DEFAULT_MAX_CONTEXT_LEN
from .dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    build_dense_attention_fixture,
    dense_attention_reference_with_custom_mask,
    dense_fixture_inputs,
    expected_dense_output_from_inputs,
    run_dense_forward,
)
from .mla_attention import DEFAULT_DEVICE as MLA_DEFAULT_DEVICE
from .mla_attention import DEFAULT_DTYPE as MLA_DEFAULT_DTYPE
from .mla_attention import DEFAULT_HIDDEN_SIZE as MLA_DEFAULT_HIDDEN_SIZE
from .mla_attention import (
    DEFAULT_KV_LORA_RANK,
)
from .mla_attention import DEFAULT_MAX_CONTEXT_LEN as MLA_DEFAULT_MAX_CONTEXT_LEN
from .mla_attention import (
    MLA_ATOL,
    MLA_RTOL,
    MLAAttentionCase,
    build_mla_attention_fixture,
    expected_mla_output_from_inputs,
    mla_attention_reference_with_custom_mask,
    mla_fixture_inputs,
    run_mla_forward,
)


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


def _make_eagle_verify_input(
    case,
    batch,
    *,
    topk: int,
    device: str,
):
    draft_token_num = _check_target_verify_case(case)
    _, custom_mask = _make_custom_masks(case, topk=topk, device=device)
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

    return EagleVerifyInput(
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


def _prepare_target_verify_batch(batch, case, device: str) -> None:
    prefix_lens = torch.tensor(case.prefix_lens, dtype=torch.int32, device=device)
    batch.seq_lens = prefix_lens
    batch.seq_lens_cpu = torch.tensor(case.prefix_lens, dtype=torch.int32, device="cpu")
    batch.seq_lens_sum = sum(case.prefix_lens)


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
    fixture.forward_batch.spec_info = _make_eagle_verify_input(
        case,
        fixture.forward_batch,
        topk=topk,
        device=device,
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


def run_dense_eagle_draft_extend_case(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DENSE_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DENSE_DEFAULT_DTYPE,
    device: str = DENSE_DEFAULT_DEVICE,
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
    fixture.forward_batch.spec_info = _make_eagle_draft_extend_input(
        case,
        fixture.forward_batch,
        device=device,
    )
    inputs = dense_fixture_inputs(fixture)
    expected = expected_dense_output_from_inputs(fixture, case, inputs, None)

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        actual = run_dense_forward(fixture, fixture.forward_batch, inputs)

    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)


def run_mla_eagle_verify_case(
    testcase,
    case: MLAAttentionCase,
    *,
    topk: int,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    hidden_size: int = MLA_DEFAULT_HIDDEN_SIZE,
    max_context_len: int = MLA_DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = MLA_DEFAULT_DTYPE,
    device: str = MLA_DEFAULT_DEVICE,
):
    fixture = build_mla_attention_fixture(
        testcase,
        case,
        kv_lora_rank=kv_lora_rank,
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


def run_mla_eagle_draft_extend_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
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
