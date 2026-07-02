import json
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.dllm.tp_local_vocab_state import (
    FLOAT32_EXACT_INT_LIMIT,
    argmax_max_prob_from_logits_output,
    can_pack_vocab_ids_as_float32,
    local_vocab_state_from_logits,
    low_confidence_transfer_mask,
    merge_gathered_packed_vocab_state,
    merge_vocab_states,
    pack_vocab_state_for_tp_gather,
    VocabState,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    _slice_prefill_logits_output,
)

ENDPOINT_BASELINE_URL_ENV = "SGLANG_DLLM_BASELINE_URL"
ENDPOINT_TP_LOCAL_URL_ENV = "SGLANG_DLLM_TP_LOCAL_VOCAB_URL"
ENDPOINT_TP_LOCAL_TRACE_ENV = "SGLANG_DLLM_TP_LOCAL_VOCAB_TRACE_JSONL"


def _dense_argmax_and_max_prob(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    max_values, argmax_ids = torch.max(logits, dim=-1)
    max_probs = torch.exp(max_values - torch.logsumexp(logits, dim=-1))
    return argmax_ids.long(), max_probs


def test_low_confidence_tp_state_matches_dense_logits():
    torch.manual_seed(0)
    logits = torch.randn(9, 23, dtype=torch.float32)
    input_ids = torch.tensor([99, 99, 3, 99, 4, 99, 99, 7, 99], dtype=torch.long)
    mask_id = 99
    threshold = 0.19

    dense_argmax_ids, dense_max_probs = _dense_argmax_and_max_prob(logits)
    dense_transfer = low_confidence_transfer_mask(
        input_ids=input_ids,
        argmax_ids=dense_argmax_ids,
        max_probs=dense_max_probs,
        mask_id=mask_id,
        threshold=threshold,
    )

    shard_sizes = [5, 11, 7]
    states = []
    vocab_start = 0
    for shard_size in shard_sizes:
        shard_logits = logits[:, vocab_start : vocab_start + shard_size]
        states.append(
            local_vocab_state_from_logits(
                local_logits=shard_logits,
                vocab_start=vocab_start,
            )
        )
        vocab_start += shard_size

    merged = merge_vocab_states(states)
    merged_transfer = low_confidence_transfer_mask(
        input_ids=input_ids,
        argmax_ids=merged.argmax_ids,
        max_probs=merged.max_probs,
        mask_id=mask_id,
        threshold=threshold,
    )

    torch.testing.assert_close(merged.max_probs, dense_max_probs)
    torch.testing.assert_close(merged.logsumexp, torch.logsumexp(logits.float(), dim=-1))
    assert torch.equal(merged.argmax_ids, dense_argmax_ids)
    assert torch.equal(merged_transfer, dense_transfer)


def test_local_vocab_state_ignores_padded_vocab_entries():
    dense_logits = torch.tensor(
        [
            [0.0, 0.3, -0.2, 0.1, 1.0, 0.9],
            [1.2, -0.4, 0.5, 0.2, -1.0, 0.7],
        ],
        dtype=torch.float32,
    )
    dense_argmax_ids, dense_max_probs = _dense_argmax_and_max_prob(dense_logits)

    first_state = local_vocab_state_from_logits(
        local_logits=dense_logits[:, :4],
        vocab_start=0,
    )
    second_state = local_vocab_state_from_logits(
        local_logits=torch.cat(
            [
                dense_logits[:, 4:],
                torch.full((dense_logits.shape[0], 3), 1000.0),
            ],
            dim=-1,
        ),
        vocab_start=4,
        valid_vocab_size=2,
    )

    merged = merge_vocab_states([first_state, second_state])

    torch.testing.assert_close(merged.max_probs, dense_max_probs)
    torch.testing.assert_close(
        merged.logsumexp,
        torch.logsumexp(dense_logits.float(), dim=-1),
    )
    assert torch.equal(merged.argmax_ids, dense_argmax_ids)


def test_joint_threshold_penalty_matches_dense_logits():
    logits = torch.tensor(
        [
            [0.2, 1.0, 0.1, -0.1, 0.0, 0.3, -0.4, 0.6],
            [0.5, 0.2, 1.4, 0.8, 0.1, 1.7, 0.4, -0.2],
            [1.3, 0.1, -0.5, 1.6, 0.2, 0.0, 1.5, 0.4],
            [0.0, 0.7, 0.2, -0.3, 1.9, 0.8, 0.1, 1.8],
        ],
        dtype=torch.float32,
    )
    penalty_token_ids = torch.tensor([-1, 5, 3, 6], dtype=torch.long)
    penalty_lambda = 0.75

    dense_penalized = logits.clone()
    row_ids = torch.arange(logits.shape[0])
    penalized_rows = penalty_token_ids >= 0
    dense_penalized[row_ids[penalized_rows], penalty_token_ids[penalized_rows]] -= (
        penalty_lambda
    )
    dense_argmax_ids, dense_max_probs = _dense_argmax_and_max_prob(dense_penalized)

    first_state = local_vocab_state_from_logits(
        local_logits=logits[:, :3],
        vocab_start=0,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )
    second_state = local_vocab_state_from_logits(
        local_logits=logits[:, 3:6],
        vocab_start=3,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )
    third_state = local_vocab_state_from_logits(
        local_logits=logits[:, 6:],
        vocab_start=6,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )

    merged = merge_vocab_states([first_state, second_state, third_state])

    torch.testing.assert_close(merged.max_probs, dense_max_probs)
    torch.testing.assert_close(
        merged.logsumexp,
        torch.logsumexp(dense_penalized.float(), dim=-1),
    )
    assert torch.equal(merged.argmax_ids, dense_argmax_ids)


@pytest.mark.parametrize(
    ("seed", "rows", "vocab_size", "shard_sizes"),
    [
        (0, 1, 7, [2, 3, 2]),
        (1, 8, 31, [5, 9, 1, 16]),
        (20260622, 13, 67, [17, 3, 19, 28]),
    ],
)
def test_random_low_confidence_tp_state_matches_dense_with_padding_and_penalty(
    seed: int,
    rows: int,
    vocab_size: int,
    shard_sizes: list[int],
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(rows, vocab_size, generator=generator, dtype=torch.float32)
    penalty_token_ids = torch.randint(
        low=0,
        high=vocab_size + 1,
        size=(rows,),
        generator=generator,
        dtype=torch.long,
    ) - 1
    penalty_lambda = 0.625

    penalized = logits.clone()
    row_ids = torch.arange(rows)
    penalized_rows = penalty_token_ids >= 0
    penalized[row_ids[penalized_rows], penalty_token_ids[penalized_rows]] -= (
        penalty_lambda
    )
    dense_argmax_ids, dense_max_probs = _dense_argmax_and_max_prob(penalized)

    states = []
    vocab_start = 0
    for shard_size in shard_sizes:
        shard_logits = logits[:, vocab_start : vocab_start + shard_size]
        padded_logits = torch.cat(
            [
                shard_logits,
                torch.full((rows, 3), 10000.0, dtype=torch.float32),
            ],
            dim=-1,
        )
        states.append(
            local_vocab_state_from_logits(
                local_logits=padded_logits,
                vocab_start=vocab_start,
                valid_vocab_size=shard_size,
                penalized_token_ids=penalty_token_ids,
                penalty_lambda=penalty_lambda,
            )
        )
        vocab_start += shard_size

    merged = merge_vocab_states(states)

    torch.testing.assert_close(merged.max_probs, dense_max_probs)
    torch.testing.assert_close(
        merged.logsumexp,
        torch.logsumexp(penalized.float(), dim=-1),
    )
    assert torch.equal(merged.argmax_ids, dense_argmax_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_local_vocab_state_matches_reference_cuda():
    from sglang.srt.dllm.tp_local_vocab_kernel import (
        local_vocab_state_from_logits_triton,
    )

    torch.manual_seed(1)
    logits = torch.randn(6, 41, device="cuda", dtype=torch.float32)
    logits[:, 37:] = 1000.0

    actual = local_vocab_state_from_logits_triton(
        local_logits=logits,
        vocab_start=17,
        valid_vocab_size=37,
    )
    expected = local_vocab_state_from_logits(
        local_logits=logits,
        vocab_start=17,
        valid_vocab_size=37,
    )

    torch.testing.assert_close(actual.max_values, expected.max_values)
    torch.testing.assert_close(actual.max_probs, expected.max_probs)
    torch.testing.assert_close(actual.logsumexp, expected.logsumexp)
    assert torch.equal(actual.argmax_ids, expected.argmax_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_local_vocab_state_applies_penalty_cuda():
    from sglang.srt.dllm.tp_local_vocab_kernel import (
        local_vocab_state_from_logits_triton,
    )

    torch.manual_seed(2)
    logits = torch.randn(5, 29, device="cuda", dtype=torch.float32)
    penalty_token_ids = torch.tensor([11, -1, 17, 30, 27], device="cuda")
    penalty_lambda = 0.5

    actual = local_vocab_state_from_logits_triton(
        local_logits=logits,
        vocab_start=9,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )
    expected = local_vocab_state_from_logits(
        local_logits=logits,
        vocab_start=9,
        penalized_token_ids=penalty_token_ids,
        penalty_lambda=penalty_lambda,
    )

    torch.testing.assert_close(actual.max_values, expected.max_values)
    torch.testing.assert_close(actual.max_probs, expected.max_probs)
    torch.testing.assert_close(actual.logsumexp, expected.logsumexp)
    assert torch.equal(actual.argmax_ids, expected.argmax_ids)


def test_argmax_max_prob_uses_compact_state_without_full_logits():
    state = VocabState(
        max_values=torch.tensor([1.0, 2.0, 3.0]),
        argmax_ids=torch.tensor([7, 8, 9]),
        logsumexp=torch.tensor([1.5, 2.5, 3.5]),
        max_probs=torch.tensor([0.5, 0.6, 0.7]),
    )
    logits_output = LogitsProcessorOutput(
        next_token_logits=None,
        full_logits=None,
        dllm_vocab_state=state,
    )

    argmax_ids, max_probs = argmax_max_prob_from_logits_output(
        logits_output,
        start=1,
        end=3,
    )

    assert torch.equal(argmax_ids, torch.tensor([8, 9]))
    torch.testing.assert_close(max_probs, torch.tensor([0.6, 0.7]))


def test_prefill_cuda_graph_slices_dllm_vocab_state_without_next_logits():
    state = VocabState(
        max_values=torch.tensor([1.0, 2.0, 3.0]),
        argmax_ids=torch.tensor([7, 8, 9]),
        logsumexp=torch.tensor([1.5, 2.5, 3.5]),
        max_probs=torch.tensor([0.5, 0.6, 0.7]),
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    full_logits = torch.arange(15, dtype=torch.float32).view(3, 5)
    output = LogitsProcessorOutput(
        next_token_logits=None,
        hidden_states=hidden_states,
        full_logits=full_logits,
        dllm_vocab_state=state,
    )

    sliced = _slice_prefill_logits_output(
        output,
        2,
        is_dllm=True,
        is_speculative=False,
    )

    assert sliced.next_token_logits is None
    torch.testing.assert_close(sliced.hidden_states, hidden_states[:2])
    torch.testing.assert_close(sliced.full_logits, full_logits[:2])
    torch.testing.assert_close(sliced.dllm_vocab_state.max_values, state.max_values[:2])
    assert torch.equal(sliced.dllm_vocab_state.argmax_ids, state.argmax_ids[:2])
    torch.testing.assert_close(sliced.dllm_vocab_state.logsumexp, state.logsumexp[:2])
    torch.testing.assert_close(sliced.dllm_vocab_state.max_probs, state.max_probs[:2])


def test_argmax_max_prob_falls_back_to_full_logits_without_compact_state():
    logits = torch.tensor(
        [
            [0.0, 4.0, 1.0],
            [2.5, -1.0, 2.5],
            [-3.0, -2.0, -1.0],
            [0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    logits_output = LogitsProcessorOutput(
        next_token_logits=None,
        full_logits=logits,
        dllm_vocab_state=None,
    )

    argmax_ids, max_probs = argmax_max_prob_from_logits_output(
        logits_output,
        start=1,
        end=3,
    )

    expected_argmax_ids, expected_max_probs = _dense_argmax_and_max_prob(logits[1:3])
    assert torch.equal(argmax_ids, expected_argmax_ids)
    torch.testing.assert_close(max_probs, expected_max_probs)


def test_merge_vocab_states_tie_breaks_equal_max_values_by_token_id():
    states = [
        VocabState(
            max_values=torch.tensor([5.0, 7.0, 1.0]),
            argmax_ids=torch.tensor([10, 12, 30]),
            logsumexp=torch.log(torch.tensor([2.0, 3.0, 5.0])),
            max_probs=torch.empty(3),
        ),
        VocabState(
            max_values=torch.tensor([5.0, 6.0, 1.0]),
            argmax_ids=torch.tensor([8, 9, 25]),
            logsumexp=torch.log(torch.tensor([4.0, 7.0, 11.0])),
            max_probs=torch.empty(3),
        ),
        VocabState(
            max_values=torch.tensor([4.0, 7.0, 2.0]),
            argmax_ids=torch.tensor([6, 11, 40]),
            logsumexp=torch.log(torch.tensor([13.0, 17.0, 19.0])),
            max_probs=torch.empty(3),
        ),
    ]

    merged = merge_vocab_states(states)

    assert torch.equal(merged.argmax_ids, torch.tensor([8, 11, 40]))
    torch.testing.assert_close(merged.max_values, torch.tensor([5.0, 7.0, 2.0]))
    torch.testing.assert_close(
        merged.logsumexp,
        torch.log(torch.tensor([19.0, 27.0, 35.0])),
    )
    torch.testing.assert_close(
        merged.max_probs,
        torch.exp(merged.max_values - merged.logsumexp),
    )


def test_can_pack_vocab_ids_as_float32_boundary():
    assert can_pack_vocab_ids_as_float32(0) is True
    assert can_pack_vocab_ids_as_float32(FLOAT32_EXACT_INT_LIMIT - 1) is True
    assert can_pack_vocab_ids_as_float32(FLOAT32_EXACT_INT_LIMIT) is True
    assert can_pack_vocab_ids_as_float32(-1) is False
    assert can_pack_vocab_ids_as_float32(FLOAT32_EXACT_INT_LIMIT + 1) is False


def test_float32_vocab_id_roundtrip_boundary_documents_guard():
    exact_id = torch.tensor([FLOAT32_EXACT_INT_LIMIT])
    inexact_id = torch.tensor([FLOAT32_EXACT_INT_LIMIT + 1])

    assert torch.equal(exact_id.float().long(), exact_id)
    assert not torch.equal(inexact_id.float().long(), inexact_id)


def test_packed_vocab_state_preserves_exact_ids_at_float32_boundary():
    states = [
        VocabState(
            max_values=torch.tensor([1.0, 4.0, 6.0], dtype=torch.float32),
            argmax_ids=torch.tensor(
                [FLOAT32_EXACT_INT_LIMIT, 9, FLOAT32_EXACT_INT_LIMIT],
                dtype=torch.long,
            ),
            logsumexp=torch.tensor([2.0, 5.0, 7.0], dtype=torch.float32),
            max_probs=torch.empty(3, dtype=torch.float32),
        ),
        VocabState(
            max_values=torch.tensor([1.0, 4.0, 5.0], dtype=torch.float32),
            argmax_ids=torch.tensor(
                [FLOAT32_EXACT_INT_LIMIT - 1, 7, 11],
                dtype=torch.long,
            ),
            logsumexp=torch.tensor([3.0, 6.0, 6.0], dtype=torch.float32),
            max_probs=torch.empty(3, dtype=torch.float32),
        ),
    ]

    expected = merge_vocab_states(states)
    gathered = torch.stack([pack_vocab_state_for_tp_gather(state) for state in states])
    actual = merge_gathered_packed_vocab_state(gathered)

    assert torch.equal(actual.argmax_ids, expected.argmax_ids)
    assert actual.argmax_ids[0].item() == FLOAT32_EXACT_INT_LIMIT - 1
    assert actual.argmax_ids[1].item() == 7
    assert actual.argmax_ids[2].item() == FLOAT32_EXACT_INT_LIMIT
    torch.testing.assert_close(actual.max_values, expected.max_values)
    torch.testing.assert_close(actual.logsumexp, expected.logsumexp)
    torch.testing.assert_close(actual.max_probs, expected.max_probs)


def test_packed_vocab_state_merge_matches_legacy_merge_with_tie_break():
    states = [
        VocabState(
            max_values=torch.tensor([5.0, 7.0, -1.0], dtype=torch.float32),
            argmax_ids=torch.tensor([10, 12, 30]),
            logsumexp=torch.tensor([5.1, 7.5, 2.0], dtype=torch.float32),
            max_probs=torch.empty(3),
        ),
        VocabState(
            max_values=torch.tensor([5.0, 6.0, 0.0], dtype=torch.float32),
            argmax_ids=torch.tensor([8, 9, 25]),
            logsumexp=torch.tensor([5.2, 6.5, 2.5], dtype=torch.float32),
            max_probs=torch.empty(3),
        ),
        VocabState(
            max_values=torch.tensor([4.0, 7.0, 1.0], dtype=torch.float32),
            argmax_ids=torch.tensor([6, 11, 40]),
            logsumexp=torch.tensor([5.3, 7.7, 3.0], dtype=torch.float32),
            max_probs=torch.empty(3),
        ),
    ]

    expected = merge_vocab_states(states)
    gathered = torch.stack([pack_vocab_state_for_tp_gather(state) for state in states])

    actual = merge_gathered_packed_vocab_state(gathered)

    assert torch.equal(actual.max_values, expected.max_values)
    assert torch.equal(actual.logsumexp, expected.logsumexp)
    assert torch.equal(actual.argmax_ids, expected.argmax_ids)
    assert actual.argmax_ids[0].item() == 8
    assert actual.argmax_ids[1].item() == 11
    torch.testing.assert_close(actual.max_probs, expected.max_probs)


def test_pack_vocab_state_requires_float32_state_values():
    base = VocabState(
        max_values=torch.tensor([1.0], dtype=torch.float32),
        argmax_ids=torch.tensor([5]),
        logsumexp=torch.tensor([1.5], dtype=torch.float32),
        max_probs=torch.tensor([0.6], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="max_values"):
        pack_vocab_state_for_tp_gather(
            VocabState(
                max_values=base.max_values.double(),
                argmax_ids=base.argmax_ids,
                logsumexp=base.logsumexp,
                max_probs=base.max_probs,
            )
        )

    with pytest.raises(ValueError, match="logsumexp"):
        pack_vocab_state_for_tp_gather(
            VocabState(
                max_values=base.max_values,
                argmax_ids=base.argmax_ids,
                logsumexp=base.logsumexp.double(),
                max_probs=base.max_probs,
            )
        )


def test_packed_gather_gate_uses_vocab_upper_bound_and_env():
    import sglang.srt.layers.logits_processor as logits_processor
    from sglang.srt.environ import envs
    from sglang.srt.dllm.tp_local_vocab_kernel import (
        LOCAL_VOCAB_STATE_TRITON_MAX_BLOCK_VOCAB,
        can_use_local_vocab_state_triton,
    )

    assert can_pack_vocab_ids_as_float32(151669)
    assert not can_pack_vocab_ids_as_float32(FLOAT32_EXACT_INT_LIMIT + 1)
    assert can_use_local_vocab_state_triton(
        LOCAL_VOCAB_STATE_TRITON_MAX_BLOCK_VOCAB
    )
    assert not can_use_local_vocab_state_triton(
        LOCAL_VOCAB_STATE_TRITON_MAX_BLOCK_VOCAB + 1
    )

    with envs.SGLANG_DLLM_TP_LOCAL_VOCAB_PACKED_GATHER.override(True):
        assert logits_processor._should_pack_dllm_vocab_state_for_gather(151669)
        assert not logits_processor._should_pack_dllm_vocab_state_for_gather(
            FLOAT32_EXACT_INT_LIMIT + 1
        )

    with envs.SGLANG_DLLM_TP_LOCAL_VOCAB_PACKED_GATHER.override(False):
        assert not logits_processor._should_pack_dllm_vocab_state_for_gather(151669)


def test_dllm_vocab_state_tp_merge_uses_rank_uniform_packed_or_legacy_path():
    from sglang.srt.environ import envs
    from sglang.srt.layers.logits_processor import _merge_dllm_vocab_state_across_tp

    states = [
        VocabState(
            max_values=torch.tensor([2.0, 5.0], dtype=torch.float32),
            argmax_ids=torch.tensor([12, 40], dtype=torch.long),
            logsumexp=torch.tensor([3.0, 6.0], dtype=torch.float32),
            max_probs=torch.empty(2),
        ),
        VocabState(
            max_values=torch.tensor([2.0, 4.0], dtype=torch.float32),
            argmax_ids=torch.tensor([9, 30], dtype=torch.long),
            logsumexp=torch.tensor([4.0, 7.0], dtype=torch.float32),
            max_probs=torch.empty(2),
        ),
        VocabState(
            max_values=torch.tensor([1.0, 6.0], dtype=torch.float32),
            argmax_ids=torch.tensor([7, 35], dtype=torch.long),
            logsumexp=torch.tensor([5.0, 8.0], dtype=torch.float32),
            max_probs=torch.empty(2),
        ),
    ]

    class FakeTPGroup:
        def __init__(self, all_states):
            self.all_states = all_states
            self.calls = []
            self.float_call_index = 0

        def all_gather_into_tensor(self, output, input_tensor):
            self.calls.append(
                (tuple(output.shape), tuple(input_tensor.shape), input_tensor.dtype)
            )
            if input_tensor.dim() == 2:
                packed_states = [
                    pack_vocab_state_for_tp_gather(state)
                    for state in self.all_states
                ]
                output.copy_(torch.cat(packed_states, dim=0))
                return
            if input_tensor.dtype == torch.long:
                output.copy_(torch.cat([state.argmax_ids for state in self.all_states]))
                return

            values = [state.max_values for state in self.all_states]
            if self.float_call_index == 1:
                values = [state.logsumexp for state in self.all_states]
            output.copy_(torch.cat(values))
            self.float_call_index += 1

    with envs.SGLANG_DLLM_TP_LOCAL_VOCAB_PACKED_GATHER.override(True):
        packed_group = FakeTPGroup(states)
        packed = _merge_dllm_vocab_state_across_tp(
            local_state=states[0],
            tp_group=packed_group,
            tp_size=len(states),
            global_max_vocab_id_inclusive=151669,
        )
        assert len(packed_group.calls) == 1
        assert packed_group.calls[0][0] == (len(states) * 2, 3)

        legacy_group = FakeTPGroup(states)
        legacy = _merge_dllm_vocab_state_across_tp(
            local_state=states[0],
            tp_group=legacy_group,
            tp_size=len(states),
            global_max_vocab_id_inclusive=FLOAT32_EXACT_INT_LIMIT + 1,
        )
        assert len(legacy_group.calls) == 3

    torch.testing.assert_close(packed.max_probs, legacy.max_probs)
    assert torch.equal(packed.max_values, legacy.max_values)
    assert torch.equal(packed.logsumexp, legacy.logsumexp)
    assert torch.equal(packed.argmax_ids, legacy.argmax_ids)


def test_logits_processor_tp_local_vocab_gate(monkeypatch):
    import sglang.srt.layers.logits_processor as logits_processor
    from sglang.srt.environ import envs

    processor = SimpleNamespace(
        use_attn_tp_group=False,
        do_tensor_parallel_all_gather_dp_attn=False,
    )
    lm_head = SimpleNamespace(weight=torch.empty(1))
    should_use = logits_processor.LogitsProcessor._should_use_dllm_tp_local_vocab

    monkeypatch.setattr(
        logits_processor,
        "get_global_server_args",
        lambda: SimpleNamespace(dllm_algorithm="LowConfidence"),
    )

    with envs.SGLANG_DLLM_TP_LOCAL_VOCAB.override(False):
        assert should_use(processor, lm_head) is False

    with envs.SGLANG_DLLM_TP_LOCAL_VOCAB.override(True):
        assert should_use(processor, lm_head) is False
        monkeypatch.setattr(logits_processor, "_is_npu", True)
        assert should_use(processor, lm_head) is False
        monkeypatch.setattr(logits_processor, "_is_npu", False)

        monkeypatch.setattr(
            logits_processor,
            "get_global_server_args",
            lambda: SimpleNamespace(dllm_algorithm="JointThreshold"),
        )
        assert should_use(processor, lm_head) is False

        monkeypatch.setattr(
            logits_processor,
            "get_global_server_args",
            lambda: SimpleNamespace(dllm_algorithm="LowConfidence"),
        )
        assert (
            should_use(
                SimpleNamespace(
                    use_attn_tp_group=True,
                    do_tensor_parallel_all_gather_dp_attn=False,
                ),
                lm_head,
            )
            is False
        )
        assert (
            should_use(
                SimpleNamespace(
                    use_attn_tp_group=False,
                    do_tensor_parallel_all_gather_dp_attn=True,
                ),
                lm_head,
            )
            is False
        )
        assert should_use(processor, SimpleNamespace()) is False

        global_added_lm_head = SimpleNamespace(
            weight=torch.empty(1),
            num_added_embeddings=2,
            shard_indices=SimpleNamespace(num_added_elements=0),
        )
        clean_lm_head = SimpleNamespace(
            weight=torch.empty(1),
            num_added_embeddings=0,
            shard_indices=SimpleNamespace(num_added_elements=0),
        )
        lora_wrapped_lm_head = SimpleNamespace(
            weight=torch.empty(1),
            num_added_embeddings=0,
            base_layer=SimpleNamespace(
                shard_indices=SimpleNamespace(
                    org_vocab_start_index=4,
                    num_org_elements=8,
                    num_added_elements=0,
                )
            ),
        )
        assert should_use(processor, global_added_lm_head) is False
        assert should_use(processor, clean_lm_head) is True
        assert should_use(processor, lora_wrapped_lm_head) is False


def test_dllm_vocab_state_logit_scale_matches_full_logits_dtype_order(monkeypatch):
    import sglang.srt.layers.logits_processor as logits_processor

    logits = torch.tensor(
        [[-4.15625, -4.125, -4.0], [1.125, 1.25, 1.375]],
        dtype=torch.bfloat16,
    )
    scale = 2.764019822896123
    hidden_states = torch.empty(2, 1)
    processor = SimpleNamespace(
        vocab_size=3,
        logit_scale=scale,
        final_logit_softcapping=None,
        _dllm_tp_local_vocab_logged=True,
        _gather_dp_attn_hidden_states=lambda hidden, metadata: (hidden, hidden),
        _compute_lm_head=lambda hidden, lm_head: logits.clone(),
    )
    lm_head = SimpleNamespace(
        weight=torch.empty(1),
        shard_indices=SimpleNamespace(org_vocab_start_index=0, num_org_elements=3),
    )

    monkeypatch.setattr(
        logits_processor,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=1),
    )

    state = logits_processor.LogitsProcessor._get_dllm_vocab_state(
        processor,
        hidden_states,
        lm_head,
        SimpleNamespace(),
    )
    expected_logits = logits.clone()
    expected_logits.mul_(scale)
    expected_state = local_vocab_state_from_logits(
        local_logits=expected_logits.float(),
        vocab_start=0,
        valid_vocab_size=3,
    )

    torch.testing.assert_close(state.max_values, expected_state.max_values)
    torch.testing.assert_close(state.max_probs, expected_state.max_probs)
    torch.testing.assert_close(state.logsumexp, expected_state.logsumexp)
    assert torch.equal(state.argmax_ids, expected_state.argmax_ids)


def test_endpoint_trace_evidence_requires_compact_path_record(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "framework": "sglang",
                "path": "full_vocab_materialized",
                "component": "unit.full",
            }
        )
        + "\n"
    )
    with pytest.raises(AssertionError, match="consumer_sufficient_compact"):
        _assert_compact_trace_evidence(trace_path)

    trace_path.write_text(
        json.dumps(
            {
                "framework": "sglang",
                "path": "consumer_sufficient_compact",
                "component": "sglang.logits_processor._get_dllm_vocab_state",
                "tp_size": 1,
            }
        )
        + "\n"
    )
    _assert_compact_trace_evidence(trace_path)


def test_dllm_vocab_state_softcap_uses_float32_logits(monkeypatch):
    import sglang.srt.layers.logits_processor as logits_processor

    logits = torch.tensor([[0.5, 1.5, -2.0]], dtype=torch.bfloat16)
    hidden_states = torch.empty(1, 1)
    processor = SimpleNamespace(
        vocab_size=3,
        logit_scale=None,
        final_logit_softcapping=2.0,
        _dllm_tp_local_vocab_logged=True,
        _gather_dp_attn_hidden_states=lambda hidden, metadata: (hidden, hidden),
        _compute_lm_head=lambda hidden, lm_head: logits,
    )

    def fake_softcap_inplace(values, cap):
        assert values.dtype == torch.float32
        values.copy_(cap * torch.tanh(values / cap))

    monkeypatch.setattr(logits_processor, "fused_softcap", fake_softcap_inplace)
    monkeypatch.setattr(
        logits_processor,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=1),
    )

    state = logits_processor.LogitsProcessor._get_dllm_vocab_state(
        processor,
        hidden_states,
        SimpleNamespace(weight=torch.empty(1)),
        SimpleNamespace(),
    )
    expected_logits = 2.0 * torch.tanh(logits.float() / 2.0)
    expected_state = local_vocab_state_from_logits(
        local_logits=expected_logits,
        vocab_start=0,
        valid_vocab_size=3,
    )

    assert state.max_values.dtype == torch.float32
    assert state.logsumexp.dtype == torch.float32
    torch.testing.assert_close(state.max_values, expected_state.max_values)
    torch.testing.assert_close(state.max_probs, expected_state.max_probs)
    assert torch.equal(state.argmax_ids, expected_state.argmax_ids)


def test_consumer_state_trace_emits_full_and_compact_records(tmp_path):
    from sglang.srt.dllm.consumer_state_trace import (
        emit_compact_vocab_state_trace,
        emit_full_vocab_trace,
    )
    from sglang.srt.environ import envs

    trace_path = tmp_path / "trace.jsonl"
    full_logits = torch.empty(2, 5, dtype=torch.float32)
    local_logits = torch.empty(2, 3, dtype=torch.bfloat16)
    state = VocabState(
        max_values=torch.empty(2, dtype=torch.float32),
        argmax_ids=torch.empty(2, dtype=torch.int64),
        logsumexp=torch.empty(2, dtype=torch.float32),
        max_probs=torch.empty(2, dtype=torch.float32),
    )

    with envs.SGLANG_CONSUMER_STATE_TRACE_JSONL.override(str(trace_path)):
        emit_full_vocab_trace(
            component="unit.full",
            full_logits=full_logits,
            vocab_size=5,
            tp_size=2,
            rank=1,
            consumer_contract="full",
        )
        emit_compact_vocab_state_trace(
            component="unit.compact",
            local_logits=local_logits,
            state=state,
            vocab_size=9,
            valid_vocab_size=3,
            tp_size=2,
            rank=1,
            packed_gather=True,
            consumer_contract="compact",
        )

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["framework"] for record in records] == ["sglang", "sglang"]
    assert records[0]["path"] == "full_vocab_materialized"
    assert records[0]["full_vocab_materialized_bytes"] == 2 * 5 * 4
    assert records[0]["tp_gather_bytes"] == 2 * 5 * 4

    assert records[1]["path"] == "consumer_sufficient_compact"
    assert records[1]["local_vocab_materialized_bytes"] == 2 * 3 * 2
    assert records[1]["avoidable_full_vocab_materialized_bytes"] == 2 * 9 * 2
    assert records[1]["compact_state_bytes"] == 2 * (4 + 8 + 4 + 4)
    assert records[1]["tp_gather_bytes"] == 2 * 2 * 3 * 4


def _endpoint_url(env_key: str) -> str | None:
    url = os.environ.get(env_key)
    if not url:
        return None
    return url.rstrip("/")


def _assert_compact_trace_evidence(trace_path) -> None:
    records = []
    with open(trace_path) as trace_file:
        for line in trace_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert any(
        record.get("framework") == "sglang"
        and record.get("path") == "consumer_sufficient_compact"
        and record.get("component") == "sglang.logits_processor._get_dllm_vocab_state"
        for record in records
    ), "missing consumer_sufficient_compact endpoint trace evidence"


def _post_generate(endpoint: str, prompt: str) -> str:
    requests = pytest.importorskip("requests")
    response = requests.post(
        f"{endpoint}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        if "text" in payload:
            return payload["text"]
        if "output" in payload:
            return payload["output"]
        return payload["outputs"][0]["text"]
    return payload[0]["text"]


def test_baseline_and_tp_local_vocab_endpoints_match_when_configured():
    baseline_url = _endpoint_url(ENDPOINT_BASELINE_URL_ENV)
    tp_local_url = _endpoint_url(ENDPOINT_TP_LOCAL_URL_ENV)
    tp_local_trace = os.environ.get(ENDPOINT_TP_LOCAL_TRACE_ENV)
    if not baseline_url or not tp_local_url:
        pytest.skip(
            f"set {ENDPOINT_BASELINE_URL_ENV} and {ENDPOINT_TP_LOCAL_URL_ENV} "
            "to compare already-running endpoints"
        )

    prompts = [
        "Name one primary color.",
        "Complete the sequence: 1, 1, 2, 3,",
    ]

    for prompt in prompts:
        assert _post_generate(tp_local_url, prompt) == _post_generate(
            baseline_url, prompt
        )
    if tp_local_trace:
        _assert_compact_trace_evidence(tp_local_trace)


def test_cuda_graph_replay_slice_preserves_compact_vocab_state():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        _slice_dllm_vocab_state,
    )

    state = VocabState(
        max_values=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        argmax_ids=torch.tensor([10, 11, 12, 13]),
        logsumexp=torch.tensor([1.5, 2.5, 3.5, 4.5]),
        max_probs=torch.tensor([0.6, 0.7, 0.8, 0.9]),
    )

    sliced = _slice_dllm_vocab_state(state, 2)

    assert sliced is not None
    torch.testing.assert_close(sliced.max_values, torch.tensor([1.0, 2.0]))
    assert torch.equal(sliced.argmax_ids, torch.tensor([10, 11]))
    torch.testing.assert_close(sliced.logsumexp, torch.tensor([1.5, 2.5]))
    torch.testing.assert_close(sliced.max_probs, torch.tensor([0.6, 0.7]))
