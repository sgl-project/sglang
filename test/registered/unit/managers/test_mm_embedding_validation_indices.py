import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch  # noqa: E402
from sglang.srt.managers.tp_worker import _mm_embedding_validation_indices  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mm_input(*offsets):
    return SimpleNamespace(mm_items=[SimpleNamespace(offsets=list(offsets))])


def _validation_batch(forward_mode, multimodal_inputs, prefix_lens, extend_lens):
    return ScheduleBatch(
        reqs=[SimpleNamespace() for _ in multimodal_inputs],
        forward_mode=forward_mode,
        multimodal_inputs=multimodal_inputs,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )


def _disagg_req(req_pool_idx, multimodal_inputs):
    return SimpleNamespace(
        kv=ReqKvInfo(req_pool_idx=req_pool_idx, kv_committed_len=4, kv_allocated_len=4),
        prefix_indices=[0, 1],
        extend_range=SimpleNamespace(length=2),
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[5],
        retracted_stain=False,
        already_computed=2,
        cached_tokens=0,
        cached_tokens_device=0,
        is_retracted=True,
        pd_rebootstrap_in_progress=False,
        multimodal_inputs=multimodal_inputs,
        beam_group=None,
        decode_batch_idx=0,
        kv_committed_len=4,
    )


def _disagg_batch(reqs):
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int64).reshape(4, 8)
        ),
        device="cpu",
        return_logprob=False,
        model_config=SimpleNamespace(vocab_size=32, is_encoder_decoder=False),
        enable_overlap=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        hisparse_coordinator=None,
        has_grammar=False,
        return_hidden_states=False,
        is_prefill_only=False,
    )
    batch.prepare_for_prebuilt()
    return batch


def test_decode_ignores_stale_short_extend_metadata():
    batch = _validation_batch(
        ForwardMode.DECODE,
        [None, None],
        [2],
        [2],
    )

    assert batch.mm_embedding_validation_indices() == []


def test_target_verify_ignores_extend_metadata():
    batch = _validation_batch(
        ForwardMode.TARGET_VERIFY,
        [_mm_input((2, 4))],
        [0],
        [4],
    )

    assert batch.mm_embedding_validation_indices() == []


@pytest.mark.parametrize(
    ("forward_mode", "expected"),
    [
        (ForwardMode.DECODE, []),
        (ForwardMode.TARGET_VERIFY, []),
        (ForwardMode.EXTEND, [1]),
        (ForwardMode.MIXED, [1]),
    ],
)
def test_batchless_worker_validation_respects_forward_mode(forward_mode, expected):
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        mm_inputs=[None, _mm_input((2, 4))],
    )

    assert _mm_embedding_validation_indices(None, forward_batch) == expected


@pytest.mark.parametrize(
    "multimodal_inputs",
    [
        [None, None],
        [None, _mm_input((2, 4))],
    ],
    ids=["text-only", "multimodal-origin"],
)
def test_disagg_prepare_merge_decode_lifecycle_skips_validation(multimodal_inputs):
    with (
        patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "SamplingBatchInfo.from_schedule_batch",
            side_effect=lambda *_args: SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False),
                merge_batch=Mock(),
            ),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            side_effect=lambda batch, **_kwargs: torch.arange(len(batch.reqs)),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.mamba_extra_buffer_enabled",
            return_value=False,
        ),
    ):
        running_batch = _disagg_batch([_disagg_req(0, multimodal_inputs[0])])
        running_batch.prepare_for_decode()
        new_batch = _disagg_batch([_disagg_req(1, multimodal_inputs[1])])

        running_batch.merge_batch(new_batch)
        running_batch.prepare_for_decode()

    assert running_batch.forward_mode == ForwardMode.DECODE
    assert running_batch.multimodal_inputs == multimodal_inputs
    assert len(running_batch.prefix_lens) == 1
    assert len(running_batch.extend_lens) == 1
    assert running_batch.mm_embedding_validation_indices() == []


@pytest.mark.parametrize("forward_mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_extend_modes_validate_only_overlapping_multimodal_rows(forward_mode):
    batch = _validation_batch(
        forward_mode,
        [None, _mm_input((6, 9)), _mm_input((20, 22))],
        [0, 4, 8],
        [4, 4, 4],
    )

    assert batch.mm_embedding_validation_indices() == [1]


@pytest.mark.parametrize("forward_mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_extend_modes_reject_misaligned_metadata(forward_mode):
    batch = _validation_batch(
        forward_mode,
        [None, _mm_input((2, 4))],
        [0],
        [4, 4],
    )

    with pytest.raises(ValueError, match="zip\\(\\) argument 2 is shorter"):
        batch.mm_embedding_validation_indices()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
