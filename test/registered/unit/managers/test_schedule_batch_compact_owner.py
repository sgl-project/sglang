from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.dflash_compact_physical_layout import (
    CompactDFlashPhysicalLayout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(index: int):
    return SimpleNamespace(
        rid=f"req-{index}",
        req_pool_idx=index,
        grammar=None,
        return_logprob=False,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        is_prefill_only=False,
        finished=lambda: False,
        beam_group=None,
    )


def _batch(indices, generations, acquire):
    reqs = [_req(index) for index in indices]
    batch = ScheduleBatch(reqs=reqs)
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.req_to_token_pool = SimpleNamespace(
        req_generation=torch.tensor([0, *generations], dtype=torch.int64)
    )
    batch.req_pool_indices = torch.tensor(indices, dtype=torch.int64)
    batch.req_pool_indices_cpu = torch.tensor(indices, dtype=torch.int64)
    batch.expected_req_generations_cpu = torch.tensor(generations, dtype=torch.int64)
    batch.acquire_owner_mask = torch.tensor(acquire, dtype=torch.bool)
    batch.seq_lens = torch.arange(1, len(reqs) + 1, dtype=torch.int64)
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.input_ids = torch.arange(len(reqs), dtype=torch.int64)
    batch.multimodal_inputs = None
    batch.sampling_info = MagicMock()
    batch.spec_info = None
    batch.return_logprob = False
    batch.return_hidden_states = False
    batch.return_hidden_states_mode = CaptureHiddenMode.NULL
    batch.has_grammar = False
    batch.is_prefill_only = False
    batch.forward_mode = ForwardMode.DECODE
    return batch


def test_owner_snapshot_reads_pool_generation_and_validates_shape():
    batch = _batch([1, 2], [7, 11], [False, False])
    batch.set_req_pool_owner_metadata([True, False])
    assert batch.expected_req_generations_cpu.tolist() == [7, 11]
    assert batch.acquire_owner_mask.tolist() == [True, False]

    with pytest.raises(RuntimeError, match="acquire mask shape mismatch"):
        batch.set_req_pool_owner_metadata([True])


def test_filter_merge_and_copy_keep_owner_metadata_aligned_and_isolated():
    batch = _batch([1, 2, 3], [11, 13, 17], [True, False, True])
    original_expected = batch.expected_req_generations_cpu
    original_acquire = batch.acquire_owner_mask

    batch.filter_batch(keep_indices=[2, 0])
    assert batch.expected_req_generations_cpu.tolist() == [17, 11]
    assert batch.acquire_owner_mask.tolist() == [True, True]
    assert batch.expected_req_generations_cpu.data_ptr() != original_expected.data_ptr()
    assert batch.acquire_owner_mask.data_ptr() != original_acquire.data_ptr()

    copied = batch.copy()
    assert copied.expected_req_generations_cpu.tolist() == [17, 11]
    assert copied.acquire_owner_mask.tolist() == [True, True]
    assert (
        copied.expected_req_generations_cpu.data_ptr()
        != batch.expected_req_generations_cpu.data_ptr()
    )

    other = _batch([1], [23], [False])
    batch.merge_batch(other)
    assert batch.expected_req_generations_cpu.tolist() == [17, 11, 23]
    assert batch.acquire_owner_mask.tolist() == [True, True, False]


def test_stale_batch_generation_is_rejected_before_owner_rebind():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1, window_size=8, block_size=4, page_size=1
    )
    owner_generation = torch.tensor([0, 7], dtype=torch.int64)
    current_generation = torch.tensor([0, 8], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="request generation mismatch"):
        layout.bind_first_use_or_assert_generation(
            torch.tensor([1]),
            owner_generation,
            current_generation,
            torch.tensor([7]),
            torch.tensor([True]),
        )
    assert owner_generation.tolist() == [0, 7]


def test_success_consumes_acquire_authority_out_of_place():
    batch = _batch([1], [7], [True])
    original = batch.acquire_owner_mask
    Scheduler._complete_req_pool_owner_acquisition(batch)
    assert batch.acquire_owner_mask.tolist() == [False]
    assert batch.acquire_owner_mask.data_ptr() != original.data_ptr()
