"""Length, compressed-slot and result bookkeeping for Ascend mixed verification."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_build_dsv4_verify_bundle,
)
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_mixed_logits_keep_verify_tokens_and_full_draft_hidden():
    # One three-token prefill and two two-token verify rows.
    hidden = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    selected = torch.tensor([2, 3, 4, 5, 6])
    aux = torch.cat([hidden + 100, hidden + 200], dim=1)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.MIXED,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        draft_extend_select_index=selected,
    )
    pruned, before_norm, aux_pruned, sample_indices, _, _ = (
        LogitsProcessor._get_pruned_states(None, hidden, None, aux, metadata)
    )
    torch.testing.assert_close(pruned, hidden[selected])
    stored = LogitsProcessor._get_hidden_states_to_store(
        None,
        hidden,
        None,
        aux,
        pruned,
        before_norm,
        aux_pruned,
        sample_indices,
        metadata,
    )
    torch.testing.assert_close(stored, aux)


@pytest.mark.parametrize(
    "prefixes,queries",
    [
        ([0, 127], [129, 6]),
        ([123, 127, 255], [5, 6, 6]),
        ([128, 129], [127, 2]),
        ([9, 8], [1, 1]),
    ],
)
def test_mixed_compressed_slots_match_completed_groups(prefixes, queries):
    bs = len(prefixes)
    full = torch.tensor(
        [
            (i + 1) * 1024 + p
            for i, (start, count) in enumerate(zip(prefixes, queries))
            for p in range(start, start + count)
        ]
    )
    c128_pages = torch.arange(bs * 8).reshape(bs, 8) + 10
    batch = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_c128_sidecar=c128_pages,
            c128_page_size=2,
        ),
        req_pool_indices_cpu=torch.arange(bs),
        out_cache_loc=full,
        out_cache_loc_dsv4=DSV4OutCacheLoc(full, full, full, full),
        token_to_kv_pool_allocator=SimpleNamespace(
            translate_loc_from_full_to_swa=lambda loc: loc + 8192,
        ),
    )
    bundle = maybe_build_dsv4_verify_bundle(
        batch,
        6,
        live_seq_lens_cpu=torch.tensor(prefixes),
        query_lens_cpu=queries,
    )
    expected_c4, expected_c128 = [], []
    for i, (start, count) in enumerate(zip(prefixes, queries)):
        for end in range(start + 1, start + count + 1):
            if end % 4 == 0:
                expected_c4.append(((i + 1) * 1024 + end - 1) // 4)
            if end % 128 == 0:
                compressed_pos = end // 128 - 1
                page = int(c128_pages[i, compressed_pos // 2])
                expected_c128.append(page * 2 + compressed_pos % 2)
    assert bundle.out_c4_loc.tolist() == expected_c4
    assert bundle.out_c128_loc.tolist() == expected_c128
    torch.testing.assert_close(bundle.out_full_loc, full)
    torch.testing.assert_close(bundle.out_swa_loc, full + 8192)


@pytest.mark.parametrize("needs_cpu_seq_lens", [True, False])
def test_mixed_relay_preserves_prefill_and_reads_exact_tail_lengths(needs_cpu_seq_lens):
    relay = SimpleNamespace(
        publish_ready=None,
        needs_cpu_seq_lens=needs_cpu_seq_lens,
        new_seq_lens_buf=torch.tensor([0, 0, 203, 0, 105]),
    )
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=torch.tensor([4, 2])),
        num_prefill_rows=1,
        forward_mode=ForwardMode.MIXED,
        seq_lens=torch.tensor([200, 100, 200]),
        prefix_lens=[190, 100, 200],
        extend_lens=[10, 6, 6],
    )
    FutureMap.resolve_seq_lens_cpu(relay, batch)
    assert batch.seq_lens.tolist() == [200, 105, 203]
    assert batch.seq_lens_cpu.tolist() == [200, 105, 203]
    assert batch.seq_lens_sum == 508


def test_mixed_result_uses_prefill_and_speculative_decode_processors():
    batch = SimpleNamespace(
        reqs=["prefill", "decode0", "decode1"],
        num_prefill_rows=1,
        forward_mode=ForwardMode.MIXED,
        decoding_reqs=["decode0", "decode1"],
        extend_lens=[128, 6, 6],
        prefix_lens=[0, 127, 255],
    )
    result = GenerationBatchResult(
        next_token_ids=torch.arange(13),
        accept_lens=torch.tensor([2, 6]),
        block_accept_lens=torch.tensor([2, 6]),
        speculative_num_draft_tokens=6,
    )
    processor = SimpleNamespace(
        process_batch_result_prefill=Mock(),
        process_batch_result_decode=Mock(),
        metrics_collector=Mock(),
    )

    def decode(dec, output):
        assert dec.reqs == ["decode0", "decode1"]
        assert dec.forward_mode == ForwardMode.DECODE
        assert output.next_token_ids.tolist() == list(range(1, 13))
        assert output.accept_lens.tolist() == [2, 6]
        output.num_correct_drafts = 6
        output.num_correct_drafts_per_req_cpu = [1, 5]

    processor.process_batch_result_decode.side_effect = decode
    with patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.get_observability",
        return_value=SimpleNamespace(enable_metrics=True),
    ):
        SchedulerBatchResultProcessor.process_batch_result_mixed(
            processor, batch, result
        )
    pre, output = processor.process_batch_result_prefill.call_args.args
    assert pre.reqs == ["prefill"]
    assert pre.decoding_reqs is None
    assert pre.forward_mode == ForwardMode.EXTEND
    assert output.next_token_ids.tolist() == [0]
    assert output.accept_lens is None
    assert result.num_correct_drafts == 6
    assert result.num_correct_drafts_per_req_cpu == [1, 5]
    assert batch.forward_mode == ForwardMode.MIXED
    processor.metrics_collector.increment_mixed_verify_step.assert_called_once()
