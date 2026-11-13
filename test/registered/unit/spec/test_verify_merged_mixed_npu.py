"""Length, compressed-slot and result bookkeeping for Ascend mixed verification."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_build_dsv4_verify_bundle,
)
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult, allocate_distinct_stream
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoDynamicOutputIndex(TorchDispatchMode):
    """Fail on boolean indexing, which synchronizes dynamic sizes on Ascend."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.index.Tensor:
            assert all(i is None or i.dtype != torch.bool for i in args[1])
        assert func not in (
            torch.ops.aten.nonzero.default,
            torch.ops.aten.masked_select.default,
        )
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("native_handle", ["cuda_stream", "npu_stream"])
def test_d2h_stream_redraw_skips_scheduler_forward_and_result_copy_aliases(
    native_handle,
):
    schedule, forward, result_copy, length_copy = [
        SimpleNamespace(**{native_handle: i}) for i in (1, 2, 3, 4)
    ]
    device_module = SimpleNamespace(
        Stream=Mock(side_effect=[forward, schedule, result_copy, length_copy])
    )
    assert (
        allocate_distinct_stream(device_module, (schedule, forward, result_copy))
        is length_copy
    )


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
        ([126, 254], [1, 1]),
        ([127, 255], [1, 1]),
        ([127, 255], [0, 0]),
    ],
)
def test_mixed_compressed_slots_match_completed_groups(prefixes, queries):
    bs = len(prefixes)
    full = torch.tensor(
        [
            (i + 1) * 1024 + p
            for i, (start, count) in enumerate(zip(prefixes, queries))
            for p in range(start, start + count)
        ],
        dtype=torch.int64,
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
    with _NoDynamicOutputIndex():
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
@pytest.mark.parametrize("private_stream", [True, False])
@pytest.mark.parametrize("mixed", [True, False])
def test_relay_reads_exact_lengths_without_copying_schedule_gather(
    needs_cpu_seq_lens, private_stream, mixed
):
    relay = FutureMap.__new__(FutureMap)
    relay.device = torch.device("cpu")
    relay.publish_ready = Mock() if private_stream else None
    relay.fwd_prepare_d2h_stream = Mock() if private_stream else None
    relay.new_seq_lens_cpu_pinned = torch.empty(5, dtype=torch.int64)
    relay.needs_cpu_seq_lens = needs_cpu_seq_lens
    relay.new_seq_lens_buf = torch.tensor([0, 0, 203, 0, 105])
    relay._publish_fresh = True
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=torch.tensor([4, 2])),
        num_prefill_rows=1 if mixed else None,
        forward_mode=ForwardMode.MIXED if mixed else ForwardMode.DECODE,
        req_pool_indices_cpu=torch.tensor([1, 4, 2] if mixed else [4, 2]),
        seq_lens=torch.tensor([200, 100, 200]),
        prefix_lens=[190, 100, 200],
        extend_lens=[10, 6, 6],
    )
    device_module = SimpleNamespace(stream=Mock(side_effect=lambda _: nullcontext()))
    cpu_guard = (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("schedule D2H"))
        if private_stream
        else nullcontext()
    )
    with (
        cpu_guard,
        patch("torch.get_device_module", return_value=device_module),
        patch("sglang.srt.managers.overlap_utils._DEBUG_ASSERT", False),
    ):
        relay.resolve_seq_lens_cpu(batch)
    expected = [200, 105, 203] if mixed else [105, 203]
    assert batch.seq_lens.tolist() == expected
    if not mixed and not needs_cpu_seq_lens:
        assert batch.seq_lens_cpu is None and batch.seq_lens_sum is None
        if private_stream:
            relay.fwd_prepare_d2h_stream.synchronize.assert_not_called()
        return
    assert batch.seq_lens_cpu.tolist() == expected
    assert batch.seq_lens_sum == sum(expected)
    if private_stream:
        relay.fwd_prepare_d2h_stream.wait_event.assert_called_once_with(
            relay.publish_ready
        )
        relay.fwd_prepare_d2h_stream.synchronize.assert_called_once_with()
        device_module.stream.assert_called_once_with(relay.fwd_prepare_d2h_stream)
        # A later copy must not change this batch's exact host lengths.
        relay.new_seq_lens_cpu_pinned.fill_(-1)
        assert batch.seq_lens_cpu.tolist() == expected


@pytest.mark.parametrize("private_stream", [True, False])
def test_legacy_mixed_tail_keeps_lengths_and_reserved_slots_consistent(private_stream):
    relay = FutureMap.__new__(FutureMap)
    relay.device = torch.device("cpu")
    relay._deferred_relay = None
    relay.publish_ready = Mock() if private_stream else None
    relay.fwd_prepare_d2h_stream = Mock() if private_stream else None
    relay.new_seq_lens_cpu_pinned = torch.empty(5, dtype=torch.int64)
    relay.new_seq_lens_buf = torch.tensor([0, 0, 203, 0, 105])
    relay.req_to_token = torch.arange(5 * 256).reshape(5, 256)
    batch = SimpleNamespace(
        mix_running_indices=torch.tensor([4, 2]),
        mix_running_indices_cpu=torch.tensor([4, 2]),
        seq_lens=torch.tensor([200, 100, 200]),
        seq_lens_cpu=torch.tensor([200, 100, 200]),
        out_cache_loc=torch.tensor([10, 20, 30]),
        prefix_lens=[190, 100, 200],
    )
    cpu_guard = (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("schedule D2H"))
        if private_stream
        else nullcontext()
    )
    with (
        cpu_guard,
        patch(
            "torch.get_device_module",
            return_value=SimpleNamespace(stream=lambda _: nullcontext()),
        ),
    ):
        relay.resolve_mixed_spec_tails(batch)
    assert batch.seq_lens.tolist() == batch.seq_lens_cpu.tolist() == [200, 106, 204]
    assert batch.seq_lens_sum == 510
    assert batch.prefix_lens == [190, 105, 203]
    assert batch.out_cache_loc.tolist() == [10, 4 * 256 + 105, 2 * 256 + 203]


@pytest.mark.parametrize("decode_graph", [None, False, True])
def test_mixed_result_uses_prefill_and_speculative_decode_processors(decode_graph):
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
        mixed_decode_can_run_cuda_graph=decode_graph,
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
        assert output.can_run_cuda_graph is bool(decode_graph)
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
    assert output.can_run_cuda_graph is False
    assert result.num_correct_drafts == 6
    assert result.num_correct_drafts_per_req_cpu == [1, 5]
    assert batch.forward_mode == ForwardMode.MIXED
    processor.metrics_collector.increment_mixed_verify_step.assert_called_once()
    assert processor.metrics_collector.increment_mixed_decode_graph_step.call_count == (
        1 if decode_graph else 0
    )
