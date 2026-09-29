"""CPU layout and allocation checks for the shared zigzag collectives."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from sglang.srt.layers.cp import base, padding, zigzag
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def _parallel(cp_size, rank, group=None):
    parallel = SimpleNamespace(
        attn_cp_rank=rank, attn_cp_size=cp_size, attn_cp_group=group
    )
    with (
        patch.object(base, "get_parallel", return_value=parallel),
        patch.object(zigzag, "get_parallel", return_value=parallel),
        patch.object(padding, "get_parallel", return_value=parallel),
    ):
        yield


def _make_shards(payload, cp_size, lengths, physical_padding):
    strategy = ZigzagCPStrategy(cp_size)
    shards, batches = [], []
    with (
        patch.object(zigzag, "get_device", return_value=SimpleNamespace(device="cpu")),
        patch.object(base, "_STRATEGY", strategy),
    ):
        for rank in range(cp_size):
            with _parallel(cp_size, rank):
                metadata = strategy.build_metadata(sum(lengths), lengths)
                if physical_padding:
                    pad_logical_token_to_physical(metadata)
                batch = SimpleNamespace(attn_cp_metadata=metadata)
                shards.append(strategy.shard_hidden_states(payload, batch))
                batches.append(batch)
    return strategy, shards, batches


def _noncontiguous(tensor):
    backing = tensor.new_empty((*tensor.shape[:-1], tensor.shape[-1] * 2))
    view = backing[..., ::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def _gathered_shards(shards, metadata):
    lengths = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    max_len = max(lengths)
    padded = shards[0].new_zeros((len(shards), max_len, *shards[0].shape[1:]))
    for rank, length in enumerate(lengths):
        padded[rank, :length].copy_(shards[rank][:length])
    return padded.flatten(0, 1)


@pytest.mark.parametrize("method", ["gather_hidden_states", "gather_kv_cache"])
@pytest.mark.parametrize("physical_padding", [False, True])
@pytest.mark.parametrize(
    "cp_size,lengths,shape",
    [
        (8, [512], (3,)),
        (8, [100, 33], (2, 3)),
        (4, [257, 16, 256], (3,)),
        (2, [7, 9, 24], (2, 3)),
    ],
)
def test_gather_restores_global_token_order(
    method, physical_padding, cp_size, lengths, shape
):
    num_tokens = sum(lengths)
    payload = torch.arange(num_tokens * torch.Size(shape).numel()).reshape(
        num_tokens, *shape
    )
    strategy, shards, batches = _make_shards(
        payload, cp_size, lengths, physical_padding
    )
    expected_gathered = _gathered_shards(shards, batches[0].attn_cp_metadata)

    for rank, (shard, batch) in enumerate(zip(shards, batches)):
        # Nonzero physical padding must not leak into reconstructed token order.
        logical = batch.attn_cp_metadata.per_rank_logical_token
        if logical is not None:
            shard[logical[rank] :].fill_(-1)
        local = _noncontiguous(shard)

        def gather(output, input):
            assert input.is_contiguous()
            torch.testing.assert_close(input, expected_gathered.chunk(cp_size)[rank])
            output.copy_(expected_gathered)

        group = SimpleNamespace(all_gather_into_tensor=gather)
        with _parallel(cp_size, rank, group):
            actual = getattr(strategy, method)(local, batch)
        torch.testing.assert_close(actual, payload, rtol=0, atol=0)


@pytest.mark.parametrize("physical_padding", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_symmetric_gather_allocates_equal_send_and_receive_buffers_in_pool(
    physical_padding, strided
):
    cp_size, lengths = 4, [17, 19]
    payload = torch.arange(sum(lengths) * 6).reshape(-1, 2, 3)
    strategy, shards, batches = _make_shards(
        payload, cp_size, lengths, physical_padding
    )
    gathered = _gathered_shards(shards, batches[0].attn_cp_metadata)
    send_shape = tuple(gathered.chunk(cp_size)[0].shape)
    original_new_empty, original_empty = torch.Tensor.new_empty, torch.empty
    in_pool = False
    allocations = {}

    @contextmanager
    def pool(group):
        nonlocal in_pool
        in_pool = True
        try:
            yield
        finally:
            in_pool = False

    def record_new_empty(tensor, *args, **kwargs):
        result = original_new_empty(tensor, *args, **kwargs)
        allocations[result.data_ptr()] = (in_pool, tuple(result.shape))
        return result

    def record_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations[result.data_ptr()] = (in_pool, tuple(result.shape))
        return result

    for rank, (shard, batch) in enumerate(zip(shards, batches)):
        local = _noncontiguous(shard) if strided else shard
        allocations.clear()

        def gather(output, input):
            assert input.is_contiguous()
            assert not in_pool
            assert input.data_ptr() != local.data_ptr()
            assert allocations[input.data_ptr()] == (True, send_shape)
            assert allocations[output.data_ptr()] == (True, tuple(gathered.shape))
            assert torch.equal(input, gathered.chunk(cp_size)[rank])
            output.copy_(gathered)

        group = SimpleNamespace(all_gather_into_tensor=gather)
        with (
            _parallel(cp_size, rank, group),
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ),
            patch.object(zigzag, "is_symmetric_memory_enabled", return_value=True),
            patch.object(zigzag, "is_allocation_symmetric", return_value=True),
            patch.object(zigzag, "use_symmetric_memory", side_effect=pool),
            patch.object(torch.Tensor, "new_empty", record_new_empty),
            patch.object(torch, "empty", record_empty),
        ):
            actual = strategy.gather_kv_cache(local, batch)
        torch.testing.assert_close(actual, payload, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
