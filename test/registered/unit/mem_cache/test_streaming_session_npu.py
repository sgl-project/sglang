from array import array

import pytest
import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.plan import TREE_TOKEN, TREE_WINDOW, TreeSlot
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="extra-a-test-npu")


@pytest.mark.parametrize(
    "start,window", [(127, 1024), (255, 1024), (256, 128), (191, 63), (7627, 7499)]
)
def test_private_swa_append_and_release(start, window):
    torch.npu.set_device(0)
    results = []
    with get_context().override_server_args(
        device="npu",
        page_size=128,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
    ):
        requests = ReqToTokenPool(
            size=4, max_context_len=16384, device="npu", enable_memory_saver=False
        )
        pool = SWAKVPool(
            size=16384,
            size_swa=16384,
            page_size=128,
            dtype=torch.float32,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="npu",
            token_to_kv_pool_class=NPUMHATokenToKVPool,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=16384,
            size_swa=16384,
            page_size=128,
            dtype=torch.float32,
            device="npu",
            kvcache=pool,
            need_sort=False,
            req_to_token_pool=requests,
        )
        allocator.swa_attn_allocator.free_pages = (
            allocator.swa_attn_allocator.free_pages.flip(0)
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=requests,
                token_to_kv_pool_allocator=allocator,
                page_size=128,
                sliding_window_size=window,
                enable_session_radix_cache=True,
                tree_slots=(
                    TreeSlot(component="FULL", family=TREE_TOKEN),
                    TreeSlot(component="SWA", family=TREE_WINDOW, window=window),
                ),
                tree_core_backend="python",
            )
        )
        kv_storage_of(requests, allocator).bind_window_retention(window, None)
        initial_free = (allocator.full_available_size(), allocator.swa_available_size())

        def allocate(prefix, length, last):
            prefix_cpu = torch.tensor([prefix], dtype=torch.int64, device="cpu")
            sequence_cpu = torch.tensor([length], dtype=torch.int64, device="cpu")
            return allocator.alloc_extend(
                prefix_cpu.npu(),
                prefix_cpu,
                sequence_cpu.npu(),
                sequence_cpu,
                torch.tensor([last], dtype=torch.int64, device="npu"),
                length - prefix,
            )

        other = allocate(0, 2, -1)
        other_swa = allocator.translate_loc_from_full_to_swa(other).clone()
        for layer, slots in [(1, other), (0, other_swa)]:
            pool.get_key_buffer(layer).view(-1, 1, 8)[slots] = 777
            pool.get_value_buffer(layer).view(-1, 1, 8)[slots] = 888

        session = Session(0, session_id=f"session-{start}-{window}", streaming=True)

        def make_request(length):
            request = Req(
                f"turn-{length}",
                "",
                array("q", range(length)),
                SamplingParams(max_new_tokens=1),
                session=session,
            )
            request.refresh_fill_ids()
            return request

        request = make_request(start)
        requests.alloc([request])
        row = request.kv.req_pool_idx
        full_slots = allocate(0, start, -1)
        requests.req_to_token[row, :start] = full_slots
        swa_slots = allocator.translate_loc_from_full_to_swa(full_slots).clone()
        assert not torch.equal(full_slots, swa_slots)
        for layer, slots in [(1, full_slots), (0, swa_slots)]:
            values = torch.arange(start, device="npu", dtype=torch.float32)[
                :, None, None
            ]
            pool.get_key_buffer(layer).view(-1, 1, 8)[slots] = values
            pool.get_value_buffer(layer).view(-1, 1, 8)[slots] = values + 1000
        request.kv.kv_allocated_len = request.kv.kv_committed_len = start
        cache.session.take(request)
        request.output_ids.append(start)
        request.finished_reason = FINISH_LENGTH(1)
        assert cache.session.try_cache_finished_req(request)
        evicted = False
        for prefix in range(start, start + 4):
            request = make_request(prefix + 1)
            matched = cache.session.try_match_prefix(
                MatchPrefixParams(key=list(range(prefix)), req=request)
            )
            if evicted:
                if window % 128:
                    assert matched is None
                    assert not cache.session.slots
                    assert not request.kv.holds_kv
                else:
                    assert matched.device_prefix_len == prefix // 128 * 128
                break
            assert matched is not None
            assert matched.device_prefix_len == prefix
            cache.session.take(request)
            batch = ScheduleBatch.__new__(ScheduleBatch)
            batch.tree_cache = cache
            batch.token_to_kv_pool_allocator = allocator
            batch.reqs = [request]
            batch.forward_mode = ForwardMode.EXTEND
            batch.prefix_lens = [prefix]
            batch.enable_overlap = False
            batch.maybe_evict_swa()
            evicted = request.kv.max_evicted_seqlen > 0
            kept_start = request.kv.max_evicted_seqlen
            old_full = requests.req_to_token[row, :prefix].long().clone()
            old_swa = allocator.translate_loc_from_full_to_swa(
                old_full[kept_start:]
            ).clone()
            before = [
                (
                    pool.get_key_buffer(layer).view(-1, 1, 8)[slots].clone(),
                    pool.get_value_buffer(layer).view(-1, 1, 8)[slots].clone(),
                )
                for layer, slots in [(1, old_full), (0, old_swa)]
            ]
            added = allocate(prefix, prefix + 1, int(old_full[-1].item()))
            added_swa = allocator.translate_loc_from_full_to_swa(added)
            assert added.numel() == added_swa.numel() == 1
            assert (
                int(added.item()) % 128 == int(added_swa.item()) % 128 == prefix % 128
            )
            requests.req_to_token[row, prefix] = added[0]
            for layer, slots in [(1, added), (0, added_swa)]:
                pool.get_key_buffer(layer).view(-1, 1, 8)[slots] = prefix
                pool.get_value_buffer(layer).view(-1, 1, 8)[slots] = prefix + 1000
            for index, (layer, slots) in enumerate([(1, old_full), (0, old_swa)]):
                assert torch.equal(
                    pool.get_key_buffer(layer).view(-1, 1, 8)[slots], before[index][0]
                )
                assert torch.equal(
                    pool.get_value_buffer(layer).view(-1, 1, 8)[slots], before[index][1]
                )
            request.kv.kv_allocated_len = request.kv.kv_committed_len = prefix + 1
            request.output_ids.append(prefix + 1)
            request.finished_reason = FINISH_LENGTH(1)
            assert cache.session.try_cache_finished_req(request)
            results.append(
                {
                    "start": start,
                    "window": window,
                    "prefix": prefix,
                    "full_slot": int(added.item()),
                    "swa_slot": int(added_swa.item()),
                    "evicted": evicted,
                }
            )
        if window in (63, 128, 7499):
            assert evicted, "the real eviction boundary must be exercised"
        cache.session.release_session(session.session_id)
        for layer, slots in [(1, other), (0, other_swa)]:
            assert bool(
                torch.all(pool.get_key_buffer(layer).view(-1, 1, 8)[slots] == 777)
            )
            assert bool(
                torch.all(pool.get_value_buffer(layer).view(-1, 1, 8)[slots] == 888)
            )
        allocator.free(other)
        assert (
            allocator.full_available_size(),
            allocator.swa_available_size(),
        ) == initial_free
        assert requests.available_size() == 4
    assert results


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
