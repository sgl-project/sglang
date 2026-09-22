import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")
register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


def _event_hashes(events):
    return [block_hash for event in events for block_hash in event.block_hashes]


class TestMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_hybrid_linear_kv_pool(self):
        size = 16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            enable_memory_saver=False,
            mamba_pool=None,
        )
        assert pool._transfer_full_attention_id(global_interval - 1) == 0
        assert pool._transfer_full_attention_id(2 * global_interval - 1) == 1
        with self.assertRaises(ValueError) as context:
            pool._transfer_full_attention_id(1)
        self.assertIn(
            "layer_id=1 not in full attention layers:", str(context.exception)
        )

    def test_hybrid_linear_kv_pool_npu_layer_ids_match_buffer_groups(self):
        pool = object.__new__(HybridLinearKVPool)
        pool.full_attention_layer_id_mapping = {3: 0, 7: 1}
        pool.use_mla = True

        with patch("sglang.srt.mem_cache.memory_pool._is_npu", False):
            self.assertEqual(pool.get_kv_layer_ids(), [3, 7])

        with patch("sglang.srt.mem_cache.memory_pool._is_npu", True):
            for group_count in (2, 3):
                with self.subTest(group_count=group_count):
                    pool.full_kv_pool = SimpleNamespace(
                        get_contiguous_buf_infos=lambda: (
                            list(range(2 * group_count)),
                            [],
                            [],
                        )
                    )
                    self.assertEqual(pool.get_kv_layer_ids(), [3, 7] * group_count)

    def test_mamba_pool(self):
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = get_device()
        global_interval = 4
        num_layers = 48
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=16,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )

        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=mamba_layers,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )

        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_allocator.available_size() == mamba_cache_size

        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=1,
        )
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=sampling_params,
        )

        # alloc req
        req_to_token_pool.alloc([req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert (
            req_to_token_pool.mamba_allocator.available_size() == mamba_cache_size - 1
        )

        # free req
        req_to_token_pool.free_mamba_cache(req)
        req_to_token_pool.free(req)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_allocator.available_size() == mamba_cache_size

        # alloc req without free mamba cache
        req.kv.mamba_pool_idx = None
        req_to_token_pool.alloc([req])
        req_to_token_pool.free(req)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert (
            req_to_token_pool.mamba_allocator.available_size() == mamba_cache_size - 1
        )

        # alloc again
        req_to_token_pool.alloc([req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert (
            req_to_token_pool.mamba_allocator.available_size() == mamba_cache_size - 1
        )

    def test_mamba_pool_deduplicated_conv_window_axis(self):
        class WindowFirstMambaPool(MambaPool):
            conv_window_axis = 0

        num_mamba_layers = 2
        spec_state_size = 3
        speculative_num_draft_tokens = 4
        window_size = 3
        conv_dim = 5

        pool = object.__new__(WindowFirstMambaPool)
        # Bypasses __init__, so set the device the allocator reads directly.
        pool.device = get_device()
        physical, view = pool._allocate_deduplicated_conv_window(
            conv_shape=(window_size, conv_dim),
            num_mamba_layers=num_mamba_layers,
            spec_state_size=spec_state_size,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            conv_dtype=torch.float32,
        )

        shared_window_size = speculative_num_draft_tokens + window_size - 1
        self.assertEqual(
            physical.shape,
            (
                num_mamba_layers,
                spec_state_size + 1,
                shared_window_size,
                conv_dim,
            ),
        )
        self.assertEqual(
            view.shape,
            (
                num_mamba_layers,
                spec_state_size + 1,
                speculative_num_draft_tokens,
                window_size,
                conv_dim,
            ),
        )

        physical.copy_(
            torch.arange(
                physical.numel(), dtype=physical.dtype, device=physical.device
            ).reshape_as(physical)
        )
        for step in range(speculative_num_draft_tokens):
            torch.testing.assert_close(
                view[:, :, step],
                physical[:, :, step : step + window_size],
            )
        torch.testing.assert_close(view[:, :, :-1, 1:], view[:, :, 1:, :-1])

        view[0, 0, 0, 1, 0] = -1
        self.assertEqual(view[0, 0, 1, 0, 0].item(), -1)

    def _setup_pools(self):
        """Build the hybrid req/KV pools and an allocator for pool-level tests."""
        server_args = ServerArgs(model_path="dummy", page_size=1)
        # The mamba pool reads mamba_cache_chunk_size, whose property otherwise
        # loads the HF config for self.model_path — impossible for the dummy model.
        # Mirror the property's default for a dummy HF config: FLA_CHUNK_SIZE.
        server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
        set_global_server_args_for_scheduler(server_args)
        size = 128
        dtype = torch.bfloat16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=mamba_layers,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )

        def make_dummy_req():
            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            req = Req(
                rid=0,
                origin_input_text="",
                origin_input_ids=array("q"),
                sampling_params=sampling_params,
            )
            req_to_token_pool.alloc([req])
            return req

        return allocator, req_to_token_pool, make_dummy_req

    # Qwen4-Exp's PLE N-gram window is 2 wide (ngram_size=3) and its "no history"
    # sentinel is the eos id; pick a recognisable one for the tests.
    NGRAM_CONTEXT_LEN = 2
    NGRAM_EOS = 248044

    def _setup_pool_with_ngram(self, ngram_context_len: int = NGRAM_CONTEXT_LEN):
        server_args = ServerArgs(model_path="dummy", page_size=1)
        server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
        set_global_server_args_for_scheduler(server_args)
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            cache_params = Mamba2CacheParams(shape=shape, layers=[0])
        return HybridReqToTokenPool(
            size=10,
            mamba_size=20,
            mamba_spec_state_size=10,
            max_context_len=128,
            device=get_device(),
            enable_memory_saver=False,
            cache_params=cache_params,
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
            ngram_context_len=ngram_context_len,
            ngram_eos_token_id=self.NGRAM_EOS,
        )

    # Slot-sibling parity: each test below pins one way a mamba slot changes owner.

    def test_slot_siblings_registered(self):
        """Enabled PLE side states register on the pool that owns the slots;
        disabled ones stay off so the host-offload payload keeps its legacy shape."""
        _, base_pool, _ = self._setup_pools()
        # The default hybrid setup has no PLE config: no siblings ride along.
        self.assertEqual(len(base_pool.mamba_pool._slot_siblings), 0)
        pool = self._setup_pool_with_ngram()
        self.assertEqual(len(pool.mamba_pool._slot_siblings), 1)

    def test_ngram_clear_slots_resets_window(self):
        """A recycled slot must not carry its previous owner's N-gram window;
        the sibling reset must ride the same deferred ``clear_slots`` call."""
        pool = self._setup_pool_with_ngram()
        mamba_pool = pool.mamba_pool
        ngram = pool.ngram_pool

        victim = pool.mamba_allocator.alloc(1)
        ngram.context[victim.long()] = 777  # poison, as a real request's history
        mamba_pool.clear_slots(victim)
        self.assertTrue(
            torch.all(ngram.context[victim.long()] == self.NGRAM_EOS),
            f"clear_slots left a dirty N-gram row: {ngram.context[victim.long()]}",
        )

    def test_ngram_copy_from_copies_window(self):
        """copy_from carries the window, so radix cow gets the cached prefix's state."""
        pool = self._setup_pool_with_ngram()
        mamba_pool = pool.mamba_pool
        ngram = pool.ngram_pool

        src = pool.mamba_allocator.alloc(1)
        dst = pool.mamba_allocator.alloc(1)
        window = torch.tensor(
            [[55, 66]], dtype=ngram.context.dtype, device=ngram.context.device
        )
        ngram.context[src.long()] = window

        mamba_pool.copy_from(src, dst)
        self.assertTrue(
            torch.equal(ngram.context[dst.long()], window),
            f"copy_from lost the N-gram window: got {ngram.context[dst.long()]}",
        )

    def test_ngram_cpu_offload_roundtrip(self):
        """The window survives a host offload round-trip along with mamba state."""
        pool = self._setup_pool_with_ngram()
        mamba_pool = pool.mamba_pool
        ngram = pool.ngram_pool

        indices = pool.mamba_allocator.alloc(2)
        window = torch.tensor(
            [[11, 12], [13, 14]],
            dtype=ngram.context.dtype,
            device=ngram.context.device,
        )
        ngram.context[indices.long()] = window

        saved = mamba_pool.get_cpu_copy(indices)
        ngram.context[indices.long()] = self.NGRAM_EOS  # simulate slot reuse
        mamba_pool.load_cpu_copy(saved, indices)

        self.assertTrue(
            torch.equal(ngram.context[indices.long()], window),
            f"offload round-trip lost the window: got {ngram.context[indices.long()]}",
        )

    def test_ngram_pool_absent_keeps_legacy_offload_shape(self):
        """Disabled pool stays inert: legacy 2-tuple offload payload, no sibling."""
        pool = self._setup_pool_with_ngram(ngram_context_len=0)
        self.assertIsNone(pool.ngram_pool.context)
        self.assertEqual(len(pool.mamba_pool._slot_siblings), 0)

        src = pool.mamba_allocator.alloc(1)
        payload = pool.mamba_pool.get_cpu_copy(src)
        self.assertEqual(len(payload), 2)
        pool.mamba_pool.load_cpu_copy(payload, src)

    def test_mamba_track_aligned_lens_math(self):
        """Floor division must swallow the scheduler's `aligned + 1` (_force_track_h),
        or the PLE side states snapshot one token past the mamba state."""
        from types import SimpleNamespace

        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        def aligned_for(chunk_size, track_seqlens, prefix_lens):
            server_args = ServerArgs(model_path="dummy", page_size=1)
            server_args._mamba_cache_chunk_size = chunk_size
            set_global_server_args_for_scheduler(server_args)
            fake = SimpleNamespace(
                mamba_track_mask=torch.tensor([True] * len(track_seqlens)),
                mamba_track_seqlens=torch.tensor(track_seqlens, dtype=torch.int64),
                extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
            )
            return ForwardBatch.mamba_track_aligned_lens(fake).tolist()

        # normal: track_seqlens = prefix + extend_input_len
        self.assertEqual(
            aligned_for(64, [100 + 64, 100 + 100, 100 + 127], [100, 100, 100]),
            [64, 64, 64],
        )
        # _force_track_h with chunk > 64: track_seqlens = aligned + 1
        self.assertEqual(aligned_for(128, [100 + 128 + 1], [100]), [128])
        self.assertEqual(aligned_for(128, [100 + 256 + 1], [100]), [256])
        # branching point inside the chunk, also handed over as +1
        self.assertEqual(aligned_for(64, [100 + 64 + 1], [100]), [64])
        # a masked-off row carries -1 and must come out non-positive, so the
        # caller's clamp(min=0) routes it harmlessly
        self.assertLessEqual(aligned_for(64, [-1], [100])[0], 0)

        # restore the chunk size the rest of the suite expects
        server_args = ServerArgs(model_path="dummy", page_size=1)
        server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
        set_global_server_args_for_scheduler(server_args)

    def test_mamba_pool_cpu_offload(self):
        """MambaPool.get_cpu_copy / load_cpu_copy round-trips conv and temporal state."""
        _, req_to_token_pool, _ = self._setup_pools()
        mamba_pool = req_to_token_pool.mamba_pool
        n = 3
        indices = req_to_token_pool.mamba_allocator.alloc(n)
        self.assertIsNotNone(indices)

        # Write known sentinel values at the allocated slots.
        for conv in mamba_pool.mamba_cache.conv:
            conv[:, indices] = 1.0
        mamba_pool.mamba_cache.temporal[:, indices] = 2.0

        # Save to CPU.
        conv_cpu, temporal_cpu = mamba_pool.get_cpu_copy(indices)

        # Verify CPU tensors match what was written.
        for i, conv in enumerate(mamba_pool.mamba_cache.conv):
            expected = conv[:, indices].cpu()
            self.assertTrue(
                torch.allclose(conv_cpu[i].float(), expected.float()),
                f"conv[{i}] CPU copy mismatch",
            )
        expected_t = mamba_pool.mamba_cache.temporal[:, indices].cpu()
        self.assertTrue(
            torch.allclose(temporal_cpu.float(), expected_t.float()),
            "temporal CPU copy mismatch",
        )

        # Zero out GPU slots and restore from CPU copy.
        for conv in mamba_pool.mamba_cache.conv:
            conv[:, indices] = 0.0
        mamba_pool.mamba_cache.temporal[:, indices] = 0.0

        mamba_pool.load_cpu_copy((conv_cpu, temporal_cpu), indices)

        # Verify restored values match the sentinels.
        for conv in mamba_pool.mamba_cache.conv:
            restored = conv[:, indices]
            self.assertTrue(
                torch.all(restored == 1.0),
                "conv not restored after load_cpu_copy",
            )
        self.assertTrue(
            torch.all(mamba_pool.mamba_cache.temporal[:, indices] == 2.0),
            "temporal not restored after load_cpu_copy",
        )

    def test_hybrid_kv_pool_cpu_offload(self):
        """HybridLinearKVPool.get_cpu_copy / load_cpu_copy saves and restores both
        the full-attention KV cache and Mamba state in a single round-trip."""
        allocator, req_to_token_pool, _ = self._setup_pools()
        mamba_pool = req_to_token_pool.mamba_pool
        hybrid_pool = allocator._kvcache  # HybridLinearKVPool

        self.assertIsInstance(hybrid_pool, HybridLinearKVPool)

        n_tokens = 4
        kv_indices = allocator.alloc(n_tokens)
        self.assertIsNotNone(kv_indices)
        mamba_indices = req_to_token_pool.mamba_allocator.alloc(1)
        self.assertIsNotNone(mamba_indices)

        # Write sentinel values into KV buffers (all full-attention layers).
        for layer_id in range(hybrid_pool.full_kv_pool.layer_num):
            hybrid_pool.full_kv_pool.k_buffer[layer_id][kv_indices] = 3.0
            hybrid_pool.full_kv_pool.v_buffer[layer_id][kv_indices] = 4.0

        # Write sentinel values into Mamba state.
        for conv in mamba_pool.mamba_cache.conv:
            conv[:, mamba_indices] = 5.0
        mamba_pool.mamba_cache.temporal[:, mamba_indices] = 6.0

        # --- Round-trip with Mamba indices provided ---
        cpu_copy = allocator.get_cpu_copy(kv_indices, mamba_indices=mamba_indices)
        kv_cpu, mamba_cpu = cpu_copy
        self.assertIsNotNone(
            mamba_cpu, "mamba_cpu should be saved when mamba_indices given"
        )

        # Zero out GPU.
        for layer_id in range(hybrid_pool.full_kv_pool.layer_num):
            hybrid_pool.full_kv_pool.k_buffer[layer_id][kv_indices] = 0.0
            hybrid_pool.full_kv_pool.v_buffer[layer_id][kv_indices] = 0.0
        for conv in mamba_pool.mamba_cache.conv:
            conv[:, mamba_indices] = 0.0
        mamba_pool.mamba_cache.temporal[:, mamba_indices] = 0.0

        allocator.load_cpu_copy(cpu_copy, kv_indices, mamba_indices=mamba_indices)

        # Verify KV restored.
        for layer_id in range(hybrid_pool.full_kv_pool.layer_num):
            self.assertTrue(
                torch.all(
                    hybrid_pool.full_kv_pool.k_buffer[layer_id][kv_indices] == 3.0
                ),
                f"k_buffer layer {layer_id} not restored",
            )
            self.assertTrue(
                torch.all(
                    hybrid_pool.full_kv_pool.v_buffer[layer_id][kv_indices] == 4.0
                ),
                f"v_buffer layer {layer_id} not restored",
            )

        # Verify Mamba restored.
        for conv in mamba_pool.mamba_cache.conv:
            self.assertTrue(
                torch.all(conv[:, mamba_indices] == 5.0),
                "conv not restored after load_cpu_copy",
            )
        self.assertTrue(
            torch.all(mamba_pool.mamba_cache.temporal[:, mamba_indices] == 6.0),
            "temporal not restored after load_cpu_copy",
        )

        # --- Without mamba_indices: mamba_cpu must be None ---
        cpu_copy_no_mamba = allocator.get_cpu_copy(kv_indices, mamba_indices=None)
        _, mamba_cpu_none = cpu_copy_no_mamba
        self.assertIsNone(
            mamba_cpu_none, "mamba_cpu should be None when mamba_indices=None"
        )


if __name__ == "__main__":
    unittest.main()
