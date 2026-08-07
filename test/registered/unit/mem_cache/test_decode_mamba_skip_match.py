"""
Unit tests for skip_mamba_match + cow_mamba control in decode-side radix cache
for mamba/SSM hybrid models (PD disaggregation).

Verifies:
1. skip_mamba_match=True matching works when mamba state is present
   (the flag is a no-op on nodes that already have mamba_value).
2. cow_mamba=False does NOT trigger mamba slot allocation during matching.
3. lock_ref stays balanced when matching and releasing with skip_mamba_match.
4. cow_mamba=True allocates mamba slot (guards single-server behavior).

Tombstone-straddling (matching past nodes where mamba_value=None) is tested
indirectly through the UnifiedRadixCache component pathway (not MambaRadixCache).
This file guards the skip_mamba_match flag plumbing in MambaRadixCache.

Usage:
    FLASHINFER_DISABLE_VERSION_CHECK=1 python -m pytest \
        test/registered/unit/mem_cache/test_decode_mamba_skip_match.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")

import unittest
from array import array

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device


def _setup_tree_and_allocator():
    """Create a MambaRadixCache with allocator for testing."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
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
    mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]

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
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
    )
    tree = MambaRadixCache(params=params)
    return tree, allocator, req_to_token_pool


class TestDecodeMambaSkipMatch(unittest.TestCase):
    """Test skip_mamba_match flag plumbing and cow_mamba control."""

    @classmethod
    def setUpClass(cls):
        cls.tree, cls.allocator, cls.req_to_token_pool = _setup_tree_and_allocator()
        cls.mamba_allocator = cls.req_to_token_pool.mamba_allocator

    def setUp(self):
        """Allocate a fresh request for each test."""
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        self.req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=sampling_params,
        )
        self.req_to_token_pool.alloc([self.req])
        self.addCleanup(self._cleanup_req)

    def _cleanup_req(self):
        if self.req.mamba_pool_idx is not None:
            self.req_to_token_pool.free_mamba_cache(self.req)
        self.req_to_token_pool.free(self.req)

    def _make_donor_and_insert(self, token_ids):
        """Insert a prefix with a real mamba_value from a donor request."""
        donor = Req(
            rid=999,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.req_to_token_pool.alloc([donor])
        kv_indices = self.allocator.alloc(len(token_ids))
        self.tree.insert(
            InsertParams(
                key=RadixKey(array("q", token_ids)),
                value=kv_indices[: len(token_ids)],
                mamba_value=donor.mamba_pool_idx.unsqueeze(0),
            )
        )
        return donor

    def test_skip_mamba_match_true_matches_with_mamba(self):
        """skip_mamba_match=True: matches normally when mamba_value IS present."""
        tok_ids = [1, 2, 3, 4]
        donor = self._make_donor_and_insert(tok_ids)

        result = self.tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tok_ids)),
                cow_mamba=False,
                req=self.req,
                skip_mamba_match=True,
            )
        )
        self.assertEqual(
            len(result.device_indices),
            len(tok_ids),
            "skip_mamba_match=True should match full prefix",
        )

        self.req_to_token_pool.free_mamba_cache(donor)
        self.req_to_token_pool.free(donor)

    def test_skip_mamba_match_false_matches_with_mamba(self):
        """skip_mamba_match=False: matches normally when mamba_value IS present."""
        tok_ids = [10, 11, 12, 13]
        donor = self._make_donor_and_insert(tok_ids)

        result = self.tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tok_ids)),
                cow_mamba=False,
                req=self.req,
                skip_mamba_match=False,
            )
        )
        self.assertEqual(
            len(result.device_indices),
            len(tok_ids),
            "skip_mamba_match=False should match full prefix when mamba is present",
        )

        self.req_to_token_pool.free_mamba_cache(donor)
        self.req_to_token_pool.free(donor)

    def test_cow_mamba_false_no_mamba_alloc(self):
        """cow_mamba=False: matching does not consume an extra mamba slot.

        In PD decode, state comes from prefill via RDMA, so even when the
        request already has a mamba slot (from req_to_token_pool.alloc),
        cow_mamba=False should not allocate a second one.
        """
        tok_ids = [20, 21, 22]
        donor = self._make_donor_and_insert(tok_ids)

        # Free self.req's mamba slot so we can check whether cow allocates one.
        self.req_to_token_pool.free_mamba_cache(self.req)
        self.req.mamba_pool_idx = None

        mamba_avail_before = self.mamba_allocator.available_size()

        result = self.tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tok_ids)),
                cow_mamba=False,
                req=self.req,
                skip_mamba_match=False,
            )
        )
        self.assertGreater(len(result.device_indices), 0)
        self.assertEqual(
            self.mamba_allocator.available_size(),
            mamba_avail_before,
            "cow_mamba=False must not allocate a mamba slot",
        )

        self.req_to_token_pool.free_mamba_cache(donor)
        self.req_to_token_pool.free(donor)

    def test_cow_mamba_true_allocates_mamba(self):
        """cow_mamba=True: matching allocates a mamba slot for COW.

        Guards the non-PD single-server behavior: a request without a mamba
        slot gets one allocated during prefix matching.
        """
        tok_ids = [30, 31, 32]
        donor = self._make_donor_and_insert(tok_ids)

        # Free self.req's mamba slot so cow allocates a fresh one.
        self.req_to_token_pool.free_mamba_cache(self.req)
        self.req.mamba_pool_idx = None

        mamba_avail_before = self.mamba_allocator.available_size()

        result = self.tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tok_ids)),
                cow_mamba=True,
                req=self.req,
                skip_mamba_match=False,
            )
        )
        self.assertGreater(len(result.device_indices), 0)
        self.assertEqual(
            self.mamba_allocator.available_size(),
            mamba_avail_before - 1,
            "cow_mamba=True allocates one mamba slot",
        )

        self.req_to_token_pool.free_mamba_cache(donor)
        self.req_to_token_pool.free(donor)

    def test_skip_mamba_match_lock_ref_balance(self):
        """lock-ref stays balanced with skip_mamba_match=True."""
        tok_ids = [40, 41, 42]
        donor = self._make_donor_and_insert(tok_ids)

        result = self.tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tok_ids)),
                cow_mamba=False,
                req=self.req,
                skip_mamba_match=True,
            )
        )
        self.assertGreater(len(result.device_indices), 0)

        full_prot_before = self.tree.full_protected_size()
        mamba_prot_before = self.tree.mamba_protected_size()

        self.tree.inc_lock_ref(result.last_device_node)
        self.tree.dec_lock_ref(result.last_device_node)

        self.assertEqual(
            self.tree.full_protected_size(),
            full_prot_before,
            "Full-KV protected size balanced after inc+dec",
        )
        self.assertEqual(
            self.tree.mamba_protected_size(),
            mamba_prot_before,
            "Mamba protected size balanced after inc+dec",
        )

        self.req_to_token_pool.free_mamba_cache(donor)
        self.req_to_token_pool.free(donor)


if __name__ == "__main__":
    unittest.main()
