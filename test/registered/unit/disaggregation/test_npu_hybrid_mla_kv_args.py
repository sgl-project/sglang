"""NPU PD registration uses each hybrid pool's token-KV layout and draft layer count."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    get_kv_transfer_buf_infos,
    setup_state_kv_args,
)
from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_MLA = dict(size=16, page_size=4, dtype=torch.bfloat16, device="cpu")
_TARGET_LAYERS = 6


def _recurrent_state(layer_ids):
    """The recurrent-state pool a hybrid pool registers beside its token KV."""
    state = torch.zeros(len(layer_ids), 4, 8)
    return SimpleNamespace(
        get_contiguous_buf_infos=lambda: (
            [state.data_ptr()],
            [state.nbytes],
            [state[0].nbytes],
        ),
        get_state_dim_per_tensor=lambda: [8],
        get_state_layer_ids=lambda: list(layer_ids),
        get_state_slice_outer_counts=lambda: [1],
        get_state_conv_shard_groups=lambda: [None],
    )


def _npu_mla(layer_num, **kwargs):
    return NPUMLATokenToKVPool(
        **_MLA,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        layer_num=layer_num,
        enable_memory_saver=False,
        **kwargs,
    )


def _hybrid_npu_mla(full_layer_ids, state_layer_ids):
    return HybridLinearKVPool(
        **_MLA,
        head_num=1,
        head_dim=16,
        full_attention_layer_ids=list(full_layer_ids),
        mamba_pool=_recurrent_state(state_layer_ids),
        use_mla=True,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        full_kv_pool_class=NPUMLATokenToKVPool,
    )


def _register(target, draft, *, mooncake_page_dcp=False):
    allocator = SimpleNamespace(storage=None, state=None, get_kvcache=lambda: target)
    storage = StaticKVStorage(
        None,
        allocator,
        draft_token_to_kv_pool=draft,
        num_hidden_layers=_TARGET_LAYERS,
    )
    kv_args = KVArgs()
    kv_args.kv_data_ptrs, kv_args.kv_data_lens, kv_args.kv_item_lens = (
        get_kv_transfer_buf_infos(storage)
    )
    if draft is not None:
        draft_ptrs, draft_lens, draft_item_lens = draft.get_contiguous_buf_infos()
        kv_args.kv_data_ptrs += draft_ptrs
        kv_args.kv_data_lens += draft_lens
        kv_args.kv_item_lens += draft_item_lens
        kv_args.num_draft_entries = len(draft_ptrs)
    # Use the NPU layer-ID layout for these CPU tensors.
    with patch.object(memory_pool, "_is_npu", True):
        kv_args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=target,
            draft_token_to_kv_pool=draft,
            num_draft_entries=kv_args.num_draft_entries,
            num_hidden_layers=_TARGET_LAYERS,
            storage=storage,
        )
    setup_state_kv_args(
        kv_args,
        storage,
        draft_token_to_kv_pool=draft,
        total_kv_layers=_TARGET_LAYERS,
        mooncake_page_dcp=mooncake_page_dcp,
    )
    return kv_args


class TestNpuHybridMlaKvArgs(CustomTestCase):
    def setUp(self):
        parallel = patch.object(
            memory_pool_npu,
            "get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
        )
        parallel.start()
        self.addCleanup(parallel.stop)
        # Full-attention layers 0, 2, 4; recurrent layers 1, 3, 5.
        self.target = _hybrid_npu_mla((0, 2, 4), (1, 3, 5))

    def _assert_target_registered(self, kv_args):
        self.assertEqual(kv_args.kv_buf_groups, 2)
        self.assertEqual(kv_args.hidden_kv_layers, _TARGET_LAYERS)
        self.assertTrue(kv_args.is_hybrid_mla_backend)
        self.assertIn(StateType.MAMBA, kv_args.state_types)

    def test_hybrid_dsa_constructs_the_npu_kpool(self):
        pool = HybridLinearKVPool(
            **_MLA,
            head_num=1,
            head_dim=16,
            full_attention_layer_ids=[0],
            mamba_pool=_recurrent_state([1]),
            use_mla=True,
            kv_lora_rank=16,
            qk_rope_head_dim=0,
            use_dsa=True,
            index_head_dim=128,
            kv_cache_dim=16,
            index_kpool=4,
            index_kpool_compress=True,
            max_running_requests=2,
            full_kv_pool_class=NPUMLATokenToKVPool,
        )
        self.assertIsInstance(pool.full_kv_pool, NPUMLATokenToKVPool)
        self.assertTrue(pool.full_kv_pool.kpool_use_compress)

    def test_a_hybrid_draft_counts_the_layers_of_its_token_kv_view(self):
        """Hybrid drafts expose the KV layer count through their token view;
        PD registration must not read a missing layer count from the wrapper."""
        draft = _hybrid_npu_mla((0, 2), (1,))
        self.assertIsNot(draft.layer_router.wrapped_view(), draft)
        kv_args = _register(self.target, draft)
        self._assert_target_registered(kv_args)
        self.assertEqual(kv_args.draft_kv_layers, 2)

    def test_a_plain_draft_counts_its_own_layers(self):
        kv_args = _register(self.target, _npu_mla(layer_num=1))
        self._assert_target_registered(kv_args)
        self.assertEqual(kv_args.draft_kv_layers, 1)

    def test_no_draft_registers_no_draft_layers(self):
        kv_args = _register(self.target, None)
        self._assert_target_registered(kv_args)
        self.assertEqual(kv_args.draft_kv_layers, 0)

    def test_mooncake_page_dcp_registers_plain_pp_local_buffer_groups(self):
        target = _npu_mla(
            layer_num=2,
            start_layer=2,
            end_layer=4,
            index_head_dim=8,
            indexer_layer_ids=(3,),
        )
        kv_args = _register(target, _npu_mla(layer_num=1), mooncake_page_dcp=True)

        self.assertEqual(kv_args.kv_layer_ids, [])
        # Two latent K buffers, two RoPE buffers, one index-K and two draft buffers.
        self.assertEqual(list(kv_args.dcp_page_layer_ids), [2, 3, 2, 3, 3, 6, 6])
        self.assertEqual(
            list(kv_args.dcp_page_global_slots),
            [False, False, False, False, True, True, True],
        )

    def test_mooncake_page_dcp_preserves_hybrid_full_attention_layer_ids(self):
        kv_args = _register(self.target, _npu_mla(layer_num=1), mooncake_page_dcp=True)

        self.assertEqual(list(kv_args.dcp_page_layer_ids), [0, 2, 4, 0, 2, 4, 6, 6])
        self.assertEqual(list(kv_args.dcp_page_layer_ids), kv_args.kv_layer_ids)
        self.assertEqual(
            list(kv_args.dcp_page_global_slots),
            [False, False, False, False, False, False, True, True],
        )


if __name__ == "__main__":
    unittest.main()
