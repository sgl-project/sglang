"""NPU PD registration uses each hybrid pool's token-KV layout and draft layer count."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import build_kv_layer_ids, setup_state_kv_args
from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.kv.alloc import runner_ids
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
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


def _npu_mla(layer_num, *, is_draft_worker=False):
    return NPUMLATokenToKVPool(
        **_MLA,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        layer_num=layer_num,
        enable_memory_saver=False,
        is_draft_worker=is_draft_worker,
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


def _register(target, draft):
    allocator = SimpleNamespace(storage=None, state=None, get_kvcache=lambda: target)
    storage = StaticKVStorage(
        None,
        allocator,
        draft_token_to_kv_pool=draft,
        num_hidden_layers=_TARGET_LAYERS,
    )
    kv_args = KVArgs()
    kv_args.num_draft_entries = (
        len(draft.get_contiguous_buf_infos()[0]) if draft is not None else 0
    )
    with patch("sglang.srt.mem_cache.memory_pool._is_npu", True):
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
        self.assertEqual(kv_args.kv_layer_ids[:6], [0, 2, 4] * 2)
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
        kv_args = _register(self.target, _npu_mla(layer_num=1, is_draft_worker=True))
        self._assert_target_registered(kv_args)
        self.assertEqual(kv_args.draft_kv_layers, 1)

    def test_no_draft_registers_no_draft_layers(self):
        kv_args = _register(self.target, None)
        self._assert_target_registered(kv_args)
        self.assertEqual(kv_args.draft_kv_layers, 0)

    def test_hybrid_writes_and_pd_use_token_interleaving_by_default(self):
        with (
            patch.object(
                memory_pool_npu,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=2, attn_dcp_rank=1),
            ),
            patch.object(runner_ids, "_is_npu", True),
            patch.object(
                runner_ids,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=True, dcp_size=2, dcp_rank=1),
            ),
        ):
            target = _hybrid_npu_mla((0, 2, 4), (1, 3, 5))
            draft = _npu_mla(layer_num=1, is_draft_worker=True)
            ids = runner_ids.RunnerIds(
                req_to_token=torch.zeros((1, 8), dtype=torch.int32),
                token_to_kv_pool_allocator=None,
                token_to_kv_pool=target,
                page_size=4,
                device="cpu",
            )
            kv_args = _register(target, draft)
        self.assertEqual(
            ids.space(IdSpaceKind.FULL).write(torch.arange(8)).tolist(),
            [-1, 0, -1, 1, -1, 2, -1, 3],
        )
        self._assert_target_registered(kv_args)
        self.assertEqual(target.full_kv_pool.dcp_size, 2)
        self.assertEqual(
            target.full_kv_pool.get_contiguous_buf_infos()[2], [128] * 3 + [64] * 3
        )
        self.assertEqual(draft.get_contiguous_buf_infos()[2], [256, 128])

    def test_dsa_indexer_registers_as_state_with_draft_layer_ids(self):
        def dsa_pool(layers, draft=False):
            return NPUMLATokenToKVPool(
                **_MLA,
                kv_lora_rank=16,
                qk_rope_head_dim=8,
                layer_num=layers,
                index_head_dim=128,
                enable_memory_saver=False,
                is_draft_worker=draft,
            )

        target, draft = dsa_pool(2), dsa_pool(1, draft=True)
        args = _register(target, draft)
        self.assertEqual(args.state_types, [StateType.DSA])
        self.assertEqual(args.state_layer_ids, [[0, 1, _TARGET_LAYERS]])
        self.assertEqual(
            args.state_data_ptrs[0],
            target.get_state_buf_infos()[0] + draft.get_state_buf_infos()[0],
        )
        self.assertEqual(len(target.get_contiguous_buf_infos()[0]), 4)
        self.assertEqual(args.num_draft_entries, 2)
        self.assertEqual(args.kv_buf_groups, 2)
        self.assertEqual(args.kv_layer_ids, [0, 1, 0, 1, 6, 6])

    def test_plain_and_hybrid_dsa_register_indexer_and_compressed_tails(self):
        for hybrid in (False, True):
            for draft_hybrid in (None, False, True):
                with self.subTest(hybrid=hybrid, draft_hybrid=draft_hybrid):
                    kwargs = dict(
                        **_MLA,
                        kv_lora_rank=16,
                        qk_rope_head_dim=0,
                        index_head_dim=128,
                        kv_cache_dim=16,
                        index_kpool=4,
                        index_kpool_compress=True,
                        max_running_requests=2,
                    )

                    def pool(layers, wrapped):
                        if wrapped:
                            return HybridLinearKVPool(
                                **kwargs,
                                head_num=1,
                                head_dim=16,
                                full_attention_layer_ids=list(range(layers)),
                                mamba_pool=_recurrent_state([layers]),
                                use_mla=True,
                                use_dsa=True,
                                full_kv_pool_class=NPUMLATokenToKVPool,
                            )
                        return NPUMLATokenToKVPool(
                            **kwargs, layer_num=layers, enable_memory_saver=False
                        )

                    target = pool(2, hybrid)
                    draft = pool(1, draft_hybrid) if draft_hybrid is not None else None
                    args = _register(target, draft)
                    pools = [target.full_kv_pool if hybrid else target]
                    if draft is not None:
                        pools.append(draft.full_kv_pool if draft_hybrid else draft)
                    self.assertEqual(args.state_types.count(StateType.DSA), 1)
                    self.assertEqual(args.state_types.count(StateType.DSA_TAIL), 1)
                    index = args.state_types.index(StateType.DSA)
                    self.assertEqual(
                        args.state_layer_ids[index],
                        [0, 1] + ([_TARGET_LAYERS] if draft is not None else []),
                    )
                    kv_ptrs = {
                        ptr for p in pools for ptr in p.get_contiguous_buf_infos()[0]
                    }
                    for state_type, method in (
                        (StateType.DSA, "get_state_buf_infos"),
                        (StateType.DSA_TAIL, "get_compress_tail_buf_infos"),
                    ):
                        state = args.state_types.index(state_type)
                        expected = [getattr(p, method)() for p in pools]
                        for actual, column in (
                            (args.state_data_ptrs[state], 0),
                            (args.state_data_lens[state], 1),
                            (args.state_item_lens[state], 2),
                        ):
                            self.assertEqual(
                                actual, sum((e[column] for e in expected), [])
                            )
                        self.assertTrue(
                            set(args.state_data_ptrs[state]).isdisjoint(kv_ptrs)
                        )

    def test_plain_mla_writes_use_token_interleaving_by_default(self):
        with (
            patch.object(
                memory_pool_npu,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=4, attn_dcp_rank=2),
            ),
            patch.object(runner_ids, "_is_npu", True),
            patch.object(
                runner_ids,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=True, dcp_size=4, dcp_rank=2),
            ),
        ):
            pool = _npu_mla(layer_num=1)
            ids = runner_ids.RunnerIds(
                req_to_token=torch.zeros((1, 8), dtype=torch.int32),
                token_to_kv_pool_allocator=None,
                token_to_kv_pool=pool,
                page_size=4,
                device="cpu",
            )
        self.assertEqual(pool.dcp_size, 4)
        self.assertEqual(
            ids.space(IdSpaceKind.FULL).write(torch.arange(8)).tolist(),
            [-1, -1, 0, -1, -1, -1, 1, -1],
        )


if __name__ == "__main__":
    unittest.main()
