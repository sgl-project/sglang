import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import is_mla_backend, setup_state_kv_args
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_hybrid_pool(*, page_size: int = 128) -> HybridLinearKVPool:
    full_kv_pool = object.__new__(NPUMLATokenToKVPool)
    full_kv_pool.layer_num = 6
    full_kv_pool.page_size = page_size

    pool = object.__new__(HybridLinearKVPool)
    pool.use_mla = True
    pool.page_size = page_size
    pool.full_kv_pool = full_kv_pool
    pool.full_attention_layer_id_mapping = {
        layer_id: index for index, layer_id in enumerate(range(1, 7))
    }
    pool.mamba_pool = SimpleNamespace(
        get_contiguous_buf_infos=lambda: ([100], [200], [20]),
        get_state_dim_per_tensor=lambda: [64],
        get_state_layer_ids=lambda: [1],
        get_state_conv_shard_groups=lambda: [[]],
        get_state_slice_outer_counts=lambda: [1],
    )
    return pool


def _make_draft_pool(*, page_size: int = 128):
    return SimpleNamespace(
        page_size=page_size,
        get_contiguous_buf_infos=lambda: (
            list(range(1000, 1010)),
            [4096] * 10,
            [512] * 10,
        ),
    )


class TestKimiK3DSparkPD(unittest.TestCase):
    def test_hybrid_npu_mla_uses_flat_transfer(self):
        pool = _make_hybrid_pool()
        pool.full_kv_pool.get_contiguous_buf_infos = lambda: (
            list(range(12)),
            [4096] * 12,
            [512] * 12,
        )
        self.assertTrue(is_mla_backend(pool))
        self.assertEqual(pool.get_kv_layer_ids(), list(range(1, 7)) * 2)

    def test_draft_kv_does_not_change_target_mla_grouping(self):
        kv_args = KVArgs()
        # Six latent-K buffers followed by six K-RoPE buffers.
        kv_args.kv_data_ptrs = list(range(12))

        setup_state_kv_args(
            kv_args,
            _make_hybrid_pool(),
            draft_token_to_kv_pool=_make_draft_pool(),
            total_kv_layers=24,
            draft_kv_as_state=True,
        )

        self.assertEqual(kv_args.kv_buf_groups, 2)
        self.assertEqual(len(kv_args.kv_data_ptrs), 12)
        self.assertEqual(
            kv_args.state_types,
            [StateType.MAMBA, StateType.DRAFT_KV],
        )
        self.assertEqual(kv_args.state_data_ptrs[1], list(range(1000, 1010)))

    def test_draft_kv_requires_matching_page_size(self):
        kv_args = KVArgs()
        kv_args.kv_data_ptrs = list(range(12))

        with self.assertRaisesRegex(ValueError, "same page size"):
            setup_state_kv_args(
                kv_args,
                _make_hybrid_pool(),
                draft_token_to_kv_pool=_make_draft_pool(page_size=64),
                total_kv_layers=24,
                draft_kv_as_state=True,
            )

    def test_ascend_draft_state_bypasses_target_mla_grouping(self):
        manager = object.__new__(AscendKVManager)
        manager.kv_args = SimpleNamespace(
            prefill_start_layer=0,
            kv_buf_groups=2,
            total_kv_layers=24,
            mla_compression_ratios=None,
        )
        src = list(range(10))
        dst = list(range(100, 110))

        sliced_src, sliced_dst, count = manager.get_mla_kv_ptrs_with_pp(
            src, dst, StateType.DRAFT_KV
        )

        self.assertEqual(sliced_src, src)
        self.assertEqual(sliced_dst, dst)
        self.assertEqual(count, 10)


if __name__ == "__main__":
    unittest.main()
