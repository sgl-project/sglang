import unittest

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    FP4MXBlock16KVCacheMethod,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

SIZE, PAGE, HEADS, HDIM, LAYERS = 256, 16, 2, 64, 2


def _fp4_pool() -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=SIZE,
        page_size=PAGE,
        dtype=torch.uint8,
        head_num=HEADS,
        head_dim=HDIM,
        layer_num=LAYERS,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
        quant_method=FP4MXBlock16KVCacheMethod(),
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestFP4KVScalesTravelWithKV(unittest.TestCase):
    def test_pd_registers_block_scales(self):
        pool = _fp4_pool()
        kv_args = KVArgs()
        setup_state_kv_args(kv_args, pool)

        self.assertIn(StateType.BLOCK_SCALE, kv_args.state_types)
        i = kv_args.state_types.index(StateType.BLOCK_SCALE)
        scale_ptrs = {t.data_ptr() for t in pool.k_scale_buffer + pool.v_scale_buffer}
        self.assertEqual(set(kv_args.state_data_ptrs[i]), scale_ptrs)

    def test_retraction_copy_restores_block_scales(self):
        pool = _fp4_pool()
        old = torch.arange(PAGE, 3 * PAGE, device="cuda")
        new = torch.arange(8 * PAGE, 10 * PAGE, device="cuda")
        for buf in pool.k_scale_buffer + pool.v_scale_buffer:
            buf[old] = 7
            buf[new] = 99  # left by the previous owner of the new slots

        pool.load_cpu_copy(pool.get_cpu_copy(old), new)

        for buf in pool.k_scale_buffer + pool.v_scale_buffer:
            self.assertTrue(torch.all(buf[new] == 7))


if __name__ == "__main__":
    unittest.main()
