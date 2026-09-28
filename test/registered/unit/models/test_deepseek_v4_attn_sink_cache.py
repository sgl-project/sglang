import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

N_HEADS = 16


def _make_attn(attn_tp_size: int, attn_tp_rank: int) -> deepseek_v4.MqaAttentionBase:
    attn = deepseek_v4.MqaAttentionBase.__new__(deepseek_v4.MqaAttentionBase)
    nn.Module.__init__(attn)
    attn.n_heads = N_HEADS
    attn.attn_tp_size = attn_tp_size
    attn.attn_tp_rank = attn_tp_rank
    attn.n_local_heads = N_HEADS // attn_tp_size
    attn.compress_ratio = 0
    attn.attn_sink = nn.Parameter(torch.arange(N_HEADS, dtype=torch.float32))
    attn._attn_sink_local = None
    return attn


def _load_new_sink(attn: deepseek_v4.MqaAttentionBase) -> torch.Tensor:
    # Simulate online update: weight loader writes to attn_sink in-place
    new_sink = torch.randn(N_HEADS)
    attn.attn_sink.data.copy_(new_sink)
    return new_sink


class TestDeepseekV4AttnSinkCache(CustomTestCase):
    def test_cache_is_stale_without_refresh(self):
        attn = _make_attn(attn_tp_size=2, attn_tp_rank=1)
        cache = attn._local_attn_sink()

        new_sink = _load_new_sink(attn)

        # Param is updated, but forward still reads the pre-update cache.
        torch.testing.assert_close(attn.attn_sink, new_sink)
        self.assertIs(attn._local_attn_sink(), cache)
        torch.testing.assert_close(cache[:8], torch.arange(8, 16, dtype=torch.float32))
        self.assertFalse(torch.equal(cache[:8], new_sink[8:]))

    def test_post_load_weights_refreshes_cache(self):
        attn = _make_attn(attn_tp_size=2, attn_tp_rank=1)
        cache = attn._local_attn_sink()
        model = deepseek_v4.DeepseekV4ForCausalLM.__new__(
            deepseek_v4.DeepseekV4ForCausalLM
        )
        nn.Module.__init__(model)
        layer = SimpleNamespace(
            self_attn=attn, refresh_mhc_norm_weight_cache=lambda: None
        )
        model.model = SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])

        new_sink = _load_new_sink(attn)
        with patch.object(deepseek_v4, "_FP8_WO_A_GEMM", False):
            model.post_load_weights()

        torch.testing.assert_close(cache[:8], new_sink[8:])

    def test_refresh_updates_tp_local_cache_in_place(self):
        attn = _make_attn(attn_tp_size=2, attn_tp_rank=1)
        cache = attn._local_attn_sink()
        data_ptr = cache.data_ptr()

        new_sink = _load_new_sink(attn)
        attn.refresh_attn_sink_cache()

        # CUDA graph captures the cache address, so it must remain the same after refresh
        self.assertIs(attn._local_attn_sink(), cache)
        self.assertEqual(cache.data_ptr(), data_ptr)
        torch.testing.assert_close(cache[:8], new_sink[8:])
        torch.testing.assert_close(cache[8:], torch.zeros(56))

    def test_refresh_uses_build_time_head_slice(self):
        # The cache is built under the decode attn TP layout; after refresh, TP state is restored to default layout
        attn = _make_attn(attn_tp_size=4, attn_tp_rank=2)
        cache = attn._local_attn_sink()
        attn.attn_tp_size, attn.attn_tp_rank, attn.n_local_heads = 1, 0, N_HEADS

        new_sink = _load_new_sink(attn)
        attn.refresh_attn_sink_cache()

        torch.testing.assert_close(cache[:4], new_sink[8:12])
        torch.testing.assert_close(cache[4:], torch.zeros(60))

    def test_refresh_before_first_use_is_noop(self):
        attn = _make_attn(attn_tp_size=2, attn_tp_rank=0)
        attn.refresh_attn_sink_cache()
        self.assertIsNone(attn._attn_sink_local)

        new_sink = _load_new_sink(attn)
        torch.testing.assert_close(attn._local_attn_sink()[:8], new_sink[:8])

    def test_attn_tp1_uses_parameter_directly(self):
        attn = _make_attn(attn_tp_size=1, attn_tp_rank=0)
        attn.refresh_attn_sink_cache()
        self.assertIs(attn._local_attn_sink(), attn.attn_sink)
        self.assertIsNone(attn._attn_sink_local)


if __name__ == "__main__":
    unittest.main()
