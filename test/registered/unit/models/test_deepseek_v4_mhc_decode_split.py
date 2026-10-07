"""DeepSeek-V4 mHC: the decoder layer tells hc_mix_stats_sinkhorn whether the forward
is decode/verify (where SM120 may split K more finely) or prefill, for both
hyper-connections. Runs the real forward_hc_pre_from_prev, posts and mix_stats on a
mocked layer, stopping at the kernel."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
import sglang.srt.models.deepseek_v4_mhc as deepseek_v4_mhc
from sglang.kernels.ops.layernorm import mhc as mhc_kernels
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS, HC, HIDDEN = 6, 4, 5120

DECODE_MODES = (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY)
PREFILL_MODES = (
    ForwardMode.EXTEND,
    ForwardMode.MIXED,
    ForwardMode.DRAFT_EXTEND_V2,
    ForwardMode.SPLIT_PREFILL,
)


def _residual():
    """A stand-in for a bf16 CUDA [rows, hc, hidden] residual."""
    flat = mock.Mock(shape=(ROWS, HC * HIDDEN))
    flat.is_contiguous.return_value = True
    residual = mock.Mock(is_cuda=True, dtype=torch.bfloat16, shape=(ROWS, HC, HIDDEN))
    residual.flatten.return_value = flat
    return residual


def _sublayer():
    cfg = SimpleNamespace(mult=HC, sinkhorn_iters=20, rms_eps=1e-6, eps=1e-6)
    return SimpleNamespace(
        cfg=cfg,
        fn=object(),
        scale=object(),
        base=object(),
        tf32_parts=None,
        bf16_parts=None,
    )


def _layer():
    layer = mock.MagicMock()
    layer.hc_cfg = SimpleNamespace(pre_from_prev=False)
    layer.attn_hc = _sublayer()
    layer.ffn_hc = _sublayer()
    layer._can_fuse_attn_mhc = False
    layer._can_fuse_ffn_mhc = False
    layer.self_attn.maybe_use_decode_attn_tp.return_value = nullcontext()
    return layer


class TestMhcDecodeFlag(CustomTestCase):
    def _decode_flags(self, mode):
        """The ``decode`` argument of each hc_mix_stats_sinkhorn call of one layer."""
        residual = _residual()
        kernel = mock.Mock(return_value=(object(), object(), object()))
        with (
            override_platform(is_blackwell=True, is_sm90=False, is_sm100=False),
            mock.patch.object(torch.version, "cuda", "13.0"),
            mock.patch.object(deepseek_v4_mhc, "_is_gfx95_supported", False),
            mock.patch.object(mhc_kernels, "hc_mix_stats_sinkhorn", kernel),
            mock.patch.object(deepseek_v4_mhc, "combine"),
            mock.patch.object(
                deepseek_v4_mhc,
                "_post_fusion",
                return_value=mock.Mock(residual=residual),
            ),
        ):
            deepseek_v4.DeepseekV4DecoderLayer.forward_hc_pre_from_prev(
                _layer(),
                positions=object(),
                state=mock.Mock(residual=residual),
                input_ids=object(),
                forward_batch=SimpleNamespace(forward_mode=mode),
                input_ids_global=object(),
            )
        flat = residual.flatten.return_value
        self.assertEqual(kernel.call_count, 2)
        for call in kernel.call_args_list:
            self.assertIs(call.args[0], flat)
        return [call.kwargs["decode"] for call in kernel.call_args_list]

    def test_decode_and_verify_select_decode_split(self):
        for mode in DECODE_MODES:
            with self.subTest(mode=mode.name):
                self.assertEqual(self._decode_flags(mode), [True, True])

    def test_prefill_keeps_default_split(self):
        for mode in PREFILL_MODES:
            with self.subTest(mode=mode.name):
                self.assertEqual(self._decode_flags(mode), [False, False])

    def test_direct_mix_stats_callers_default_to_prefill(self):
        residual = _residual()
        kernel = mock.Mock(return_value=(object(), object(), object()))
        with (
            override_platform(is_blackwell=True, is_sm90=False, is_sm100=False),
            mock.patch.object(torch.version, "cuda", "13.0"),
            mock.patch.object(deepseek_v4_mhc, "_is_gfx95_supported", False),
            mock.patch.object(mhc_kernels, "hc_mix_stats_sinkhorn", kernel),
        ):
            deepseek_v4_mhc.mix_stats(_sublayer(), residual)
        self.assertIs(kernel.call_args.kwargs["decode"], False)


if __name__ == "__main__":
    unittest.main()
