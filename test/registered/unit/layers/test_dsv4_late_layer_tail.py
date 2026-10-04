import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    late_layer_tail_layout,
)
from sglang.srt.layers.attention.deepseek_v4_backend import late_layer_tail_lens
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


def _batch(extend_lens, **kwargs):
    fields = dict(
        extend_seq_lens_cpu=extend_lens,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _layout(extend_lens, seq_lens, tail_len):
    return late_layer_tail_layout(
        extend_lens_cpu=extend_lens,
        seq_lens_cpu=seq_lens,
        tail_len=tail_len,
        window=128,
        device=torch.device("cpu"),
    )


class TestLateLayerTail(CustomTestCase):
    def test_tail_lens(self):
        self.assertEqual(late_layer_tail_lens(_batch([300, 50])), [128, 128])
        batch = _batch(
            [300, 300, 300],
            return_logprob=True,
            extend_logprob_start_lens_cpu=[100, 172, 300],
        )
        self.assertEqual(late_layer_tail_lens(batch), [300, 128, 128])
        batch = _batch([300], capture_hidden_mode=CaptureHiddenMode.FULL)
        self.assertEqual(late_layer_tail_lens(batch), [300])
        batch.spec_algorithm = SpeculativeAlgorithm.DSPARK
        self.assertEqual(late_layer_tail_lens(batch), [128])

    def test_whole_extend_floors(self):
        # 100 cached tokens: rows before the last window stop at the extend start.
        token_indices, tail_lens, floor = _layout([300, 200], [400, 200], [300, 128])
        self.assertEqual(tail_lens, [300, 128])
        self.assertEqual(
            token_indices.tolist(), list(range(300)) + list(range(372, 500))
        )
        self.assertEqual(floor.tolist(), [100] * 172 + [272] * 128 + [72] * 128)
        _, _, floor = _layout([300], [400], [300])
        self.assertEqual(floor.tolist(), [100] * 172 + [272] * 128)

    def test_last_window_matches_plain_tail(self):
        _, _, plain = _layout([300, 90], [400, 90], [128, 128])
        _, _, whole = _layout([300, 90], [400, 90], [300, 90])
        self.assertEqual(whole[172:].tolist(), plain.tolist())


if __name__ == "__main__":
    unittest.main()
