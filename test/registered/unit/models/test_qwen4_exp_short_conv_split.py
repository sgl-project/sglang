"""The Qwen4-Exp PLE prefill short conv stays under the int32 element limit."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHANNELS, KERNEL, DILATION = 4, 4, 3
STATE_LEN = (KERNEL - 1) * DILATION


class TestQwen4ExpShortConvSplit(CustomTestCase):
    def test_padded_prefill_conv_is_split_under_the_limit(self):
        # One long request among short ones: the padded conv input is
        # 4 rows x 4 channels x (37 + 9) = 736 elements.
        lengths = [37, 1, 3, 2]
        layer = qwen4_exp.Qwen4ExpPLELayer.__new__(qwen4_exp.Qwen4ExpPLELayer)
        nn.Module.__init__(layer)
        layer.layer_id = 0
        layer.conv_channels = CHANNELS
        layer.short_conv_dilation = DILATION
        layer.short_conv_state_len = STATE_LEN
        layer.conv1d = nn.Conv1d(
            CHANNELS,
            CHANNELS,
            KERNEL,
            groups=CHANNELS,
            dilation=DILATION,
            bias=False,
        )
        # Small integers keep every sum exact, so split and unsplit results match.
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            layer.conv1d.weight.copy_(
                torch.randint(-2, 3, layer.conv1d.weight.shape, generator=gen)
            )
        x = torch.randint(-3, 4, (sum(lengths), CHANNELS), generator=gen).float()
        initial_state = torch.randint(
            -3, 4, (len(lengths) + 1, CHANNELS, STATE_LEN), generator=gen
        ).float()

        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
            spec_algorithm=None,
            spec_info=None,
            out_cache_loc=None,
            global_num_token_non_padded_cpu=None,
            extend_seq_lens=torch.tensor(lengths),
            extend_seq_lens_cpu=lengths,
            req_pool_indices=torch.arange(1, len(lengths) + 1),
            mamba_track_indices=None,
            mamba_track_mask=None,
        )
        pool = Mock(get_mamba_indices=lambda indices: indices)

        def run(limit):
            state = initial_state.clone()
            pool.short_conv_layer_cache = lambda layer_id: state
            inputs = []
            real_conv1d = F.conv1d

            def conv1d(conv_input, *args, **kwargs):
                inputs.append(conv_input.numel())
                return real_conv1d(conv_input, *args, **kwargs)

            with (
                patch.object(qwen4_exp, "get_req_to_token_pool", return_value=pool),
                patch.object(qwen4_exp, "_SHORT_CONV_MAX_NUMEL", limit, create=True),
                patch.object(qwen4_exp.F, "conv1d", conv1d),
            ):
                batch = qwen4_exp._prepare_ple_batch(
                    torch.zeros(sum(lengths), dtype=torch.long),
                    forward_batch,
                    ngram_size=None,
                    ngram_eos_token_id=None,
                )
                output = layer._short_conv(x, forward_batch, batch)
            return output, inputs

        expected_output, inputs = run(2**31 - 1)
        self.assertEqual(inputs, [len(lengths) * CHANNELS * (max(lengths) + STATE_LEN)])
        # 200 splits by rows; 100 also splits each row's columns.
        for limit in (200, 100):
            with self.subTest(limit=limit):
                output, inputs = run(limit)
                self.assertGreater(len(inputs), 1)
                self.assertLess(max(inputs), limit)
                self.assertTrue(torch.equal(output, expected_output))


if __name__ == "__main__":
    unittest.main()
