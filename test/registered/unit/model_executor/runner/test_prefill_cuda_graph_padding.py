import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.bcg import PrefillCPBCGInput
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))

    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
            extend_prefix_lens_cpu=[0],
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_cp_local_capacity_overflow_uses_next_capture_bucket(self):
        runner = self._make_runner()
        runner.capture_num_tokens = [2048, 2304, 2560]
        runner.max_num_tokens = 2560
        runner.enable_cp_v2_bcg_capture = True
        runner.prefill_cp_bcg_input = PrefillCPBCGInput(
            input_embeds=torch.empty(0),
            positions=torch.empty(0),
            bucket_local_tokens={2048: 512, 2304: 576, 2560: 640},
        )
        forward_batch = self._make_forward_batch(2048)
        forward_batch.batch_size = 3
        forward_batch.seq_lens_cpu = [1534, 161, 353]
        forward_batch.extend_seq_lens_cpu = [1534, 161, 353]
        forward_batch.extend_prefix_lens_cpu = [0, 0, 0]
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=4,
            )
        )

        with patch(
            "sglang.srt.layers.cp.bcg.get_cp_padding_align_size",
            return_value=8,
        ):
            selected_buckets = []
            for cp_rank in range(4):
                with get_parallel().override(attn_cp_rank=cp_rank, attn_cp_size=4):
                    selected_buckets.append(
                        runner.select_replay_num_tokens(
                            num_tokens=2048,
                            extend_seq_lens=[1534, 161, 353],
                        )
                    )

            self.assertEqual(selected_buckets, [2304, 2304, 2304, 2304])
            with get_parallel().override(attn_cp_rank=0, attn_cp_size=4):
                self.assertTrue(runner.can_run_graph(forward_batch))

    def test_cp_bucket_search_preserves_two_x_padding_limit(self):
        runner = self._make_runner()
        runner.capture_num_tokens = [2048, 2304]
        runner.max_num_tokens = 2304
        runner.enable_cp_v2_bcg_capture = True
        runner.prefill_cp_bcg_input = PrefillCPBCGInput(
            input_embeds=torch.empty(0),
            positions=torch.empty(0),
            bucket_local_tokens={2048: 512, 2304: 576},
        )

        self.assertIsNone(
            runner.prefill_cp_bcg_input.select_replay_bucket(
                num_tokens=1024,
                required_local_tokens=520,
                capture_num_tokens=runner.capture_num_tokens,
                max_padding_factor=2,
            )
        )

    def test_cp_bucket_search_falls_back_when_no_capture_has_capacity(self):
        runner = self._make_runner()
        runner.capture_num_tokens = [2048, 2304]
        runner.max_num_tokens = 2304
        runner.enable_cp_v2_bcg_capture = True
        runner.prefill_cp_bcg_input = PrefillCPBCGInput(
            input_embeds=torch.empty(0),
            positions=torch.empty(0),
            bucket_local_tokens={2048: 512, 2304: 516},
        )
        forward_batch = self._make_forward_batch(2048)
        forward_batch.batch_size = 3
        forward_batch.extend_seq_lens_cpu = [1534, 161, 353]
        forward_batch.extend_prefix_lens_cpu = [0, 0, 0]
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=4,
            )
        )

        with (
            get_parallel().override(attn_cp_rank=0, attn_cp_size=4),
            patch(
                "sglang.srt.layers.cp.bcg.get_cp_padding_align_size",
                return_value=8,
            ),
        ):
            self.assertFalse(runner.can_run_graph(forward_batch))

    def test_cp_load_batch_uses_selected_larger_bucket(self):
        class StopAfterRecordingFill(Exception):
            pass

        class RecordingRegistry:
            padded_num_tokens = None

            def fill_from(self, _source, **kwargs):
                self.padded_num_tokens = kwargs["padded_num_tokens"]
                raise StopAfterRecordingFill

        runner = self._make_runner()
        runner.capture_num_tokens = [2048, 2304]
        runner.max_num_tokens = 2304
        runner.enable_cp_v2_bcg_capture = True
        runner.prefill_cp_bcg_input = PrefillCPBCGInput(
            input_embeds=torch.empty(0),
            positions=torch.empty(0),
            bucket_local_tokens={2048: 512, 2304: 576},
        )
        runner.buffer_registry = RecordingRegistry()
        forward_batch = self._make_forward_batch(2048)
        forward_batch.batch_size = 3
        forward_batch.extend_seq_lens_cpu = [1534, 161, 353]
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=4,
            )
        )

        with (
            get_parallel().override(attn_cp_rank=0, attn_cp_size=4),
            patch(
                "sglang.srt.layers.cp.bcg.get_cp_padding_align_size",
                return_value=8,
            ),
            self.assertRaises(StopAfterRecordingFill),
        ):
            runner.load_batch(forward_batch)

        self.assertEqual(runner.buffer_registry.padded_num_tokens, 2304)


if __name__ == "__main__":
    unittest.main()
