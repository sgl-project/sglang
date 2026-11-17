import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.last_layer_ffn import graph_rows_for_batch, stage_graph_indices
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def batch(lens):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        is_prefill_only=False,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        lora_ids=None,
        mm_inputs=None,
        extend_seq_lens_cpu=lens,
        extend_prefix_lens_cpu=[0] * len(lens),
        extend_is_final_cpu=[True] * len(lens),
        batch_size=len(lens),
        input_ids=torch.zeros(sum(lens), dtype=torch.long),
        extend_seq_lens=torch.tensor(lens, dtype=torch.int32),
        input_embeds=None,
    )


class GraphTests(CustomTestCase):
    def test_graph_row_buckets(self):
        for n, expected in ((1, 1), (2, 4), (4, 4), (5, 16), (16, 16), (17, 0)):
            self.assertEqual(graph_rows_for_batch(batch([128] * n)), expected)

    def test_unsupported_uses_original_graph(self):
        for key, value in [
            ("return_logprob", True),
            ("tbo_split_seq_index", 0),
            ("tbo_parent_token_range", (0, 128)),
            ("tbo_children", []),
            ("extend_is_final_cpu", [False]),
            ("extend_prefix_lens_cpu", [64]),
            ("capture_hidden_mode", CaptureHiddenMode.FULL),
            ("input_embeds", torch.ones(1)),
            ("return_pooled_hidden_states", True),
            ("_original_forward_mode", ForwardMode.MIXED),
            ("global_forward_mode", ForwardMode.MIXED),
        ]:
            b = batch([128])
            setattr(b, key, value)
            self.assertEqual(graph_rows_for_batch(b), 0)

    def test_graph_key_switches_back_to_original_for_fallbacks(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._ffn_graph_enabled = True
        runner._capture_chunked_prefix = False
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(enable_last_layer_ffn_pruning=True)
        )
        b = batch([128, 128])
        self.assertEqual(runner._shape_key(256, b).variant_label, "last_layer_ffn_4")
        b.return_logprob = True
        self.assertIsNone(runner._shape_key(256, b).variant_label)
        b.return_logprob = False
        self.assertEqual(runner._shape_key(256, b).variant_label, "last_layer_ffn_4")
        b.tbo_split_seq_index = 0
        self.assertIsNone(runner._shape_key(256, b).variant_label)
        b.tbo_split_seq_index = None
        runner.model_runner.model.enable_last_layer_ffn_pruning = False
        self.assertIsNone(runner._shape_key(256, b).variant_label)

    def test_live_indices_and_unused_rows(self):
        buf = torch.full((16,), 999, dtype=torch.int64)
        ptr = buf.data_ptr()
        stage_graph_indices(buf, torch.tensor([129, 257, 126], dtype=torch.int32), 4)
        self.assertEqual(buf[:4].tolist(), [128, 385, 511, 0])
        stage_graph_indices(buf, torch.tensor([111, 299, 102], dtype=torch.int32), 4)
        self.assertEqual(buf[:4].tolist(), [110, 409, 511, 0])
        stage_graph_indices(buf, torch.tensor([128], dtype=torch.int32), 4)
        self.assertEqual(buf[:4].tolist(), [127, 0, 0, 0])
        self.assertEqual(buf.data_ptr(), ptr)


if __name__ == "__main__":
    unittest.main()
