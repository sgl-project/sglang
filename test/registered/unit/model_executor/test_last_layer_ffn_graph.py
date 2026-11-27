import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.last_layer_ffn import (
    FFN_GRAPH_ROWS,
    eligible_batch,
    graph_rows_for_batch,
    select_graph_row_buckets,
    select_graph_token_buckets,
    stage_graph_indices,
)
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
    def test_capture_rows_cover_all_reachable_batches(self):
        for maximum in range(0, 65):
            expected = {
                next(rows for rows in FFN_GRAPH_ROWS if bs <= rows)
                for bs in range(1, min(maximum, 16) + 1)
            }
            self.assertEqual(set(select_graph_row_buckets(maximum)), expected)
        self.assertEqual(select_graph_row_buckets(None), FFN_GRAPH_ROWS)
        self.assertEqual(select_graph_row_buckets(3), (1, 4))
        self.assertEqual(select_graph_row_buckets(5), (1, 4, 8))
        self.assertEqual(select_graph_row_buckets(9), (1, 4, 8, 16))

    def test_graph_row_buckets_follow_each_model(self):
        qwen = 512
        self.assertEqual(graph_rows_for_batch(batch([512]), qwen), 1)
        self.assertEqual(graph_rows_for_batch(batch([256, 256]), qwen), 4)
        self.assertEqual(graph_rows_for_batch(batch([128] * 4), qwen), 4)
        self.assertEqual(graph_rows_for_batch(batch([128]), qwen), 0)
        self.assertEqual(graph_rows_for_batch(batch([512] * 5), qwen), 8)
        llama = 2048
        self.assertEqual(graph_rows_for_batch(batch([512]), llama), 0)
        self.assertEqual(graph_rows_for_batch(batch([512] * 4), llama), 4)
        self.assertEqual(graph_rows_for_batch(batch([2048]), llama), 1)
        self.assertEqual(graph_rows_for_batch(batch([128] * 4), llama), 0)

    def test_qwen3_6144_capture_and_three_request_rows(self):
        self.assertEqual(
            select_graph_token_buckets("qwen3", [4096, 6144, 8192]), (4096, 6144, 8192)
        )
        self.assertEqual(
            select_graph_token_buckets("qwen3", [4096, 8192]), (4096, 8192)
        )
        self.assertEqual(graph_rows_for_batch(batch([2048] * 3), 512), 4)
        self.assertEqual(graph_rows_for_batch(batch([6144]), 512), 1)
        indices = torch.full((4,), -1, dtype=torch.int64)
        stage_graph_indices(indices, torch.tensor([2048] * 3, dtype=torch.int32), 4)
        self.assertEqual(indices.tolist(), [2047, 4095, 6143, 0])

    def test_capture_plan_uses_existing_buckets_above_minimum(self):
        capture = [128, 512, 640, 1024, 2048, 2560, 4608, 6144, 8192, 16384, 512]
        for model_type, minimum in [("qwen2", 512), ("qwen3", 512), ("llama", 2048)]:
            self.assertEqual(
                select_graph_token_buckets(model_type, capture),
                tuple(sorted({n for n in capture if n >= minimum})),
            )
        self.assertEqual(FFN_GRAPH_ROWS, (1, 4, 8, 16))

    def test_graph_cap_does_not_limit_eager(self):
        for size, rows in [(5, 8), (8, 8), (9, 16), (16, 16), (17, 0), (32, 0)]:
            b = batch([512] * size)
            self.assertTrue(eligible_batch(b, 512))
            self.assertEqual(graph_rows_for_batch(b, 512), rows)
        self.assertFalse(eligible_batch(batch([8] * 32), 512))
        self.assertFalse(eligible_batch(batch([512] * 32), None))

    def test_large_variable_batch_indices_reset_padding(self):
        lengths = [100 + i for i in range(9)]
        buf = torch.full((16,), 999, dtype=torch.int64)
        stage_graph_indices(buf, torch.tensor(lengths, dtype=torch.int32), 16)
        expected = torch.tensor(lengths).cumsum(0).sub(1).tolist() + [0] * 7
        self.assertEqual(buf.tolist(), expected)
        stage_graph_indices(buf, torch.tensor([512] * 5, dtype=torch.int32), 8)
        self.assertEqual(buf[:8].tolist(), [511, 1023, 1535, 2047, 2559, 0, 0, 0])

    def test_cached_prefix_uses_suffix_rows_and_keeps_metadata(self):
        b = batch([300, 212])
        b.extend_prefix_lens_cpu = [1024, 256]
        self.assertEqual(graph_rows_for_batch(b, 512), 4)
        self.assertEqual(graph_rows_for_batch(b, 2048), 0)
        buf = torch.full((4,), -1, dtype=torch.int64)
        stage_graph_indices(buf, b.extend_seq_lens, 4)
        self.assertEqual(buf.tolist(), [299, 511, 0, 0])
        self.assertEqual(b.extend_prefix_lens_cpu, [1024, 256])
        b.extend_is_final_cpu = [False, True]
        self.assertEqual(graph_rows_for_batch(b, 512), 0)

    def test_unsupported_uses_original_graph(self):
        for key, value in [
            ("return_logprob", True),
            ("tbo_split_seq_index", 0),
            ("tbo_parent_token_range", (0, 2048)),
            ("tbo_children", []),
            ("extend_is_final_cpu", [False]),
            ("extend_prefix_lens_cpu", [-1]),
            ("capture_hidden_mode", CaptureHiddenMode.FULL),
            ("input_embeds", torch.ones(1)),
            ("return_pooled_hidden_states", True),
            ("_original_forward_mode", ForwardMode.MIXED),
            ("global_forward_mode", ForwardMode.MIXED),
        ]:
            b = batch([2048])
            setattr(b, key, value)
            self.assertEqual(graph_rows_for_batch(b, 512), 0)
        self.assertEqual(graph_rows_for_batch(batch([2048]), 512), 1)

    def test_graph_key_switches_back_to_original_for_fallbacks(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._ffn_graph_enabled = True
        runner._ffn_graph_rows = FFN_GRAPH_ROWS
        runner._capture_chunked_prefix = False
        runner._ffn_graph_token_buckets = frozenset({512, 2048, 4096, 8192})
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                last_layer_ffn_min_extend_tokens=512,
            )
        )
        runner._ffn_graph_rows = (1,)
        self.assertIsNone(runner._shape_key(2048, batch([1024, 1024])).variant_label)
        runner._ffn_graph_rows = FFN_GRAPH_ROWS
        b = batch([1024, 1024])
        self.assertEqual(runner._shape_key(2048, b).variant_label, "last_layer_ffn_4")
        self.assertIsNone(runner._shape_key(1024, b).variant_label)
        short = batch([128, 128])
        self.assertIsNone(runner._shape_key(256, short).variant_label)
        b.return_logprob = True
        self.assertIsNone(runner._shape_key(2048, b).variant_label)
        b.return_logprob = False
        self.assertEqual(runner._shape_key(2048, b).variant_label, "last_layer_ffn_4")
        b.tbo_split_seq_index = 0
        self.assertIsNone(runner._shape_key(2048, b).variant_label)
        b.tbo_split_seq_index = None
        runner.model_runner.model.last_layer_ffn_min_extend_tokens = None
        self.assertIsNone(runner._shape_key(2048, b).variant_label)
        runner.model_runner.model.last_layer_ffn_min_extend_tokens = 512
        runner.model_runner.model.last_layer_ffn_min_extend_tokens = 2048
        runner._ffn_graph_token_buckets = frozenset({2048, 8192})
        self.assertIsNone(runner._shape_key(512, batch([512])).variant_label)
        self.assertEqual(
            runner._shape_key(2048, batch([2048])).variant_label, "last_layer_ffn_1"
        )

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
