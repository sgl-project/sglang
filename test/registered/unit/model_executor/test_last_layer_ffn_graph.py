import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.last_layer_ffn import (
    FFN_GRAPH_ROWS,
    graph_rows_for_batch,
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
    def test_graph_row_buckets_follow_each_model(self):
        qwen = 512
        self.assertEqual(graph_rows_for_batch(batch([512]), qwen), 1)
        self.assertEqual(graph_rows_for_batch(batch([256, 256]), qwen), 4)
        self.assertEqual(graph_rows_for_batch(batch([128] * 4), qwen), 4)
        self.assertEqual(graph_rows_for_batch(batch([128]), qwen), 0)
        self.assertEqual(graph_rows_for_batch(batch([512] * 5), qwen), 0)
        llama = 2048
        self.assertEqual(graph_rows_for_batch(batch([512]), llama), 0)
        self.assertEqual(graph_rows_for_batch(batch([512] * 4), llama), 4)
        self.assertEqual(graph_rows_for_batch(batch([2048]), llama), 1)
        self.assertEqual(graph_rows_for_batch(batch([128] * 4), llama), 0)

    def test_capture_plan_skips_unmeasured_buckets(self):
        capture = [128, 256, 512, 640, 1024, 2048, 4096, 8192, 16384]
        self.assertEqual(
            select_graph_token_buckets("qwen2", capture),
            (512, 2048, 4096, 8192, 16384),
        )
        self.assertEqual(
            select_graph_token_buckets("qwen3", capture),
            (512, 2048, 4096, 8192),
        )
        self.assertEqual(select_graph_token_buckets("llama", capture), (2048, 8192))
        self.assertEqual(
            len(select_graph_token_buckets("qwen3", capture)) * len(FFN_GRAPH_ROWS),
            8,
        )
        self.assertNotIn(16, FFN_GRAPH_ROWS)

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
        runner._capture_chunked_prefix = False
        runner._ffn_graph_token_buckets = frozenset({512, 2048, 4096, 8192})
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                enable_last_layer_ffn_pruning=True,
                last_layer_ffn_min_extend_tokens=512,
            )
        )
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
        runner.model_runner.model.enable_last_layer_ffn_pruning = False
        self.assertIsNone(runner._shape_key(2048, b).variant_label)
        runner.model_runner.model.enable_last_layer_ffn_pruning = True
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
