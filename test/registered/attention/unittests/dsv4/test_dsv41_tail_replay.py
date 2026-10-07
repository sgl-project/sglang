import copy
import unittest
from dataclasses import replace

import torch

from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
    DSV4_PAGE_SIZE,
    DSV4AttentionCase,
    _make_forward_batch,
    build_dsv4_attention_fixture,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTailMetadataReplay(CustomTestCase):
    def test_eager_tail_then_short_graph_replay(self):
        case = DSV4AttentionCase(
            name="tail_then_graph",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(0, 0),
            extend_lens=(256, 256),
        )
        fixture = build_dsv4_attention_fixture(
            self, case, max_context_len=1024, swa_size=2048
        )
        self.addCleanup(fixture.runner._server_args_override.restore)
        backend = fixture.backend
        backend.enable_decoder_swa_bounded_replay = True
        eager = fixture.forward_batch

        def batch(rows, prefixes=None):
            return _make_forward_batch(
                replace(
                    case,
                    prefix_lens=(0,) * len(rows) if prefixes is None else prefixes,
                    extend_lens=rows,
                ),
                fixture.runner,
                max_context_len=1024,
                device="cuda",
            )

        capture_batch = batch((512,))
        capture_batch.max_seq_len_override = 1024
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            backend.init_forward_metadata(eager)
            self.assertEqual(
                backend.tail_forward_metadata.late_layer_tail.extend_seq_lens_cpu,
                [128, 128],
            )
            captured = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                capture_batch
            )
            self.assertEqual(
                backend.tail_forward_metadata.late_layer_tail.extend_seq_lens_cpu, [128]
            )
            slots = captured.core_attn_metadata.raw_out_loc
            output = torch.empty_like(slots)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output.copy_(slots)

            for rows, prefixes, tail_rows in [
                ((44,), None, list(range(44))),
                ((24, 24), None, list(range(48))),
                ((80,), None, list(range(80))),
                ((127,), None, list(range(127))),
                ((128,), (128,), list(range(128))),
                ((129,), None, list(range(1, 129))),
                ((256,), None, list(range(128, 256))),
                ((320, 160), None, list(range(192, 320)) + list(range(352, 480))),
                ((320, 160), (512, 256), list(range(192, 320)) + list(range(352, 480))),
                ((44,), (512,), list(range(44))),
                ((24, 24), (256, 256), list(range(48))),
                ((511,), None, list(range(383, 511))),
                ((512,), None, list(range(384, 512))),
            ] * 4:
                with self.subTest(rows=rows, prefixes=prefixes):
                    backend.init_forward_metadata(eager)
                    self.assertIsNotNone(backend.tail_forward_metadata)
                    live = batch(rows, prefixes)
                    static = copy.copy(live)
                    static.max_seq_len_override = 1024
                    static.out_cache_loc = torch.nn.functional.pad(
                        live.out_cache_loc, (0, 512 - sum(rows))
                    )
                    backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                        captured, live, static_forward_batch=static
                    )
                    self.assertEqual(
                        backend.tail_forward_metadata.late_layer_tail.extend_seq_lens_cpu,
                        [min(n, 128) for n in rows],
                    )
                    torch.testing.assert_close(
                        backend.tail_forward_metadata.late_layer_tail.positions,
                        live.positions[tail_rows],
                    )
                    published = BlockIds(
                        torch.arange(sum(rows), device="cuda").view(-1, 1),
                        list(rows),
                    )
                    with enable_breakable_cuda_graph():
                        backend._publish_candidate_metadata(published)
                        graph.replay()
                    self.assertIs(
                        backend.forward_metadata.candidate_metadata, published
                    )
                    torch.testing.assert_close(
                        backend.tail_forward_metadata.candidate_metadata.blocks,
                        published.blocks[tail_rows],
                    )
                    self.assertIs(captured.core_attn_metadata.raw_out_loc, slots)
                    torch.testing.assert_close(
                        output,
                        static.out_cache_loc.to(slots.dtype),
                    )


if __name__ == "__main__":
    unittest.main()
