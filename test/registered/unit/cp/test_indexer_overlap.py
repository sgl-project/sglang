"""CP overlap stream ownership and live BCG row contracts (CPU only)."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp import indexer_overlap as overlap
from sglang.srt.layers.cp.bcg import indexer_overlap as bridge
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCPIndexerOverlap(CustomTestCase):
    def test_ready_event_precedes_scoring_and_join_covers_errors(self):
        for score, fails in ((True, False), (False, False), (True, True)):
            with self.subTest(score=score, fails=fails):
                operations = []
                main = SimpleNamespace(
                    wait_stream=lambda other: operations.append("join")
                )
                side = SimpleNamespace(
                    wait_event=lambda event: operations.append("wait-ready")
                )
                event = SimpleNamespace(
                    record=lambda stream: operations.append("ready")
                )
                result = SimpleNamespace(
                    record_stream=lambda stream: operations.append("retain")
                )

                def forward_cuda(**kwargs):
                    operations.append("index-key-collective")
                    if score:
                        kwargs["on_logits_start"]()
                        operations.append("score")
                    return "topk"

                def gather(kv, batch):
                    operations.append("kv-collective")
                    if fails:
                        raise RuntimeError("collective failed")
                    return result

                with (
                    patch.object(torch.cuda, "current_stream", return_value=main),
                    patch.object(torch.cuda, "Event", return_value=event),
                    patch.object(torch.cuda, "stream", return_value=nullcontext()),
                    patch("sglang.srt.runtime_context.get_stream", return_value=side),
                    patch.object(
                        overlap,
                        "get_cp_strategy",
                        return_value=SimpleNamespace(gather_kv_cache=gather),
                    ),
                ):
                    call = lambda: overlap.run_indexer_with_kv_gather(
                        SimpleNamespace(forward_cuda=forward_cuda),
                        None,
                        None,
                        None,
                        object(),
                        0,
                        torch.empty(1, 1),
                    )
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "collective failed"):
                            call()
                    else:
                        self.assertEqual(call(), ("topk", result))
                expected = ["index-key-collective", "ready"]
                if score:
                    expected.append("score")
                expected += ["wait-ready", "kv-collective", "join"]
                if not fails:
                    expected.append("retain")
                self.assertEqual(operations, expected)

    def test_fp8_kv_uses_byte_transport_without_changing_values(self):
        kv = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.float8_e4m3fn)
        stream = SimpleNamespace(
            wait_stream=lambda other: None, wait_event=lambda event: None
        )

        def gather(value, batch):
            self.assertEqual(value.dtype, torch.uint8)
            return value.repeat(2, 1)

        def forward_cuda(**kwargs):
            kwargs["on_logits_start"]()
            return None

        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(
                torch.cuda,
                "Event",
                return_value=SimpleNamespace(record=lambda stream: None),
            ),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.Tensor, "record_stream", lambda tensor, stream: None),
            patch("sglang.srt.runtime_context.get_stream", return_value=stream),
            patch.object(
                overlap,
                "get_cp_strategy",
                return_value=SimpleNamespace(gather_kv_cache=gather),
            ),
        ):
            _, actual = overlap.run_indexer_with_kv_gather(
                SimpleNamespace(forward_cuda=forward_cuda),
                None,
                None,
                None,
                None,
                0,
                kv,
            )
        self.assertEqual(actual.dtype, kv.dtype)
        torch.testing.assert_close(
            actual.float(), kv.float().repeat(2, 1), rtol=0, atol=0
        )

    def test_bcg_reuses_buffers_with_live_prefix_and_padding_counts(self):
        output = torch.full((8, 3), 123, dtype=torch.int32)
        kv_output = torch.full((16, 4), float("nan"))
        x = torch.zeros(8, 4)
        positions = torch.arange(8)
        batch = SimpleNamespace()
        context = SimpleNamespace(forward_batch=batch)
        # Different live batches reuse the same capture capacity. The second
        # rank-local shard contains a physical padding row, then shrinks again.
        for physical, logical, global_rows, prefix in (
            (8, 8, 16, 0),
            (4, 3, 7, 90000),
            (1, 1, 1, 90007),
        ):
            batch.extend_seq_lens_cpu = [global_rows]
            batch.extend_prefix_lens_cpu = [prefix]

            def run(indexer, actual_x, q, pos, actual_batch, layer, kv):
                self.assertIs(actual_batch, batch)
                self.assertEqual(actual_x.shape[0], physical)
                self.assertEqual(kv.shape[0], physical)
                return (
                    torch.full((physical, 3), prefix, dtype=torch.int32),
                    torch.full((global_rows, 4), float(prefix)),
                )

            with (
                patch.object(
                    bridge, "get_tc_piecewise_forward_context", return_value=context
                ),
                patch.object(
                    bridge,
                    "cp_interleave_indexer_rows",
                    return_value=(physical, logical),
                ),
                patch.object(bridge, "run_indexer_with_kv_gather", side_effect=run),
            ):
                bridge._indexer_with_kv(None, x, x, positions, output, 0, x, kv_output)
            self.assertTrue(torch.all(output[:logical] == prefix))
            self.assertTrue(torch.all(output[logical:] == -1))
            self.assertTrue(torch.all(kv_output[:global_rows] == prefix))
            self.assertTrue(torch.all(kv_output[global_rows:] == 0))


if __name__ == "__main__":
    unittest.main()
