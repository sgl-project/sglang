"""DeepSeek MoE transport isolates overlapping microbatches."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup
from sglang.srt.layers.layer_boundary.row_transport import TransportRows
from sglang.srt.models.deepseek_common import moe_transport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOverlappedTransport(unittest.TestCase):
    def test_interleaved_microbatches_keep_events_and_buffers_separate(self):
        trace = []
        compute = SimpleNamespace(
            wait_event=lambda event: trace.append(("wait", event.key))
        )
        comm = SimpleNamespace(wait_stream=lambda stream: trace.append(("fork",)))
        buffers = {}

        def buffer(key, n, h, dtype, device):
            value = torch.empty(n, h, dtype=dtype, device=device)
            buffers[key] = value
            return value

        def gather(out, value, batch):
            trace.append(("gather", batch.sub))
            out.copy_(torch.cat((value, value + 10)))

        def event(key):
            return SimpleNamespace(
                key=key, record=lambda stream: trace.append(("record", key))
            )

        def combine(out, value, sizes, *, event_key):
            trace.append(("combine", event_key))
            out.copy_(value[: out.shape[0]] * 2)
            return event(event_key)

        layer = SimpleNamespace(mlp=SimpleNamespace())
        inputs = (torch.arange(6.0).view(2, 3), torch.tensor([[7.0, 11.0, 13.0]]))
        batches = [SimpleNamespace(sub=i) for i in range(2)]
        lengths = [4, 2]
        with (
            patch(
                "sglang.srt.models.deepseek_common.moe_transport.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch(
                "sglang.srt.models.deepseek_common.moe_transport.SHARED_EXPERT_LOCAL",
                False,
            ),
            patch.object(torch.cuda, "current_stream", return_value=compute),
            patch.object(
                torch.cuda, "stream", side_effect=lambda stream: nullcontext()
            ),
            patch.object(moe_transport, "get_dp_tbo_comm_stream", return_value=comm),
            patch.object(
                moe_transport, "get_tbo_persistent_buffer", side_effect=buffer
            ),
            patch.object(
                moe_transport, "get_global_dp_buffer_len", side_effect=lengths
            ),
            patch.object(moe_transport, "get_local_dp_buffer_len", side_effect=[2, 1]),
            patch.object(
                moe_transport, "get_dp_global_num_tokens", side_effect=[[2, 2], [1, 1]]
            ),
            patch.object(moe_transport, "dp_gather_partial", side_effect=gather),
            patch.object(moe_transport, "_tbo_event", side_effect=event),
            patch.object(
                moe_transport, "dp_reduce_scatterv_async", side_effect=combine
            ),
        ):
            calls = [
                moe_transport.OverlappedMoeTransport(
                    layer.mlp,
                    x,
                    batch,
                    i,
                    TransportRows(
                        Layout(frozenset()),
                        Layout(frozenset()),
                        Layout(frozenset()),
                        SumGroup.MOE_OUTPUT,
                    ),
                )
                for i, (x, batch) in enumerate(zip(inputs, batches))
            ]
            for call in calls:
                with self.assertRaisesRegex(RuntimeError, "combine wait"):
                    call.prepare_output(call.input)
                call.issue_gather()
            for call in calls:
                x = call.wait_gather()
                call.issue_combine(x * 3)
            for call, expected in zip(calls, inputs):
                output = call.wait_combine()
                torch.testing.assert_close(call.prepare_output(output), expected * 6)
                self.assertIsNone(call.gather_keepalive)
                self.assertIsNone(call.combine_keepalive)
                self.assertIsNone(call.global_hidden)
                with self.assertRaisesRegex(RuntimeError, "not issued"):
                    call.wait_combine()
        self.assertEqual(
            [x for x in trace if x[0] == "gather"], [("gather", 0), ("gather", 1)]
        )
        self.assertIsNot(buffers[("gh", 0)], buffers[("gh", 1)])


if __name__ == "__main__":
    unittest.main()
