import unittest

import torch


@unittest.skipUnless(
    hasattr(torch, "npu")
    and torch.npu.is_available()
    and torch.npu.get_device_name().replace(" ", "").startswith("Ascend910B"),
    "requires 910B",
)
class TestHostTable(unittest.TestCase):
    def test_signed_rows_scales_and_shard_boundaries(self):
        from sglang.srt.hardware_backend.npu.dsv4.engram_host_uva import (
            NpuHostInt8Engram,
        )

        table = NpuHostInt8Engram(
            5, 64, torch.device("npu", torch.npu.current_device()), chunk_rows=2
        )
        try:
            weights = torch.arange(320).reshape(5, 64).to(torch.int8)
            table.weight.copy_(weights)
            table.scale.copy_(torch.tensor([126, 128], dtype=torch.uint8).expand(5, 2))
            ids = torch.tensor([[14, 10, 9], [12, 15, 10]], device=table.device)
            rows = (
                (weights.float().reshape(5, 2, 32) * torch.tensor([0.5, 2])[:, None])
                .reshape(5, 64)
                .bfloat16()
            )
            expected = torch.stack(
                [
                    rows[4],
                    rows[0],
                    torch.zeros_like(rows[0]),
                    rows[2],
                    torch.zeros_like(rows[0]),
                    rows[0],
                ]
            ).view(2, 3, 64)
            torch.testing.assert_close(
                table.lookup(ids, 10).cpu(), expected, rtol=0, atol=0
            )
            self.assertEqual(table.lookup(ids[:0], 10).shape, (0, 3, 64))
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = table.lookup(ids, 10)
            ids.fill_(11)
            graph.replay()
            torch.testing.assert_close(
                output.cpu(), rows[1].expand(2, 3, -1), rtol=0, atol=0
            )
            del graph, output
        finally:
            table.close()


if __name__ == "__main__":
    unittest.main()
