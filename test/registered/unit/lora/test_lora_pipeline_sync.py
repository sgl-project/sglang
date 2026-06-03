"""
Unit tests for per-layer pipelined LoRA loading synchronization primitives.
"""

import unittest

import torch

from sglang.srt.lora.lora_pipeline_sync import LoRAPipelineFlag


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestLoRAPipelineFlag(unittest.TestCase):
    def test_flag_initial_state(self):
        """Flag starts ready (no load in progress)."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        # wait_until_ready should be a no-op
        flag.wait_until_ready(torch.cuda.current_stream())

    def test_mark_loading_and_ready(self):
        """Test basic mark_loading -> mark_ready cycle."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        stream = torch.cuda.Stream()

        flag.mark_loading()
        # Simulate DMA on load stream
        with torch.cuda.stream(stream):
            t = torch.zeros(256, 256, device="cuda:0")
            t.fill_(1.0)
        flag.mark_ready(stream)

        flag.wait_until_ready(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
        self.assertEqual(t[0, 0].item(), 1.0)

    def test_cross_stream_synchronization(self):
        """Compute stream waits for loading stream to finish."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        load_stream = torch.cuda.Stream()

        flag.mark_loading()
        with torch.cuda.stream(load_stream):
            large_tensor = torch.zeros(1024, 1024, device="cuda:0")
            large_tensor.fill_(42.0)
        flag.mark_ready(load_stream)

        # On compute stream: wait for flag
        flag.wait_until_ready(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
        self.assertEqual(large_tensor[0, 0].item(), 42.0)

    def test_no_wait_when_not_loading(self):
        """wait_until_ready is a no-op when no load is in progress."""
        flag = LoRAPipelineFlag(torch.device("cuda:0"))
        # Should not block
        flag.wait_until_ready(torch.cuda.current_stream())
        flag.wait_until_ready(torch.cuda.current_stream())

    def test_multiple_flags_independent(self):
        """Multiple flags operate independently."""
        flag1 = LoRAPipelineFlag(torch.device("cuda:0"))
        flag2 = LoRAPipelineFlag(torch.device("cuda:0"))
        stream = torch.cuda.Stream()

        flag1.mark_loading()
        # flag2 should still be ready (no-op wait)
        flag2.wait_until_ready()

        flag1.mark_ready(stream)
        stream.synchronize()


if __name__ == "__main__":
    unittest.main()
