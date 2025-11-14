import logging
import unittest

import torch

from sglang.srt.distributed.device_communicators.mtml_wrapper import pymtml as pynvml

try:
    import torch_musa

    _TORCH_MUSA_AVAILABLE = True
except ImportError:
    _TORCH_MUSA_AVAILABLE = False


@unittest.skipUnless(_TORCH_MUSA_AVAILABLE, "torch_musa is not available")
class TestNVMLP2PStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pynvml.nvmlInit()

    @classmethod
    def tearDownClass(cls):
        pynvml.nvmlShutdown()

    def test_nvlink_p2p_status(self):
        physical_device_ids = list(range(torch.musa.device_count()))
        assert (
            len(physical_device_ids) > 1
        ), "This test requires at least 2 MUSA devices to check P2P status."
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    assert p2p_status == pynvml.NVML_P2P_STATUS_OK


if __name__ == "__main__":
    unittest.main()
