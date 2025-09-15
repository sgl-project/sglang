import logging
import unittest

import torch

from sglang.srt.utils import is_musa

_is_musa = is_musa()

if _is_musa:
    try:
        import torch_musa

        from sglang.srt.distributed.device_communicators.mtml_wrapper import (
            pymtml as pynvml,
        )
    except ImportError as e:
        logging.info("Failed to import pymtml with %r", e)


class TestNVMLP2PStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pynvml.nvmlInit()

    @classmethod
    def tearDownClass(cls):
        pynvml.nvmlShutdown()

    def test_nvlink_p2p_status(self):
        physical_device_ids = list(range(torch.musa.device_count()))
        assert len(physical_device_ids) == 8
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    assert p2p_status != pynvml.NVML_P2P_STATUS_OK


if __name__ == "__main__":
    unittest.main()
