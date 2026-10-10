# SPDX-License-Identifier: Apache-2.0
"""Repeated Ovis requests must survive conditioning and shape cache changes."""

import os
import unittest

from sglang.multimodal_gen.test.server.test_server_ovis_image import (
    create_ovis_image_server,
)
from sglang.multimodal_gen.test.server.test_server_ovis_image import (
    test_repeated_generation_and_request_state as check_request_state,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, terminate_and_kill_process_tree

register_cuda_ci(est_time=300, stage="base-b", runner_config="diffusion-1-gpu-b200")


class TestOvisImageHTTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_ovis_image_server(
            os.environ.get("MODEL_PATH", "ATH-MaaS/Ovis-Image-7B")
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "server", None) is not None:
            try:
                terminate_and_kill_process_tree(cls.server.process)
            finally:
                cls.server.cleanup()

    def test_request_lifecycle(self):
        check_request_state(self.server)


if __name__ == "__main__":
    unittest.main()
