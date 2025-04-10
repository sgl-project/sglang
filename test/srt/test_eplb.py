import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
import sglang as sgl


class TestEPLB(CustomTestCase):
    def test_eplb_e2e(self):
        engine = sgl.Engine(model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST)
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
