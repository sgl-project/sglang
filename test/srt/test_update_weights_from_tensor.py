import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestUpdateWeightsFromTensor(unittest.TestCase):
    def test_update_weights(self):
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST, DEFAULT_URL_FOR_TEST, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )
       
        TODO

        kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
