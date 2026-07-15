import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import BatchTokenIDOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDetokenizerManager(unittest.TestCase):
    def test_handle_batch_token_id_out_passthrough_without_tokenizer(self):
        decode = MagicMock(
            side_effect=AssertionError("skip-tokenizer path must not attempt to decode")
        )
        manager = SimpleNamespace(
            tokenizer=None,
            _decode_batch_token_id_output=decode,
        )
        recv_obj = MagicMock(spec=BatchTokenIDOutput)

        result = DetokenizerManager.handle_batch_token_id_out(manager, recv_obj)

        self.assertIs(result, recv_obj)
        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
