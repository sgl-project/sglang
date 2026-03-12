import types
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class _FakeTokenizerManager:
    def generate_request(self, _obj, _request):
        async def _gen():
            yield {
                "text": "ok",
                "meta_info": {
                    "top_logprobs": [[float("-inf"), -1, None]],
                },
            }

        return _gen()


class TestGenerateRequestResponseSerialization(unittest.IsolatedAsyncioTestCase):
    async def test_non_stream_generate_serializes_non_finite_logprobs(self):
        fake_state = types.SimpleNamespace(tokenizer_manager=_FakeTokenizerManager())
        fake_obj = types.SimpleNamespace(stream=False)

        with patch.object(http_server, "_global_state", fake_state):
            resp = await http_server.generate_request(fake_obj, request=None)

        self.assertEqual(resp.media_type, "application/json")
        self.assertIn(b"null", resp.body)
        self.assertNotIn(b"-Infinity", resp.body)


if __name__ == "__main__":
    unittest.main()
