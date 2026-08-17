import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEngineNgramApi(unittest.TestCase):
    def test_ngram_corpus_id_is_keyword_only(self):
        methods = (
            EngineBase.generate,
            Engine.generate,
            Engine.async_generate,
            HttpServerEngineAdapter.generate,
        )

        for method in methods:
            with self.subTest(method=method.__qualname__):
                parameter = inspect.signature(method).parameters["ngram_corpus_id"]
                self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)


if __name__ == "__main__":
    unittest.main()
