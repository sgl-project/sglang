"""Check the backend's raw-K/V selection without constructing CUDA resources."""

import ast
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang.srt
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFaSkipKvCacheGating(unittest.TestCase):
    def test_raw_kv_requires_opt_in_and_structural_preconditions(self):
        path = (
            Path(next(iter(sglang.srt.__path__)))
            / "layers/attention/flashattention_backend.py"
        )
        values = [
            node.value
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute) and target.attr == "fa_skip_kv_cache"
        ]
        self.assertEqual(len(values), 1)
        predicate = compile(ast.Expression(values[0]), str(path), "eval")

        for opted_in, embedding, unchunked, uncached, mla in itertools.product(
            (False, True), repeat=5
        ):
            with self.subTest(
                opted_in=opted_in,
                embedding=embedding,
                unchunked=unchunked,
                uncached=uncached,
                mla=mla,
            ):
                schedule = SimpleNamespace(
                    prefill_only_disable_kv_cache=opted_in,
                    chunked_prefill_size=-1 if unchunked else 4096,
                )
                model = SimpleNamespace(is_embedding=embedding)
                memory = SimpleNamespace(disable_radix_cache=uncached)
                actual = eval(
                    predicate,
                    {
                        "get_schedule": lambda: schedule,
                        "get_model": lambda: model,
                        "get_memory": lambda: memory,
                        "self": SimpleNamespace(use_mla=mla),
                    },
                )
                self.assertEqual(
                    actual,
                    opted_in and embedding and unchunked and uncached and not mla,
                )


if __name__ == "__main__":
    unittest.main()
