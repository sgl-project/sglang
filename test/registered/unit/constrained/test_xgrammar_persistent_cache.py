import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from sglang.srt.constrained import xgrammar_persistent_cache as cache_module
from sglang.srt.constrained.xgrammar_persistent_cache import (
    PersistentXGrammarCache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerInfo:
    def serialize_json(self) -> str:
        return '{"vocab":["a","b"]}'


class _FakeCompiledGrammar:
    def __init__(self, value: str) -> None:
        self.value = value

    def serialize_json(self) -> str:
        return json.dumps({"value": self.value})

    @classmethod
    def deserialize_json(
        cls,
        serialized: str,
        tokenizer_info: _FakeTokenizerInfo,
    ) -> "_FakeCompiledGrammar":
        del tokenizer_info
        payload = json.loads(serialized)
        if set(payload) != {"value"} or not isinstance(payload["value"], str):
            raise ValueError("invalid compiled grammar")
        return cls(payload["value"])


class TestPersistentXGrammarCache(CustomTestCase):
    def _make_cache(
        self,
        directory: str,
        *,
        deserialize_bytes_per_second: int = 10**18,
        max_bytes: int = 1024**2,
    ) -> PersistentXGrammarCache:
        return PersistentXGrammarCache(
            tokenizer_info=_FakeTokenizerInfo(),
            cache_directory=directory,
            max_bytes=max_bytes,
            deserialize_bytes_per_second=deserialize_bytes_per_second,
            local_compile_speedup=2,
            compiler_identity={
                "any_whitespace": True,
                "compiler_threads": 2,
                "override_stop_tokens": [],
                "vocab_size": 2,
            },
        )

    def test_serialized_round_trip_and_corruption_are_fail_closed(self) -> None:
        compile_calls = 0

        def compile_grammar() -> _FakeCompiledGrammar:
            nonlocal compile_calls
            compile_calls += 1
            return _FakeCompiledGrammar("root")

        with (
            tempfile.TemporaryDirectory(prefix="xgrammar-cache-test-") as cache_dir,
            patch.object(cache_module, "CompiledGrammar", _FakeCompiledGrammar),
            patch.dict(
                os.environ,
                {"SGLANG_XGRAMMAR_CACHE_SESSION_ID": "session-1"},
                clear=False,
            ),
        ):
            cache = self._make_cache(cache_dir)
            first = cache.get_or_compile(
                key_type="ebnf",
                key_string='root ::= "ab"',
                compile_fn=compile_grammar,
            )
            self.assertEqual(first.source, "compile")
            self.assertEqual(first.grammar.value, "root")
            self.assertEqual(
                set(first.phase_seconds),
                {
                    "native_compile",
                    "serialize",
                    "entry_write",
                    "account_prune",
                    "policy_serialized",
                },
            )

            second_cache = self._make_cache(cache_dir)
            second = second_cache.get_or_compile(
                key_type="ebnf",
                key_string='root ::= "ab"',
                compile_fn=compile_grammar,
            )
            self.assertEqual(second.source, "disk")
            self.assertEqual(second.grammar.value, "root")
            self.assertEqual(compile_calls, 1)

            entry = next(second_cache.entries.glob("*.json"))
            entry.write_text("corrupt", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cache entry is invalid"):
                second_cache.get_or_compile(
                    key_type="ebnf",
                    key_string='root ::= "ab"',
                    compile_fn=compile_grammar,
                )
            self.assertEqual(compile_calls, 1)

    def test_adaptive_local_compile_marker_and_checksum(self) -> None:
        compile_calls = 0

        def compile_grammar() -> _FakeCompiledGrammar:
            nonlocal compile_calls
            compile_calls += 1
            return _FakeCompiledGrammar("root")

        with (
            tempfile.TemporaryDirectory(
                prefix="xgrammar-local-compile-test-"
            ) as cache_dir,
            patch.object(cache_module, "CompiledGrammar", _FakeCompiledGrammar),
        ):
            cache = self._make_cache(
                cache_dir,
                deserialize_bytes_per_second=1,
            )
            first = cache.get_or_compile(
                key_type="ebnf",
                key_string='root ::= "ab"',
                compile_fn=compile_grammar,
            )
            self.assertEqual(first.source, "local_compile")
            entry = next(cache.entries.glob("*.json"))
            marker = entry.read_bytes()
            self.assertTrue(marker.startswith(cache._LOCAL_COMPILE_MAGIC))
            self.assertEqual(len(marker), cache._LOCAL_COMPILE_MARKER_BYTES)

            peer = self._make_cache(
                cache_dir,
                deserialize_bytes_per_second=1,
            )
            second = peer.get_or_compile(
                key_type="ebnf",
                key_string='root ::= "ab"',
                compile_fn=compile_grammar,
            )
            self.assertEqual(second.source, "local_compile")
            self.assertEqual(
                set(second.phase_seconds),
                {"adaptive_marker_read", "native_compile"},
            )
            self.assertEqual(compile_calls, 2)

            corrupted_marker = bytearray(marker)
            corrupted_marker[-1] ^= 1
            entry.write_bytes(corrupted_marker)
            with self.assertRaisesRegex(RuntimeError, "cache entry is invalid"):
                peer.get_or_compile(
                    key_type="ebnf",
                    key_string='root ::= "ab"',
                    compile_fn=compile_grammar,
                )
            self.assertEqual(compile_calls, 2)

    def test_concurrent_accounting_and_session_recovery(self) -> None:
        with (
            tempfile.TemporaryDirectory(
                prefix="xgrammar-accounting-test-"
            ) as cache_dir,
            patch.object(cache_module, "CompiledGrammar", _FakeCompiledGrammar),
            patch.dict(
                os.environ,
                {"SGLANG_XGRAMMAR_CACHE_SESSION_ID": "session-1"},
                clear=False,
            ),
        ):
            cache = self._make_cache(cache_dir)

            def compile_unique(index: int) -> None:
                cache.get_or_compile(
                    key_type="ebnf",
                    key_string=f'root ::= "ab" /* {index} */',
                    compile_fn=lambda: _FakeCompiledGrammar(str(index)),
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(compile_unique, range(64)))

            entry_bytes = sum(
                entry.stat().st_size for entry in cache.entries.glob("*.json")
            )
            self.assertEqual(cache._read_size_ledger(), entry_bytes)

            cache._size_ledger.write_bytes(b"invalid")
            with self.assertRaisesRegex(RuntimeError, "size ledger is invalid"):
                self._make_cache(cache_dir)

            os.environ["SGLANG_XGRAMMAR_CACHE_SESSION_ID"] = "session-2"
            recovered = self._make_cache(cache_dir)
            self.assertEqual(recovered._read_size_ledger(), entry_bytes)


if __name__ == "__main__":
    unittest.main()
