"""CPU unit tests for the fixed system-prompt prefix feature.

Covers prompt-text / file resolution, mamba-chunk alignment, the token prefix
split probe (including BOS handling and probe-disable fallback), the mamba
boundary seeder, and the mamba donate-length cap that retains only the shared
system-prefix state. No server / engine is launched.
"""

import os
import tempfile
import unittest

from sglang.srt.system_prompt import (
    SystemPrefix,
    aligned_prefix_len,
    cap_mamba_donate_len,
    is_system_prefix_warm,
    resolve_system_prompt,
    seed_mamba_boundary,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.mem_cache.radix_cache import _system_kv_cap

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_MARKERS = {"<s>": 1, "<|system|>": 10, "<|user|>": 11, "<|assistant|>": 12}


class _StubTokenizer:
    """Minimal chat-template tokenizer for exercising SystemPrefix."""

    def __init__(self, top_bos: bool, stateful: bool = False, require_user: bool = False, noisy: bool = False):
        self._id = {c: i + 20 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
        for d in "0123456789":
            self._id[d] = 30 + int(d)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self._top_bos = top_bos
        self._stateful = stateful
        self._require_user = require_user
        self._noisy = noisy
        self._noise = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        if self._require_user and not any(m["role"] == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        if self._noisy:
            self._noise += 1
            out = f"<{self._noise}N>"
        else:
            out = ""
        if self._stateful:
            out += f"<{len(messages)}>"
        if self._top_bos:
            out += "<s>"
        for m in messages:
            out += f"<|{m['role']}|>{m['content']}"
        if add_generation_prompt:
            out += "<|assistant|>"
        return out

    def encode(self, text, **kw):
        ids, i = [], 0
        while i < len(text):
            m = next((x for x in _MARKERS if text.startswith(x, i)), None)
            if m:
                ids.append(_MARKERS[m])
                i += len(m)
            else:
                ids.append(self._id.get(text[i], 7))
                i += 1
        return ids


class TestSystemPromptArgs(CustomTestCase):
    def test_resolve_literal(self):
        server_args = _FakeArgs(system_prompt="You are a helpful assistant.")
        self.assertEqual(
            resolve_system_prompt(server_args), "You are a helpful assistant."
        )

    def test_resolve_none(self):
        self.assertIsNone(resolve_system_prompt(_FakeArgs(None)))

    def test_resolve_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("system from file\n")
            path = f.name
        try:
            self.assertEqual(resolve_system_prompt(_FakeArgs(path)), "system from file")
        finally:
            os.unlink(path)

    def test_aligned_prefix_len(self):
        self.assertEqual(aligned_prefix_len(0, 64), 0)
        self.assertEqual(aligned_prefix_len(64, 64), 64)
        self.assertEqual(aligned_prefix_len(65, 64), 128)
        self.assertEqual(aligned_prefix_len(100, 32), 128)
        self.assertEqual(aligned_prefix_len(30, 0), 30)


class _FakeReq:
    def __init__(self, boundary):
        self.system_prefix_boundary = boundary


class TestKvCap(CustomTestCase):
    def test_kv_cap_aligned(self):
        req = _FakeReq(128)
        self.assertEqual(_system_kv_cap(req, 64), 128)
        self.assertEqual(_system_kv_cap(req, 128), 128)

    def test_kv_cap_floor_to_page(self):
        req = _FakeReq(130)
        self.assertEqual(_system_kv_cap(req, 64), 128)

    def test_kv_cap_disabled_when_unset(self):
        self.assertIsNone(_system_kv_cap(_FakeReq(None), 64))
        self.assertIsNone(_system_kv_cap(_FakeReq(0), 64))


class TestMambaBoundary(CustomTestCase):
    def test_seed_aligned(self):
        self.assertEqual(seed_mamba_boundary(128, 64), 128)
        self.assertEqual(seed_mamba_boundary(256, 64), 256)

    def test_seed_unaligned_returns_none(self):
        self.assertIsNone(seed_mamba_boundary(96, 64))

    def test_seed_none_or_bad(self):
        self.assertIsNone(seed_mamba_boundary(None, 64))
        self.assertIsNone(seed_mamba_boundary(0, 64))
        self.assertIsNone(seed_mamba_boundary(128, 0))


class TestSystemPrefixWarm(CustomTestCase):
    """The warm/cold predicate driving the mamba slot-factor optimization.

    A request is warm only when the shared system-prefix boundary is fully
    covered by the matched prefix AND the mamba boundary state already lives in
    the tree (match left no branching point). Each cold branch below guards a
    distinct failure mode: if the predicate flipped to always-True, cold
    requests (warm-up / post-eviction) would skip tracking, never (re-)donate
    the boundary state, and every subsequent request would run from a stale /
    zero mamba state; if it flipped to always-False, the slot-factor savings
    vanish and concurrency regresses to pool/3.
    """

    def test_warm_when_boundary_cached(self):
        self.assertTrue(is_system_prefix_warm(64, 64, None))
        self.assertTrue(is_system_prefix_warm(64, 100, None))

    def test_cold_when_boundary_unset(self):
        # Feature off / client-supplied system message: no boundary -> never
        # warm (default path unchanged).
        self.assertFalse(is_system_prefix_warm(None, 100, None))
        self.assertFalse(is_system_prefix_warm(0, 100, None))

    def test_cold_first_request_prefix_short(self):
        # Warm-up pass: tree empty, matched prefix shorter than the boundary
        # -> must be cold so it tracks and donates the shared boundary state.
        self.assertFalse(is_system_prefix_warm(64, 0, None))
        self.assertFalse(is_system_prefix_warm(64, 32, None))

    def test_cold_after_mamba_eviction(self):
        # KV prefix still covers the boundary but the mamba state was evicted,
        # so the match returned a branching point (non-None) to re-track ->
        # must be cold so the boundary state is re-captured, not skipped.
        self.assertFalse(is_system_prefix_warm(64, 100, 64))


class TestMambaDonateCap(CustomTestCase):
    """The cap that makes the tree retain only the shared system-prefix state.

    Each case guards a distinct failure mode of ``cap_mamba_donate_len``: the
    cap fires only in extra_buffer mode and only when the boundary is set, else
    the donate length passes through unchanged. no_buffer must NOT be capped
    (its donate is the live chunk-end state, not the boundary state, so capping
    would store a wrong state at the shared boundary node).
    """

    def test_caps_suffix_donate_in_extra_buffer(self):
        # Guards the core fix: a suffix donate (cache_len 200 > boundary 64)
        # must be capped to the boundary so it maps to the shared node and the
        # suffix state is freed via mamba_exist. Removing the min -> per-request
        # suffix states leak into the tree (the bug being fixed).
        self.assertEqual(cap_mamba_donate_len(200, 64, True), 64)

    def test_no_cap_in_no_buffer(self):
        # Guards the safety gate: no_buffer donates the live chunk-end state, so
        # capping its key to the boundary would store a WRONG state at the shared
        # boundary node (silent output corruption). Removing the extra_buffer
        # guard -> this turns red and the corruption path reopens.
        self.assertEqual(cap_mamba_donate_len(200, 64, False), 200)

    def test_passthrough_when_feature_off(self):
        # Default path stays byte-identical: with no boundary the donate length
        # is unchanged in both modes. Removing the boundary-None check -> red.
        self.assertEqual(cap_mamba_donate_len(200, None, True), 200)
        self.assertEqual(cap_mamba_donate_len(200, None, False), 200)

    def test_none_cache_len_passthrough(self):
        # mamba_last_track_seqlen is None mid-decode / before any track; the cap
        # must not crash on it. Removing the cache_len-None guard -> TypeError.
        self.assertIsNone(cap_mamba_donate_len(None, 64, True))


class TestSystemPrefix(CustomTestCase):
    def _render(self, tok, msgs, gen):
        return tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen))

    def _run(self, top_bos, chunk):
        tok = _StubTokenizer(top_bos=top_bos)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(chunk, encode_kwargs={})
        self.assertTrue(sp.enabled, "identity probe should enable caching")
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}]
        got = sp.split_request(msgs, encode_kwargs={})
        real_sys = self._render(tok, msgs[:1], gen=False)
        full = self._render(tok, msgs, gen=True)
        # system tokens preserved verbatim up to their true length
        self.assertEqual(got[: len(real_sys)], real_sys)
        # prefix padded to a chunk boundary
        self.assertEqual(sp.prefix_len % chunk, 0)
        # rest (after the padded prefix) equals the full-render tail
        self.assertEqual(got[sp.prefix_len :], full[len(real_sys) :])
        # padding extends the prefix to the chunk boundary without reducing the
        # system tokens already rendered
        self.assertEqual(got[: len(real_sys)], real_sys)
        self.assertEqual(sp.prefix_len % chunk, 0)

    def test_split_qwen_style(self):
        for chunk in (30, 64):
            with self.subTest(chunk=chunk):
                self._run(top_bos=False, chunk=chunk)

    def test_split_llama_style_top_bos(self):
        for chunk in (30, 64):
            with self.subTest(chunk=chunk):
                self._run(top_bos=True, chunk=chunk)

    def test_split_disabled_on_identity_mismatch(self):
        # Non-deterministic template: each render gets a fresh leading token, so
        # no stable system prefix can be cached -> caching must be disabled and
        # split_request must return None (fall back to the full render).
        tok = _StubTokenizer(top_bos=False, noisy=True)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(30, encode_kwargs={})
        self.assertFalse(sp.enabled)
        self.assertIsNone(
            sp.split_request(
                [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}],
                encode_kwargs={},
            )
        )

    def test_measured_path_user_required(self):
        # Qwen-style template rejects a bare system render; the measured path
        # must still produce an aligned prefix and a split that reproduces full.
        tok = _StubTokenizer(top_bos=False, require_user=True)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(30, encode_kwargs={})
        self.assertTrue(sp.enabled)
        self.assertTrue(sp.prefix_len % 30 == 0)
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}]
        # The cached prefix is the PADDED system; build the reference full render
        # from the same injected (padded) content SystemPrefix actually uses.
        full_msgs = [sp.inject_message(), msgs[1]]
        full = self._render(tok, full_msgs, gen=True)
        got = sp.split_request(full_msgs, encode_kwargs={})
        self.assertEqual(got, full)
        self.assertEqual(got[sp.prefix_len :], full[sp.prefix_len :])

    def test_prefix_len_matches_boundary(self):
        tok = _StubTokenizer(top_bos=False)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(12, encode_kwargs={})
        self.assertEqual(seed_mamba_boundary(sp.prefix_len, 12), sp.prefix_len)
        self.assertEqual(sp.prefix_len % 12, 0)

    def test_auto_strip_default_system(self):
        # Qwen2.5-VL-style template: rendering user alone auto-prepends a default
        # system block. auto-strip must recover it and reconstruct `full`.
        tok = _StubTokenizer(top_bos=False, require_user=True, stateful=True)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(30, encode_kwargs={})
        self.assertTrue(sp.enabled)
        self.assertIsNotNone(sp._auto_prefix)
        msgs = [sp.inject_message(), {"role": "user", "content": "U"}]
        full = self._render(tok, msgs, gen=True)
        got = sp.split_request(msgs, encode_kwargs={})
        self.assertIsNotNone(got)
        self.assertEqual(got, full)


class _FakeArgs:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt


if __name__ == "__main__":
    unittest.main()