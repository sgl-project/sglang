"""CPU unit tests for the fixed system-prompt prefix feature.

Covers prompt-text / file resolution, mamba-chunk alignment (truncate-down),
the token prefix split probe (including BOS handling and probe-disable
fallback), the truncate-down invariants (byte-identical token stream, shared
prefix on the chunk grid), the short-system disable guard, the startup probe
gate, the static-pool mamba slot budget, the mamba boundary seeder, and the
mamba donate-length cap that retains only the shared system-prefix state. No
server / engine is launched.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.system_prompt import (
    SystemPrefix,
    aligned_prefix_len_down,
    cap_mamba_donate_len,
    is_system_prefix_warm,
    probe_system_prefix_or_disable,
    resolve_system_prompt,
    reseed_mamba_boundary_after_match,
    seed_mamba_boundary,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.mem_cache.radix_cache import _system_kv_cap

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_MARKERS = {"<s>": 1, "「": 10, "」": 11, "』": 12}


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
            out += "」"
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

    def test_aligned_prefix_len_down(self):
        # Derived property: the shared prefix must land ON the checkpoint grid
        # by rounding DOWN, never past the native system block (padding up
        # would alter the token stream; the crash the down-rounding prevents is
        # a boundary state that no checkpoint can back).
        self.assertEqual(aligned_prefix_len_down(0, 64), 0)
        self.assertEqual(aligned_prefix_len_down(64, 64), 64)
        self.assertEqual(aligned_prefix_len_down(65, 64), 64)
        self.assertEqual(aligned_prefix_len_down(100, 32), 96)
        self.assertEqual(aligned_prefix_len_down(30, 0), 30)


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


class TestReseedAfterMatch(CustomTestCase):
    """The post-match re-derivation that keeps the boundary seed alive.

    Bug regression: ``init_next_round_input`` overwrote
    ``mamba_branching_seqlen`` with the match result unconditionally, so on an
    empty tree (first warm-up request) the ``__init__``-seeded boundary died and
    prefill tracked the prompt tail instead of the boundary; the finish donate
    then stored the tail state under the boundary-capped key, poisoning the
    shared node for every later warm request. Each case below guards one
    branch of the re-derivation.
    """

    def test_first_request_restores_seed(self):
        # Empty tree: match returns prefix 0 and branching None -> the seed
        # must come back so prefill tracks the boundary state.
        self.assertEqual(
            reseed_mamba_boundary_after_match(64, 0, None),
            64,
        )

    def test_cold_partial_prefix_restores_seed(self):
        # KV partially cached but boundary not covered, no branching point ->
        # still the warm-up pass; track the boundary.
        self.assertEqual(
            reseed_mamba_boundary_after_match(64, 32, None),
            64,
        )

    def test_match_branching_wins(self):
        # Mamba state evicted (match found a branching point to re-track):
        # the match's recovery plan must not be clobbered by the seed.
        self.assertEqual(
            reseed_mamba_boundary_after_match(64, 32, 32),
            32,
        )

    def test_warm_stays_none(self):
        # Boundary fully covered, no branching point: warm request. Returning
        # the seed here would flip is_system_prefix_warm to False and cost the
        # request a full ping-pong path (concurrency regression to pool/3).
        self.assertIsNone(reseed_mamba_boundary_after_match(64, 64, None))
        self.assertIsNone(reseed_mamba_boundary_after_match(64, 100, None))

    def test_feature_off_passthrough(self):
        # Default path stays byte-identical: without a boundary the match
        # result passes through untouched, warm or not.
        self.assertIsNone(reseed_mamba_boundary_after_match(None, 0, None))
        self.assertEqual(reseed_mamba_boundary_after_match(None, 0, 32), 32)


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
        full = self._render(tok, msgs, gen=True)
        real_sys = self._render(tok, msgs[:1], gen=False)
        # truncate-down invariants (see TestTruncateDownInvariants)
        self.assertEqual(got, full)
        self.assertEqual(sp.prefix_len, (len(real_sys) // chunk) * chunk)
        self.assertEqual(sp.prefix_len % chunk, 0)
        self.assertEqual(got[: sp.prefix_len], list(sp.prefix_ids))
        # inject_message carries the ORIGINAL text: no padding ever added
        self.assertEqual(sp.inject_message()["content"], "SYS")

    def test_split_qwen_style(self):
        for chunk in (3, 4):
            with self.subTest(chunk=chunk):
                self._run(top_bos=False, chunk=chunk)

    def test_split_llama_style_top_bos(self):
        for chunk in (3, 4):
            with self.subTest(chunk=chunk):
                self._run(top_bos=True, chunk=chunk)

    def test_split_disabled_on_identity_mismatch(self):
        # Non-deterministic template: each render gets a fresh leading token, so
        # no stable system prefix can be cached -> caching must be disabled and
        # split_request must return None (fall back to the full render).
        tok = _StubTokenizer(top_bos=False, noisy=True)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(3, encode_kwargs={})
        self.assertFalse(sp.enabled)
        self.assertIsNone(
            sp.split_request(
                [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}],
                encode_kwargs={},
            )
        )

    def test_measured_path_user_required(self):
        # Qwen-style template rejects a bare system render; the measured path
        # must still produce an aligned prefix and a split that reproduces the
        # native full render (inject_message = original text).
        tok = _StubTokenizer(top_bos=False, require_user=True)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(4, encode_kwargs={})
        self.assertTrue(sp.enabled)
        self.assertTrue(sp.prefix_len % 4 == 0)
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}]
        full = self._render(tok, msgs, gen=True)
        got = sp.split_request(msgs, encode_kwargs={})
        self.assertEqual(got, full)

    def test_prefix_len_matches_boundary(self):
        tok = _StubTokenizer(top_bos=False)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(12, encode_kwargs={})
        self.assertEqual(seed_mamba_boundary(sp.prefix_len, 12), sp.prefix_len)
        self.assertEqual(sp.prefix_len % 12, 0)

    def test_auto_strip_default_system(self):
        # Qwen2.5-VL-style template: rendering user alone auto-prepends a default
        # system block. auto-strip must recover it and reconstruct the native
        # `full` render exactly (no padding — the Qwen3.8-27B crash regression:
        # auto-strip used to skip chunk alignment entirely).
        tok = _StubTokenizer(top_bos=False, require_user=True, stateful=True)
        sp = SystemPrefix(tok, "SYS")
        sp.prepare(4, encode_kwargs={})
        self.assertTrue(sp.enabled)
        self.assertIsNotNone(sp._auto_prefix)
        self.assertEqual(sp.prefix_len % 4, 0)
        msgs = [sp.inject_message(), {"role": "user", "content": "U"}]
        full = self._render(tok, msgs, gen=True)
        got = sp.split_request(msgs, encode_kwargs={})
        self.assertIsNotNone(got)
        self.assertEqual(got, full)


class TestTruncateDownInvariants(CustomTestCase):
    """Derived property: truncate-down never changes the token stream.

    For every chunk size, the shared prefix must be the native system block
    truncated DOWN to the grid, and prefix + tail + rest must reproduce the
    native full render token-for-token. A regression to pad-up (or to an
    off-grid boundary) turns these red: pad-up changes `got` (extra pad tokens
    not in `full`), an off-grid boundary breaks the `% chunk` check and would
    put the boundary between mamba checkpoints — the half-enabled crash state.
    """

    SYS_TEXT = "S" * 40  # "<|system|>" + 40 chars -> 50 tokens with this stub

    def test_boundary_is_floor_and_stream_is_native(self):
        for chunk in (5, 8, 16, 30, 64):
            with self.subTest(chunk=chunk):
                tok = _StubTokenizer(top_bos=False)
                sp = SystemPrefix(tok, self.SYS_TEXT)
                sp.prepare(chunk, encode_kwargs={})
                real_sys = tok.encode(
                    tok.apply_chat_template(
                        [{"role": "system", "content": self.SYS_TEXT}],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                if len(real_sys) < chunk:
                    # covered by TestShortSystemDisables; here every chunk fits
                    self.assertFalse(sp.enabled)
                    continue
                self.assertTrue(sp.enabled)
                self.assertEqual(sp.boundary_len, (len(real_sys) // chunk) * chunk)
                self.assertEqual(sp.prefix_len, sp.boundary_len)
                self.assertEqual(sp.suffix_len, len(real_sys) - sp.boundary_len)
                msgs = [
                    {"role": "system", "content": self.SYS_TEXT},
                    {"role": "user", "content": "U"},
                ]
                full = tok.encode(
                    tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                )
                got = sp.split_request(msgs, encode_kwargs={})
                self.assertEqual(got, full)
                self.assertEqual(got[: sp.boundary_len], list(sp.prefix_ids))


class TestShortSystemDisables(CustomTestCase):
    """Bug regression (Qwen3.8-27B crash): a system block shorter than one
    mamba chunk used to flow through with an unusable boundary — every request
    stayed cold (3 slots) while the pool was sized for warm (1 slot/req) and
    the first batch crashed the mamba allocator. A block that leaves no
    chunk-aligned boundary must disable the feature entirely.
    """

    def test_short_system_disables(self):
        tok = _StubTokenizer(top_bos=False)
        sp = SystemPrefix(tok, "SYS")  # 13 tokens with this stub
        sp.prepare(30, encode_kwargs={})
        self.assertFalse(sp.enabled)
        self.assertEqual(sp.boundary_len, 0)
        self.assertIsNone(sp.prefix_ids)
        self.assertIsNone(
            sp.split_request(
                [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}],
                encode_kwargs={},
            )
        )


class TestStartupProbeGate(CustomTestCase):
    """Bug regression (Qwen3.8-27B crash): the mamba pool sized itself for the
    warm steady state (ratio 1) purely from ``server_args.system_prompt`` being
    set, before any probe ran. When the runtime probe later failed, every
    request was cold on a warm-sized pool -> over-admission crash. The startup
    gate must clear the flag (end-to-end feature-off) whenever the probe fails.
    """

    def _args(self, chunk=16):
        return _FakeArgs(
            "SYS-TEXT", model_path="/tmp/m", mamba_cache_chunk_size=chunk
        )

    def test_probe_ok_keeps_flag(self):
        args = self._args(chunk=8)
        tok = _StubTokenizer(top_bos=False)
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tok
        ):
            probe_system_prefix_or_disable(args)
        self.assertIsNotNone(args.system_prompt)

    def test_probe_reject_short_system_clears_flag(self):
        # chunk 64 > the stub's 13-token system block: probe disables -> the
        # gate must clear server_args.system_prompt before schedulers fork.
        args = self._args(chunk=64)
        tok = _StubTokenizer(top_bos=False)
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tok
        ):
            probe_system_prefix_or_disable(args)
        self.assertIsNone(args.system_prompt)

    def test_tokenizer_load_failure_clears_flag(self):
        # A probe that cannot even load the tokenizer must fall back to the
        # native path rather than risk the half-enabled state.
        args = self._args()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("boom"),
        ):
            probe_system_prefix_or_disable(args)
        self.assertIsNone(args.system_prompt)

    def test_unset_flag_is_noop(self):
        args = _FakeArgs(None)
        probe_system_prefix_or_disable(args)
        self.assertIsNone(args.system_prompt)


class TestStaticMambaSlotBudget(CustomTestCase):
    """Bug regression (Qwen3.8-27B crash): on the static (non-shared) hybrid
    mamba pool the PrefillAdder charged nothing per request, so a cold burst
    (each req taking running + ping-pong = 3 slots) over-admitted past the
    ratio-1-sized pool and died in the fail-loud allocator. The slot budget
    must charge cold requests the full factor and warm requests 1.
    """

    def _adder(self, static: bool, lazy: bool = False):
        adder = SimpleNamespace(
            _static_mamba_slots=static,
            tree_cache=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(
                    # main's pool exposes the initial tracking slots directly
                    # (2 with overlap, 1 in lazy mode) instead of the lazy flag.
                    mamba_initial_tracking_slots=1 if lazy else 2
                )
            ),
        )
        return adder

    def _req(self, boundary, prefix_len=0, branching=None, has_pool_idx=False):
        return SimpleNamespace(
            system_prefix_boundary=boundary,
            prefix_len=prefix_len,
            mamba_branching_seqlen=branching,
            kv=SimpleNamespace(
                mamba_pool_idx=0 if has_pool_idx else None
            ),
        )

    def _budget(self, adder, req):
        from sglang.srt.managers.schedule_policy import PrefillAdder

        return PrefillAdder._mamba_slot_budget_for_req(adder, req)

    def test_cold_charged_full_factor(self):
        # Warm-up burst: no boundary coverage -> 3 slots (running+ping-pong).
        adder = self._adder(static=True)
        self.assertEqual(self._budget(adder, self._req(64, 0, None)), 3)
        self.assertEqual(self._budget(adder, self._req(64, 32, None)), 3)

    def test_warm_charged_one(self):
        # Steady state: boundary covered, no branching -> sentinel-only
        # ping-pong, 1 slot.
        adder = self._adder(static=True)
        self.assertEqual(self._budget(adder, self._req(64, 64, None)), 1)
        self.assertEqual(self._budget(adder, self._req(64, 100, None)), 1)

    def test_cold_after_eviction(self):
        # Branching point from the match (mamba state evicted) -> cold again.
        adder = self._adder(static=True)
        self.assertEqual(self._budget(adder, self._req(64, 100, 64)), 3)

    def test_lazy_pool_charges_two(self):
        adder = self._adder(static=True, lazy=True)
        self.assertEqual(self._budget(adder, self._req(64, 0, None)), 2)

    def test_zero_when_not_static(self):
        # Shared pool / non-mamba: slot budgeting happens via the token
        # reserve; charging here too would double-count.
        adder = self._adder(static=False)
        self.assertEqual(self._budget(adder, self._req(64, 0, None)), 0)

    def test_zero_when_pool_idx_held(self):
        # Re-added request that already holds its mamba slot: alloc will not
        # re-allocate, so admission must not charge again.
        adder = self._adder(static=True)
        self.assertEqual(self._budget(adder, self._req(64, 0, None, has_pool_idx=True)), 0)

    def test_feature_off_charges_cold_factor(self):
        # Without --system-prompt every request is "cold" in slot terms (full
        # ping-pong); the factor-3 charge matches alloc_req_slots' accounting.
        adder = self._adder(static=True)
        self.assertEqual(self._budget(adder, self._req(None, 0, None)), 3)

    def test_match_headroom(self):
        # Bug regression (match-COW crash, deep waiting queue): the scheduler
        # must stop pulling new requests once the remaining slot budget is
        # below a worst-case cold request, because init_next_round_input's
        # mamba COW allocates BEFORE add_one_req can reject. Headroom = the
        # cold factor on the static pool (lazy variant = 2), on the shared
        # pool too, and 0 (pre-check disabled) without a mamba pool.
        from sglang.srt.managers.schedule_policy import PrefillAdder

        self.assertEqual(PrefillAdder._mamba_match_headroom(self._adder(static=True)), 3)
        self.assertEqual(
            PrefillAdder._mamba_match_headroom(self._adder(static=True, lazy=True)), 2
        )
        no_mamba = SimpleNamespace(
            _mamba_slot_cost=0, _static_mamba_slots=False, tree_cache=None
        )
        self.assertEqual(PrefillAdder._mamba_match_headroom(no_mamba), 0)
        shared = SimpleNamespace(
            _mamba_slot_cost=64,
            _static_mamba_slots=False,
            tree_cache=self._adder(static=True).tree_cache,
        )
        self.assertEqual(PrefillAdder._mamba_match_headroom(shared), 3)


class _FakeArgs:
    def __init__(
        self,
        system_prompt,
        model_path="/tmp/model",
        tokenizer_path=None,
        trust_remote_code=True,
        mamba_cache_chunk_size=64,
    ):
        self.system_prompt = system_prompt
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.trust_remote_code = trust_remote_code
        self.mamba_cache_chunk_size = mamba_cache_chunk_size


if __name__ == "__main__":
    unittest.main()
