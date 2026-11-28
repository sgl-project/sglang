"""The `auto_high_throughput` mamba capacity policy: when it may act, and what
it resolves to when it may not.

The policy exists because `_mamba_radix_cache_resolution` runs before the
passes that disable the overlap schedule, and the lazy track buffer requires
that schedule. These tests pin both halves: the eligibility predicate, and the
ratio the configurator then computes from the resolved strategy.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VERIFIED_ARCH = "Qwen3_5MoeForConditionalGeneration"


def _view(**kw):
    """A resolved view with the fields the policy reads."""
    base = dict(
        mamba_radix_cache_strategy="auto_high_throughput",
        uses_mamba_radix_cache=True,
        disable_radix_cache=False,
        disable_overlap_schedule=False,
        disaggregation_mode="null",
        speculative_algorithm=None,
        linear_attn_backend="triton",
        page_size=1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _with_arch(arch):
    return patch(
        "sglang.srt.arg_groups.overrides.model_config_of",
        return_value=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])),
    )


class _unset_skip_env:
    """Context with SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK absent from os.environ.

    `is_set()` reads os.environ, so "unset" has to be constructed rather than
    assumed: another test may have left the variable behind.
    """

    def __enter__(self):
        import os

        self._name = envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.name
        self._had = self._name in os.environ
        self._old = os.environ.pop(self._name, None)
        return self

    def __exit__(self, *exc):
        import os

        if self._had:
            os.environ[self._name] = self._old
        else:
            os.environ.pop(self._name, None)
        return False


def _resolve(view, arch=VERIFIED_ARCH):
    from sglang.srt.arg_groups.overrides import _mamba_high_throughput_capacity

    with _with_arch(arch):
        return _mamba_high_throughput_capacity(view)


def _ratio(
    *,
    strategy,
    skip_env=None,
    capacity=False,
    disable_overlap=False,
    disable_radix=False,
):
    """The ratio the configurator computes for a published configuration."""
    from sglang.srt import runtime_context as rc
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    fake = SimpleNamespace(server_args=SimpleNamespace())
    overrides = dict(
        disable_radix_cache=disable_radix,
        disable_overlap_schedule=disable_overlap,
        mamba_radix_cache_strategy=strategy,
        mamba_high_throughput_capacity=capacity,
    )
    ctx = rc.get_context()
    # "no env" has to be constructed: is_set() reads os.environ, which another
    # test may have left populated.
    env_ctx = (
        _unset_skip_env()
        if skip_env is None
        else envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(skip_env)
    )
    with env_ctx:
        with ctx.override_server_args(**overrides):
            return KVCacheConfigurator._calculate_mamba_ratio(fake)


class TestMambaAutoHighThroughput(CustomTestCase):
    def test_case_a_default_auto_is_unchanged(self):
        """`auto` resolves as before and reserves five slots per request."""
        view = _view(mamba_radix_cache_strategy="auto")
        self.assertEqual(_resolve(view), {})
        self.assertEqual(_ratio(strategy="extra_buffer"), 5)

    def test_case_b_eligible_deployment_takes_three_slots(self):
        declared = _resolve(_view())
        self.assertEqual(declared["mamba_radix_cache_strategy"], "extra_buffer_lazy")
        self.assertTrue(declared["mamba_high_throughput_capacity"])
        self.assertEqual(_ratio(strategy="extra_buffer_lazy", capacity=True), 3)

    def test_case_c_explicit_strategy_is_never_overridden(self):
        for strategy in ("extra_buffer", "extra_buffer_lazy", "no_buffer"):
            with self.subTest(strategy=strategy):
                self.assertEqual(
                    _resolve(_view(mamba_radix_cache_strategy=strategy)), {}
                )

    def test_case_d_explicit_env_wins_over_the_policy(self):
        """A user who turned the skip off keeps it off, and pays the slot."""
        from sglang.srt.arg_groups.overrides import mamba_skip_decode_lock_of

        cfg = SimpleNamespace(mamba_high_throughput_capacity=True)
        with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False):
            self.assertFalse(mamba_skip_decode_lock_of(cfg))
        with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(True):
            self.assertTrue(mamba_skip_decode_lock_of(cfg))
        self.assertEqual(
            _ratio(strategy="extra_buffer_lazy", capacity=True, skip_env=False), 4
        )
        cfg_off = SimpleNamespace(mamba_high_throughput_capacity=False)
        with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(True):
            self.assertTrue(mamba_skip_decode_lock_of(cfg_off))

    def test_case_e_overlap_off_falls_back(self):
        declared = _resolve(_view(disable_overlap_schedule=True))
        self.assertEqual(declared["mamba_radix_cache_strategy"], "extra_buffer")
        self.assertFalse(declared["mamba_high_throughput_capacity"])
        # And the fallback is a configuration the configurator accepts.
        self.assertEqual(_ratio(strategy="extra_buffer", disable_overlap=True), 4)

    def test_case_f_pd_decode_falls_back(self):
        declared = _resolve(_view(disaggregation_mode="decode"))
        self.assertEqual(declared["mamba_radix_cache_strategy"], "extra_buffer")
        self.assertFalse(declared["mamba_high_throughput_capacity"])

    def test_case_g_model_without_extra_buffer_support_falls_back(self):
        """The architecture gate is capability, not a benchmarked-model list."""
        declared = _resolve(_view(), arch="LlamaForCausalLM")
        self.assertEqual(declared["mamba_radix_cache_strategy"], "extra_buffer")
        self.assertFalse(declared["mamba_high_throughput_capacity"])

    def test_capability_gate_not_an_evidence_gate(self):
        """Another extra-buffer architecture is eligible. The policy is an
        opt-in gated on capability; the measured scope lives in the flag help,
        not in a name list, so this must not silently become a whitelist."""
        from sglang.srt.arg_groups.overrides import mamba_high_throughput_eligible

        with _with_arch("Qwen3NextForCausalLM"):
            eligible, reason = mamba_high_throughput_eligible(_view())
        self.assertTrue(eligible, reason)

    def test_case_h_radix_cache_off_keeps_one_slot_per_request(self):
        declared = _resolve(_view(disable_radix_cache=True))
        self.assertFalse(declared["mamba_high_throughput_capacity"])
        for capacity in (False, True):
            with self.subTest(capacity=capacity):
                self.assertEqual(
                    _ratio(
                        strategy="extra_buffer_lazy",
                        capacity=capacity,
                        disable_radix=True,
                    ),
                    1,
                )

    def test_log_states_the_effective_slots_not_a_fixed_three(self):
        """eligible does not mean three slots: an explicit env=0 leaves the lock
        held, and the lazy strategy then reserves four."""
        from sglang.srt.arg_groups.overrides import (
            _mamba_high_throughput_capacity,
            mamba_high_throughput_slots,
        )

        self.assertEqual(mamba_high_throughput_slots(True), 3)
        self.assertEqual(mamba_high_throughput_slots(False), 4)

        cases = [
            (None, "on", "policy", 3),
            (True, "on", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1", 3),
            (False, "off", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0", 4),
        ]
        for env_value, state, source, slots in cases:
            with self.subTest(env=env_value):
                ctx = (
                    envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(env_value)
                    if env_value is not None
                    else _unset_skip_env()
                )
                with ctx:
                    with self.assertLogs(
                        "sglang.srt.arg_groups.overrides", level="INFO"
                    ) as cm:
                        with _with_arch(VERIFIED_ARCH):
                            declared = _mamba_high_throughput_capacity(_view())
                line = "\n".join(cm.output)
                self.assertEqual(
                    declared["mamba_radix_cache_strategy"], "extra_buffer_lazy"
                )
                self.assertIn(f"decode-lock skip {state}", line)
                self.assertIn(source, line)
                self.assertIn(f"reserves {slots} mamba state slots", line)
                # And the ratio the configurator computes agrees with the log.
                self.assertEqual(
                    _ratio(
                        strategy="extra_buffer_lazy", capacity=True, skip_env=env_value
                    ),
                    slots,
                )

    def test_fallback_log_names_the_reason(self):
        from sglang.srt.arg_groups.overrides import _mamba_high_throughput_capacity

        with self.assertLogs("sglang.srt.arg_groups.overrides", level="INFO") as cm:
            with _with_arch(VERIFIED_ARCH):
                _mamba_high_throughput_capacity(_view(disable_overlap_schedule=True))
        self.assertIn("falling back to extra_buffer", "\n".join(cm.output))
        self.assertIn("overlap schedule is off", "\n".join(cm.output))

    def test_case_i_effective_ratio_matches_the_manual_two_flag_setup(self):
        """The policy's effective configuration, not just its log line.

        Manual = extra_buffer_lazy plus the env flag. Auto = the policy. Both
        have to produce the same ratio.
        """
        manual = _ratio(strategy="extra_buffer_lazy", skip_env=True)
        auto = _ratio(strategy="extra_buffer_lazy", capacity=True)
        self.assertEqual(manual, 3)
        self.assertEqual(auto, manual)

    def test_case_j_resolution_is_idempotent(self):
        view = _view()
        first = _resolve(view)
        second = _resolve(view)
        self.assertEqual(first, second)
        # Re-resolving the already-resolved strategy is a no-op, which is what
        # the pass sees when it runs against a published configuration.
        self.assertEqual(
            _resolve(
                _view(mamba_radix_cache_strategy=first["mamba_radix_cache_strategy"])
            ),
            {},
        )

    def test_model_override_still_fires_under_the_new_token(self):
        """A model that pins its own strategy while the user left it on auto must
        keep doing so under `auto_high_throughput`, or asking for throughput
        would lose that model its verified choice."""
        from sglang.srt.arg_groups.model_override_base import mamba_strategy_is_auto

        for token, expected in (
            ("auto", True),
            ("auto_high_throughput", True),
            ("extra_buffer", False),
            ("extra_buffer_lazy", False),
            ("no_buffer", False),
        ):
            with self.subTest(token=token):
                self.assertEqual(
                    mamba_strategy_is_auto(
                        SimpleNamespace(mamba_radix_cache_strategy=token)
                    ),
                    expected,
                )

    def test_eligibility_reasons_are_specific(self):
        from sglang.srt.arg_groups.overrides import mamba_high_throughput_eligible

        cases = {
            "overlap schedule is off": _view(disable_overlap_schedule=True),
            "PD decode worker": _view(disaggregation_mode="decode"),
            "radix cache is disabled": _view(disable_radix_cache=True),
            "speculative decoding": _view(speculative_algorithm="EAGLE"),
            "hybrid-mamba radix cache": _view(uses_mamba_radix_cache=False),
        }
        for fragment, view in cases.items():
            with self.subTest(fragment=fragment):
                with _with_arch(VERIFIED_ARCH):
                    eligible, reason = mamba_high_throughput_eligible(view)
                self.assertFalse(eligible)
                self.assertIn(fragment, reason)
        with _with_arch(VERIFIED_ARCH):
            eligible, reason = mamba_high_throughput_eligible(_view())
        self.assertTrue(eligible)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
