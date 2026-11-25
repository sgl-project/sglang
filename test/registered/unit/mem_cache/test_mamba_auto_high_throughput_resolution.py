"""`auto_high_throughput` through the real resolution pipeline.

The unit tests next door call the policy pass directly. These run
``ServerArgs.resolve_once()`` on a throwaway model config, so the passes that
disable the overlap schedule after the mamba strategy was first resolved
actually execute. That ordering is the whole reason the policy resolves late,
so mocking it away would test nothing.
"""

import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

HYBRID_ARCH = "Qwen3_5MoeForConditionalGeneration"

MINI_CONFIG = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "vocab_size": 512,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16",
    # MLA shape fields: required by some hybrid arches, inert extras for the
    # rest (same reason test_model_overrides carries them).
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
}

# An architecture resolves against its own model_type.
MODEL_TYPES = {
    "Qwen3_5MoeForConditionalGeneration": "qwen3_5_moe",
    "Glm5NextForConditionalGeneration": "glm5_next",
    "LlamaForCausalLM": "llama",
}


class _NoSkipEnv:
    """SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK absent from os.environ.

    `is_set()` reads os.environ, so "unset" must be constructed: another test
    in the same process may have left the variable behind.
    """

    def __enter__(self):
        self._name = envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.name
        self._had = self._name in os.environ
        self._old = os.environ.pop(self._name, None)
        return self

    def __exit__(self, *exc):
        if self._had:
            os.environ[self._name] = self._old
        else:
            os.environ.pop(self._name, None)
        return False


class TestAutoHighThroughputResolution(CustomTestCase):
    def _resolve(self, *, arch=HYBRID_ARCH, config_extra=None, **server_kwargs):
        from sglang.srt.server_args import ServerArgs

        server_kwargs.setdefault("device", "cuda")
        config = dict(
            MINI_CONFIG,
            architectures=[arch],
            model_type=MODEL_TYPES.get(arch, "qwen3_5_moe"),
        )
        config.update(config_extra or {})
        d = tempfile.mkdtemp(prefix="auto_ht_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump(config, fh)
        sa = ServerArgs(model_path=d, **server_kwargs)
        sa.resolve_once()
        return sa

    @staticmethod
    def _effective(sa, field):
        from sglang.srt.arg_groups.model_override_base import resolved_view

        return getattr(resolved_view(sa), field)

    @staticmethod
    def _skip_lock(sa):
        """The effective decode-lock answer, read through the same function the
        pool sizing and the radix cache read through."""
        from sglang.srt.arg_groups.model_override_base import resolved_view
        from sglang.srt.arg_groups.overrides import mamba_skip_decode_lock_of

        return mamba_skip_decode_lock_of(resolved_view(sa))

    def _strategy(self, sa):
        return self._effective(sa, "mamba_radix_cache_strategy")

    # 1
    def test_default_auto_is_untouched(self):
        with _NoSkipEnv():
            sa = self._resolve()
        self.assertEqual(self._strategy(sa), "extra_buffer")
        self.assertFalse(self._effective(sa, "mamba_high_throughput_capacity"))

    # 2
    def test_eligible_auto_high_throughput_takes_the_lazy_buffer(self):
        with _NoSkipEnv():
            sa = self._resolve(mamba_radix_cache_strategy="auto_high_throughput")
        self.assertEqual(self._strategy(sa), "extra_buffer_lazy")
        self.assertTrue(self._effective(sa, "mamba_high_throughput_capacity"))

    # 3: pipeline parallelism disables overlap in a pass that runs after the
    # mamba strategy was first resolved.
    def test_pipeline_parallel_overlap_disable_forces_the_fallback(self):
        with _NoSkipEnv():
            sa = self._resolve(
                mamba_radix_cache_strategy="auto_high_throughput", pp_size=2
            )
        self.assertTrue(self._effective(sa, "disable_overlap_schedule"))
        self.assertEqual(self._strategy(sa), "extra_buffer")
        self.assertFalse(self._effective(sa, "mamba_high_throughput_capacity"))

    # 6: overlap is already off when the early pass runs, so `auto`'s own branch
    # picks no_buffer there -- the same choice plain `auto` would make. The
    # policy must not turn that into a lazy buffer.
    def test_explicitly_disabled_overlap_takes_autos_own_choice(self):
        with _NoSkipEnv():
            ht = self._resolve(
                mamba_radix_cache_strategy="auto_high_throughput",
                disable_overlap_schedule=True,
            )
            plain = self._resolve(disable_overlap_schedule=True)
        self.assertEqual(self._strategy(ht), self._strategy(plain))
        self.assertNotEqual(self._strategy(ht), "extra_buffer_lazy")
        self.assertFalse(self._effective(ht, "mamba_high_throughput_capacity"))

    # 7 / 8
    def test_explicit_env_decides_the_lock_not_the_policy(self):
        for value in (False, True):
            with self.subTest(env=value):
                with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(value):
                    sa = self._resolve(
                        mamba_radix_cache_strategy="auto_high_throughput"
                    )
                    # The strategy still resolves; only the lock differs.
                    self.assertEqual(self._strategy(sa), "extra_buffer_lazy")
                    self.assertEqual(self._skip_lock(sa), value)

    def test_policy_supplies_the_lock_when_the_env_is_unset(self):
        with _NoSkipEnv():
            sa = self._resolve(mamba_radix_cache_strategy="auto_high_throughput")
            self.assertTrue(self._skip_lock(sa))
            sa_default = self._resolve()
            self.assertFalse(self._skip_lock(sa_default))

    # 10
    def test_radix_cache_off_leaves_the_policy_inert(self):
        with _NoSkipEnv():
            sa = self._resolve(
                mamba_radix_cache_strategy="auto_high_throughput",
                disable_radix_cache=True,
            )
        self.assertFalse(self._effective(sa, "mamba_high_throughput_capacity"))
        self.assertFalse(self._skip_lock(sa))

    def test_explicit_strategy_survives_resolution(self):
        for strategy in ("extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                with _NoSkipEnv():
                    sa = self._resolve(mamba_radix_cache_strategy=strategy)
                self.assertEqual(self._strategy(sa), strategy)
                self.assertFalse(self._effective(sa, "mamba_high_throughput_capacity"))

    # 4: the sparse-head embedding path disables overlap in a pass that runs
    # right after the mamba strategy was first resolved.
    def test_sparse_head_overlap_disable_forces_the_fallback(self):
        with _NoSkipEnv():
            with envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.override("mean"):
                sa = self._resolve(mamba_radix_cache_strategy="auto_high_throughput")
        self.assertTrue(self._effective(sa, "disable_overlap_schedule"))
        self.assertEqual(self._strategy(sa), "extra_buffer")
        self.assertFalse(self._effective(sa, "mamba_high_throughput_capacity"))

    # 9: a model that pins its own strategy while the user is on an auto token
    # keeps winning, so asking for throughput cannot cost that model its own
    # verified choice.
    def test_model_override_wins_over_the_policy(self):
        with _NoSkipEnv():
            ht = self._resolve(
                arch="Glm5NextForConditionalGeneration",
                mamba_radix_cache_strategy="auto_high_throughput",
            )
            plain = self._resolve(arch="Glm5NextForConditionalGeneration")
        self.assertEqual(self._strategy(ht), self._strategy(plain))
        self.assertFalse(self._effective(ht, "mamba_high_throughput_capacity"))

    # Where the policy cannot act, its outcome has to equal plain `auto`'s, field
    # by field. `auto` is not always `extra_buffer`: on a model that does not
    # route through the hybrid-mamba radix cache, and with the radix cache off,
    # the early pass declares nothing and `auto` survives unresolved.
    PARITY_FIELDS = (
        "mamba_radix_cache_strategy",
        "uses_mamba_radix_cache",
        "mamba_high_throughput_capacity",
    )

    def _parity(self, **kw):
        with _NoSkipEnv():
            plain = self._resolve(mamba_radix_cache_strategy="auto", **kw)
            ht = self._resolve(mamba_radix_cache_strategy="auto_high_throughput", **kw)
        for field in self.PARITY_FIELDS:
            self.assertEqual(
                self._effective(plain, field), self._effective(ht, field), field
            )
        self.assertEqual(self._skip_lock(plain), self._skip_lock(ht))
        return plain, ht

    def test_non_mamba_model_matches_plain_auto(self):
        """A non-mamba architecture must not even raise: an auto token asked the
        engine to decide, and plain `auto` decides nothing here."""
        plain, ht = self._parity(arch="LlamaForCausalLM")
        self.assertEqual(self._strategy(ht), "auto")
        self.assertFalse(self._effective(ht, "uses_mamba_radix_cache"))

    def test_radix_cache_off_matches_plain_auto(self):
        plain, ht = self._parity(disable_radix_cache=True)
        self.assertEqual(self._strategy(ht), "auto")

    def test_pipeline_parallel_fallback_matches_plain_auto(self):
        plain, ht = self._parity(pp_size=2)
        self.assertEqual(self._strategy(ht), "extra_buffer")

    def test_explicit_strategy_on_a_non_mamba_model_still_fails_fast(self):
        """Relaxing the guard for auto tokens must not relax it for an explicit
        strategy, which is what the guard was written for."""
        with self.assertRaisesRegex(ValueError, "needs mamba state"):
            with _NoSkipEnv():
                self._resolve(
                    arch="LlamaForCausalLM", mamba_radix_cache_strategy="extra_buffer"
                )

    def test_no_unresolved_token_survives_resolution(self):
        """Whatever else happens, the sentinel must not reach runtime readers."""
        cases = [
            {},
            {"pp_size": 2},
            {"disable_overlap_schedule": True},
            {"disable_radix_cache": True},
        ]
        for extra in cases:
            with self.subTest(**extra):
                with _NoSkipEnv():
                    sa = self._resolve(
                        mamba_radix_cache_strategy="auto_high_throughput", **extra
                    )
                self.assertNotEqual(self._strategy(sa), "auto_high_throughput")
                self.assertIn(
                    self._strategy(sa),
                    # `auto` is a legal end state: it is where plain `auto` also
                    # stops when the early pass declines to resolve.
                    ("auto", "extra_buffer", "extra_buffer_lazy", "no_buffer"),
                )


if __name__ == "__main__":
    unittest.main()
