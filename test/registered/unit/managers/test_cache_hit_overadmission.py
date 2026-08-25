import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.cache_hit_overadmission import (
    evaluate_cache_hit_overadmission,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def make_req(
    *,
    input_tokens=100,
    cached_tokens=95,
    max_new_tokens=64,
    ignore_eos=False,
    **overrides,
):
    values = dict(
        is_retracted=False,
        output_ids=[],
        sampling_params=SimpleNamespace(
            ignore_eos=ignore_eos,
            max_new_tokens=max_new_tokens,
        ),
        session=None,
        multimodal_inputs=None,
        input_embeds=None,
        positional_embed_overrides=None,
        host_hit_length=0,
        swa_host_hit_length=0,
        storage_hit_length=0,
        mamba_host_hit_length=0,
        full_untruncated_fill_ids=list(range(input_tokens)),
        prefix_indices=list(range(cached_tokens)),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def evaluate(req):
    return evaluate_cache_hit_overadmission(
        req,
        min_hit_ratio=0.95,
        max_uncached_prefill_tokens=128,
        max_new_tokens=64,
    )


class TestCacheHitOveradmissionEligibility(unittest.TestCase):
    def test_accepts_inclusive_boundaries(self):
        decision = evaluate(make_req())

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "admitted")
        self.assertEqual(decision.hit_ratio, 0.95)
        self.assertEqual(decision.remaining_new_tokens, 64)

        uncached_boundary = evaluate(make_req(input_tokens=3000, cached_tokens=2872))
        self.assertTrue(uncached_boundary.allowed)
        self.assertEqual(uncached_boundary.uncached_tokens, 128)

    def test_rejects_hit_ratio_below_boundary(self):
        decision = evaluate(make_req(cached_tokens=94))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "hit_ratio")

    def test_rejects_uncached_tokens_above_boundary(self):
        decision = evaluate(make_req(input_tokens=3000, cached_tokens=2871))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "uncached_prefill")
        self.assertEqual(decision.uncached_tokens, 129)

    def test_rejects_output_budget_above_boundary(self):
        decision = evaluate(make_req(max_new_tokens=65))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_new_tokens")

    def test_rejects_ignore_eos(self):
        decision = evaluate(make_req(ignore_eos=True))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "ignore_eos")

    def test_rejects_non_device_cache_hits(self):
        decision = evaluate(make_req(host_hit_length=95))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "non_device_cache")

    def test_rejects_request_types_with_non_fresh_state(self):
        cases = {
            "retracted": dict(is_retracted=True),
            "continuation": dict(output_ids=[1]),
            "session": dict(session=SimpleNamespace()),
            "unsupported_input": dict(multimodal_inputs=SimpleNamespace()),
        }
        for reason, overrides in cases.items():
            with self.subTest(reason=reason):
                decision = evaluate(make_req(**overrides))
                self.assertFalse(decision.allowed)
                self.assertEqual(decision.reason, reason)

        input_embeds = evaluate(make_req(input_embeds=[[0.0]]))
        self.assertFalse(input_embeds.allowed)
        self.assertEqual(input_embeds.reason, "unsupported_input")


if __name__ == "__main__":
    unittest.main()
