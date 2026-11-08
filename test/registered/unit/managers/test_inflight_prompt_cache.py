"""Regression tests for duplicate prefill requests in the same scheduling step."""

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.inflight_prompt_cache import InflightPromptCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_req(
    tokens=range(64), *, extra_key=None, cache_salt=None, prefix_len=0, **overrides
):
    req = SimpleNamespace(
        origin_input_ids=array("q", tokens),
        output_ids=array("q"),
        prefix_len=prefix_len,
        extra_key=extra_key,
        cache_salt=cache_salt,
        skip_radix_cache_insert=False,
        return_logprob=False,
        multimodal_inputs=None,
        input_embeds=None,
        positional_embed_overrides=None,
        session=None,
        beam_group=None,
        is_retracted=False,
        kv=SimpleNamespace(is_kv_released=False),
        finished=lambda: False,
    )
    for name, value in overrides.items():
        setattr(req, name, value)
    return req


class TestInflightPromptCache(CustomTestCase):
    def test_follower_waits_only_until_leader_checkpoint(self):
        """Same-step duplicates must wait for the radix checkpoint, not prefill twice."""
        cache = InflightPromptCache()
        leader = make_req()
        follower = make_req()
        cache.admit(leader, is_chunked=False)
        self.assertTrue(cache.should_wait(follower))
        self.assertFalse(cache.should_wait(make_req(prefix_len=63)))

        leader.prefix_len = len(leader.origin_input_ids)
        self.assertFalse(cache.should_wait(follower))

    def test_namespaces_and_prompt_contents_do_not_cross(self):
        """A tenant or adapter mismatch must never delay a distinct request."""
        cache = InflightPromptCache()
        cache.admit(
            make_req(extra_key="adapter-a", cache_salt="tenant-a"), is_chunked=False
        )
        self.assertFalse(
            cache.should_wait(make_req(extra_key="adapter-a", cache_salt="tenant-b"))
        )
        self.assertFalse(
            cache.should_wait(make_req(extra_key="adapter-b", cache_salt="tenant-a"))
        )
        self.assertFalse(
            cache.should_wait(
                make_req(
                    tokens=[*range(63), 99],
                    extra_key="adapter-a",
                    cache_salt="tenant-a",
                )
            )
        )

    def test_aborted_leader_does_not_strand_followers(self):
        """A leader released without checkpoint must let the follower recompute."""
        cache = InflightPromptCache()
        leader = make_req()
        cache.admit(leader, is_chunked=False)
        self.assertTrue(cache.should_wait(make_req()))
        leader.finished = lambda: True
        leader.kv.is_kv_released = True
        cache.prune()
        self.assertFalse(cache.should_wait(make_req()))

    def test_unsupported_inputs_and_chunks_do_not_become_leaders(self):
        """Uncacheable or partial KV must not suppress a follower's prefill."""
        for leader, is_chunked in (
            (make_req(return_logprob=True), False),
            (make_req(multimodal_inputs=object()), False),
            (make_req(positional_embed_overrides=object()), False),
            (make_req(), True),
        ):
            cache = InflightPromptCache()
            cache.admit(leader, is_chunked=is_chunked)
            self.assertFalse(cache.should_wait(make_req()))


if __name__ == "__main__":
    unittest.main()
