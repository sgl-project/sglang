import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_ROPE_KWARGS = dict(
    head_size=64,
    rotary_dim=64,
    max_position=256,
    base=500000,
    is_neox_style=False,
)


class TestRopeCacheInvalidation(CustomTestCase):
    """`get_rope` hands out a process-wide shared module, so a model teardown
    that meta-izes or frees its module tree also kills the cache entry for every
    later model. Rebuilding is the only way that stays correct — a dead cos/sin
    cache makes the in-place RoPE ops no-op instead of raising.
    """

    def setUp(self):
        cpu_patch = patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        cpu_patch.start()
        self.addCleanup(cpu_patch.stop)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        _ROPE_DICT.clear()

    def tearDown(self):
        _ROPE_DICT.clear()

    def test_rebuilds_after_meta_invalidation(self):
        rope = get_rope(**_ROPE_KWARGS)
        expected = rope.cos_sin_cache.clone()

        rope.to(device="meta")
        self.assertEqual(rope.cos_sin_cache.device.type, "meta")

        rebuilt = get_rope(**_ROPE_KWARGS)
        self.assertIsNot(rebuilt, rope)
        self.assertNotEqual(rebuilt.cos_sin_cache.device.type, "meta")
        self.assertTrue(torch.equal(rebuilt.cos_sin_cache, expected))

    def test_rebuilds_after_storage_release(self):
        rope = get_rope(**_ROPE_KWARGS)
        expected = rope.cos_sin_cache.clone()

        rope.cos_sin_cache.untyped_storage().resize_(0)

        rebuilt = get_rope(**_ROPE_KWARGS)
        self.assertIsNot(rebuilt, rope)
        self.assertTrue(torch.equal(rebuilt.cos_sin_cache, expected))

    def test_live_entry_is_still_shared(self):
        rope = get_rope(**_ROPE_KWARGS)
        self.assertIs(get_rope(**_ROPE_KWARGS), rope)

    def test_meta_build_keeps_sharing(self):
        # Meta-device construction passes build meta buffers on purpose; they
        # must keep sharing one entry rather than rebuilding on every layer.
        with torch.device("meta"):
            rope = get_rope(**_ROPE_KWARGS)
            self.assertEqual(rope.cos_sin_cache.device.type, "meta")
            self.assertIs(get_rope(**_ROPE_KWARGS), rope)

        # ...and the real build afterwards must not inherit the meta entry.
        rebuilt = get_rope(**_ROPE_KWARGS)
        self.assertIsNot(rebuilt, rope)
        self.assertNotEqual(rebuilt.cos_sin_cache.device.type, "meta")


if __name__ == "__main__":
    unittest.main()
