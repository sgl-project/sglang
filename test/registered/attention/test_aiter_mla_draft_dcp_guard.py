"""Guard: aiter refuses an MLA draft model under decode context parallelism.

An MLA draft (EAGLE, NEXTN, STANDALONE) with ``--dcp-size > 1`` crashed at
startup inside the aiter extend with an opaque ``AttributeError`` ("'NoneType'
object has no attribute 'shape'"): the MLA model code sends the draft extend
through its DCP decode branch without the step's keys. The guard fails loudly
instead. It is inert for targets, non-MLA drafts, DCP 1, and ROCm DSA drafts that
shard their KV like the target.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _reject(*, dsa_draft_shards=False, **overrides):
    kwargs = dict(
        use_mla=True,
        dcp_world_size=2,
        is_draft_worker=True,
        hf_config=SimpleNamespace(),
    )
    kwargs.update(overrides)
    with patch.object(
        aiter_backend, "dsa_draft_shards_under_dcp", return_value=dsa_draft_shards
    ):
        AiterAttnBackend._reject_mla_draft_under_dcp(**kwargs)


class TestAiterMlaDraftDcpGuard(CustomTestCase):
    def test_raises_for_an_mla_draft_under_dcp(self):
        with self.assertRaises(ValueError) as ctx:
            _reject()
        self.assertIn("--dcp-size 2", str(ctx.exception))
        self.assertIn("MLA draft", str(ctx.exception))

    def test_inert_for_targets_non_mla_drafts_and_dcp1(self):
        _reject(is_draft_worker=False)
        _reject(use_mla=False)
        _reject(dcp_world_size=1)

    def test_inert_for_a_dsa_draft_that_shards_like_the_target(self):
        _reject(dsa_draft_shards=True)


if __name__ == "__main__":
    unittest.main()
