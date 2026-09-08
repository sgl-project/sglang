"""RecoverSSM accepts linear drafts and rejects incompatible replay modes."""

import unittest
from types import SimpleNamespace

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRecoverSSMConfig(CustomTestCase):
    def test_none_mode_requires_linear_drafts_and_no_replayssm(self):
        for mode in ("full", "none"):
            for topk in (None, 1, 2):
                for replay, replay_spec in (
                    (False, False),
                    (True, False),
                    (False, True),
                ):
                    with self.subTest(
                        mode=mode, topk=topk, replay=replay, replay_spec=replay_spec
                    ):
                        cfg = SimpleNamespace(
                            gdn_mtp_cache_mode=mode,
                            speculative_eagle_topk=topk,
                            enable_linear_replayssm=replay,
                            enable_linear_replayssm_spec=replay_spec,
                        )
                        if mode == "none" and (topk == 2 or replay or replay_spec):
                            with self.assertRaises(ValueError):
                                ServerArgs._validate_gdn_mtp_cache_mode(cfg)
                        else:
                            ServerArgs._validate_gdn_mtp_cache_mode(cfg)

    def test_validation_observes_resolved_overrides(self):
        cfg = SimpleNamespace(
            gdn_mtp_cache_mode="full",
            speculative_eagle_topk=2,
            enable_linear_replayssm=False,
            enable_linear_replayssm_spec=False,
            _resolved_overrides=[("test", {"gdn_mtp_cache_mode": "none"})],
        )
        with self.assertRaisesRegex(ValueError, "linear draft chain"):
            ServerArgs._validate_gdn_mtp_cache_mode(cfg)


if __name__ == "__main__":
    unittest.main()
