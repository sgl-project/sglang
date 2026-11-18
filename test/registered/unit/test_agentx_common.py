"""Unit tests for the InferenceX row normalization in sglang/test/agentx_common.py.

Needs a checkout of InferenceX at INFERENCEX_COMMIT: set AGENTX_INFERENCEX_E2E
to its ``inferencex-e2e`` directory, otherwise the tests are skipped.
"""

import os
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

E2E = os.environ.get("AGENTX_INFERENCEX_E2E", "")


@unittest.skipUnless(E2E, "AGENTX_INFERENCEX_E2E is not set")
class TestNormalizedEntry(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, E2E)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(E2E)

    def normalize(self, **row):
        from sglang.test.agentx_common import _normalized_entry

        return _normalized_entry({"tp": 2, "kv-offloading": "none", **row})

    def test_defaults(self):
        row = self.normalize(**{"conc-list": [1, 4]})
        self.assertEqual(row["spec-decoding"], "none")
        self.assertEqual(row["dcp-size"], 1)
        self.assertEqual(row["ep"], 1)
        self.assertEqual(row["conc-list"], [1, 4])

    def test_conc_range(self):
        row = self.normalize(**{"conc-start": 1, "conc-end": 10})
        self.assertEqual(row["conc-list"], [1, 2, 4, 8, 10])

    def test_given_values_kept(self):
        row = self.normalize(
            **{"conc-list": [1], "ep": 2, "dcp-size": 2, "spec-decoding": "mtp"}
        )
        self.assertEqual(
            (row["ep"], row["dcp-size"], row["spec-decoding"]), (2, 2, "mtp")
        )

    def test_null_ep_is_one_and_zero_stays(self):
        self.assertEqual(self.normalize(**{"conc-list": [1], "ep": None})["ep"], 1)
        self.assertEqual(self.normalize(**{"conc-list": [1], "ep": 0})["ep"], 0)

    def test_null_dcp_size_rejected(self):
        # InferenceX's schema rejects it, so the harness must not accept it either.
        with self.assertRaises(Exception):
            self.normalize(**{"conc-list": [1], "dcp-size": None})


if __name__ == "__main__":
    unittest.main()
