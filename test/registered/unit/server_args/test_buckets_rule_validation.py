# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Malformed --prompt-tokens-buckets / --generation-tokens-buckets must always be rejected.

The numeric-parameter checks in validate_buckets_rule previously sat behind
`assert False` in the except branch. Under `python -O` asserts are stripped,
so a rule like ['tse', 'not-a-float', '2', '8'] fell through the except
branch with no error at all and the bad config was silently accepted.

Pinned here: both malformed shapes raise ValueError (not AssertionError, so
the guard survives python -O), and valid rules still pass.

    python -m pytest test/registered/unit/server_args/test_buckets_rule_validation.py -v
"""

import unittest

from sglang.srt.arg_groups.validation_hook import validate_buckets_rule
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ARG = "--generation-tokens-buckets"


class TestBucketsRuleValidation(unittest.TestCase):
    def test_tse_non_numeric_params_rejected(self):
        with self.assertRaises(ValueError):
            validate_buckets_rule(ARG, ["tse", "not-a-float", "2", "8"])

    def test_tse_missing_param_rejected(self):
        # IndexError path: fewer than 4 tokens already raises on the length
        # assert; the except branch covers parse failures.
        with self.assertRaises((ValueError, AssertionError)):
            validate_buckets_rule(ARG, ["tse", "1", "2"])

    def test_custom_non_numeric_rejected(self):
        with self.assertRaises(ValueError):
            validate_buckets_rule(ARG, ["custom", "1", "abc"])

    def test_valid_rules_accepted(self):
        validate_buckets_rule(ARG, ["tse", "128", "2", "8"])
        validate_buckets_rule(ARG, ["default"])
        validate_buckets_rule(ARG, ["custom", "1", "2", "4"])

    def test_rejection_is_not_assertion(self):
        # The guard must not rely on assert: python -O strips asserts, which
        # is what made the malformed rules silently pass before this fix.
        for bad in (["tse", "x", "2", "8"], ["custom", "abc"]):
            with self.assertRaises(ValueError) as cm:
                validate_buckets_rule(ARG, bad)
            self.assertNotIsInstance(cm.exception, AssertionError)


if __name__ == "__main__":
    unittest.main()
