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
"""`--cache-salt-ttl-seconds` refuses what it cannot bound.

The TTL is a retention bound, so a configuration it only half covers is worse
than no TTL at all: it reads as a guarantee. Every tier that keeps a copy of
the KV outside the device radix tree keys that copy by a salt-free token hash
and offers no per-key delete, so each flag that builds one has to be rejected
-- `--enable-hierarchical-cache` is not the only one.

    python -m pytest test/registered/unit/server_args/test_cache_salt_ttl_gates.py -v
"""

import unittest

import msgspec

from sglang.srt.arg_groups.validation_hook import validate_cache_salt_ttl
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _validate(**overrides) -> ServerArgs:
    sa = ServerArgs(model_path="dummy")
    fields = {"cache_salt_ttl_seconds": 10.0}
    fields.update(overrides)
    for name, value in fields.items():
        msgspec.Struct.__setattr__(sa, name, value)
    validate_cache_salt_ttl(sa)
    return sa


class TestCacheSaltTtlGates(unittest.TestCase):
    def test_a_plain_ttl_launch_is_accepted(self):
        """An over-broad gate here would reject every TTL launch."""
        sa = _validate()
        # The ceiling on a client-shortened TTL defaults to the server's own.
        self.assertEqual(sa.cache_salt_ttl_max_seconds, 10.0)

    def test_the_ttl_is_inert_when_unset(self):
        # Every gate below has to stay out of the way of a default server.
        _validate(cache_salt_ttl_seconds=None, enable_hierarchical_cache=True)

    def test_every_flag_that_builds_a_second_tier_is_refused(self):
        for name, override in (
            ("hierarchical cache", {"enable_hierarchical_cache": True}),
            ("storage backend", {"hicache_storage_backend": "file"}),
            (
                "retraction backup",
                {"disaggregation_decode_retraction_backup": "host_pool"},
            ),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "no per-key delete"):
                    _validate(**override)

    def test_a_second_clock_or_an_unsupported_tree_is_refused(self):
        with self.assertRaisesRegex(ValueError, "tokenizer-worker-num"):
            _validate(tokenizer_worker_num=2)
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
            with self.assertRaisesRegex(ValueError, "expire_cache_salts"):
                _validate()
        with envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.override(True):
            with self.assertRaisesRegex(ValueError, "C\\+\\+"):
                _validate()

    def test_a_client_ceiling_below_the_server_default_is_refused(self):
        # The clamp is min(requested, max), so a ceiling under the default
        # would silently shorten every request that asked for nothing.
        with self.assertRaisesRegex(ValueError, "must be >="):
            _validate(cache_salt_ttl_max_seconds=5.0)

    def test_zeroize_is_refused_where_it_cannot_order_against_the_reads(self):
        with self.assertRaisesRegex(ValueError, "two-batch-overlap"):
            _validate(cache_salt_ttl_zeroize=True, enable_two_batch_overlap=True)
        # ...but the flag alone is fine.
        _validate(cache_salt_ttl_zeroize=True)


if __name__ == "__main__":
    unittest.main()
