"""Per-cell contract of the DSA compatibility table (RFC #31774).

Guards: defaults equal the historical if/elif spec over the full input
product; rejection fires exactly on unsupported cells; the table stays in
sync with ``DSA_CHOICES``.
"""

import itertools
import unittest
from typing import Optional, Tuple

from sglang.srt.arg_groups.dsa_compat import (
    DSA_COMPAT_TABLE,
    KV_BUCKETS,
    PHASES,
    PLATFORM_HIP,
    PLATFORM_SM100,
    PLATFORMS,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    check_dsa_backend_compat,
    lookup_cell,
    resolve_dsa_default_backends,
    supported_backends,
)
from sglang.srt.server_args import DSA_CHOICES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _historical_defaults(
    sm_major: int,
    hip: bool,
    kv_cache_dtype: str,
    user_set_prefill: bool,
    user_set_decode: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """The pre-table resolver branches, kept verbatim as the spec."""
    if not user_set_prefill and not user_set_decode and hip:
        return "tilelang", "tilelang"
    if kv_cache_dtype == "fp8_e4m3":
        default = "trtllm" if sm_major >= 10 else "flashmla_kv"
        return (
            None if user_set_prefill else default,
            None if user_set_decode else default,
        )
    return (
        None if user_set_prefill else "flashmla_sparse",
        None if user_set_decode else ("trtllm" if sm_major >= 10 else "fa3"),
    )


class TestDsaDefaultsMatchHistorical(CustomTestCase):

    def test_full_input_product(self):
        for (
            sm_major,
            hip,
            kv_cache_dtype,
            user_set_prefill,
            user_set_decode,
        ) in itertools.product(
            (8, 9, 10, 12),
            (False, True),
            ("auto", "bfloat16", "fp8_e5m2", "fp8_e4m3"),
            (False, True),
            (False, True),
        ):
            expected = _historical_defaults(
                sm_major, hip, kv_cache_dtype, user_set_prefill, user_set_decode
            )
            got = resolve_dsa_default_backends(
                sm_major=sm_major,
                hip=hip,
                kv_cache_dtype=kv_cache_dtype,
                user_set_prefill=user_set_prefill,
                user_set_decode=user_set_decode,
            )
            self.assertEqual(
                got,
                expected,
                msg=(
                    f"defaults diverged for sm_major={sm_major} hip={hip} "
                    f"kv={kv_cache_dtype} user_set_prefill={user_set_prefill} "
                    f"user_set_decode={user_set_decode}"
                ),
            )


class TestDsaCompatValidation(CustomTestCase):

    def test_rejects_exactly_the_unsupported_cells(self):
        for backend, platform, kv_dtype, phase in itertools.product(
            DSA_CHOICES, PLATFORMS, KV_BUCKETS, PHASES
        ):
            cell = lookup_cell(
                backend=backend, platform=platform, kv_dtype=kv_dtype, phase=phase
            )
            kwargs = dict(
                kv_cache_dtype=kv_dtype,
                prefill_backend=backend if phase == "prefill" else None,
                decode_backend=backend if phase == "decode" else None,
                sm_major=10 if platform == PLATFORM_SM100 else 9,
                hip=platform == PLATFORM_HIP,
            )
            if cell.status == STATUS_UNSUPPORTED:
                with self.assertRaises(ValueError, msg=f"expected reject: {cell}"):
                    check_dsa_backend_compat(**kwargs)
            else:
                check_dsa_backend_compat(**kwargs)

    def test_error_names_supported_alternatives(self):
        # the actionable hint from #31346: an fp8-capable backend, plus the
        # keep-backend-switch-dtype route
        with self.assertRaises(ValueError) as ctx:
            check_dsa_backend_compat(
                kv_cache_dtype="fp8_e4m3",
                prefill_backend="tilelang",
                decode_backend=None,
                sm_major=9,
                hip=False,
            )
        msg = str(ctx.exception)
        self.assertIn("flashmla_kv", msg)
        self.assertIn("bfloat16", msg)


class TestDsaCompatTableWellFormed(CustomTestCase):

    def test_backends_exist_in_dsa_choices(self):
        for cell in DSA_COMPAT_TABLE:
            self.assertIn(cell.backend, DSA_CHOICES)

    def test_axis_values_are_valid(self):
        for cell in DSA_COMPAT_TABLE:
            self.assertIn(cell.platform, PLATFORMS)
            self.assertIn(cell.kv_dtype, KV_BUCKETS)
            self.assertIn(cell.phase, PHASES)

    def test_unsupported_cells_carry_reason_and_evidence(self):
        for cell in DSA_COMPAT_TABLE:
            if cell.status == STATUS_UNSUPPORTED:
                self.assertTrue(cell.reason, msg=f"missing reason: {cell}")
                self.assertTrue(cell.evidence, msg=f"missing evidence: {cell}")

    def test_default_is_unambiguous(self):
        for platform, kv_dtype, phase in itertools.product(
            PLATFORMS, KV_BUCKETS, PHASES
        ):
            priorities = [
                cell.default_priority
                for cell in DSA_COMPAT_TABLE
                if cell.platform == platform
                and cell.kv_dtype == kv_dtype
                and cell.phase == phase
                and cell.status == STATUS_SUPPORTED
                and cell.default_priority is not None
            ]
            if priorities:
                self.assertEqual(
                    priorities.count(max(priorities)),
                    1,
                    msg=f"ambiguous default for {platform}/{kv_dtype}/{phase}",
                )

    def test_only_supported_cells_can_default(self):
        for cell in DSA_COMPAT_TABLE:
            if cell.default_priority is not None:
                self.assertEqual(cell.status, STATUS_SUPPORTED, msg=str(cell))

    def test_supported_backends_is_nonempty_where_something_rejects(self):
        # every rejection must be able to name an alternative
        rejecting = {
            (cell.platform, cell.kv_dtype, cell.phase)
            for cell in DSA_COMPAT_TABLE
            if cell.status == STATUS_UNSUPPORTED
        }
        for platform, kv_dtype, phase in rejecting:
            self.assertTrue(
                supported_backends(platform=platform, kv_dtype=kv_dtype, phase=phase),
                msg=f"no supported alternative for {platform}/{kv_dtype}/{phase}",
            )


if __name__ == "__main__":
    unittest.main()
