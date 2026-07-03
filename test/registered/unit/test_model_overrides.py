"""Unit tests for the model-override machinery: whitelist metadata (V3a)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import dataclasses
import unittest
from typing import Optional

from sglang.srt.arg_groups.arg_utils import A, Arg, model_overridable_fields
from sglang.test.test_utils import CustomTestCase


@dataclasses.dataclass
class _FakeArgs:
    plain: A[int, "help text only"] = 0
    resolved_by_model: A[str, Arg(help="x", model_overridable=True)] = "auto"
    also_resolved: A[Optional[int], Arg(help="y", model_overridable=True)] = None
    metadata_but_not_overridable: A[bool, Arg(help="z")] = False


class TestModelOverridableWhitelist(CustomTestCase):
    def test_arg_defaults_to_not_overridable(self):
        self.assertFalse(Arg().model_overridable)

    def test_whitelist_derivation_from_annotated_metadata(self):
        self.assertEqual(
            model_overridable_fields(_FakeArgs),
            frozenset({"resolved_by_model", "also_resolved"}),
        )

    def test_server_args_whitelist_empty_at_skeleton(self):
        # No ServerArgs field is tagged yet: the V3 sweeps whitelist fields
        # one family at a time. This pin makes accidental tagging visible.
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(model_overridable_fields(ServerArgs), frozenset())


if __name__ == "__main__":
    unittest.main()
