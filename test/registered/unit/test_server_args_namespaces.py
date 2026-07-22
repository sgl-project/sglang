"""Coverage lint for the ServerArgs -> RuntimeContext namespace split.

Every ServerArgs field must carry an ``NS("<path>")`` marker in its ``Annotated``
metadata, and every path must be one of the known domains. This is the guardrail
that fails when an upstream PR adds a ServerArgs field without assigning it a
namespace (the property that retires the old hand-maintained mirror file).
"""

import dataclasses
import unittest

from sglang.srt.arg_groups.arg_utils import namespace_of
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Locked taxonomy (global_context/11-server-args-namespace-split.md).
VALID_NAMESPACES = {
    "parallel",
    "device",
    "model",
    "schedule",
    "memory",
    "spec",
    "lora",
    "mm",
    "disagg",
    "serving",
    "observability",
    "exec.kernel",
    "exec.moe",
    "exec.graph",
    "exec.comm",
    "exec.mamba",
    "exec.overlap",
    "exec.offload",
    "exec.dllm",
    "exec.deterministic",
    "exec.features",
}


def _field_names():
    return {f.name for f in dataclasses.fields(ServerArgs)}


class TestServerArgsNamespaces(CustomTestCase):
    def test_every_field_has_a_namespace(self):
        nsmap = namespace_of(ServerArgs)
        missing = sorted(_field_names() - set(nsmap))
        self.assertFalse(
            missing,
            "ServerArgs fields missing an NS(...) marker "
            f"(assign a namespace in server_args.py): {missing}",
        )

    def test_all_namespaces_are_known(self):
        nsmap = namespace_of(ServerArgs)
        bad = {f: p for f, p in nsmap.items() if p not in VALID_NAMESPACES}
        self.assertFalse(bad, f"unknown namespace paths (typo or new domain?): {bad}")

    def test_namespace_map_covers_all_fields(self):
        nsmap = namespace_of(ServerArgs)
        self.assertEqual(set(nsmap), _field_names())
        self.assertGreaterEqual(len(nsmap), 440)


if __name__ == "__main__":
    unittest.main()
