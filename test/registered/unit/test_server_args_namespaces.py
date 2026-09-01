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
    def test_no_module_shadows_a_bag_accessor(self):
        """An accessor name bound twice in one module is a silent wrong read.

        This has happened twice. Once a module imported `get_model` from the
        context and a same-named helper from elsewhere, and once `get_device`
        -- which names three different things in this tree: the bag accessor,
        the device-string utility, and a platform method. The second import
        wins, the converted line calls the wrong callable, and the failure is
        an AttributeError on whichever branch reaches it, which for a
        per-pass recorder or a specific accelerator can be none of the ones a
        CPU suite runs. Nothing else notices; a name scan looks fine.
        """
        import ast
        import collections
        import pathlib as _pathlib

        import sglang

        srt = _pathlib.Path(sglang.__file__).resolve().parent / "srt"
        context_module = ast.parse(
            (srt / "runtime_context.py").read_text(encoding="utf-8-sig")
        )
        accessors = {
            node.name
            for node in context_module.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("get_")
        }
        self.assertGreater(len(accessors), 15, "the accessor derivation broke")

        shadowed = []
        for path in sorted(srt.rglob("*.py")):
            if path.name == "runtime_context.py":
                continue
            source = path.read_text(encoding="utf-8-sig")
            if "runtime_context" not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                self.fail(f"unparsable module in the census: {path}")
            bindings = collections.defaultdict(set)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    origin = node.module or ""
                    kind = (
                        "context"
                        if origin.endswith("runtime_context")
                        else f"{origin or '.'}"
                    )
                    for alias in node.names:
                        bindings[alias.asname or alias.name].add((kind, node.lineno))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        bindings[(alias.asname or alias.name).split(".")[0]].add(
                            ("import", node.lineno)
                        )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    bindings[node.name].add(("def", node.lineno))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            bindings[target.id].add(("assign", node.lineno))
            for name, where in bindings.items():
                if name not in accessors:
                    continue
                kinds = {kind for kind, _ in where}
                # The same accessor imported from the context more than once
                # (module level plus a lazy import inside a function) is one
                # object under one name; a *different* origin is the hazard.
                if "context" in kinds and kinds - {"context"}:
                    shadowed.append(
                        f"{path.relative_to(srt)}: {name} <- "
                        + ", ".join(
                            f"{k}@{l}" for k, l in sorted(where, key=lambda w: w[1])
                        )
                    )
        self.assertEqual(
            shadowed,
            [],
            "a bag accessor shares its name with another binding in the same "
            "module, so the converted reads call whichever import came last; "
            "alias one of them:\n  " + "\n  ".join(shadowed),
        )

    def test_the_readers_agree_with_the_namespace_metadata(self):
        """Two independent sources say where a leaf lives; they must match.

        The metadata is one source and the ~2000 hand-written reads
        (`get_schedule().chunked_prefill_size`) are the other. Checking the
        projection against the metadata cannot catch a field assigned to the
        wrong group -- both sides come from the same marker, so the check is
        true by construction. The readers are written by hand, so a
        disagreement means one of the two is wrong, and every reader on the
        losing side raises `has no leaf/subgroup` at runtime on whichever
        branch reaches it first.
        """
        import ast
        import pathlib as _pathlib

        import sglang

        srt = _pathlib.Path(sglang.__file__).resolve().parent / "srt"
        mapping = namespace_of(ServerArgs)
        accessors = {
            f"get_{group}" for group in {p.split(".")[0] for p in mapping.values()}
        }

        sites = 0
        disagreements = []
        for path in sorted(srt.rglob("*.py")):
            source = path.read_text(encoding="utf-8-sig")
            if not any(name in source for name in accessors):
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                self.fail(f"unparsable module in the census: {path}")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                chain, cursor = [], node
                while isinstance(cursor, ast.Attribute):
                    chain.append(cursor.attr)
                    cursor = cursor.value
                if not (
                    isinstance(cursor, ast.Call)
                    and isinstance(cursor.func, ast.Name)
                    and cursor.func.id in accessors
                ):
                    continue
                chain.reverse()
                field = chain[-1]
                if field not in mapping:
                    continue
                sites += 1
                read = [cursor.func.id[len("get_") :]] + chain[:-1]
                if read[:2] == ["parallel", "config"]:
                    # `config` on `get_parallel()` is the tier hop, not a
                    # sub-namespace: bare names there are the live topology.
                    del read[1]
                if mapping[field].split(".") != read:
                    disagreements.append(
                        f"{path.relative_to(srt)}:{node.lineno} reads "
                        f"{'.'.join(read)}.{field}, metadata says "
                        f"{mapping[field]}.{field}"
                    )
        self.assertEqual(
            disagreements,
            [],
            "a reader and the namespace metadata disagree about where a leaf "
            "lives; one of them is wrong:\n  " + "\n  ".join(disagreements),
        )
        self.assertGreater(
            sites,
            1500,
            f"only {sites} bag reads were matched; the scan broke and this "
            "check stopped covering anything",
        )

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
