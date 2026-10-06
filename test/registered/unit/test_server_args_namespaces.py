"""Coverage lint for the ServerArgs -> RuntimeContext namespace split.

Every ServerArgs field must resolve to a namespace, and every path must be one of
the known domains. A field gets its namespace from the ``arg_groups/fields/``
class that declares it -- each carries the ``_NS_PATH`` it stands for, so the
module a declaration lives in *is* the answer, and there is no per-field marker
to forget. (``NS("<path>")`` survives for the one shape a class cannot express:
an ad-hoc dataclass spanning namespaces, which the config-bag tests build.)

This is the guardrail that fails when an upstream PR adds a field to a namespace
class that has no ``_NS_PATH``, or adds one outside the taxonomy below.
"""

import ast
import unittest

import msgspec
import msgspec.structs

from sglang.srt.arg_groups.arg_utils import namespace_of
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=31, suite="base-a-test-cpu")

# Supported runtime configuration namespaces.
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
    return {f.name for f in msgspec.structs.fields(ServerArgs)}


def _config_reads(tree, mapping):
    """Yield ``(node, namespace, leaf)`` for each config leaf a module reads.

    A read starts at a bag accessor such as ``get_exec()`` and follows the
    sub-namespaces it names (``get_exec().moe``); the first other name is the
    leaf. Attributes past the leaf belong to its value, so
    ``get_parallel().tp_group.device`` reads ``tp_group``, not the ``device``
    leaf.
    """
    accessors = {f"get_{path.split('.')[0]}" for path in mapping.values()}
    namespaces = {
        ".".join(path.split(".")[:depth])
        for path in mapping.values()
        for depth in range(1, path.count(".") + 2)
    }
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
        read = [cursor.func.id[len("get_") :]]
        if read == ["parallel"] and chain[:1] == ["config"]:
            # `config` on `get_parallel()` is the tier hop, not a
            # sub-namespace: bare names there are the live topology.
            chain = chain[1:]
        for index, name in enumerate(chain):
            if name in mapping or ".".join(read + [name]) not in namespaces:
                break
            read.append(name)
        else:
            continue
        # A leaf is checked at the node that ends with it, so once.
        if index == len(chain) - 1 and name in mapping:
            yield node, ".".join(read), name


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
        self.assertTrue(accessors, "no runtime context accessors found")

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

        The metadata is one source and the ~2800 hand-written reads
        (`get_schedule().chunked_prefill_size`) are the other. Checking the
        projection against the metadata cannot catch a field assigned to the
        wrong group -- both sides come from the same marker, so the check is
        true by construction. The readers are written by hand, so a
        disagreement means one of the two is wrong, and every reader on the
        losing side raises `has no leaf/subgroup` at runtime on whichever
        branch reaches it first.
        """
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
            for node, read, leaf in _config_reads(tree, mapping):
                sites += 1
                if mapping[leaf] != read:
                    disagreements.append(
                        f"{path.relative_to(srt)}:{node.lineno} reads "
                        f"{read}.{leaf}, metadata says {mapping[leaf]}.{leaf}"
                    )
        self.assertEqual(
            disagreements,
            [],
            "a reader and the namespace metadata disagree about where a leaf "
            "lives; one of them is wrong:\n  " + "\n  ".join(disagreements),
        )
        self.assertGreater(
            sites,
            0,
            f"only {sites} bag reads were matched; the scan broke and this "
            "check stopped covering anything",
        )

    def test_the_scan_reads_the_leaf_a_chain_names(self):
        """Wrong reads are caught through sub-namespaces, and an attribute of a
        leaf's value is not mistaken for a leaf."""
        mapping = namespace_of(ServerArgs)

        def disagreements(expression):
            return [
                (read, leaf)
                for _, read, leaf in _config_reads(ast.parse(expression), mapping)
                if mapping[leaf] != read
            ]

        for expression in (
            "get_exec().moe.moe_a2a_backend",
            "get_parallel().config.tp_size",
            # A process group is a value, not a namespace.
            "get_parallel().tp_group.device",
        ):
            with self.subTest(expression):
                self.assertEqual(disagreements(expression), [])
        for expression, read in (
            ("get_exec().moe_a2a_backend", ("exec", "moe_a2a_backend")),
            (
                "get_exec().moe.chunked_prefill_size",
                ("exec.moe", "chunked_prefill_size"),
            ),
            ("get_schedule().device", ("schedule", "device")),
        ):
            with self.subTest(expression):
                self.assertEqual(disagreements(expression), [read])

    def test_all_namespaces_are_known(self):
        nsmap = namespace_of(ServerArgs)
        bad = {f: p for f, p in nsmap.items() if p not in VALID_NAMESPACES}
        self.assertFalse(bad, f"unknown namespace paths (typo or new domain?): {bad}")

    def test_namespace_map_covers_all_fields(self):
        nsmap = namespace_of(ServerArgs)
        self.assertEqual(set(nsmap), _field_names())


if __name__ == "__main__":
    unittest.main()
