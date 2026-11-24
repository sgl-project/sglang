import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPreCaptureGate(CustomTestCase):
    """graph_capture must not reach the Gluon backend on other devices.

    The pre-capture rendezvous build runs inside graph_capture, which every
    backend executes. Without an explicit gate it imports the backend and
    probes the device on CUDA and on AMD GPUs other than gfx950 -- and, before
    the kernel PR lands, logs a warning on every capture.
    """

    def test_graph_capture_gates_on_aiter_and_gfx95(self):
        import inspect

        from sglang.srt.distributed import parallel_state

        src = inspect.getsource(parallel_state)
        marker = "gluon_eligible = "
        self.assertIn(marker, src, "the pre-capture build must be gated")
        line = next(l for l in src.splitlines() if marker in l)
        self.assertIn("_use_aiter", line)
        self.assertIn("is_gfx95_supported()", line)

    def test_backend_is_not_imported_when_ineligible(self):
        """The import sits behind the gate, not in front of it."""
        import inspect

        from sglang.srt.distributed import parallel_state

        src = inspect.getsource(parallel_state)
        gate_at = src.index("gluon_eligible = ")
        import_at = src.index("gluon_tp_ar_norm_quant as _g")
        self.assertLess(
            gate_at,
            import_at,
            "the gate must precede the backend import so other devices never load it",
        )


class TestEveryBackendImportIsGated(CustomTestCase):
    """No function may load the Gluon backend before checking it can run.

    The backend is gfx950 + aiter only. Importing it to ask is already too
    much: on CUDA and other AMD parts it loads a module and queries the device
    for an answer that is always no, and before the kernel PR lands the import
    itself fails.

    The check walks the AST of the *importing function*, because a
    module-level ``_use_aiter = ...`` would otherwise satisfy a naive
    text-order check while the function still imports unguarded.
    """

    BACKEND = "gluon_tp_ar_norm_quant"
    GATES = ("_use_aiter", "is_gfx95_supported")

    def _sources(self):
        """Every file under sglang/ that names the backend, discovered.

        Deliberately not a hardcoded list: a future import site in a module
        nobody thought to add here is exactly the case this guards.
        """
        import pathlib

        import sglang

        root = pathlib.Path(sglang.__file__).parent
        own = root / "srt" / "distributed" / "device_communicators"
        for path in sorted(root.rglob("*.py")):
            if path.parent == own and path.name.startswith(self.BACKEND):
                continue  # the backend itself
            text = path.read_text(encoding="utf-8", errors="ignore")
            if self.BACKEND in text:
                yield path, text

    def _importing_functions(self, text):
        """(function node, line of the backend import) for each importer."""
        import ast

        tree = ast.parse(text)
        out = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.ImportFrom) and any(
                    a.name == self.BACKEND for a in node.names
                ):
                    out.append((fn, node.lineno))
        return out

    def _gate_lines(self, fn):
        """Lines inside fn where eligibility is decided.

        Counts a direct ``if _use_aiter and is_gfx95_supported()`` and the
        assignment form ``eligible = _use_aiter and is_gfx95_supported()``,
        since gating through a local is just as effective.
        """
        import ast

        gates = set(self.GATES)
        lines = []
        aliases = set()

        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                if names & gates:
                    lines.append(node.lineno)
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            aliases.add(t.id)

        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if names & (gates | aliases):
                lines.append(node.lineno)
        return lines

    def test_a_gate_precedes_every_backend_import(self):
        seen = []
        for path, text in self._sources():
            for fn, import_line in self._importing_functions(text):
                seen.append(f"{path.stem}.{fn.name}")
                gates = [l for l in self._gate_lines(fn) if l < import_line]
                self.assertTrue(
                    gates,
                    f"{path}:{import_line} in {fn.name}() imports "
                    f"{self.BACKEND} with no {' / '.join(self.GATES)} check "
                    "before it: other devices would load the backend to be "
                    "told no",
                )

        # Both known importers must be found, or the discovery above silently
        # stopped covering them and every assertion became vacuous.
        self.assertEqual(
            {"layernorm._try_gluon_tp_ar_norm_quant", "parallel_state.graph_capture"},
            set(seen),
            f"importer discovery found {seen}",
        )


if __name__ == "__main__":
    unittest.main()
