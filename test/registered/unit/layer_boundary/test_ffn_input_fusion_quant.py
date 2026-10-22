import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.layer_boundary.fusions import allreduce as fusions
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_READOUT,
    Fp8Input,
    NormQuantReadout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 2048
GROUPS = HIDDEN // 128


class _Norm:
    """Stands in for the stage's norm module.

    ``forward_with_allreduce_fusion_quant_per_group`` is what the Gluon backend
    hooks into; this records how the boundary calls it and returns the shape a
    ``keep_bf16`` consumer expects.
    """

    def __init__(self):
        self.quant_calls = []
        self.plain_calls = 0

    def forward_with_allreduce_fusion_quant_per_group(
        self, hidden_states, residual, use_attn_tp_group, keep_bf16
    ):
        self.quant_calls.append(
            {
                "use_attn_tp_group": use_attn_tp_group,
                "keep_bf16": keep_bf16,
                "rows": hidden_states.shape[0],
            }
        )
        rows = hidden_states.shape[0]
        bf16 = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        fp8 = torch.zeros(rows, HIDDEN, dtype=torch.float8_e4m3fn)
        scale = torch.ones(rows, GROUPS, dtype=torch.float32)
        return (bf16, fp8, scale), residual

    def forward_with_allreduce_fusion(self, hidden_states, residual, use_attn_tp_group):
        self.plain_calls += 1
        return hidden_states, residual


def _plan(norm):
    return SimpleNamespace(norm=norm, fusions=None)


class TestFfnInputFusionQuant(CustomTestCase):
    """The FFN side can reach the fused AR+norm+per-group-quant kernel."""

    def setUp(self):
        self.norm = _Norm()
        self.hidden_states = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
        self.residual = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
        self.forward_batch = SimpleNamespace(forward_mode=None)

    def _run(self, **kwargs):
        with patch.object(fusions, "aiter_ar_fusion_applies", return_value=True):
            return fusions.fused_ffn_input(
                _plan(self.norm),
                self.hidden_states,
                self.residual,
                self.forward_batch,
                **kwargs,
            )

    def test_returns_fp8_scale_and_residual_when_fused(self):
        out, residual = self._run(fuses_quant=True, keep_bf16=True)

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        bf16, fp8, scale = out
        self.assertEqual(bf16.dtype, torch.bfloat16)
        self.assertEqual(fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(fp8.shape, (8, HIDDEN))
        self.assertEqual(scale.shape, (8, GROUPS))
        self.assertIs(residual, self.residual)

        # The FFN completes the attention-TP sum, unlike the attention side.
        self.assertEqual(len(self.norm.quant_calls), 1)
        self.assertTrue(self.norm.quant_calls[0]["use_attn_tp_group"])
        self.assertTrue(self.norm.quant_calls[0]["keep_bf16"])
        self.assertEqual(self.norm.plain_calls, 0)

    def test_plain_path_when_not_fused(self):
        out, _ = self._run(fuses_quant=False)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(self.norm.quant_calls, [])
        self.assertEqual(self.norm.plain_calls, 1)

    def test_falls_back_when_the_backend_declines_the_shape(self):
        """A backend that cannot service the shape returns None, not an error."""

        def decline(*args, **kwargs):
            return None

        self.norm.forward_with_allreduce_fusion_quant_per_group = decline
        out, _ = self._run(fuses_quant=True, keep_bf16=True)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(self.norm.plain_calls, 1)


class TestFfnInputFusionsGate(CustomTestCase):
    """fuses_quant follows the stage's declared read."""

    def test_default_read_does_not_fuse_quant(self):
        norm = _Norm()
        fused = fusions.ffn_input_fusions(_plan(norm), NORM_READOUT)
        self.assertEqual(len(fused), 1)
        self.assertFalse(fused[0].run.keywords["fuses_quant"])

    def test_declared_read_without_fp8_does_not_fuse_quant(self):
        norm = _Norm()
        fused = fusions.ffn_input_fusions(_plan(norm), NormQuantReadout())
        self.assertFalse(fused[0].run.keywords["fuses_quant"])

    def test_keep_bf16_follows_the_declaration(self):
        norm = _Norm()
        with (
            patch.object(fusions, "_use_aiter", True),
            patch.object(fusions, "get_exec") as get_exec,
            patch.object(fusions, "get_bool_env_var", return_value=False),
        ):
            get_exec.return_value = SimpleNamespace(
                comm=SimpleNamespace(enable_aiter_allreduce_fusion=True)
            )
            fused = fusions.ffn_input_fusions(
                _plan(norm),
                NormQuantReadout(fp8_input=Fp8Input.TUPLE_AND_BF16),
            )
        self.assertTrue(fused[0].run.keywords["fuses_quant"])
        self.assertTrue(fused[0].run.keywords["keep_bf16"])


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
