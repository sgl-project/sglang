"""DSparkWorkerV2._compute_decode_logprobs is the one place the decode step
extracts per-token logprobs, so a subclass whose verify tail already produced
them can skip the eager extract by overriding it. These pin the seam on CPU:
the default body issues the exact compute_spec_logprobs call the decode step
used to inline, and the decode step reaches it only through the seam, with the
same operands, under the same return_logprob gate.
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.speculative.dspark_components import dspark_worker_v2
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

SEAM = "_compute_decode_logprobs"
EXTRACT = "compute_spec_logprobs"


def _calls(tree: ast.AST, *, method: str | None = None, attr_of_self: bool = False):
    """Call nodes whose callee is ``method`` (plain name, or ``self.method``)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if attr_of_self:
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and func.attr == method
            ):
                yield node
        elif isinstance(func, ast.Name) and func.id == method:
            yield node


def _method_tree(method) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(method)))


class TestDecodeLogprobSeam(CustomTestCase):
    def test_default_seam_issues_the_chain_extract(self):
        worker = SimpleNamespace(verify_num_draft_tokens=4)
        batch = object()
        logits_output = object()
        out_tokens = torch.arange(12, dtype=torch.int64).view(3, 4)
        accept = SimpleNamespace(out_tokens=out_tokens)
        with patch.object(dspark_worker_v2, EXTRACT) as extract:
            DSparkWorkerV2._compute_decode_logprobs(
                worker, batch, logits_output, accept
            )
        extract.assert_called_once()
        args, kwargs = extract.call_args
        self.assertIs(args[0], batch)
        self.assertIs(args[1], logits_output)
        # The chain layout compute_spec_logprobs indexes with chain_stride:
        # row b*w + j of the logits is out_tokens[b, j].
        self.assertTrue(torch.equal(args[2], out_tokens.reshape(-1)))
        self.assertEqual(kwargs, {"chain_stride": 4})

    def test_decode_step_reaches_the_extract_only_through_the_seam(self):
        # Inlining the extract back into _forward_decode would silently bypass
        # a subclass override, so the module's only extract call must live in
        # the seam and the decode step must call the seam with the operands
        # the extract used to receive.
        module_tree = ast.parse(inspect.getsource(dspark_worker_v2))
        extract_calls = list(_calls(module_tree, method=EXTRACT))
        self.assertEqual(len(extract_calls), 1)
        seam_tree = _method_tree(DSparkWorkerV2._compute_decode_logprobs)
        self.assertEqual(len(list(_calls(seam_tree, method=EXTRACT))), 1)

        decode_tree = _method_tree(DSparkWorkerV2._forward_decode)
        seam_calls = list(_calls(decode_tree, method=SEAM, attr_of_self=True))
        self.assertEqual(len(seam_calls), 1)
        (call,) = seam_calls
        self.assertEqual(
            [ast.unparse(a) for a in call.args], ["batch", "logits_output", "accept"]
        )
        self.assertEqual(call.keywords, [])
        gates = [
            node
            for node in ast.walk(decode_tree)
            if isinstance(node, ast.If) and any(n is call for n in ast.walk(node))
        ]
        self.assertTrue(gates)
        innermost = min(gates, key=lambda n: sum(1 for _ in ast.walk(n)))
        self.assertEqual(ast.unparse(innermost.test), "batch.return_logprob")


if __name__ == "__main__":
    unittest.main()
