from __future__ import annotations

import ast
import pathlib
import unittest

_LINT_DIR = pathlib.Path(__file__).resolve().parent


def uses_pep604_annotation(tree: ast.AST) -> bool:
    """Whether any annotation in the tree is an `X | Y` union."""
    annotations: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.extend(
                a.annotation
                for a in (
                    *node.args.args,
                    *node.args.posonlyargs,
                    *node.args.kwonlyargs,
                )
                if a.annotation is not None
            )
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)

    return any(
        isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr)
        for annotation in annotations
        for inner in ast.walk(annotation)
    )


def has_future_annotations(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in ast.walk(tree)
    )


class TestLintScriptsRunOnOldPython(unittest.TestCase):
    """These scripts run under whatever `python3` a contributor's PATH resolves to.

    A stock macOS resolves it to 3.9, where `int | None` in an annotation is
    evaluated at definition time and raises TypeError -- which blocks every
    commit rather than reporting a lint error. `from __future__ import
    annotations` defers the evaluation, so a script using those unions needs it.
    CI runs a newer interpreter and cannot catch this on its own.
    """

    def test_pep604_annotations_come_with_the_future_import(self):
        for path in sorted(_LINT_DIR.glob("*.py")):
            with self.subTest(script=path.name):
                tree = ast.parse(path.read_text())
                if uses_pep604_annotation(tree):
                    self.assertTrue(
                        has_future_annotations(tree),
                        f"{path.name} annotates with `X | Y`; add "
                        "`from __future__ import annotations` so it still runs "
                        "on the oldest python3 a contributor might have",
                    )


if __name__ == "__main__":
    unittest.main()
