"""Load selected production definitions for CPU-only NPU regression tests.

Only optional accelerator imports are isolated; method bodies are unchanged.
These tests do not validate NPU kernels or full-model imports.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[4] / "python/sglang"


def load(path, names, namespace=None):
    """Load whole definitions or a class's selected methods, without rewriting them."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = []
    for name, methods in names.items():
        node = next((n for n in tree.body if getattr(n, "name", None) == name))
        if methods is not None:
            node.bases = []
            node.keywords = []
            node.body = [n for n in node.body if getattr(n, "name", None) in methods]
        nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {"torch": torch, "nn": nn, "F": F, "logger": logging.getLogger(__name__)}
    scope.update(namespace or {})
    exec(compile(module, str(ROOT / path), "exec"), scope)
    return NS(**scope)


def forbidden(*args, **kwargs):
    raise AssertionError("entered an accelerator-only kernel or compiler")
