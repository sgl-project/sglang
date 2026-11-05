"""Load native function bodies and scope process-local import hooks.

Adapter calls serialize temporary module substitutions. Other threads running
real SGLang/Torch are outside this lock; do not co-host them during a simulation.
No subprocess or separate service is required.
"""

import ast
import sys
from contextlib import contextmanager
from threading import RLock

_MODULE_LOCK = RLock()
_MISSING = object()


@contextmanager
def temporary_modules(replacements):
    with _MODULE_LOCK:
        saved = {name: sys.modules.get(name, _MISSING) for name in replacements}
        try:
            sys.modules.update(replacements)
            yield
        finally:
            for name, value in saved.items():
                if value is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


def extract(path, env, names=None, cls=None):
    tree = ast.parse(path.read_text())
    nodes = (
        tree.body
        if cls is None
        else next(
            n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
    )
    selected = [
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and (names is None or n.name in names)
    ]
    if names is not None:
        if {n.name for n in selected} != set(names):
            raise ValueError(f"AFD_SOURCE_API_MISMATCH: {path}: {cls}: {names}")
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ] + selected
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        env,
    )
