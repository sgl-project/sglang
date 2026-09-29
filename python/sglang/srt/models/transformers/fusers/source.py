# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import ast
import copy
import inspect
import linecache
import textwrap


def read_forward(module):
    fn = inspect.unwrap(getattr(module.forward, "__func__", module.forward))
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    function.decorator_list = []
    return function, fn


def compile_forward(function, original):
    function = copy.deepcopy(function)
    function.returns = None
    for argument in (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ):
        argument.annotation = None
    if function.args.vararg is not None:
        function.args.vararg.annotation = None
    if function.args.kwarg is not None:
        function.args.kwarg.annotation = None
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    source = ast.unparse(tree)
    filename = f"<sglang-fused-{original.__module__}.{original.__qualname__}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__globals__)
    namespace.update(inspect.getclosurevars(original).nonlocals)
    exec(compile(source, filename, "exec"), namespace)
    fused = namespace[function.name]
    fused.__module__ = original.__module__
    fused.__qualname__ = original.__qualname__
    return fused


def self_attribute(node):
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def projection_calls(function, names):
    calls = {}
    for name in names:
        references = [
            node for node in ast.walk(function) if self_attribute(node) == name
        ]
        matches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and node.func in references
        ]
        if len(references) != 1 or len(matches) != 1:
            return None
        call = matches[0]
        if (
            len(call.args) != 1
            or call.keywords
            or not isinstance(call.args[0], ast.Name)
        ):
            return None
        calls[name] = call
    if len({call.args[0].id for call in calls.values()}) != 1:
        return None
    return calls


class ReplaceNodes(ast.NodeTransformer):
    def __init__(self, replacements):
        self.replacements = replacements

    def visit(self, node):
        return self.replacements.get(id(node)) or super().visit(node)
