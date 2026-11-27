"""Check the uncommitted SAIL extraction without importing torch or GPU libraries.

Usage: python test/manual/moe/verify_sail_megamoe_isolation.py --base <upstream-base>

This checks CUDA function/class ASTs after folding only the SAIL platform gate,
exact helper relocation ASTs, and preservation of test inventory/registrations.
It is not a byte-reproduction certificate for a pure relocation commit and does
not prove vendor kernel numerics or performance.
"""

from __future__ import annotations

import argparse
import ast
import copy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRT = "python/sglang/srt/"
SAIL = SRT + "hardware_backend/sail/moe/"


def git_source(ref: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT).decode(
        "utf-8"
    )


def working_source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class CudaProjection(ast.NodeTransformer):
    """Substitute only SAIL detection with False; do not evaluate CUDA predicates."""

    def visit_Name(self, node):
        return ast.Constant(False) if node.id == "_is_sail" else node

    def visit_Attribute(self, node):
        if ast.unparse(node) == "platform.is_sail":
            return ast.Constant(False)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "is_sail":
            assert not node.args and not node.keywords
            return ast.Constant(False)
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, bool):
                return ast.Constant(not node.operand.value)
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        values = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, bool):
                if isinstance(node.op, ast.And) and not value.value:
                    values.append(value)
                    break
                if isinstance(node.op, ast.Or) and value.value:
                    values.append(value)
                    break
                continue
            values.append(value)
        if not values:
            return ast.Constant(isinstance(node.op, ast.And))
        if len(values) == 1:
            return values[0]
        node.values = values
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return node.body if node.test.value else node.orelse
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return node.body if node.test.value else node.orelse
        return node


def dump(node):
    return ast.dump(node, include_attributes=False)


def definitions(source):
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


class RemoveNoopFallback(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == "_log_platform_fallback"
            ):
                assert not call.args and not call.keywords
                return None
        return self.generic_visit(node)


def check_cuda(base):
    total = 0
    for path in (
        SRT + "layers/moe/mega_moe.py",
        SRT + "layers/moe/mega_gate.py",
        SRT + "models/kimi_k3.py",
        SRT + "arg_groups/mega_moe_hook.py",
    ):
        before = definitions(git_source(base, path))
        after = definitions(working_source(path))
        if "_log_platform_fallback" in after:
            noop = CudaProjection().visit(
                copy.deepcopy(after["_log_platform_fallback"])
            )
            assert not noop.body, "Fallback logger is no longer a CUDA no-op"
        for name, original in before.items():
            assert name in after, f"Removed definition: {path}:{name}"
            expected = CudaProjection().visit(copy.deepcopy(original))
            actual = CudaProjection().visit(copy.deepcopy(after[name]))
            actual = RemoveNoopFallback().visit(actual)
            assert dump(expected) == dump(actual), f"CUDA AST residual: {path}:{name}"
            total += 1
    print(f"PASS: {total} pre-existing function/class CUDA ASTs match {base}")


def check_relocations(ported):
    for source, target, name in (
        (
            SRT + "layers/moe/mega_moe.py",
            SAIL + "mega_moe.py",
            "_ppu_mega_moe_mma_type",
        ),
        (SRT + "layers/moe/mega_moe.py", SAIL + "mega_moe.py", "_mega_moe_hit_state"),
        (
            SRT + "layers/moe/mega_gate.py",
            SAIL + "mega_gate.py",
            "_sail_mega_gate_available",
        ),
        (
            SRT + "arg_groups/mega_moe_hook.py",
            SAIL + "mega_moe.py",
            "_require_ppu_mega_moe_deep_gemm_apis",
        ),
    ):
        before = definitions(git_source(ported, source))[name]
        after = definitions(working_source(target))[name]
        assert dump(before) == dump(after), f"Moved helper AST residual: {name}"
    print("PASS: four relocated helper ASTs are unchanged")


def test_inventory(source):
    tree = ast.parse(source)
    inventory = {}
    for owner in tree.body:
        if not isinstance(owner, ast.ClassDef):
            continue
        for method in owner.body:
            if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"):
                count = sum(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and (
                        node.func.attr.startswith("assert")
                        or node.func.attr == "subTest"
                    )
                    for node in ast.walk(method)
                )
                inventory[(owner.name, method.name)] = count
    registrations = [
        dump(node.value)
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id.startswith("register_")
    ]
    return inventory, registrations


def check_tests(ported):
    for path in (
        "test/registered/unit/layers/moe/test_mega_moe_deepgemm_api.py",
        "test/registered/unit/server_args/test_server_args.py",
        "test/registered/unit/layers/attention/test_pla_kda_dispatch.py",
    ):
        before, old_reg = test_inventory(git_source(ported, path))
        after, new_reg = test_inventory(working_source(path))
        assert before.keys() <= after.keys(), f"Lost test methods: {path}"
        assert old_reg == new_reg, f"Changed CI registrations: {path}"
        for name, count in before.items():
            assert after[name] >= count, f"Lost assertion/subTest: {path}:{name}"
    print(
        "PASS: existing test methods, assertion counts and CI registrations preserved"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--ported", default="12139ac8e79f2693a8e35654d61598696821cf53")
    args = parser.parse_args()
    check_cuda(args.base)
    check_relocations(args.ported)
    check_tests(args.ported)
