"""CPU check that the SPEC_FUSE source rewrite still matches the pinned SGLang tree."""

import ast
import os
import pathlib
import textwrap


HERE = pathlib.Path(__file__).resolve().parent
REL = pathlib.Path("python/sglang/srt/speculative/eagle_worker_v2.py")


def _worker():
    roots = [os.environ.get("SGLANG_ROOT"), HERE.parent, "/sgl-workspace/sglang"]
    for root in roots:
        if root and (pathlib.Path(root) / REL).is_file():
            return pathlib.Path(root) / REL
    raise AssertionError(f"SGLang source tree not found for {REL}")


WORKER = _worker()


def _constants():
    tree = ast.parse((HERE / "coral_spec_fuse.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return values


def _method_source(cls_name, method_name):
    tree = ast.parse(WORKER.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    # ast strips the class-level indent from the first line; inspect.getsource does not.
                    return "    " + ast.get_source_segment(WORKER.read_text(), item)
    raise AssertionError(f"{cls_name}.{method_name} not found")


def main():
    src = _method_source("EagleDraftWorker", "_draft_extend_for_decode")
    src = textwrap.indent(textwrap.dedent(src), "    ")
    c = _constants()
    assert src.count(c["_V22_REORG_OLD"]) == 1
    assert src.count(c["_ARGMAX_OLD"]) == 1
    print("SPEC_FUSE v0.5.22 patterns match")


if __name__ == "__main__":
    main()
