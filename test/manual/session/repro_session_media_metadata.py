"""Reproduce session media metadata aliasing using only the Python standard library.

Run this file against an unmodified or patched checkout with --source CHECKOUT.
It executes that checkout's real merge and session-append methods with lightweight
metadata containers; no SGLang imports, model, GPU, or server are required.
"""

import argparse
import ast
import dataclasses
import json
from array import array
from pathlib import Path
from types import SimpleNamespace


@dataclasses.dataclass
class Metadata:
    mm_items: list
    token_modalities: list
    image_pad_len: object = None
    mrope_positions: object = None
    mrope_position_delta: object = None
    mrope_position_delta_repeated_cache: object = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    source = args.source / "python/sglang/srt/managers/schedule_batch.py"
    tree = ast.parse(source.read_text())
    wanted = {
        "MultimodalInputs": {"merge"},
        "Req": {"extend_image_inputs", "_extend_session_image_inputs"},
    }
    methods = {}
    for owner in tree.body:
        if isinstance(owner, ast.ClassDef) and owner.name in wanted:
            methods[owner.name] = [
                method
                for method in owner.body
                if isinstance(method, ast.FunctionDef)
                and method.name in wanted[owner.name]
            ]
    for owner, names in wanted.items():
        found = {method.name for method in methods.get(owner, [])}
        if found != names:
            parser.error(f"{source}: missing {owner} methods {sorted(names - found)}")

    namespace = {"dataclasses": dataclasses, "array": array}
    for owner, definitions in methods.items():
        module = ast.Module(
            body=ast.parse("from __future__ import annotations").body + definitions,
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    Metadata.merge = namespace["merge"]
    request_type = type(
        "Request",
        (SimpleNamespace,),
        {name: namespace[name] for name in wanted["Req"]},
    )
    parent = Metadata(mm_items=["saved image"], token_modalities=[0, 1, 0])
    children = []
    for modality in (2, 3):
        req = request_type(
            session=object(),
            origin_input_ids=array("q", range(6)),
            full_untruncated_fill_ids=array("q", range(6)),
            multimodal_inputs=parent,
        )
        req.extend_image_inputs(
            Metadata(mm_items=["new image"], token_modalities=[0, modality, 0])
        )
        children.append(req.multimodal_inputs)

    observed = {
        "parent": parent.token_modalities,
        "discarded_append": children[0].token_modalities,
        "sibling_append": children[1].token_modalities,
    }
    expected = {
        "parent": [0, 1, 0],
        "discarded_append": [0, 1, 0, 0, 2, 0],
        "sibling_append": [0, 1, 0, 0, 3, 0],
    }
    success = observed == expected
    print(
        json.dumps(
            {"passed": success, "observed": observed, "expected": expected}, indent=2
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
