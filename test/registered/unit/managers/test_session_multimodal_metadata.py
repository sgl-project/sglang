"""CPU regressions for session media metadata without initializing a GPU runtime."""

import ast
import dataclasses
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(scope="module")
def classes():
    # Load the actual containers and request methods, avoiding scheduler/kernel
    # imports. No implementation is copied into these CPU-only tests.
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/managers/schedule_batch.py"
    )
    tree = ast.parse(path.read_text())
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MultimodalInputs":
            definitions.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "Req":
            methods = {
                "extend_image_inputs",
                "_extend_session_image_inputs",
                "_refresh_fill_ids",
            }
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name in methods
            ]
            definitions.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + definitions,
        type_ignores=[],
    )
    namespace = {
        "__name__": __name__,
        "dataclasses": dataclasses,
        "array": array,
        "torch": torch,
        "ReqDllmMixin": object,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def request(classes, origin, parent=None):
    req = classes.Req()
    req.session = object()
    req.origin_input_ids = array("q", origin)
    req.full_untruncated_fill_ids = array("q", origin)
    req.output_ids = array("q")
    req.multimodal_inputs = parent
    return req


def media(classes, positions, *, modalities=None):
    return classes.MultimodalInputs(
        mm_items=[object()],
        mrope_positions=positions,
        mrope_position_delta=(
            (positions.max() + 1 - positions.shape[1]).reshape(1, 1)
            if positions is not None
            else None
        ),
        token_modalities=modalities,
    )


def test_image_append_fills_generated_gap_and_preserves_parent(classes):
    prefix = torch.tensor([[0, 1, 1, 2], [0, 1, 2, 3], [0, 2, 1, 3]])
    parent = media(classes, prefix)
    parent.mrope_position_delta_repeated_cache = torch.tensor([99])
    req = request(classes, range(9), parent)
    suffix = torch.tensor([[0, 1, 1], [0, 1, 2], [0, 2, 1]])
    req.extend_image_inputs(media(classes, suffix))

    expected = torch.cat(
        [prefix, torch.tensor([[4, 5]]).expand(3, -1), suffix + 6], dim=1
    )
    torch.testing.assert_close(req.multimodal_inputs.mrope_positions, expected)
    assert req.multimodal_inputs.mrope_position_delta.item() == 0
    assert req.multimodal_inputs.mrope_position_delta_repeated_cache is None
    assert len(req.multimodal_inputs.mm_items) == 2
    assert req.multimodal_inputs is not parent
    assert len(parent.mm_items) == 1
    torch.testing.assert_close(parent.mrope_positions, prefix)
    assert parent.mrope_position_delta_repeated_cache.item() == 99


def test_text_to_image_append_builds_full_position_history(classes):
    req = request(classes, range(6))
    suffix = torch.tensor([[0, 1, 1], [0, 1, 2], [0, 2, 1]])
    req.extend_image_inputs(media(classes, suffix))

    expected = torch.cat([torch.arange(3).expand(3, -1), suffix + 3], dim=1)
    torch.testing.assert_close(req.multimodal_inputs.mrope_positions, expected)
    assert req.multimodal_inputs.mrope_position_delta.shape == (1, 1)


def test_append_invalidates_fill_tokens_even_when_padding_length_is_unchanged(classes):
    req = request(classes, [10, 20, 30])
    req.origin_input_ids = array("q", [10, -101, 30])
    req.extend_image_inputs(media(classes, torch.arange(3).expand(3, -1)))
    req._refresh_fill_ids()
    assert list(req.full_untruncated_fill_ids) == [10, -101, 30]


@pytest.mark.parametrize("with_positions", [False, True])
def test_aborted_or_branched_append_preserves_parent_token_modalities(
    classes, with_positions
):
    positions = torch.arange(3).expand(3, -1) if with_positions else None
    parent = media(classes, positions, modalities=[0, 1, 0])

    # The first append may be rejected before commit, or another regular
    # session request may branch from this same saved parent.
    discarded = request(classes, range(6), parent)
    discarded.extend_image_inputs(media(classes, positions, modalities=[0, 2, 0]))
    sibling = request(classes, range(6), parent)
    sibling.extend_image_inputs(media(classes, positions, modalities=[0, 3, 0]))

    assert parent.token_modalities == [0, 1, 0]
    assert discarded.multimodal_inputs.token_modalities == [0, 1, 0, 0, 2, 0]
    assert sibling.multimodal_inputs.token_modalities == [0, 1, 0, 0, 3, 0]
    assert len(parent.mm_items) == 1
    assert len(sibling.multimodal_inputs.mm_items) == 2


def test_append_without_mrope_keeps_parent_metadata(classes):
    parent = media(classes, None)
    req = request(classes, range(6), parent)
    req.extend_image_inputs(media(classes, None))
    assert req.multimodal_inputs.mrope_positions is None
    assert len(req.multimodal_inputs.mm_items) == 2
    assert len(parent.mm_items) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
