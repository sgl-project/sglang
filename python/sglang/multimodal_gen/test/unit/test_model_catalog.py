import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
REGISTRY_PATH = REPO_ROOT / "python/sglang/multimodal_gen/registry.py"
CATALOG_PATH = REPO_ROOT / "docs/src/snippets/diffusion/model-catalog.jsx"


def _registered_model_ids() -> set[str]:
    tree = ast.parse(REGISTRY_PATH.read_text())
    model_ids = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "register_configs":
            continue

        paths = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "hf_model_paths"
            ),
            None,
        )
        if not isinstance(paths, (ast.List, ast.Tuple)):
            continue

        model_ids.update(
            item.value
            for item in paths.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )

    return model_ids


def _catalog_model_ids() -> set[str]:
    source = CATALOG_PATH.read_text()
    model_id_arrays = re.findall(r"modelIds:\s*\[(.*?)\]", source, re.DOTALL)
    return {
        model_id
        for model_id_array in model_id_arrays
        for model_id in re.findall(r'"([^"]+)"', model_id_array)
    }


def test_explicit_registry_model_ids_are_documented():
    missing_model_ids = _registered_model_ids() - _catalog_model_ids()
    assert not missing_model_ids, (
        "Add newly registered model IDs to the Supported Models catalog: "
        f"{sorted(missing_model_ids)}"
    )
