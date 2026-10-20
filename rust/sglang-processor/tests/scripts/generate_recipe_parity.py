"""Fill tests/fixtures/recipe_parity/*.json from the Python recipe bridge.

Each fixture names a checkpoint (`model`, `revision`, `model_version`) and request
`cases`. For every case this records the prompt the bridge renders, its
special-token ranges (code-point offsets), recipe's thinking mode and output
constraint kind, and the token ids as a count plus sha256, or the error recipe
raises. `tests/recipe_parity.rs` then checks the Rust formatter against it.

Run with `deepseek_recipe` installed and the checkpoint's tokenizer.json cached:
    python tests/scripts/generate_recipe_parity.py [tests/fixtures/recipe_parity/<name>.json ...]
No GPU and no SGLang server are needed; the bridge module is imported by path.
"""

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve()
FIXTURES = HERE.parents[1] / "fixtures/recipe_parity"
BRIDGE = (
    HERE.parents[4] / "python/sglang/srt/entrypoints/openai/deepseek_recipe_bridge.py"
)
sys.path.insert(0, str(HERE.parent))
from fixture_json import dump  # noqa: E402


def load_bridge():
    spec = importlib.util.spec_from_file_location("deepseek_recipe_bridge", BRIDGE)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the module's string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def token_digest(ids):
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def fill(fixture, bridge):
    tokenizer_json = Path(
        hf_hub_download(
            fixture["model"], "tokenizer.json", revision=fixture["revision"]
        )
    ).read_text(encoding="utf-8")
    encoder = bridge.RecipeEncoder(fixture["model_version"], tokenizer_json)
    defaults = fixture.get("default_kwargs") or {}
    for case in fixture["cases"]:
        for key in (
            "prompt",
            "special_ranges",
            "token_count",
            "token_sha256",
            "error",
            "thinking_mode",
            "output_constraint",
        ):
            case.pop(key, None)
        try:
            render = encoder.convert(case["request"], defaults)
        except bridge.RecipeRequestError as error:
            case["error"] = str(error)
            continue
        case["prompt"] = render.prompt
        case["special_ranges"] = [list(r) for r in render.special_ranges]
        case["thinking_mode"] = render.thinking_mode
        case["output_constraint"] = (
            render.output_constraint[0] if render.output_constraint else None
        )
        case["token_count"] = len(render.prompt_ids)
        case["token_sha256"] = token_digest(render.prompt_ids)
    fixture["recipe_version"] = bridge.recipe_version()


def main(paths):
    bridge = load_bridge()
    if not bridge.is_available():
        sys.exit("deepseek_recipe is not installed")
    paths = [Path(p) for p in paths] or sorted(FIXTURES.glob("*.json"))
    for path in paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        fill(fixture, bridge)
        path.write_text(dump(fixture), encoding="utf-8")
        errors = sum(1 for case in fixture["cases"] if "error" in case)
        print(f"{path.name}: {len(fixture['cases'])} cases, {errors} rejected")


if __name__ == "__main__":
    main(sys.argv[1:])
