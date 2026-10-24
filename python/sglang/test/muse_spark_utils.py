"""Local checkpoint resolution for the Muse-Spark serving tests."""

import json
from pathlib import Path

from sglang.srt.utils import find_local_repo_dir
from sglang.test.test_utils import try_cached_model


def try_cached_muse_spark_model(model_repo: str, *, draft: bool = False) -> str:
    """Use a complete local snapshot, or retain the Hub ID for downloading."""
    required_files = (
        ()
        if draft
        else (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "l4_200k_base",
            "processor_config.json",
            "chat_template.jinja",
            "generation_config.json",
            "hf_quant_config.json",
        )
    )
    model_dir = try_cached_model(model_repo)
    candidates = [model_dir] if model_dir != model_repo else []
    try:
        candidates.append(find_local_repo_dir(model_repo))
    except OSError:
        pass
    for model_dir in candidates:
        if not model_dir:
            continue
        snapshot = Path(model_dir)
        try:
            index = snapshot / "model.safetensors.index.json"
            if index.exists() or index.is_symlink():
                weight_map = json.loads(index.read_text())["weight_map"]
                if not isinstance(weight_map, dict) or not weight_map:
                    continue
                weights = set(weight_map.values())
            else:
                weights = {"model.safetensors"}
            files = {"config.json", *required_files, *weights}
            if not all(
                (snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
                for name in files
            ):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue

        print(f"Using complete local checkpoint for {model_repo}: {snapshot}")
        return str(snapshot.absolute())
    return model_repo
