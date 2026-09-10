# SPDX-License-Identifier: Apache-2.0

import ast
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.util import resolve_name
from pathlib import Path
from threading import Barrier

import pytest
import torch

from sglang.multimodal_gen.configs.utils import expand_path_fields
from sglang.multimodal_gen.runtime.layers.attention.mask_strategy import dict_to_3d_list
from sglang.multimodal_gen.runtime.utils.argparse import (
    FlexibleArgumentParser,
    StoreBoolean,
)
from sglang.multimodal_gen.runtime.utils.precision import (
    get_compute_dtype,
    get_mixed_precision_state,
    set_mixed_precision_policy,
)


def test_models_do_not_import_pipeline_stages():
    root = Path(__file__).resolve().parents[2]
    violations = []
    for path in sorted((root / "runtime/models").rglob("*.py")):
        package = "sglang.multimodal_gen." + str(path.parent.relative_to(root)).replace(
            "/", "."
        )
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    prefix = resolve_name("." * node.level + prefix, package)
                names = [prefix] + [f"{prefix}.{alias.name}" for alias in node.names]
            if any(
                name.startswith("sglang.multimodal_gen.runtime.pipelines_core.stages")
                for name in names
            ):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, "Models must not depend on pipeline stages: " + ", ".join(
        violations
    )


def test_argument_parser_preserves_config_and_explicit_values(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("num_gpus: 2\nuse_cache: true\n")
    parser = FlexibleArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--use-cache", action=StoreBoolean, default=False)
    args = parser.parse_args(
        ["generate", "--config", str(config), "--num_gpus=4", "--use-cache", "false"]
    )
    assert args.num_gpus == 4
    assert args.use_cache is False
    assert args._provided == {"num_gpus", "use_cache"}


def test_expand_paths_preserves_slots_and_non_path_fields():
    @dataclass(slots=True)
    class Config:
        model_path: str = "~/model"
        image_path: list = field(default_factory=lambda: ["~/image.png", None])
        model_paths: dict = field(
            default_factory=lambda: {"vae": "~/vae", "other": None}
        )
        prompt: str = "~/not-a-path"

    config = Config()
    expand_path_fields(config)
    assert config.model_path == os.path.expanduser("~/model")
    assert config.image_path == [os.path.expanduser("~/image.png"), None]
    assert config.model_paths == {"vae": os.path.expanduser("~/vae"), "other": None}
    assert config.prompt == "~/not-a-path"


def test_mixed_precision_state_is_thread_local():
    barrier = Barrier(2)

    def worker(dtype):
        assert get_compute_dtype() == torch.get_default_dtype()
        with pytest.raises(ValueError, match="Mixed precision state not set"):
            get_mixed_precision_state()
        set_mixed_precision_policy(dtype, torch.float32, output_dtype=dtype)
        barrier.wait(timeout=10)
        assert get_mixed_precision_state().output_dtype == dtype
        return get_compute_dtype()

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(worker, (torch.float16, torch.bfloat16))) == [
            torch.float16,
            torch.bfloat16,
        ]


def test_attention_mask_strategy_preserves_tensor_identity():
    mask = torch.tensor([True, False])
    strategy = {"1_0_2": mask}
    inferred = dict_to_3d_list(strategy)
    assert len(inferred) == 2
    assert inferred[1][0][2] is mask
    assert inferred[0][0][2] is None
    assert dict_to_3d_list(strategy, 1, 1, 1) == [[[None]]]
    assert dict_to_3d_list(None, 2, 1, 1) == [[[None]], [[None]]]
