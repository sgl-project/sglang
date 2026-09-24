# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mlx.core")
pytest.importorskip("transformers", minversion="5.12.1")

pytest_plugins = ["sglang.multimodal_gen.test.mlx.test_qwen_image21_pipeline"]
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="requires Apple Silicon MLX"
)


def test_pipeline_discovery_without_mlx():
    script = textwrap.dedent("""
        import sys

        sys.modules["mlx"] = None

        from sglang.multimodal_gen.registry import get_pipeline_class
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        for name in ("QwenImage21Pipeline", "QwenImage21MLXPipeline"):
            assert get_pipeline_class(name).pipeline_name == name
        model, architecture = ModelRegistry.resolve_model_cls(
            "QwenImage21Transformer2DModel"
        )
        assert model.__module__.endswith(".models.dits.qwen_image21")
        assert architecture == "QwenImage21Transformer2DModel"
        assert "mlx.core" not in sys.modules
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, SGLANG_DIFFUSION_PLATFORM_OVERRIDE="cpu"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("image_count,from_hub_cache", [(0, True), (2, False)])
def test_cli_resolves_mlx_pack_and_preserves_rgba_seeds(
    pipeline, tmp_path_factory, image_count, from_hub_cache
):
    output = tmp_path_factory.mktemp("mlx-cli")
    model_path = output / "Qwen-Image-2.1-MLX"
    model_path.symlink_to(pipeline.model_path, target_is_directory=True)
    revision = "0" * 40
    hub_cache = output / "hub"
    if from_hub_cache:
        snapshot = (
            hub_cache / "models--toxicdog--Qwen-Image-2.1-MLX/snapshots" / revision
        )
        snapshot.parent.mkdir(parents=True)
        snapshot.symlink_to(pipeline.model_path, target_is_directory=True)
    image = Image.new("RGBA", (64, 96), (70, 120, 200, 96))
    image_path = output / "reference.png"
    image.save(image_path)
    guidance = 3.71 if image_count else 1.0
    expected = [
        pipeline.generate(
            prompt="A red panda",
            width=64,
            height=64,
            num_inference_steps=3,
            seed=seed,
            images=[image] * image_count,
            guidance_scale=guidance,
        )[0]
        for seed in (17, 18)
    ]
    repo_root = Path(__file__).resolve().parents[5]
    command = [
        sys.executable,
        "-c",
        "from sglang.cli.main import main\nmain()",
        "generate",
        "--model-path",
        "toxicdog/Qwen-Image-2.1-MLX" if from_hub_cache else str(model_path),
        "--prompt",
        "A red panda",
        "--height",
        "64",
        "--width",
        "64",
        "--num-inference-steps",
        "3",
        "--num-outputs-per-prompt",
        "2",
        "--guidance-scale",
        str(guidance),
        "--seed",
        "17",
        "--warmup-mode",
        "off",
        "--save-output",
        "--output-file-path",
        str(output / "image.png"),
        "--perf-dump-path",
        str(output / "metrics.json"),
    ]
    if from_hub_cache:
        command.extend(["--revision", revision])
    if image_count:
        command.extend(["--image-path", *[str(image_path)] * image_count])
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=dict(
            os.environ,
            PYTHONPATH=str(repo_root / "python"),
            PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.15",
            PYTORCH_MPS_LOW_WATERMARK_RATIO="0.12",
            HF_HUB_OFFLINE="1",
            HF_HUB_CACHE=str(hub_cache),
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    logs = result.stdout + result.stderr
    assert result.returncode == 0, logs
    for index, reference in enumerate(expected):
        actual = Image.open(output / f"image_{index}.png")
        assert actual.mode == "RGBA" and actual.size == (64, 64)
        np.testing.assert_array_equal(np.array(actual), np.array(reference))
    peak = re.search(r"Memory usage - Max peak: ([\d.]+) MB", logs)
    assert peak is not None and float(peak[1]) > 0, logs
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["memory_checkpoints"]["mlx"]["peak_allocated_mb"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
