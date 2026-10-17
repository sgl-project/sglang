"""FLUX.2 / Klein through the ComfyUI worker, driven by the real Flux2Adapter.

Each case launches its own worker process: the scheduler client is a process
singleton, so two generators cannot coexist in one interpreter. Tiny random
checkpoints exercise every family on a GPU.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

PIPELINES = {
    "dev": "Flux2ComfyUIPipeline",
    "klein": "Flux2KleinComfyUIPipeline",
    "klein_base": "Flux2KleinBaseComfyUIPipeline",
}
LATENT_H, LATENT_W, TEXT_LEN = 8, 8, 16


def _worker(spec: dict, out_path: str) -> None:
    from types import SimpleNamespace

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux2 import (
        Flux2Executor,
    )
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    generator = DiffGenerator.from_pretrained(
        model_path=spec["model_path"],
        pipeline_class_name=PIPELINES[spec["family"]],
        num_gpus=1,
        sp_degree=1,
        comfyui_mode=True,
        dit_cpu_offload=False,
    )
    channels, joint = spec["channels"], spec["joint"]
    torch.manual_seed(0)
    rows = spec.get("rows", 1)
    x = torch.randn(rows, channels, LATENT_H, LATENT_W, device="cuda").bfloat16()
    context = torch.randn(rows, TEXT_LEN, joint, device="cuda").bfloat16()
    refs = [
        torch.randn(rows, channels, 4, 4, device="cuda").bfloat16()
        for _ in range(spec.get("refs", 0))
    ]
    guidance = (
        torch.tensor([spec["guidance"]], device="cuda")
        if spec.get("guidance") is not None
        else None
    )
    sigma = torch.tensor([spec["sigma"]], device="cuda")
    # The real executor builds the request, sends it and unpacks the reply.
    executor = Flux2Executor(
        generator,
        spec["model_path"],
        None,
        SimpleNamespace(unet_config={"dtype": torch.bfloat16}),
    )
    packed = executor.adapter.pack(
        x, sigma, context, guidance=guidance, ref_latents=refs or None
    )
    result = executor._execute_packed(packed, x, sigma)
    torch.save(result.float().cpu(), out_path)
    generator.shutdown()


def _run(tmp_path, **spec) -> torch.Tensor:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / f"out_{len(list(tmp_path.iterdir()))}.pt"
    spec_path.write_text(json.dumps(spec))
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", str(spec_path), str(out_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    return torch.load(out_path)


@pytest.fixture(scope="module")
def checkpoints():
    from safetensors.torch import save_file

    from sglang.multimodal_gen.test.unit.test_comfyui_flux2 import _bfl_state_dict

    with tempfile.TemporaryDirectory() as root:
        paths = {}
        for name, kwargs in {
            "dev": dict(guidance=True, joint=15360),
            "klein_base": dict(guidance=True, joint=7680),
            "klein": dict(guidance=False, joint=7680),
        }.items():
            torch.manual_seed(len(paths))
            path = os.path.join(root, f"{name}.safetensors")
            save_file(_bfl_state_dict(**kwargs), path)
            paths[name] = (path, kwargs["joint"])
        yield paths


def _spec(checkpoints, family, **extra):
    path, joint = checkpoints[family]
    return dict(
        family=family, model_path=path, channels=16, joint=joint, sigma=0.7, **extra
    )


@pytest.mark.parametrize("family", ["dev", "klein_base", "klein"])
def test_worker_step_returns_finite_prediction(tmp_path, checkpoints, family):
    out = _run(
        tmp_path,
        **_spec(checkpoints, family, guidance=4.0 if family != "klein" else None),
    )
    assert out.shape == (1, 16, LATENT_H, LATENT_W)
    assert torch.isfinite(out).all() and out.abs().max() > 0


def test_stacked_rows_and_reference_images_reach_the_worker(tmp_path, checkpoints):
    """Positive+negative stacked into one call, with two reference images."""
    out = _run(tmp_path, **_spec(checkpoints, "klein", rows=2, refs=2))
    assert out.shape == (2, 16, LATENT_H, LATENT_W)
    assert torch.isfinite(out).all()
    single = _run(tmp_path, **_spec(checkpoints, "klein", rows=2, refs=0))
    assert (out - single).abs().max() > 1e-3


def test_request_guidance_changes_the_embedded_guidance(tmp_path, checkpoints):
    """FluxGuidance must reach the worker instead of the config default."""
    low = _run(tmp_path, **_spec(checkpoints, "dev", guidance=1.0))
    high = _run(tmp_path, **_spec(checkpoints, "dev", guidance=8.0))
    assert (low - high).abs().max() > 1e-3


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(json.load(open(sys.argv[2])), sys.argv[3])
    else:
        sys.exit(pytest.main([__file__, "-v"]))
