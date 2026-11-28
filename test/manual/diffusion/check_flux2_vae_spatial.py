"""Two-rank CUDA/NCCL FLUX.2 VAE with full pretrained weights.

Checks native serial against Diffusers and spatial tiled decode against that
serial oracle. The old direct-per-tile decoder supplies a negative control.

Run with two CUDA GPUs and a pinned local FLUX.2 VAE snapshot::

    torchrun --standalone --nproc_per_node=2 \
        test/manual/diffusion/check_flux2_vae_spatial.py \
        --model /path/to/snapshot/vae --checkpoint-revision REV --out /tmp/vae-check

FP32 uses a scoped strict oracle policy; BF16 retains the caller's policy.
This full-weight acceptance task is intentionally separate from per-commit CI.
"""

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2 as OfficialVAE
from safetensors.torch import load_file

from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    disable_spatial_parallel_decode,
    spatial_parallel_decode_disabled,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_flux2 import (
    AutoencoderKLFlux2,
)
from sglang.multimodal_gen.test.precision_utils import strict_fp32_oracle


def difference(actual, expected, atol, rtol):
    d = (actual.float() - expected.float()).abs()
    out = d > atol + rtol * expected.float().abs()
    return dict(
        max_abs=float(d.max()),
        rmse=float(d.square().mean().sqrt()),
        out_of_tolerance=int(out.sum()),
        shape=list(actual.shape),
        finite=bool(torch.isfinite(actual).all()),
    )


def precision_policy():
    return dict(
        matmul=torch.get_float32_matmul_precision(),
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint-revision", default=None)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        tensor_parallel_degree=world, sequence_parallel_degree=1, backend="nccl"
    )
    report = dict(
        rank=rank,
        world=world,
        torch=torch.__version__,
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        checkpoint_revision=args.checkpoint_revision,
        checkpoint_sha256=hashlib.sha256(
            (args.model / "diffusion_pytorch_model.safetensors").read_bytes()
        ).hexdigest(),
        cases=[],
    )
    try:
        config = Flux2VAEConfig()
        config.use_parallel_decode = True
        config.parallel_decode_mode = "spatial_shard"
        config.update_model_arch(json.loads((args.model / "config.json").read_text()))
        vae = AutoencoderKLFlux2(config).eval()
        state = load_file(str(args.model / "diffusion_pytorch_model.safetensors"))
        vae.load_state_dict(state, strict=True)
        del state
        assert vae._spatial_parallel_decode_enabled
        report["all_weights_loaded_strict"] = True
        vae.enable_tiling()
        for dtype in (torch.float32, torch.bfloat16):
            before = precision_policy()
            context = strict_fp32_oracle() if dtype == torch.float32 else nullcontext()
            with context:
                vae.to(device=f"cuda:{rank}", dtype=dtype)
                reference = (
                    OfficialVAE.from_pretrained(
                        str(args.model), torch_dtype=dtype, local_files_only=True
                    )
                    .to(f"cuda:{rank}")
                    .eval()
                )
                reference.enable_tiling()
                atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
                for h, w in ((129, 127), (193, 129), (1, 3), (129, 127)):
                    generator = torch.Generator(device=f"cuda:{rank}").manual_seed(
                        42 + h + w
                    )
                    z = torch.randn(
                        1,
                        32,
                        h,
                        w,
                        device=f"cuda:{rank}",
                        dtype=dtype,
                        generator=generator,
                    )
                    with torch.inference_mode(), disable_spatial_parallel_decode():
                        serial = vae.decode(z)
                        official = reference.decode(z).sample
                    with torch.inference_mode():
                        parallel = vae.decode(z)
                    row = dict(
                        dtype=str(dtype),
                        precision_policy=precision_policy(),
                        latent=[h, w],
                        atol=atol,
                        rtol=rtol,
                        native_vs_official=difference(serial, official, atol, rtol),
                        parallel_vs_serial=difference(parallel, serial, atol, rtol),
                        spatial_scope_restored=not spatial_parallel_decode_disabled(),
                    )
                    if h == 129 and len(report["cases"]) == 0:
                        original = vae._decode_latents
                        vae._decode_latents = lambda tile: vae.decoder(tile)
                        try:
                            with torch.inference_mode():
                                old = vae.decode(z)
                            row["pre_fix_direct_tile_control"] = difference(
                                old, serial, atol, rtol
                            )
                        finally:
                            vae._decode_latents = original
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
                    (args.out / f"rank-{rank}.json").write_text(
                        json.dumps(report, indent=2)
                    )
                    del z, serial, official, parallel
                del reference
                torch.cuda.empty_cache()
            assert precision_policy() == before, "Oracle precision policy leaked"
        report["passed"] = all(
            x["native_vs_official"]["finite"]
            and x["parallel_vs_serial"]["finite"]
            and x["native_vs_official"]["out_of_tolerance"] == 0
            and x["parallel_vs_serial"]["out_of_tolerance"] == 0
            and x["spatial_scope_restored"]
            for x in report["cases"]
        )
        assert report["cases"][0]["pre_fix_direct_tile_control"]["out_of_tolerance"] > 0
        assert report["passed"]
    except BaseException as exc:
        report.update(passed=False, error=repr(exc))
        raise
    finally:
        (args.out / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
