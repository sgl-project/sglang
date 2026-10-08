# SPDX-License-Identifier: Apache-2.0
"""Calibrate fused Pi0.5 Linears and export a native SGLang FP8 checkpoint.

Input is a torch.save list of already preprocessed observation dictionaries:
images (camera-name -> [B,3,H,W] float tensor in [-1,1]), image_masks
(camera-name -> [B] bool), tokens/token_masks ([B,L]), and noise ([B,T,D]).
Use the checkpoint's actual preprocessing/tokenizer and robot normalization.
Use --dummy-calibration explicitly for synthetic pipeline checks. Dataset errors
never silently fall back to synthetic inputs. See the accompanying README.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.vla.pi05_quantization import (
    DEFAULT_COMPONENTS,
    replace_projections,
)


def load_observations(path: str, config, device: torch.device) -> list[dict]:
    samples = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(samples, list) or not samples:
        raise ValueError("Calibration data must be a nonempty list of observations")
    prepared = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Observation {index} must be a dictionary")
        tokens, masks, noise = (
            sample[key] for key in ("tokens", "token_masks", "noise")
        )
        if tokens.ndim != 2 or tokens.shape != masks.shape or tokens.shape[0] != 1:
            raise ValueError(
                f"Observation {index}: expected single-request tokens/token_masks"
            )
        if tokens.dtype != torch.int64 or masks.dtype != torch.bool or not masks.any():
            raise ValueError(f"Observation {index}: invalid tokens or token masks")
        if tokens.shape[1] > config.max_token_len:
            raise ValueError(
                f"Observation {index}: token length exceeds checkpoint limit"
            )
        if (
            noise.shape != (1, config.action_horizon, config.action_dim)
            or not torch.isfinite(noise).all()
        ):
            raise ValueError(f"Observation {index}: invalid fixed noise")
        images, image_masks = [], []
        for camera in config.image_keys:
            image = sample["images"][camera]
            mask = sample["image_masks"][camera]
            if (
                image.shape != (1, 3, *config.image_size)
                or not torch.isfinite(image).all()
            ):
                raise ValueError(f"Observation {index}: invalid image {camera}")
            if image.min() < -1 or image.max() > 1 or not image.is_floating_point():
                raise ValueError(
                    f"Observation {index}: {camera} must be normalized to [-1,1]"
                )
            if mask.shape != (1,) or mask.dtype != torch.bool:
                raise ValueError(f"Observation {index}: invalid camera mask {camera}")
            images.append(image.to(device=device, dtype=torch.float32))
            image_masks.append(mask.to(device))
        prepared.append(
            dict(
                images=images,
                image_masks=image_masks,
                tokens=tokens.to(device),
                token_masks=masks.to(device),
                noise=noise.to(device=device, dtype=torch.float32),
            )
        )
    return prepared


def dummy_observations(
    config, num_samples: int, seed: int, device: torch.device
) -> list[dict]:
    """Match the Thor tutorial's random image/token/noise calibration recipe."""
    from sglang.multimodal_gen.runtime.models.vlas.pi05_core import PALIGEMMA_VOCAB_SIZE

    if num_samples <= 0:
        raise ValueError("Dummy sample count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        {
            "images": [
                torch.randn(1, 3, *config.image_size, generator=generator).to(device)
                for _ in config.image_keys
            ],
            "image_masks": [
                torch.ones(1, dtype=torch.bool, device=device)
                for _ in config.image_keys
            ],
            "tokens": torch.randint(
                0, PALIGEMMA_VOCAB_SIZE, (1, config.max_token_len), generator=generator
            ).to(device),
            "token_masks": torch.ones(
                1, config.max_token_len, dtype=torch.bool, device=device
            ),
            "noise": torch.randn(
                1, config.action_horizon, config.action_dim, generator=generator
            ).to(device),
        }
        for _ in range(num_samples)
    ]


@torch.no_grad()
def run_actions(core, sample: dict, num_steps: int) -> torch.Tensor:
    kv, masks, full_attention = core.encode_prefix(
        sample["images"], sample["image_masks"], sample["tokens"], sample["token_masks"]
    )
    actions = sample["noise"].clone()
    layout = core.prepare_denoise_layout(masks, actions, full_attention)
    timesteps = torch.linspace(
        1.0, 1.0 / num_steps, num_steps, device=actions.device, dtype=torch.float32
    )
    for step, timestep_value in enumerate(timesteps):
        timestep = timestep_value.expand(actions.shape[0])
        velocity = core.denoise_step(
            masks, kv, actions, timestep, full_attention, denoise_layout=layout
        )
        if not torch.isfinite(velocity).all():
            raise ValueError(f"Nonfinite velocity at denoising step {step}")
        actions.add_(velocity, alpha=-1.0 / num_steps)
    if not torch.isfinite(actions).all():
        raise ValueError("Nonfinite output actions")
    return actions


def export_state(core, names: list[str]) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in core.state_dict().items()
        if "_quantizer." not in name
    }
    for name in names:
        layer = core.get_submodule(name)
        for quantizer_name, scale_name in (
            ("weight_quantizer", "weight_scale"),
            ("input_quantizer", "input_scale"),
        ):
            quantizer = getattr(layer, quantizer_name)
            amax = quantizer.amax.detach().float().cpu()
            if amax.numel() != 1 or not torch.isfinite(amax).all() or (amax <= 0).any():
                raise ValueError(
                    f"Missing or invalid calibrated per-tensor amax: {name}.{quantizer_name}"
                )
            state[f"{name}.{scale_name}"] = (amax / 448.0).reshape(1)
        weight = state[f"{name}.weight"].float()
        state[f"{name}.weight"] = (
            (weight / state[f"{name}.weight_scale"])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
    return state


def make_quantization_config(names: list[str]) -> dict:
    import modelopt.torch.quantization as mtq

    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    config["quant_cfg"] = [{"quantizer_name": "*", "enable": False}]
    for name in names:
        for quantizer in ("weight_quantizer", "input_quantizer"):
            config["quant_cfg"].append(
                {
                    "quantizer_name": f"{name}.{quantizer}",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                    "enable": True,
                }
            )
    return config


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--calibration-data")
    source_group.add_argument(
        "--dummy-calibration",
        action="store_true",
        help="Explicit synthetic calibration, matching the Thor tutorial fallback",
    )
    parser.add_argument("--dummy-num-samples", type=int, default=1)
    parser.add_argument("--dummy-validation-samples", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--validation-data",
        help="Separate held-out observations; compare BF16, fake-quant and native FP8",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=DEFAULT_COMPONENTS,
        default=DEFAULT_COMPONENTS,
    )
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        raise ValueError("Pi0.5 FP8 calibration requires an SM89+ CUDA GPU")
    import modelopt.torch.quantization as mtq

    from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import Pi05PolicyModel
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    torch.manual_seed(args.seed)
    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path,
        pipeline_class_name="Pi05Pipeline",
        num_gpus=1,
        warmup_mode="off",
        attention_backend="torch_sdpa",
    )
    set_global_server_args(server_args)
    # Pi0.5 reuses SRT SigLIP, whose layers read the published parallel bag.
    # The normal diffusion GPU worker publishes this before model construction;
    # this standalone entry must do the same.
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    publish(SrtServerArgs(model_path="dummy", tp_size=1), role="diffusion_gpu_worker")
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    if torch.distributed.is_initialized():
        raise ValueError("Run Pi0.5 calibration in a standalone single-GPU process")
    rendezvous = tempfile.TemporaryDirectory(prefix="pi05-fp8-rendezvous-")

    def cleanup():
        destroy_model_parallel()
        destroy_distributed_environment()
        rendezvous.cleanup()

    atexit.register(cleanup)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{rendezvous.name}/store",
        device_id=torch.device("cuda", 0),
    )
    initialize_model_parallel(sequence_parallel_degree=1)

    config = server_args.pipeline_config
    config.enable_prefix_cuda_graph = False
    config.enable_action_cuda_graph = False
    policy = Pi05PolicyModel.from_pretrained(args.model_path, config)
    if policy._fp8_projection_names:
        raise ValueError("Calibration input must be an unquantized checkpoint")
    steps = (
        args.num_steps
        if args.num_steps is not None
        else config.default_num_inference_steps
    )
    if steps <= 0:
        raise ValueError("num-steps must be positive")
    samples = (
        dummy_observations(config, args.dummy_num_samples, args.seed, policy.device)
        if args.dummy_calibration
        else load_observations(args.calibration_data, config, policy.device)
    )
    validation_samples = (
        load_observations(args.validation_data, config, policy.device)
        if args.validation_data
        else (
            dummy_observations(
                config, args.dummy_validation_samples, args.seed + 1, policy.device
            )
            if args.dummy_calibration
            else []
        )
    )
    core = policy.core_model
    baseline_actions = [
        run_actions(core, sample, steps).cpu() for sample in validation_samples
    ]
    names = replace_projections(core, args.components, quantized=False)
    quant_cfg = make_quantization_config(names)

    def forward_loop(model):
        for sample in samples:
            run_actions(model, sample, steps)

    core = mtq.quantize(core, quant_cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(core)
    fake_actions = [
        run_actions(core, sample, steps).cpu() for sample in validation_samples
    ]
    state = export_state(core, names)
    source = Path(policy.model_path)
    source_config = source / "config.json"
    payload = json.loads(source_config.read_text()) if source_config.exists() else {}
    # Save the effective architecture/config even for converted OpenPI checkpoints.
    payload.update(
        paligemma_variant=config.paligemma_variant,
        action_expert_variant=config.action_expert_variant,
        chunk_size=config.action_horizon,
        max_action_dim=config.action_dim,
        max_state_dim=config.state_dim,
        tokenizer_max_length=config.max_token_len,
        num_inference_steps=steps,
        n_action_steps=config.n_action_steps,
        image_resolution=list(config.image_size),
        min_period=config.time_embedding_min_period,
        max_period=config.time_embedding_max_period,
        empty_cameras=config.empty_cameras,
    )
    if not payload.get("input_features"):
        payload["input_features"] = {
            **{
                f"observation.images.{name}": {"type": "VISUAL"}
                for name in config.image_keys
                if not name.startswith("empty_camera_")
            },
            "observation.state": {"type": "STATE", "shape": [config.state_dim]},
        }
    if not payload.get("output_features"):
        payload["output_features"] = {"action": {"shape": [config.output_action_dim]}}
    payload["quantization_config"] = dict(
        quant_method="modelopt",
        quant_algo="FP8",
        pi05_fused_projections=True,
        components=args.components,
    )
    payload["pi05_fp8_calibration"] = dict(
        samples=len(samples),
        num_steps=steps,
        seed=args.seed,
        source_model=args.model_path,
        input_kind="synthetic" if args.dummy_calibration else "dataset",
        calibration_data=str(Path(args.calibration_data).resolve())
        if args.calibration_data
        else None,
        calibration_sha256=file_sha256(args.calibration_data)
        if args.calibration_data
        else None,
        dummy_image_distribution="standard_normal" if args.dummy_calibration else None,
        dummy_all_cameras_present=True if args.dummy_calibration else None,
        source_checkpoint_path=str(source.resolve()),
    )
    output.mkdir(parents=True)
    save_file(state, str(output / "model.safetensors"))
    (output / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    for path in source.iterdir():
        if (
            path.is_file()
            and path.name != "config.json"
            and not path.name.endswith(".safetensors.index.json")
            and (path.suffix in (".json", ".model", ".txt") or "token" in path.name)
        ):
            shutil.copy2(path, output / path.name)
    if (source / "assets").is_dir():
        shutil.copytree(source / "assets", output / "assets")
    print(f"Exported {len(names)} FP8 projections to {output}")
    if validation_samples:
        del state, core, policy
        gc.collect()
        torch.cuda.empty_cache()
        native = Pi05PolicyModel.from_pretrained(str(output), config)
        native_actions = [
            run_actions(native.core_model, sample, steps).cpu()
            for sample in validation_samples
        ]
        report = {
            "validation_metadata": {
                "input_kind": "dataset" if args.validation_data else "synthetic",
                "samples": len(validation_samples),
                "seed": args.seed + 1 if not args.validation_data else None,
            }
        }
        for label, reference in (
            ("bf16_vs_native", baseline_actions),
            ("fake_quant_vs_native", fake_actions),
        ):
            expected = torch.cat(reference).float()
            actual = torch.cat(native_actions).float()
            diff = (expected - actual).abs()
            report[label] = dict(
                max_abs_diff=diff.max().item(),
                mean_abs_diff=diff.mean().item(),
                per_dimension_max_abs_diff=diff.amax(dim=(0, 1)).tolist(),
                valid_action_max_abs_diff=diff[
                    :, : config.n_action_steps, : config.output_action_dim
                ]
                .max()
                .item(),
                valid_action_mean_abs_diff=diff[
                    :, : config.n_action_steps, : config.output_action_dim
                ]
                .mean()
                .item(),
                cosine_similarity=torch.nn.functional.cosine_similarity(
                    expected.reshape(1, -1), actual.reshape(1, -1)
                ).item(),
            )
        (output / "validation_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
