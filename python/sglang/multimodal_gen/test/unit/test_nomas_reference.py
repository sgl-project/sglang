# SPDX-License-Identifier: Apache-2.0
"""Optional H200 parity suite against a pinned, external Hy-Nomas checkout.

NOMAS_REFERENCE_ROOT=/path/to/Hy-Nomas NOMAS_PARITY_OUTPUT=/path/to/results
PYTHONPATH=python pytest -s python/sglang/multimodal_gen/test/unit/test_nomas_reference.py

Only tests import hymm. Production modules never depend on the reference repo.
The full-width smoke test additionally requires NOMAS_TEST_FULL=1.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_nomas import condition

from sglang.multimodal_gen.configs.models.dits.nomas import NomasArchConfig, NomasConfig
from sglang.multimodal_gen.runtime.models.dits.nomas import NomasTransformer3DModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
    NomasLatentSession,
)


@pytest.fixture(autouse=True, scope="module")
def _model_parallel(single_process_model_parallel):
    """The framework's parallel linears resolve the TP group at construction."""
    yield


REFERENCE_ROOT = os.environ.get("NOMAS_REFERENCE_ROOT")
pytestmark = pytest.mark.skipif(
    not REFERENCE_ROOT or not torch.cuda.is_available(),
    reason="requires CUDA and an external Hy-Nomas checkout",
)


def reference_model(config):
    sys.path.insert(0, REFERENCE_ROOT)
    from hymm.models.dit.nomas_cpt import CPTHYNomasDiffusionTransformer

    return (
        CPTHYNomasDiffusionTransformer(
            patch_size=[1, 1, 1],
            in_channels=32,
            out_channels=32,
            hidden_size=config.hidden_size,
            heads_num=config.num_attention_heads,
            mm_double_blocks_depth=config.num_layers,
            mm_single_blocks_depth=0,
            rope_dim_list=[16, 56, 56],
            text_states_dim=3584,
            text_projection="single_refiner",
            seg_ar_attention_backend="flex",
            camera_injection_mode="extrinsics_attn",
            action_config=dict(
                keyboard_dim_in=64,
                hidden_size=128,
                img_hidden_size=config.hidden_size,
                keyboard_hidden_dim=1024,
                heads_num=16,
                patch_size=[1, 1, 1],
                qk_norm=True,
                qk_norm_type="rms",
                qkv_bias=False,
                rope_dim_list=[8, 28, 28],
                rope_theta=256,
                per_segment_camera_attn=False,
                per_chunk_camera_attn=True,
                camera_chunk_size=config.camera_chunk_size,
                seg_ar_attention_backend="flex",
            ),
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.fixture(scope="module")
def fixed_checkpoint(tmp_path_factory):
    config = NomasArchConfig(hidden_size=256, num_attention_heads=2, num_layers=2)
    torch.manual_seed(123)
    ref = reference_model(config)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if p.ndim == 1 and "norm" in name and name.endswith("weight"):
                p.fill_(1)
            else:
                p.normal_(0, 0.02)
    output = Path(
        os.environ.get("NOMAS_PARITY_OUTPUT", tmp_path_factory.mktemp("nomas"))
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "fixed_random_bf16.safetensors"
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in ref.state_dict().items()},
        str(path),
        metadata={
            "seed": "123",
            "reference_revision": "98a28976f23552bb2e2cf7941986e9cd10a3118a",
            "synthetic": "true",
        },
    )
    ref = ref.cuda().bfloat16()
    ref.load_state_dict(load_file(str(path), device="cuda"), strict=True)
    manifest = dict(
        seed=123,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameters=sum(p.numel() for p in ref.parameters()),
        synthetic=True,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    (output / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
    return config, ref, path, output


def compare(actual, expected, label, report, atol=0.006, rtol=0.02):
    diff = actual.float() - expected.float()
    report[label] = {
        "max_abs": diff.abs().max().item(),
        "relative_l2": (diff.norm() / expected.float().norm().clamp_min(1e-12)).item(),
    }
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=label)


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def test_two_segment_checkpoint_and_euler_parity(fixed_checkpoint, backend):
    from hymm.pipeline.solvers.euler import EulerSolver

    config, ref, path, output = fixed_checkpoint
    native = (
        NomasTransformer3DModel(
            NomasConfig(arch_config=config), attention_backend=backend
        )
        .cuda()
        .bfloat16()
        .eval()
        .requires_grad_(False)
    )
    native.load_state_dict(load_file(str(path), device="cuda"), strict=True)
    assert ref.state_dict().keys() == native.state_dict().keys()
    assert all(
        torch.equal(v, native.state_dict()[k]) for k, v in ref.state_dict().items()
    )
    session = NomasLatentSession(native, cache_capacity=64, seed=42, steps=8)
    cache = [
        {
            key: torch.zeros(1, 64, 2, 128, device="cuda", dtype=torch.bfloat16)
            for key in ("k", "v")
        }
        for _ in range(2)
    ]
    masks = [torch.zeros(1, 64, device="cuda", dtype=torch.bool) for _ in cache]
    cond = condition("cuda", torch.bfloat16)
    ref_gen = torch.Generator().manual_seed(42)
    init_gen = torch.Generator().manual_seed(99)
    report = {}
    for segment in range(2):
        initial = torch.randn(
            1, 32, 2, 2, 2, generator=init_gen, dtype=torch.bfloat16
        ).cuda()
        actual, trace = session.generate_segment(
            (2, 2, 2), cond, initial_latents=initial, return_trace=True
        )
        assert all(item["prediction"].abs().max() > 0.01 for item in trace)
        latent = initial.clone()
        scheduler = EulerSolver(shift=5.0)
        scheduler.set_timesteps(8, device="cuda")
        kwargs = dict(
            text_states=cond.text_states,
            text_mask=cond.text_mask,
            camera_embedding=cond.camera_embedding,
            camera_has_mask=cond.camera_has_mask,
            intra_seg_txt_offsets=[list(cond.text_offsets)],
            intra_seg_vid_windows=[list(x) for x in cond.video_windows],
            kv_cache_list=cache,
            cache_mask_list=masks,
            cache_valid_len=segment * 16,
            global_frame_counter=segment * 2,
            hide_cached_action_tails=True,
            current_action_start=2,
        )
        for step, timestep in enumerate(scheduler.timesteps):
            pred, _ = ref.forward_with_cache(
                hidden_states=latent, timestep=timestep.expand(1), **kwargs
            )
            compare(
                trace[step]["prediction"],
                pred,
                f"segment{segment}.step{step}.prediction",
                report,
            )
            latent = scheduler.step(pred, timestep, latent, return_dict=False)[
                0
            ].bfloat16()
            compare(
                trace[step]["latent"],
                latent,
                f"segment{segment}.step{step}.latent",
                report,
                atol=0.04,
            )
        compare(actual, latent, f"segment{segment}.output", report, atol=0.04)
        noise = torch.randn(latent.shape, generator=ref_gen, dtype=latent.dtype).cuda()
        noised = ((1 - 0.206897) * latent.float() + 0.206897 * noise.float()).bfloat16()
        _, added = ref.forward_with_cache(
            hidden_states=noised,
            timestep=torch.tensor([206.897], device="cuda"),
            update_cache=True,
            **kwargs,
        )
        assert added == 16 and session.cache.valid_length == (segment + 1) * 16
        for layer, (a, b) in enumerate(zip(session.cache.layers, cache)):
            for key in ("k", "v"):
                compare(
                    a[key][:, : session.cache.valid_length],
                    b[key][:, : session.cache.valid_length],
                    f"segment{segment}.layer{layer}.{key}",
                    report,
                    atol=0.04,
                )
        assert torch.equal(
            session.cache.visible[:, segment * 16 : (segment + 1) * 16],
            masks[0][:, segment * 16 : (segment + 1) * 16],
        )
    # Raw camera preprocessing must preserve front padding, ordering and residual.
    poses = torch.eye(4, device="cuda").repeat(1, 5, 1, 1)
    poses[:, 1:, 0, 3] = torch.arange(1, 5, device="cuda") * 0.1
    compare(
        native.embed_camera(poses),
        ref._preprocess_camera(
            poses, None, None, 2, 2, torch.bfloat16, torch.device("cuda")
        ),
        "camera_embedding",
        report,
        atol=0,
        rtol=0,
    )
    (output / f"parity_{backend}.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "backend": backend,
                "comparisons": len(report),
                "max_relative_l2": max(x["relative_l2"] for x in report.values()),
                "max_abs": max(x["max_abs"] for x in report.values()),
            }
        )
    )


def test_full_checkpoint_schema():
    with torch.device("meta"):
        ref = reference_model(NomasArchConfig())
        native = NomasTransformer3DModel(NomasConfig())
    a = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
    b = {k: tuple(v.shape) for k, v in native.state_dict().items()}
    assert a == b
    assert sum(p.numel() for p in native.parameters()) == 8_553_434_592


@pytest.mark.skipif(
    os.environ.get("NOMAS_TEST_FULL") != "1", reason="opt-in full-width H200 test"
)
@torch.inference_mode()
def test_full_width_native_rollout():
    torch.manual_seed(123)
    torch.cuda.reset_peak_memory_stats()
    # Allocate directly on GPU; initialize every trained parameter nonzero.
    with torch.device("cuda"):
        model = (
            NomasTransformer3DModel(NomasConfig())
            .bfloat16()
            .eval()
            .requires_grad_(False)
        )
    for name, p in model.named_parameters():
        if p.ndim == 1 and "norm" in name and name.endswith("weight"):
            p.fill_(1)
        else:
            p.normal_(0, 0.02)
    init_peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    session = NomasLatentSession(model, cache_capacity=272, seed=42, steps=2)
    cond = condition("cuda", torch.bfloat16)
    for _ in range(2):
        result = session.generate_segment((2, 8, 8), cond)
        assert torch.isfinite(result).all()
    assert session.cache.valid_length == 272 and session.cache.frame_offset == 4
    report = {
        "full_parameters": sum(p.numel() for p in model.parameters()),
        "initialization_peak_gib": init_peak,
        "rollout_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "latent_shape": list(result.shape),
        "segments": 2,
        "steps_per_segment": 2,
    }
    # Full-width numerical comparison, using exactly the same random parameters.
    with torch.device("cuda"):
        ref = reference_model(NomasArchConfig()).bfloat16()
    ref.load_state_dict(model.state_dict(), strict=True)
    caches = [
        {key: torch.zeros_like(layer[key]) for key in ("k", "v")}
        for layer in session.cache.layers
    ]
    masks = [torch.zeros_like(session.cache.visible) for _ in caches]
    session.cache.reset()
    kwargs = dict(
        hidden_states=result,
        timestep=torch.tensor([206.897], device="cuda"),
        encoder_hidden_states=cond.text_states,
        text_mask=cond.text_mask,
        cache=session.cache,
        camera_embedding=cond.camera_embedding,
        camera_has_mask=cond.camera_has_mask,
        text_offsets=cond.text_offsets,
        video_windows=cond.video_windows,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual, entries = model.predict(**kwargs, collect_cache=True)
        expected, _ = ref.forward_with_cache(
            hidden_states=result,
            timestep=kwargs["timestep"],
            text_states=cond.text_states,
            text_mask=cond.text_mask,
            camera_embedding=cond.camera_embedding,
            camera_has_mask=cond.camera_has_mask,
            intra_seg_txt_offsets=[list(cond.text_offsets)],
            intra_seg_vid_windows=[list(x) for x in cond.video_windows],
            kv_cache_list=caches,
            cache_mask_list=masks,
            update_cache=True,
        )
    metrics = {}
    compare(actual, expected, "prediction", metrics, atol=0.06, rtol=0.02)
    for index, (entry, reference) in enumerate(zip(entries, caches)):
        for key in ("k", "v"):
            compare(
                entry[key],
                reference[key][:, :136],
                f"layer{index}.{key}",
                metrics,
                atol=0.06,
                rtol=0.02,
            )
    report["parity"] = metrics
    if os.environ.get("NOMAS_PARITY_OUTPUT"):
        path = Path(os.environ["NOMAS_PARITY_OUTPUT"])
        path.mkdir(parents=True, exist_ok=True)
        (path / "full_width.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "parity"}
            | {"parity_max_abs": max(x["max_abs"] for x in metrics.values())}
        )
    )


def test_long_mask_has_bounded_allocation():
    from sglang.multimodal_gen.runtime.models.dits.nomas_layers.kv import (
        NomasAttentionPlan,
    )

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    plan = NomasAttentionPlan(
        torch.ones(1, 8, device="cuda", dtype=torch.bool),
        torch.ones(1, 32768, device="cuda", dtype=torch.bool),
        2,
        16384,
        (2, 5, 8),
        ((0, 1, 0), (1, 2, 1)),
        "flex",
    )
    assert plan.mask is not None
    peak = torch.cuda.max_memory_allocated() - start
    # An eager dense boolean mask alone would exceed 2 GiB at this size.
    assert peak < 512 * 2**20, f"block-mask peak was {peak / 2**20:.1f} MiB"
    print(
        json.dumps(
            {
                "mask_query_tokens": 32776,
                "mask_key_tokens": 65544,
                "mask_peak_mib": peak / 2**20,
            }
        )
    )


@torch.inference_mode()
def test_canonical_camera_timeline_matches_upstream(fixed_checkpoint):
    config = NomasArchConfig(hidden_size=256, num_attention_heads=2, num_layers=2)
    native = (
        NomasTransformer3DModel(NomasConfig(arch_config=config))
        .eval()
        .cuda()
        .bfloat16()
    )
    ref = reference_model(config).cuda().bfloat16()
    ref.load_state_dict(native.state_dict(), strict=True)
    poses = torch.eye(4, device="cuda").reshape(1, 1, 4, 4).repeat(1, 45, 1, 1)
    poses[0, :, 0, 3] = torch.arange(45, device="cuda") / 100
    actual = native.embed_camera(poses)
    expected = ref._preprocess_camera(
        poses, None, None, 2, 2, torch.bfloat16, torch.device("cuda")
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (1, 12, 64)
    # Segment two begins at latent 10: camera samples 37..40, not 40..43.
    window = native.embed_camera(poses[:, 37:45])
    torch.testing.assert_close(actual[:, 10:], window, atol=0, rtol=0)
