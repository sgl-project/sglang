"""Two-rank encoder folding must preserve single-rank native output.

The focused CLIP check isolates the component loader and SRT tensor-parallel
layers. The tiny SD3 check covers the public server API and complete pipeline.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = 2
_TINY_SD3_MODEL = "yujiepan/stable-diffusion-3-tiny-random"
_TINY_SD3_REVISION = "abcdbb999b2d30c35d03efdce0be981e1efac0a4"


def _tiny_clip_config():
    from sglang.multimodal_gen.configs.models.encoders.clip import (
        CLIPTextArchConfig,
        CLIPTextConfig,
    )

    return CLIPTextConfig(
        arch_config=CLIPTextArchConfig(
            architectures=["CLIPTextModel"],
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            projection_dim=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=8,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            text_len=8,
        ),
        prefix="clip",
    )


def _deterministic_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260816)
    state_dict = {}
    for name, value in model.state_dict().items():
        state_dict[name] = torch.randn(
            value.shape,
            dtype=value.dtype,
            generator=generator,
        ).mul_(0.02)
    return state_dict


def _clip_checkpoint_weights(
    state_dict: dict[str, torch.Tensor],
) -> list[tuple[str, torch.Tensor]]:
    weights = []
    for name, value in state_dict.items():
        if ".qkv_proj." not in name:
            weights.append((name, value))
            continue
        for projection, shard in zip(("q", "k", "v"), value.chunk(3, dim=0)):
            weights.append((name.replace("qkv_proj", f"{projection}_proj"), shard))
    return weights


def _worker() -> int:
    from sglang.multimodal_gen.runtime.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        get_world_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
        TextEncoderLoader,
    )
    from sglang.multimodal_gen.runtime.models.encoders.clip import CLIPTextModel
    from sglang.srt.distributed import parallel_state as srt_parallel_state

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
    )
    initialize_model_parallel(
        tensor_parallel_degree=1,
        sequence_parallel_degree=world_size,
        ulysses_degree=world_size,
        ring_degree=1,
    )

    config = _tiny_clip_config()
    reference = CLIPTextModel(config).to(device).eval()
    state_dict = _deterministic_state_dict(reference)
    reference.load_state_dict(
        {name: value.to(device) for name, value in state_dict.items()}
    )

    class InMemoryTextEncoderLoader(TextEncoderLoader):
        def _get_all_weights(self, model, model_path, to_cpu):
            del model, model_path, to_cpu
            yield from _clip_checkpoint_weights(state_dict)

    config.parallel_folding_mode = "world"
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(),
        should_start_component_on_cpu=lambda component_name: False,
    )
    folded = InMemoryTextEncoderLoader().load_model(
        "unused",
        config,
        server_args,
        dtype="fp32",
        component_starts_on_cpu=False,
    )
    folded.eval()

    fold_group = get_world_group()
    assert folded._encoder_tp_group is fold_group
    assert folded.text_model.encoder.layers[0].mlp.fc2.tp_size == world_size
    assert get_tp_group().world_size == 1

    input_ids = torch.tensor([[1, 7, 11, 2]], device=device)
    with torch.no_grad():
        expected = reference(input_ids=input_ids).last_hidden_state
        actual = folded(input_ids=input_ids).last_hidden_state

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert get_tp_group().world_size == 1
    assert srt_parallel_state.get_tp_group().world_size == 1
    assert srt_parallel_state.get_attn_tp_group().world_size == 1

    if rank == 0:
        print("ENCODER_FOLD_SRT_PARITY PASS", flush=True)
    torch.distributed.barrier()
    cleanup_dist_env_and_memory()
    return 0


def _generate_tiny_sd3(*, fold: bool):
    from openai import OpenAI

    from sglang.multimodal_gen.test.server.test_server_utils import (
        ServerManager,
        get_generate_fn,
    )
    from sglang.multimodal_gen.test.server.testcase_configs import (
        DiffusionSamplingParams,
    )
    from sglang.multimodal_gen.test.test_utils import (
        find_free_port,
        image_bytes_to_numpy,
    )

    sampling_params = DiffusionSamplingParams(
        prompt="a red cube",
        output_size="64x64",
        extras={"num_inference_steps": 2, "seed": 0, "guidance_scale": 1.0},
    )
    encoder_mode = "fold" if fold else "replicate"
    parallel_args = f"--num-gpus 2 --ulysses-degree 2 --encoder-parallel {encoder_mode}"
    extra_args = " ".join(
        [
            "--model-type diffusion",
            "--backend sglang",
            "--model-id stable-diffusion-3-medium",
            f"--served-model-name {_TINY_SD3_MODEL}",
            f"--revision {_TINY_SD3_REVISION}",
            "--strict-ports",
            parallel_args,
        ]
    )
    manager = ServerManager(
        model=_TINY_SD3_MODEL,
        port=find_free_port(),
        wait_deadline=600,
        extra_args=extra_args,
    )
    ctx = manager.start()
    try:
        client = OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{ctx.port}/v1",
            timeout=600,
            max_retries=0,
        )
        model_ids = [model.id for model in client.models.list().data]
        assert _TINY_SD3_MODEL in model_ids

        generate = get_generate_fn(
            model_path=_TINY_SD3_MODEL,
            modality="image",
            sampling_params=sampling_params,
        )
        _, content = generate("tiny_sd3_encoder_fold_e2e", client)
        log = ctx.log_tail(lines=500)
        assert "Using native sglang backend" in log
        assert "[TextEncodingStage]" in log
        return image_bytes_to_numpy(content)
    finally:
        ctx.cleanup()


class TestEncoderFoldSrtTwoGpu(CustomTestCase):
    def test_folded_pipeline_matches_replicated_encoder(self):
        if not current_platform.is_cuda():
            self.skipTest("CUDA-only test")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")

        from sglang.multimodal_gen.test.test_utils import (
            compute_mean_abs_diff,
            compute_psnr,
            compute_ssim,
        )

        reference = _generate_tiny_sd3(fold=False)
        folded = _generate_tiny_sd3(fold=True)
        ssim = compute_ssim(folded, reference)
        psnr = compute_psnr(folded, reference)
        mean_abs_diff = compute_mean_abs_diff(folded, reference)
        print(
            "ENCODER_FOLD_E2E_PARITY "
            f"ssim={ssim:.6f} psnr={psnr:.6f} mad={mean_abs_diff:.6f}",
            flush=True,
        )
        # BF16 TP reductions may move the final uint8 output slightly. A wrong
        # runtime group produces multi-level pixel drift, not this rounding noise.
        self.assertGreaterEqual(ssim, 0.98)
        self.assertLessEqual(mean_abs_diff, 2.0)

    def test_folded_srt_clip_matches_single_rank(self):
        if not current_platform.is_cuda():
            self.skipTest("CUDA-only test")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={_WORLD}",
                "--master-port=29617",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
        self.assertEqual(proc.returncode, 0, "folded SRT CLIP diverged")
        self.assertIn("ENCODER_FOLD_SRT_PARITY PASS", proc.stdout)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()
