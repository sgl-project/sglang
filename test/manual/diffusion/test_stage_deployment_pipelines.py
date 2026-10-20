# SPDX-License-Identifier: Apache-2.0
"""Real encoder/denoiser/decoder deployments without per-field payload contracts."""

import asyncio
import base64
import sys
from io import BytesIO

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.deployment.client import (
    EncodingClient,
)
from sglang.multimodal_gen.runtime.disaggregation.deployment.payload import fork_request
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.test.single_test_file.test_disagg_server import (
    HOST,
    DisaggCluster,
    _require_gpus,
)
from sglang.multimodal_gen.test.test_utils import find_free_port


class _StageCluster(DisaggCluster):
    def _warmup(self):
        # each test sends its own representative image/video requests
        pass


@pytest.mark.parametrize(
    "model,num_frames,kind",
    [
        ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 5, "standard"),
        ("Tongyi-MAI/Z-Image-Turbo", 1, "standard"),
        ("Qwen/Qwen-Image", 1, "standard"),
        ("black-forest-labs/FLUX.2-klein-4B", 1, "image"),
        ("black-forest-labs/FLUX.1-schnell", 1, "flux"),
        ("Efficient-Large-Model/Sana_600M_512px_diffusers", 1, "sana"),
        ("circlestone-labs/Anima-Base-v1.0-Diffusers", 1, "anima"),
    ],
)
def test_external_contracts_match_monolithic(model, num_frames, kind):
    _check_pipeline(model, num_frames, kind)


@pytest.mark.parametrize("parallelism", ["tp2", "ulysses2"])
def test_multirank_workers_match_monolithic(parallelism):
    _check_pipeline("Tongyi-MAI/Z-Image-Turbo", 1, "standard", parallelism)


def _check_pipeline(model, num_frames, kind, parallelism="single"):
    _require_gpus(3)
    contracts = [
        {"name": "InputValidationStage"},
        {
            "name": "TextEncodingStage",
            "components": ["text_encoder", "tokenizer"],
        },
        {
            "name": "LatentPreparationStage",
            "components": ["scheduler", "transformer"],
        },
        {"name": "TimestepPreparationStage", "components": ["scheduler"]},
        {
            "name": "denoising_stage",
            "components": ["transformer"],
        },
        {
            "name": "decoding_stage",
            "components": ["vae"],
        },
        {"name": "decoding_stage_postprocess"},
    ]
    if kind in ("flux", "sana"):
        contracts[1]["name"] = "prompt_encoding_stage_primary"
    if kind == "flux":
        contracts[1]["components"] += ["text_encoder_2", "tokenizer_2"]
    if kind in ("sana", "anima"):
        contracts[2:4] = reversed(contracts[2:4])
    if kind == "image":
        contracts.insert(2, {"name": "ImageVAEEncodingStage", "components": ["vae"]})
    if kind == "anima":
        contracts.insert(
            2,
            {
                "name": "AnimaTextConditioningStage",
                "components": ["text_conditioner", "t5_tokenizer"],
            },
        )
    requests = [
        dict(
            prompt="A mountain landscape in daylight",
            negative_prompt="",
            width=width,
            height=256 if num_frames > 1 else 512,
            num_frames=num_frames,
            num_inference_steps=2 + 2 * index,
            seed=42 + index,
            quality="exact",
            save_output=False,
            return_file_paths_only=False,
        )
        for index, width in enumerate((256, 320) if num_frames > 1 else (512, 640))
    ]
    if kind in ("flux", "sana", "anima"):
        requests[1]["num_outputs_per_prompt"] = 2
    if kind in ("sana", "anima"):
        requests[1]["negative_prompt"] = "blurry, low quality"
    if kind == "image":
        image = BytesIO()
        Image.new("RGB", (247, 239), (40, 90, 160)).save(image, format="PNG")
        requests[0]["image_path"] = [
            "data:image/png;base64,"
            + base64.b64encode(image.getvalue()).decode("ascii")
        ]
    topology = dict(tp_size=1, sp_degree=1, ulysses_degree=1, ring_degree=1)
    if parallelism == "tp2":
        topology["tp_size"] = 2
    elif parallelism == "ulysses2":
        topology.update(sp_degree=2, ulysses_degree=2)
    ranks = topology["tp_size"] * topology["sp_degree"]
    common_args = {
        "encoder_parallel": "replicate",
        "vae_config.use_parallel_decode": False,
    }
    prepared, expected = [], []
    with DiffGenerator.from_pretrained(
        model_path=model,
        num_gpus=ranks,
        base_gpu_id=0,
        scheduler_port=find_free_port(HOST),
        master_port=find_free_port(HOST),
        warmup_mode="off",
        **topology,
        **common_args,
    ) as generator:
        for request in requests:
            req = prepare_request(
                server_args=generator.server_args,
                sampling_params=SamplingParams.from_user_sampling_params_args(
                    model, server_args=generator.server_args, **request
                ),
            )
            # prepare separate request contexts for the real local and remote paths
            prepared.append(
                prepare_request(
                    server_args=generator.server_args,
                    sampling_params=SamplingParams.from_user_sampling_params_args(
                        model, server_args=generator.server_args, **request
                    ),
                )
            )
            result = generator._send_to_scheduler_and_wait_for_response([req])
            assert not result.error, result.error
            assert isinstance(result.output, torch.Tensor)
            assert torch.isfinite(result.output).all()
            expected.append(result.output.cpu())

    groups = {
        "encoder": list(range(3 if kind in ("image", "anima") else 2)),
        "denoiser": [
            stage["name"]
            for stage in contracts
            if stage["name"]
            in ("LatentPreparationStage", "TimestepPreparationStage", "denoising_stage")
        ],
        "decoder": ["decoding_stage", "decoding_stage_postprocess"],
    }
    role_args = {
        role: [
            "--warmup-mode",
            "off",
            "--encoder-parallel",
            "replicate",
            "--vae-config.use-parallel-decode",
            "false",
        ]
        for role in (*groups, "server")
    }
    for key, value in topology.items():
        role_args["denoiser"] += ["--" + key.replace("_", "-"), str(value)]
    if parallelism == "tp2":
        role_args["encoder"] += ["--tp-size", "2", "--sp-degree", "1"]
    layout = {"encoder": [0], "denoiser": [1], "decoder": [2]}
    if ranks > 1:
        layout = {
            "encoder": [0, 1] if parallelism == "tp2" else [0],
            "denoiser": [1, 2],
            "decoder": [0],
        }
    with _StageCluster(
        model=model,
        name="external_" + parallelism + "_" + model.split("/")[-1],
        gpu_layout=layout,
        extra_role_args=role_args,
        stage_groups=groups,
        stage_contracts=contracts,
    ) as cluster:

        async def check():
            async with EncodingClient(
                f"tcp://{HOST}:{cluster.base_port}", device="cuda:1"
            ) as client:
                invalid = fork_request(prepared[0])
                invalid.seed = [1, 2]
                with pytest.raises(RuntimeError, match="seed"):
                    await client.encode(invalid, ["result.output"])
                first, resized = await asyncio.gather(
                    *(client.encode(req, ["result.output"]) for req in prepared)
                )
                repeated = await client.encode(prepared[0], ["result.output"])
                for result, reference in zip(
                    (first, resized, repeated), (*expected, expected[0])
                ):
                    actual = result["result.output"]
                    assert actual.device == torch.device("cuda:1")
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

        asyncio.run(check())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
