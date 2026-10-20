# SPDX-License-Identifier: Apache-2.0
"""Real image/video pipelines with external contracts and index selectors.

Run on two idle NVIDIA GPUs with Mooncake installed. These placements keep
generators and the request scheduler within the compute role, rather than
claiming that arbitrary stateful boundaries are transferable.
"""

import asyncio

import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.deployment.client import (
    EncodingClient,
)
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
    "model,num_frames",
    [
        ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 5),
        ("Tongyi-MAI/Z-Image-Turbo", 1),
        ("Qwen/Qwen-Image", 1),
    ],
)
def test_external_contracts_match_monolithic(model, num_frames):
    _require_gpus(2)
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
            "outputs": ["latents"],
            "components": ["transformer"],
        },
        {
            "name": "decoding_stage",
            "inputs": ["latents"],
            "outputs": ["output"],
            "components": ["vae"],
        },
        {
            "name": "decoding_stage_postprocess",
            "inputs": ["output"],
            "outputs": ["output", "result.output"],
        },
    ]
    requests = [
        dict(
            prompt="A mountain landscape in daylight",
            negative_prompt="",
            width=width,
            height=256,
            num_frames=num_frames,
            num_inference_steps=2,
            seed=42,
            quality="exact",
            save_output=False,
            return_file_paths_only=False,
        )
        for width in (256, 320)
    ]
    prepared, expected = [], []
    with DiffGenerator.from_pretrained(
        model_path=model,
        num_gpus=1,
        base_gpu_id=0,
        scheduler_port=find_free_port(HOST),
        master_port=find_free_port(HOST),
        warmup_mode="off",
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
        "compute": [0, 1, 2, 3, "denoising_stage"],
        "decode": [5],
        "postprocess": ["decoding_stage_postprocess"],
    }
    with _StageCluster(
        model=model,
        name="external_" + str(num_frames) + "_" + model.split("/")[-1],
        gpu_layout={"compute": [0], "decode": [1], "postprocess": [1]},
        extra_role_args={
            role: ["--warmup-mode", "off"] for role in (*groups, "server")
        },
        stage_groups=groups,
        stage_contracts=contracts,
    ) as cluster:

        async def check():
            async with EncodingClient(
                f"tcp://{HOST}:{cluster.base_port}", device="cuda:1"
            ) as client:
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
