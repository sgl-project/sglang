# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
    extract_transfer_fields,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.ipc_cuda import (
    attach_cuda_tensors,
    detach_cuda_tensors,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _comfyui_req() -> Req:
    return Req(
        sampling_params=SamplingParams(prompt=" "),
        extra={"comfyui_session_id": "run-1"},
        vae_image_sizes=[(64, 48)],
        image_embeds=[],
        latents=torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4),
        prompt_embeds=[torch.ones(2, 8, device="cuda", dtype=torch.float32)],
    )


def test_detach_keeps_fields_disagg_extract_drops() -> None:
    _require_cuda()
    req = _comfyui_req()
    original_latents = req.latents

    skeleton, tensors = detach_cuda_tensors(req)

    assert req.latents is original_latents
    assert skeleton.extra["comfyui_session_id"] == "run-1"
    assert skeleton.vae_image_sizes == [(64, 48)]
    assert skeleton.image_embeds == []
    assert skeleton.latents is None
    assert skeleton.prompt_embeds == [None]
    assert set(tensors) == {
        "\x1f".join(("latents",)),
        "\x1f".join(("prompt_embeds", "0")),
    }
    assert torch.equal(tensors["\x1f".join(("latents",))], original_latents)

    _, scalar_fields = extract_transfer_fields(req)
    assert "vae_image_sizes" not in scalar_fields


def test_attach_restores_gpu_tensors_and_non_tensor_fields() -> None:
    _require_cuda()
    req = _comfyui_req()
    skeleton, tensors = detach_cuda_tensors(req)

    restored = attach_cuda_tensors(skeleton, tensors)

    assert restored.extra["comfyui_session_id"] == "run-1"
    assert restored.vae_image_sizes == [(64, 48)]
    assert restored.image_embeds == []
    assert restored.latents.device.type == "cuda"
    assert restored.prompt_embeds[0].device.type == "cuda"
    assert torch.equal(restored.latents, req.latents)
    assert torch.equal(restored.prompt_embeds[0], req.prompt_embeds[0])


class _FakeScheduler:
    def __init__(self, *, comfyui_mode: bool, gpu_id: int = 0):
        self.server_args = SimpleNamespace(
            comfyui_mode=comfyui_mode,
            sp_degree=2,
            tp_size=1,
            enable_cfg_parallel=False,
        )
        self.gpu_id = gpu_id
        self.worker = SimpleNamespace(local_rank=0)
        self.pyobj_payloads: list = []
        self.tensor_payloads: list = []
        self._skeleton = None
        self._tensors = None

    def _is_multi_rank(self) -> bool:
        return True

    def _broadcast_to_all_ranks(self, data):
        self.pyobj_payloads.append(data)
        if data is not None:
            self._skeleton = data
            return data
        return self._skeleton

    def _broadcast_tensor_dict_to_all_ranks(self, data):
        self.tensor_payloads.append(data)
        if data is not None:
            self._tensors = data
            return data
        return self._tensors


def test_comfyui_recv_rebuilds_non_rank0_without_disagg_extract() -> None:
    _require_cuda()
    req = _comfyui_req()
    recv_reqs = [(b"id", req)]

    rank0 = _FakeScheduler(comfyui_mode=True, gpu_id=0)
    assert SchedulerDisaggMixin._broadcast_recv_reqs(rank0, recv_reqs) is recv_reqs
    assert len(rank0.tensor_payloads) == 1
    assert rank0.tensor_payloads[0]

    rank1 = _FakeScheduler(comfyui_mode=True, gpu_id=1)
    rank1._skeleton = rank0._skeleton
    rank1._tensors = rank0._tensors
    rebuilt = SchedulerDisaggMixin._broadcast_recv_reqs(rank1, None)

    assert rebuilt[0][0] == b"id"
    other = rebuilt[0][1]
    assert other is not req
    assert other.extra["comfyui_session_id"] == "run-1"
    assert other.vae_image_sizes == [(64, 48)]
    assert other.image_embeds == []
    assert other.latents.device.type == "cuda"
    assert torch.equal(other.latents, req.latents)
    assert torch.equal(other.prompt_embeds[0], req.prompt_embeds[0])
