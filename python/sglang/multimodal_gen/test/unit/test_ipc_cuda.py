# SPDX-License-Identifier: Apache-2.0

import pickle
from dataclasses import dataclass

import pytest
import torch

from sglang.multimodal_gen.runtime.ipc_cuda import (
    CudaIpcRef,
    materialize_cuda_refs,
    spill_cuda_tensors,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA IPC tests")


def test_pickle_copies_cuda_tensor_through_host() -> None:
    _require_cuda()
    tensor = torch.ones(16, 90, 160, device="cuda", dtype=torch.bfloat16)
    assert len(pickle.dumps(tensor)) > tensor.nbytes


def test_ipc_handle_pickle_is_much_smaller_than_tensor() -> None:
    _require_cuda()
    tensor = torch.ones(16, 90, 160, device="cuda", dtype=torch.bfloat16)
    ref = CudaIpcRef.from_tensor(tensor)
    assert len(pickle.dumps(ref)) < 2048
    rebuilt = ref.materialize()
    assert rebuilt.device.type == "cuda"
    assert rebuilt.shape == tensor.shape
    assert rebuilt.dtype == tensor.dtype
    assert torch.equal(rebuilt, tensor)


def test_spill_does_not_mutate_caller_req_tensors() -> None:
    _require_cuda()

    @dataclass
    class _Holder:
        latents: torch.Tensor
        prompt_embeds: list

    latents = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    embeds = [torch.ones(3, 5, device="cuda", dtype=torch.float32)]
    holder = _Holder(latents=latents, prompt_embeds=embeds)

    spilled = spill_cuda_tensors(holder)
    assert isinstance(spilled.latents, CudaIpcRef)
    assert isinstance(spilled.prompt_embeds[0], CudaIpcRef)
    assert holder.latents is latents
    assert holder.prompt_embeds[0] is embeds[0]

    restored = materialize_cuda_refs(spilled)
    assert torch.equal(restored.latents, latents)
    assert torch.equal(restored.prompt_embeds[0], embeds[0])


def test_cross_process_ipc_roundtrip() -> None:
    _require_cuda()
    ctx = torch.multiprocessing.get_context("spawn")
    tensor = torch.arange(64, device="cuda", dtype=torch.float32).reshape(4, 16)
    ref = CudaIpcRef.from_tensor(tensor)
    queue = ctx.Queue()
    proc = ctx.Process(target=_cross_process_child, args=(queue, pickle.dumps(ref)))
    proc.start()
    try:
        shape, dtype, first, last = queue.get(timeout=30)
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join()
    assert proc.exitcode == 0
    assert shape == (4, 16)
    assert dtype == "torch.float32"
    assert first == 0.0
    assert last == 63.0


def _cross_process_child(queue, payload: bytes) -> None:
    ref = pickle.loads(payload)
    rebuilt = ref.materialize()
    queue.put(
        (
            tuple(rebuilt.shape),
            str(rebuilt.dtype),
            float(rebuilt.reshape(-1)[0]),
            float(rebuilt.reshape(-1)[-1]),
        )
    )
