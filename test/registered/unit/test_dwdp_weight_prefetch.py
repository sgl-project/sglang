"""DWDP peer expert-weight prefetch, end to end without a model.

Runs on whichever VMM backend the host has, and the expected bytes are the same
either way: this is the one gate that holds CUDA and XPU to an identical result,
so a backend whose granularity or handle semantics differ fails here rather than
as drifted logits in a model test.

Drives the real transport / weight buffer / weight manager over synthetic expert
weights, so the pieces a model test cannot isolate are pinned here:

  - the local shard must land at the layout's offset inside the composite VA,
    which depends on the allocation granularity the backend reports;
  - every peer expert must arrive byte-exact, including the two experts whose
    bytes straddle the local handle's edge pages and are seeded once instead of
    prefetched into pool pages;
  - reading a layer must be safe while the layer two ahead is already being
    prefetched into the same double-buffer slot.

Expert sizes are deliberately not a page multiple; a page-multiple shape hides
every edge case above.

Run with ``python test_dwdp_weight_prefetch.py`` (relaunches under torchrun).
"""

from __future__ import annotations

import atexit
import os
from types import SimpleNamespace
from typing import Dict, Tuple

import pytest
import torch
import torch.distributed as dist

from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    build_layer_weight_specs,
)
from sglang.srt.layers.moe.dwdp.transport import DWDPTransport
from sglang.srt.layers.moe.dwdp.weight_buffer import WeightBuffer, fill_edge_experts
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager
from sglang.srt.utils import get_device, get_device_module, is_cuda, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_xpu_ci(est_time=60, suite="nightly-xpu-4-gpu", nightly=True)
register_cuda_ci(est_time=90, stage="extra-b", runner_config="4-gpu-h100")

# the two backends get_vmm_backend() knows; DWDP has no CPU path to fall back to
pytestmark = pytest.mark.skipif(
    not (is_cuda() or is_xpu()), reason="requires a CUDA or Intel XPU device"
)

# 3 per rank: DWDP needs the count divisible by the world size, and a rank with
# 3 experts keeps an interior one while both of its edges belong to a peer
_NUM_EXPERTS = 3 * int(os.environ.get("WORLD_SIZE", 2))
# more than two layers so a third layer reuses the first layer's buffer slot
_NUM_LAYERS = 3
_HIDDEN = 128
# expert_bytes is 512 * _INTERMEDIATE for w13 and 256 * _INTERMEDIATE for w2, so
# 6145 puts both a few pages apart and off any power-of-two page boundary; the
# local handle's first and last page then hold bytes of experts this rank does
# not own, which a page-multiple expert size would hide
_INTERMEDIATE = 6145
_DTYPE = torch.bfloat16
_WEIGHT_NAMES = ("w13_weight", "w2_weight")

WeightKey = Tuple[int, str]


def _local_device_id() -> int:
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    get_device_module().set_device(device_id)
    return device_id


def _gloo_group() -> dist.ProcessGroup:
    if not dist.is_initialized():
        _local_device_id()
        dist.init_process_group(backend="gloo")
        atexit.register(dist.destroy_process_group)
    return dist.group.WORLD


def _expert_shape(name: str) -> Tuple[int, ...]:
    if name == "w13_weight":
        return (2 * _INTERMEDIATE, _HIDDEN)
    return (_HIDDEN, _INTERMEDIATE)


def _reference_weight(layer_idx: int, name: str, device: torch.device) -> torch.Tensor:
    # seeded from the key, not hash(), so every rank builds the same weights
    seed = 1000 * layer_idx + _WEIGHT_NAMES.index(name)
    generator = torch.Generator().manual_seed(seed)
    full = torch.randn(
        (_NUM_EXPERTS,) + _expert_shape(name), generator=generator, dtype=torch.float32
    )
    return full.to(dtype=_DTYPE).to(device)


class _DwdpFixture:
    """The transport/buffer/manager stack one rank would build during setup()."""

    def __init__(self) -> None:
        group = _gloo_group()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device_id = _local_device_id()
        device = torch.device(get_device(), self.device_id)

        self.layout = DwdpExpertLayout(
            num_routed_experts=_NUM_EXPERTS,
            dwdp_size=self.world_size,
            dwdp_rank=self.rank,
        )
        local_start = self.layout.local_expert_start
        local_end = self.layout.local_expert_end

        self.references: Dict[WeightKey, torch.Tensor] = {
            (layer_idx, name): _reference_weight(layer_idx, name, device)
            for layer_idx in range(_NUM_LAYERS)
            for name in _WEIGHT_NAMES
        }
        # the transport frees these, so hand it copies of the reference rows
        local_params = {
            key: reference[local_start:local_end].clone()
            for key, reference in self.references.items()
        }
        specs = build_layer_weight_specs(local_params, _NUM_EXPERTS)

        transport = DWDPTransport.create(
            layer_weight_specs=specs,
            local_params=local_params,
            group=SimpleNamespace(cpu_group=group, device_group=group),
            layout=self.layout,
            device_id=self.device_id,
        )
        self.buffer = WeightBuffer.create(
            layer_weight_specs=specs,
            handles=transport.handle_set,
            local_start=local_start,
            local_end=local_end,
            dwdp_size=self.world_size,
            device_id=self.device_id,
        )
        fill_edge_experts(
            self.buffer,
            transport.peer_views,
            local_start=local_start,
            local_end=local_end,
            peer_ranges=self.layout.peer_ranges,
        )
        self.manager = DWDPWeightManager(
            weight_buffer=self.buffer,
            peer_views=transport.peer_views,
            peer_ranges=self.layout.peer_ranges,
            moe_layer_indices=list(range(_NUM_LAYERS)),
            weight_names=list(_WEIGHT_NAMES),
            dwdp_rank=self.rank,
            dwdp_size=self.world_size,
            transport=transport,
        )

    def release(self) -> None:
        self.manager.release()
        get_device_module().synchronize(self.device_id)


@pytest.fixture
def dwdp():
    if dist.is_initialized() and dist.get_world_size() < 2:
        pytest.skip("needs at least 2 ranks")
    fixture = _DwdpFixture()
    try:
        yield fixture
    finally:
        fixture.release()
        dist.barrier(group=_gloo_group())


def test_local_shard_lands_at_the_layout_offset(dwdp: _DwdpFixture) -> None:
    """The local handle is mapped into the middle of the composite VA and the full
    tensor starts pre_padding bytes into the reservation. Get either offset wrong
    and the rank's own experts read as garbage, before any peer is involved."""
    start, end = dwdp.layout.local_expert_start, dwdp.layout.local_expert_end
    for (layer_idx, name), reference in dwdp.references.items():
        full = dwdp.buffer.get_full_tensor(layer_idx, name)
        assert full.shape == reference.shape
        assert torch.equal(full[start:end], reference[start:end]), (
            f"rank {dwdp.rank} lost its own experts for layer {layer_idx} {name}"
        )


def test_prefetch_reconstructs_every_expert(dwdp: _DwdpFixture) -> None:
    """The payoff property: after the prefetch for a layer completes, that layer's
    tensor holds all _NUM_EXPERTS experts even though the rank stores only its own
    shard. Snapshots are taken inside each layer's window because two layers share
    a buffer slot -- reading layer 0 after layer 2 prefetched would see layer 2."""
    manager = dwdp.manager
    manager.prefetch_first_layers()

    observed: Dict[WeightKey, torch.Tensor] = {}
    for layer_idx in range(_NUM_LAYERS):
        manager.wait_prefetch(layer_idx)
        for name in _WEIGHT_NAMES:
            observed[(layer_idx, name)] = dwdp.buffer.get_full_tensor(
                layer_idx, name
            ).clone()
        manager.record_compute_and_prefetch_next(layer_idx)
    get_device_module().synchronize(dwdp.device_id)

    for key, reference in dwdp.references.items():
        layer_idx, name = key
        mismatched = (observed[key] != reference).any(
            dim=tuple(range(1, reference.dim()))
        )
        assert not mismatched.any(), (
            f"rank {dwdp.rank} layer {layer_idx} {name}: experts "
            f"{mismatched.nonzero().flatten().tolist()} differ"
        )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
