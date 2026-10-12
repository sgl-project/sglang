"""DWDP peer expert-weight prefetch, end to end without a model.

Runs on whichever VMM backend the host has, and the expected bytes are the same
either way: this is the one gate that holds CUDA and XPU to an identical result,
so a backend whose granularity or handle semantics differ fails here rather than
as drifted logits in a model test.

Drives the real transport / weight buffer / weight manager over synthetic expert
weights, so the pieces a model test cannot isolate are pinned here; each case
says which. Expert sizes are deliberately not a page multiple, since a
page-multiple shape hides every edge case they cover.

Run with ``python test_dwdp_weight_prefetch.py`` (relaunches under torchrun).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

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
from sglang.srt.utils.vmm_backend import get_vmm_backend
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.kernels.utils import (
    gloo_group,
    local_device_id,
    multigpu_pytest_main,
)

register_xpu_ci(est_time=60, suite="nightly-xpu-4-gpu", nightly=True)
register_cuda_ci(est_time=90, stage="extra-b", runner_config="4-gpu-h100")

# the two backends get_vmm_backend() knows; DWDP has no CPU path to fall back to
pytestmark = pytest.mark.skipif(
    not (is_cuda() or is_xpu()), reason="requires a CUDA or Intel XPU device"
)

# 3 per rank: DWDP needs the count divisible by the world size, and a rank with
# 3 experts keeps an interior one while both of its edges belong to a peer
_NUM_EXPERTS = 3 * int(os.environ.get("WORLD_SIZE", 2))
# more layers than the 3 slots a remapping backend uses, so on every backend a
# later layer reuses an earlier layer's buffer slot
_NUM_LAYERS = 4
_HIDDEN = 128
# expert_bytes is 512 * _INTERMEDIATE for w13 and 256 * _INTERMEDIATE for w2, so
# 6145 puts both a few pages apart and off any power-of-two page boundary; the
# local handle's first and last page then hold bytes of experts this rank does
# not own, which a page-multiple expert size would hide
_INTERMEDIATE = 6145
_DTYPE = torch.bfloat16
_WEIGHT_NAMES = ("w13_weight", "w2_weight")

WeightKey = Tuple[int, str]


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


def _local_params(
    references: Dict[WeightKey, torch.Tensor], layout: DwdpExpertLayout
) -> Dict[WeightKey, torch.Tensor]:
    # the transport frees these, so hand it copies of the reference rows
    return {
        key: reference[layout.local_expert_start : layout.local_expert_end].clone()
        for key, reference in references.items()
    }


class _DwdpFixture:
    """The transport/buffer/manager stack one rank would build during setup()."""

    def __init__(self) -> None:
        group = gloo_group()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device_id = local_device_id()
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
        local_params = _local_params(self.references, self.layout)
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
        dist.barrier(group=gloo_group())


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
    shard. Snapshots are taken inside each layer's window because layers share
    buffer slots -- reading layer 0 after a later layer took its slot would see
    that layer."""
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


def _fd_is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    else:
        return True


def test_failed_peer_import_closes_every_fd(monkeypatch: pytest.MonkeyPatch) -> None:
    """A peer import that fails partway must not leak the fds the exchange handed
    over: every received fd, and the exported ones the backend leaves to the
    caller, are closed when DWDPTransport.create raises."""
    if dist.get_world_size() < 2:
        pytest.skip("needs at least 2 ranks")
    group = gloo_group()
    device_id = local_device_id()
    device = torch.device(get_device(), device_id)
    layout = DwdpExpertLayout(
        num_routed_experts=_NUM_EXPERTS,
        dwdp_size=dist.get_world_size(),
        dwdp_rank=dist.get_rank(),
    )
    references = {
        (layer_idx, name): _reference_weight(layer_idx, name, device)
        for layer_idx in range(_NUM_LAYERS)
        for name in _WEIGHT_NAMES
    }
    local_params = _local_params(references, layout)

    backend = get_vmm_backend(device_id)
    export_handles = backend.export_handles
    exported_fds: List[int] = []
    received_fds: List[Optional[int]] = []

    def recording_export(
        handles: List[int], cpu_group: dist.ProcessGroup, rank: int
    ) -> Tuple[List[bytes], List[int], bool]:
        fabric_handles, posix_fds, use_fabric = export_handles(handles, cpu_group, rank)
        exported_fds.extend(posix_fds)
        return fabric_handles, posix_fds, use_fabric

    def failing_import(
        fabric_handle: Optional[bytes],
        fd: Optional[int],
        *,
        use_fabric: bool,
        peer_rank: int,
        size: int,
    ) -> int:
        received_fds.append(fd)
        raise RuntimeError("injected peer import failure")

    monkeypatch.setattr(backend, "export_handles", recording_export)
    monkeypatch.setattr(backend, "import_handle", failing_import)
    with pytest.raises(RuntimeError, match="injected peer import failure"):
        DWDPTransport.create(
            layer_weight_specs=build_layer_weight_specs(local_params, _NUM_EXPERTS),
            local_params=local_params,
            group=SimpleNamespace(cpu_group=group, device_group=group),
            layout=layout,
            device_id=device_id,
        )
    if any(fd is None for fd in received_fds):
        pytest.skip("FABRIC handles carry no fds")
    else:
        pass
    caller_owned = exported_fds if backend.owns_exported_fds() else []
    still_open = [fd for fd in received_fds + caller_owned if _fd_is_open(fd)]
    dist.barrier(group=group)
    assert received_fds, "the injected failure never reached a peer import"
    assert not still_open, f"rank {dist.get_rank()} leaked fds {still_open}"


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
