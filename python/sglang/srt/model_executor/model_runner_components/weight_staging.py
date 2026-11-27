from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch

# slot alignment: a slot is viewed in its parameter's dtype, which needs an offset its element size divides
_SLOT_ALIGNMENT = 64


@dataclass(slots=True)
class WeightStaging:
    """A second copy of a runner's parameters (``--weight-update-staging``).

    Peers write a weight update into `buffer` while the runner keeps computing with the live parameters; `commit`
    then copies it over them on the device. The live parameters stay where they are, so captured CUDA graphs and the
    addresses kernels were launched with remain valid, which swapping the two copies would break.
    """

    # one allocation, so a transport registers one block
    buffer: torch.Tensor
    # (address, numel, element_size) of each parameter's slot in `buffer`, the layout runners publish to peers
    slot_info: dict[str, tuple[int, int, int]]
    # the same for the live parameters when the region was allocated; they must not move afterwards
    live_info: dict[str, tuple[int, int, int]]
    # (live parameter, its slot viewed as that parameter), for every named parameter: two that share storage share
    # the slot, and each is listed, so a loader is redirected whichever of the two names it writes
    _slots: list[tuple[torch.nn.Parameter, torch.Tensor]] = field(repr=False)
    # False until the slots hold the live bytes: peers write only the parameters they train, the rest must not
    # reach the live weights as uninitialized memory
    _mirrors_live: bool = False

    @classmethod
    def allocate(cls, model: torch.nn.Module) -> WeightStaging:
        # shortcut: refuses the one layout known to change a parameter's shape for a load; upgrade to comparing
        # shapes around restore_weights_before_loading if another quant method starts to do that.
        assert not any(
            getattr(
                getattr(module, "quant_method", None),
                "use_flashinfer_trtllm_moe",
                False,
            )
            for module in model.modules()
        ), (
            "--weight-update-staging is not supported with the flashinfer TRT-LLM MoE backend: it repacks expert "
            "weights into another shape for a load, and a staging slot keeps the shape its parameter had at startup"
        )
        offsets_by_address: dict[int, tuple[int, int]] = {}
        named_offsets: list[tuple[str, torch.nn.Parameter, int]] = []
        total_nbytes = 0
        for name, param in model.named_parameters():
            nbytes = param.numel() * param.element_size()
            assert param.is_contiguous(), f"{name} is not contiguous"
            if nbytes and param.data_ptr() in offsets_by_address:
                offset, shared_nbytes = offsets_by_address[param.data_ptr()]
                assert shared_nbytes == nbytes, (
                    f"{name} shares storage with a parameter of {shared_nbytes} bytes but has {nbytes}"
                )
            else:
                offset = total_nbytes
                offsets_by_address[param.data_ptr()] = (offset, nbytes)
                total_nbytes += -(-nbytes // _SLOT_ALIGNMENT) * _SLOT_ALIGNMENT
            named_offsets.append((name, param, offset))
        # two parameters over the same bytes share a slot (above); a partial overlap would get two slots, and the
        # install of one would undo the other
        ranges = sorted(
            (address, address + nbytes)
            for address, (_, nbytes) in offsets_by_address.items()
        )
        assert all(end <= start for (_, end), (start, _) in zip(ranges, ranges[1:])), (
            "two parameters overlap in memory without being the same storage"
        )

        device = next(model.parameters()).device
        buffer = torch.empty(total_nbytes, dtype=torch.uint8, device=device)
        slot_info, live_info, slots = {}, {}, []
        for name, param, offset in named_offsets:
            nbytes = param.numel() * param.element_size()
            slot_info[name] = (
                buffer.data_ptr() + offset,
                param.numel(),
                param.element_size(),
            )
            live_info[name] = (param.data_ptr(), param.numel(), param.element_size())
            if nbytes:
                slot = (
                    buffer[offset : offset + nbytes].view(param.dtype).view(param.shape)
                )
                slots.append((param, slot))
        return cls(
            buffer=buffer, slot_info=slot_info, live_info=live_info, _slots=slots
        )

    def mirror_live(self) -> None:
        """Before peers write: make every slot hold its live parameter's bytes, unless it already does."""
        if self._mirrors_live:
            return
        for param, slot in self._slots:
            slot.copy_(param.data)
        self._mirrors_live = True

    @contextmanager
    def receiving_loads(self) -> Iterator[None]:
        """While open, the model's parameters are their slots, so a weight loader (`model.load_weights`) that
        copies into them fills the staging region. Only for the scheduler thread, between two forward launches:
        kernels already launched and captured CUDA graphs hold the live addresses, and the live storage is back
        before the next forward reads a parameter."""
        live = [param.data for param, _ in self._slots]
        for param, slot in self._slots:
            param.data = slot
        try:
            yield
            rebound = sum(
                param.data_ptr() != slot.data_ptr() for param, slot in self._slots
            )
            # a loader that replaces a parameter's tensor wrote somewhere that is neither live nor staged
            assert not rebound, (
                f"the weight loader rebound {rebound} parameters instead of copying into them; "
                "it cannot fill the staging region"
            )
        finally:
            for (param, _), data in zip(self._slots, live):
                param.data = data

    def commit(self) -> None:
        """Install what peers wrote. The caller must have let every forward launched earlier finish."""
        for param, slot in self._slots:
            param.data.copy_(slot)

    def discard(self) -> None:
        """The slots no longer hold the live bytes, after an aborted update left part of it in them or after an
        in-place load changed the live weights: mirror them again before the next update."""
        self._mirrors_live = False
