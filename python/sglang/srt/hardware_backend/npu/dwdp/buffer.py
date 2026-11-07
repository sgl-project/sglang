"""Composite ND views: resident local shards plus two remote-only page slots.

Remote bytes sharing a resident boundary page are copied into that page's
padding. Only the owning rank's logical expert slice is exposed to peers.
"""

import torch

from sglang.srt.layers.moe.dwdp.layout import PageAlignedLayout


def tensor_from_pointer(pointer, nbytes, *, shape, dtype, device):
    import torch_npu

    # Non-owning storage: the buffer/transport must outlive every tensor view.
    storage = torch_npu._C._construct_storage_from_data_pointer(pointer, device, nbytes)
    stride, steps = [], 1
    for dimension in reversed(shape):
        stride.append(steps)
        steps *= dimension
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        dict(
            size=shape,
            stride=tuple(reversed(stride)),
            nbytes=nbytes,
            dtype=dtype,
            data_ptr=pointer,
            storage_offset=0,
            device=device,
            npu_format=2,
            layout=torch.strided,
            requires_grad=False,
        ),
        storage,
    )


class NpuWeightBuffer:
    def __init__(self, specs, transport, world_size, rank):
        self.layer_indices = sorted(specs)
        self._slots = {li: i % 2 for i, li in enumerate(self.layer_indices)}
        self._transport = transport
        self._tensors = {}
        self._reservations, self._mappings = [], []
        self.layouts = {}
        page_size = 8 * transport.granularity
        slot_sizes = [0, 0]
        for li, weights in specs.items():
            total = 0
            for name, spec in weights.items():
                experts = spec.shape[0]
                local = experts // world_size
                layout = PageAlignedLayout.compute(
                    expert_bytes=spec.nbytes // experts,
                    num_experts=experts,
                    local_start=rank * local,
                    local_end=(rank + 1) * local,
                    granularity=transport.granularity,
                    pool_granularity=page_size,
                    handle_phys_size=transport.local_allocation((li, name)).size,
                )
                self.layouts[li, name] = layout
                total += layout.pre_size + layout.post_size
            slot = self._slots[li]
            slot_sizes[slot] = max(slot_sizes[slot], total)
        # Physical pages belong to the transport; all composite mappings are
        # removed before it frees either pool pages or resident source handles.
        self.slot_sizes = slot_sizes
        self._pool = [
            [transport.allocate(page_size) for _ in range(size // page_size)]
            for size in slot_sizes
        ]
        try:
            for li, weights in specs.items():
                page_offset = 0
                for name, spec in weights.items():
                    layout = self.layouts[li, name]
                    pointer = transport.reserve(layout.total_size)
                    self._reservations.append(pointer)
                    for offset, size in (
                        (0, layout.pre_size),
                        (layout.pre_size + layout.mnnvl_size, layout.post_size),
                    ):
                        for page in range(size // page_size):
                            address = pointer + offset + page * page_size
                            allocation = self._pool[self._slots[li]][page_offset]
                            transport.map(address, page_size, allocation.handle)
                            self._mappings.append(address)
                            page_offset += 1
                    address = pointer + layout.pre_size
                    allocation = transport.local_allocation((li, name))
                    transport.map(address, layout.mnnvl_size, allocation.handle)
                    self._mappings.append(address)
                    self._tensors[li, name] = tensor_from_pointer(
                        pointer + layout.pre_padding,
                        spec.nbytes,
                        shape=spec.shape,
                        dtype=spec.dtype,
                        device=transport.device,
                    )
        except Exception:
            self.release()
            raise

    def buffer_index_for_layer(self, layer):
        return self._slots[layer]

    def get_full_tensor(self, layer, name):
        return self._tensors[layer, name]

    def release(self):
        self._tensors.clear()
        while self._mappings:
            self._transport.unmap(self._mappings[-1])
            self._mappings.pop()
        while self._reservations:
            self._transport.release_address(self._reservations[-1])
            self._reservations.pop()
