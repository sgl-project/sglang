"""NPU peer copies and batch checks on top of the shared DWDP scheduler."""

import torch

from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager


class NpuDWDPWeightManager(DWDPWeightManager):
    def __init__(self, weight_buffer, transport, specs, layout, device):
        self._npu_transport = transport
        self._specs = specs
        self._epoch = 0
        self._pending = [None, None]
        # The top-level NPU manager owns transport teardown, including the
        # barrier between dropping peer mappings and freeing resident sources.
        super().__init__(
            weight_buffer=weight_buffer,
            peer_views={},
            peer_ranges=layout.peer_ranges,
            moe_layer_indices=sorted(specs),
            weight_names=list(next(iter(specs.values()))),
            dwdp_rank=layout.dwdp_rank,
            dwdp_size=layout.dwdp_size,
            device_module=torch.npu,
            device=device,
        )

    def _prefetch_layer_per_slice(self, layer_idx):
        for name, spec in self._specs[layer_idx].items():
            full = self._weight_buffer.get_full_tensor(layer_idx, name)
            shard_bytes = spec.nbytes // self._dwdp_size
            for peer in range(self._dwdp_size):
                if peer != self._dwdp_rank:
                    self._npu_transport.copy_shard(
                        self._dwdp_rank,
                        peer,
                        (layer_idx, name),
                        full.data_ptr() + peer * shard_bytes,
                        shard_bytes,
                        self._copy_stream,
                    )

    def prefetch_layer(self, layer_idx):
        super().prefetch_layer(layer_idx)
        slot = self._weight_buffer.buffer_index_for_layer(layer_idx)
        self._pending[slot] = (self._epoch, layer_idx)

    def prefetch_first_layers(self):
        self._epoch += 1
        super().prefetch_first_layers()

    def _check_prefetch(self, layer_idx):
        slot = self._weight_buffer.buffer_index_for_layer(layer_idx)
        if self._pending[slot] != (self._epoch, layer_idx):
            raise RuntimeError(
                f"NPU DWDP layer {layer_idx} has no current-batch prefetch"
            )

    def wait_prefetch(self, layer_idx):
        self._check_prefetch(layer_idx)
        super().wait_prefetch(layer_idx)

    def record_compute_and_prefetch_next(self, layer_idx):
        self._check_prefetch(layer_idx)
        super().record_compute_and_prefetch_next(layer_idx)
