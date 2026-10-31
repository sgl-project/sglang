"""Device<->host KV transfer on Intel XPU is an identity, per kernel pair.

backup_from_device_all_layer then load_to_device_per_layer must return the exact
bytes, which the e2e tests cannot check since they compare greedy continuations.

Kernel pairs covered (see pool_host/mha.py and pool_host/mla.py,
load_to_device_per_layer / backup_from_device_all_layer):

    io_backend  mem_layout          load kernel                         backup kernel
    ----------  ------------------  ----------------------------------  ----------------------------------
    kernel      layer_first         transfer_kv_per_layer               transfer_kv_all_layer
    kernel      page_first          transfer_kv_per_layer_pf_lf         transfer_kv_all_layer_lf_pf
    kernel      page_head           transfer_kv_per_layer_ph_lf         transfer_kv_all_layer_lf_ph
    direct      layer_first         transfer_kv_direct                  transfer_kv_direct
    direct      page_first_direct   transfer_kv_per_layer_direct_pf_lf  transfer_kv_all_layer_direct_lf_pf
    kernel      layer_first (MLA)   transfer_kv_per_layer_mla           transfer_kv_all_layer_mla
    kernel      page_first (MLA)    transfer_kv_per_layer_mla_pf_lf     transfer_kv_all_layer_mla_lf_pf
    kernel      page_first (Mamba)  transfer_kv_mamba_pf_lf             transfer_kv_mamba_lf_pf
    direct      page_first_direct   transfer_kv_per_layer_direct_pf_lf  transfer_kv_all_layer_direct_lf_pf
                (Mamba)
"""

import unittest

import torch

from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-a-test-1-gpu-xpu")

XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, "xpu") else False

DEVICE = "xpu"
PAGE_SIZE = 16
NUM_PAGES = 8
POOL_TOKENS = PAGE_SIZE * NUM_PAGES
LAYER_NUM = 4
DTYPE = torch.bfloat16


def _fill_random(buffers) -> list[torch.Tensor]:
    """Randomize each buffer and return clones to compare against."""
    for buf in buffers:
        buf.copy_(torch.randn(buf.shape, dtype=torch.float32, device=buf.device))
    return [buf.clone() for buf in buffers]


def _transfer_indices(io_backend: str):
    """One page-aligned run of tokens, host slots shifted by a page.

    The shift keeps host and device indices from coinciding, so a kernel that
    ignores one of the two index tensors still round-trips wrong. Placement
    mirrors CacheController.move_indices: the kernel backend wants both tensors
    on the device, the direct backend wants device_indices on the host.
    """
    n = PAGE_SIZE * (NUM_PAGES - 1)
    device_indices = torch.arange(n, dtype=torch.int64, device=DEVICE)
    host_indices = torch.arange(PAGE_SIZE, PAGE_SIZE + n, dtype=torch.int64)
    if io_backend == "kernel":
        return host_indices.to(DEVICE), device_indices, n
    return host_indices, device_indices.cpu(), n


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
class TestHiCacheTransferRoundTripXpu(CustomTestCase):
    def _assert_round_trip(
        self, host, device_pool, device_buffers, io_backend, num_layers=LAYER_NUM
    ):
        reference = _fill_random(device_buffers)
        host_indices, device_indices, n = _transfer_indices(io_backend)

        host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        torch.xpu.synchronize()
        for buf in device_buffers:
            buf.zero_()
        torch.xpu.synchronize()
        for layer_id in range(num_layers):
            host.load_to_device_per_layer(
                device_pool, host_indices, device_indices, layer_id, io_backend
            )
        torch.xpu.synchronize()

        for i, (got, want) in enumerate(zip(device_buffers, reference)):
            differing = int((got[:n] != want[:n]).sum())
            self.assertEqual(
                differing,
                0,
                f"buffer {i}: {differing}/{got[:n].numel()} elements changed across "
                f"the {io_backend} round trip -> the transfer is not an identity",
            )

    def _mha_round_trip(self, io_backend: str, layout: str):
        device_pool = MHATokenToKVPool(
            size=POOL_TOKENS,
            page_size=PAGE_SIZE,
            dtype=DTYPE,
            head_num=2,
            head_dim=128,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        host = MHATokenToKVPoolHost(
            buffer_info=device_pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout=layout,
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            write_back_staging=device_pool.hicache_write_back_staging,
        )
        buffers = [device_pool.k_buffer[i] for i in range(LAYER_NUM)]
        buffers += [device_pool.v_buffer[i] for i in range(LAYER_NUM)]
        self._assert_round_trip(host, device_pool, buffers, io_backend)

    def _mla_round_trip(self, io_backend: str, layout: str):
        device_pool = MLATokenToKVPool(
            size=POOL_TOKENS,
            page_size=PAGE_SIZE,
            dtype=DTYPE,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
        )
        host = MLATokenToKVPoolHost(
            buffer_info=device_pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout=layout,
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            target_layer_num=device_pool.layer_num,
        )
        # MLA keeps the latent KV in one buffer per layer, not a K/V pair.
        buffers = [device_pool.kv_buffer[i] for i in range(LAYER_NUM)]
        self._assert_round_trip(host, device_pool, buffers, io_backend)

    def _mamba_round_trip(self, io_backend: str, layout: str, num_drafts: int = 0):
        def state(layers):
            return MambaStateBufferInfo(
                temporal=torch.empty(
                    (layers, POOL_TOKENS, 4, 32), dtype=torch.float32, device=DEVICE
                ),
                conv=(
                    torch.empty(
                        (layers, POOL_TOKENS, 3, 64), dtype=DTYPE, device=DEVICE
                    ),
                ),
            )

        infos = (state(LAYER_NUM), *(state(1) for _ in range(num_drafts)))
        host = MambaPoolHost(
            infos[0],
            host_to_device_ratio=2.0,
            host_size=0,
            device_capacity=POOL_TOKENS,
            layout=layout,
            mtp_draft_buffer_infos=infos[1:],
        )
        # Per-layer views, so the comparison covers each target and draft layer.
        buffers = [
            buf[layer]
            for info in infos
            for buf in (info.temporal, *info.conv)
            for layer in range(buf.shape[0])
        ]
        self._assert_round_trip(
            host, None, buffers, io_backend, num_layers=host.num_mamba_layers
        )

    def test_mha_kernel_layer_first(self):
        self._mha_round_trip("kernel", "layer_first")

    def test_mha_kernel_page_first(self):
        self._mha_round_trip("kernel", "page_first")

    def test_mha_kernel_page_head(self):
        self._mha_round_trip("kernel", "page_head")

    def test_mha_direct_layer_first(self):
        self._mha_round_trip("direct", "layer_first")

    def test_mha_direct_page_first_direct(self):
        self._mha_round_trip("direct", "page_first_direct")

    def test_mla_kernel_layer_first(self):
        self._mla_round_trip("kernel", "layer_first")

    def test_mla_kernel_page_first(self):
        self._mla_round_trip("kernel", "page_first")

    def test_mamba_kernel_page_first(self):
        self._mamba_round_trip("kernel", "page_first")

    def test_mamba_direct_page_first_direct(self):
        self._mamba_round_trip("direct", "page_first_direct")

    def test_mamba_kernel_page_first_mtp_draft(self):
        # Draft layers are separate allocations; backup stacks them for the XPU op.
        self._mamba_round_trip("kernel", "page_first", num_drafts=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
