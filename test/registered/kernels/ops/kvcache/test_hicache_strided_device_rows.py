"""HiCache host transfers address strided device KV rows by their stride.

A per-layer device view can step its slots by more than one row: the unified
pool's token-major entries hold every layer's K and V in one slot. Backing
such rows up by the row width copies neighbouring bytes instead, and the
reload later restores them as if they were this token's KV -- silently. This
round-trips strided device rows through the host pool and compares them bit
for bit.

    python -m pytest test/registered/kernels/ops/kvcache/test_hicache_strided_device_rows.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_xpu_ci(est_time=15, suite="stage-b-test-1-gpu-xpu")

_DEVICE = "xpu" if is_xpu() else "cuda"

_L, _H, _D, _ROWS = 2, 2, 64, 64
# One slot holds every layer's K and V plus a gap, as a unified-pool entry.
_E = 2 * _L * _H * _D + 128


def _strided_device_pool():
    pool = MHATokenToKVPool(
        size=_ROWS - 1,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=_H,
        head_dim=_D,
        layer_num=_L,
        device=_DEVICE,
        enable_memory_saver=False,
    )
    backing = torch.randn(_ROWS * _E, dtype=torch.bfloat16, device=_DEVICE)
    pool.k_buffer = [
        backing.as_strided((_ROWS, _H, _D), (_E, _D, 1), 2 * l * _H * _D)
        for l in range(_L)
    ]
    pool.v_buffer = [
        backing.as_strided((_ROWS, _H, _D), (_E, _D, 1), (2 * l + 1) * _H * _D)
        for l in range(_L)
    ]
    pool._init_data_ptrs_and_strides()
    return pool


def _xpu_kernels_take_row_stride() -> bool:
    import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel

    schema = str(torch.ops.sgl_kernel.transfer_kv_per_layer.default._schema)
    return "src_row_stride" in schema


@unittest.skipUnless(
    torch.cuda.is_available() or is_xpu(), "requires a CUDA or XPU device"
)
class TestHiCacheStridedDeviceRows(CustomTestCase):
    def test_backup_and_load_keep_every_slot(self):
        for layout in ("layer_first", "page_first"):
            with self.subTest(layout=layout):
                device = _strided_device_pool()
                host = MHATokenToKVPoolHost(
                    buffer_info=device.get_device_pool_infos()[0].buffer_info,
                    host_to_device_ratio=2.0,
                    host_size=0,
                    page_size=1,
                    layout=layout,
                    pin_memory=True,
                    device="cpu",
                    allocator_type="default",
                    device_capacity_tokens=device.size
                    if device.host_capacity_tokens is None
                    else device.host_capacity_tokens,
                    start_layer=device.start_layer,
                    end_layer=device.end_layer,
                    write_back_staging=device.hicache_write_back_staging,
                )
                self.addCleanup(host.destroy)
                # XPU has no JIT path; its AOT kernels step rows by src/dst_row_stride.
                if is_xpu():
                    if not _xpu_kernels_take_row_stride():
                        self.skipTest("sgl-kernel-xpu predates the row-stride kwargs")
                elif not host.can_use_jit:
                    self.skipTest("strided device rows need the JIT HiCache kernels")

                src = torch.tensor([3, 7, 20, 41], device=_DEVICE)
                dst = torch.tensor([5, 9, 30, 50], device=_DEVICE)
                want = [
                    (k[src].clone(), v[src].clone())
                    for k, v in zip(device.k_buffer, device.v_buffer)
                ]
                host_slots = host.alloc(len(src)).to(_DEVICE)

                host.backup_from_device_all_layer(device, host_slots, src, "kernel")
                for layer in range(_L):
                    host.load_to_device_per_layer(
                        device, host_slots, dst, layer, "kernel"
                    )
                torch.get_device_module(_DEVICE).synchronize()

                for layer, (k, v) in enumerate(want):
                    self.assertTrue(torch.equal(device.k_buffer[layer][dst], k))
                    self.assertTrue(torch.equal(device.v_buffer[layer][dst], v))
                # The staged write-back reads whole device pages: packed rows only.
                self.assertEqual(host.device_row_stride_bytes, _E * 2)
                self.assertFalse(host.can_use_write_back_jit)


if __name__ == "__main__":
    unittest.main()
