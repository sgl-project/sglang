"""Device-side tests for MHATokenToKVPool with asymmetric KV (head_dim != v_head_dim).

Covers the wiring the kernel-level tests cannot see: that the pool derives
``v_row_dim`` from ``v_head_dim`` and threads it into the fused store_cache kernel.
A mis-wired ``v_row_dim`` still writes the right bytes into the right K slots, so
the untouched-slot assertions are what pin the V width and stride down.

Skipped on CPU -- the fused path is CUDA/HIP only.

    python -m pytest test/registered/unit/mem_cache/test_asymmetric_mha_pool.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

_HAS_CUDA = torch.cuda.is_available()
_HAS_NATIVE_FP8_CUDA = (
    _HAS_CUDA
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (8, 9)
)

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

DTYPE = torch.bfloat16
HEAD_NUM = 2
POOL_SIZE = 63  # buffers get POOL_SIZE + page_size rows
NUM_WRITES = 16

# (head_dim, v_head_dim). Both orderings, since nothing may assume K is wider.
# The last pair is wide enough for the split heuristic to pick num_split=2.
ASYM_DIM_PAIRS = [(192, 128), (128, 192), (512, 256)]


def _build_pool(head_dim: int, v_head_dim: int, dtype=DTYPE) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=1,
        dtype=dtype,
        head_num=HEAD_NUM,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


@unittest.skipUnless(_HAS_CUDA, "fused store_cache path requires CUDA")
class TestAsymmetricMHAPoolRowDims(unittest.TestCase):
    def test_v_row_dim_tracks_v_head_dim(self):
        for head_dim, v_head_dim in ASYM_DIM_PAIRS:
            with self.subTest(head_dim=head_dim, v_head_dim=v_head_dim):
                pool = _build_pool(head_dim, v_head_dim)
                self.assertEqual(pool.row_dim, HEAD_NUM * head_dim)
                self.assertEqual(pool.v_row_dim, HEAD_NUM * v_head_dim)

    def test_v_row_dim_defaults_to_row_dim_when_symmetric(self):
        pool = _build_pool(128, 128)
        self.assertEqual(pool.v_row_dim, pool.row_dim)

    def test_swa_dims_override_row_dims(self):
        # A hybrid sliding-window model builds a second pool through the swa_*
        # parameters, which override head_num/head_dim/v_head_dim wholesale. Both
        # of MiMoV2's pools are asymmetric, so v_row_dim has to follow
        # swa_v_head_dim rather than the full pool's v_head_dim.
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=1,
            dtype=DTYPE,
            head_num=HEAD_NUM,
            head_dim=512,
            v_head_dim=256,
            swa_head_num=1,
            swa_head_dim=192,
            swa_v_head_dim=128,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        self.assertEqual(pool.row_dim, 1 * 192)
        self.assertEqual(pool.v_row_dim, 1 * 128)

    def test_swa_v_head_dim_falls_back_to_v_head_dim(self):
        # swa_v_head_dim omitted: head_dim comes from swa_head_dim but v_head_dim
        # does not, so the two are read from different sources. Pinned because a
        # pool built this way is asymmetric in a way neither config states.
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=1,
            dtype=DTYPE,
            head_num=HEAD_NUM,
            head_dim=512,
            v_head_dim=256,
            swa_head_num=1,
            swa_head_dim=192,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        self.assertEqual(pool.row_dim, 1 * 192)
        self.assertEqual(pool.v_row_dim, 1 * 256)


@unittest.skipUnless(_HAS_CUDA, "fused store_cache path requires CUDA")
class TestAsymmetricMHAPoolSetKVBuffer(unittest.TestCase):
    """set_kv_buffer round-trip through the fused kernel, per dim pair."""

    def _run_roundtrip(self, head_dim: int, v_head_dim: int):
        pool = _build_pool(head_dim, v_head_dim)
        k_buf, v_buf = pool.k_buffer[0], pool.v_buffer[0]
        self.assertEqual(tuple(k_buf.shape[1:]), (HEAD_NUM, head_dim))
        self.assertEqual(tuple(v_buf.shape[1:]), (HEAD_NUM, v_head_dim))

        itemsize = pool.store_dtype.itemsize
        self.assertTrue(
            can_use_store_cache(pool.row_dim * itemsize, pool.v_row_dim * itemsize),
            "fused store_cache unavailable; the naive fallback is also correct, so "
            "this test would pass without covering anything",
        )

        # Seed every slot so an over-wide V write shows up on a slot never targeted.
        k_buf.copy_(torch.randn_like(k_buf))
        v_buf.copy_(torch.randn_like(v_buf))
        k_before, v_before = k_buf.clone(), v_buf.clone()

        # Slot 0 is the reserved padding slot store_cache skips; target [1, num_slots).
        num_slots = k_buf.shape[0]
        loc = torch.randperm(num_slots - 1, device="cuda")[:NUM_WRITES] + 1
        cache_k = torch.randn(
            (NUM_WRITES, HEAD_NUM, head_dim), dtype=DTYPE, device="cuda"
        )
        cache_v = torch.randn(
            (NUM_WRITES, HEAD_NUM, v_head_dim), dtype=DTYPE, device="cuda"
        )

        pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, cache_k, cache_v)

        self.assertTrue(torch.equal(k_buf[loc], cache_k), "K target slots")
        self.assertTrue(torch.equal(v_buf[loc], cache_v), "V target slots")

        untouched = torch.ones(num_slots, dtype=torch.bool, device="cuda")
        untouched[loc] = False
        self.assertTrue(
            torch.equal(k_buf[untouched], k_before[untouched]),
            "K bled outside its target slots",
        )
        self.assertTrue(
            torch.equal(v_buf[untouched], v_before[untouched]),
            "V bled outside its target slots (wrong row width or stride)",
        )

    def test_asymmetric_roundtrip(self):
        for head_dim, v_head_dim in ASYM_DIM_PAIRS:
            with self.subTest(head_dim=head_dim, v_head_dim=v_head_dim):
                self._run_roundtrip(head_dim, v_head_dim)

    def test_symmetric_roundtrip_unchanged(self):
        self._run_roundtrip(128, 128)


@unittest.skipUnless(_HAS_CUDA, "prefix-valid tiled kernel requires CUDA")
class TestAsymmetricPrefixValidGuard(unittest.TestCase):
    """set_kv_buffer_prefix_valid's tiled kernel takes one row width for both
    tensors, so it must refuse asymmetric KV rather than truncate V."""

    def _call_prefix_valid(self, pool, head_dim, v_head_dim):
        rows = 2
        loc_2d = torch.tensor([[1, 2]], dtype=torch.int64, device="cuda")
        commit_lens = torch.tensor([rows], dtype=torch.int32, device="cuda")
        cache_k = torch.randn((rows, HEAD_NUM, head_dim), dtype=DTYPE, device="cuda")
        cache_v = torch.randn((rows, HEAD_NUM, v_head_dim), dtype=DTYPE, device="cuda")
        pool.set_kv_buffer_prefix_valid(
            SimpleNamespace(layer_id=0, k_scale=None, v_scale=None),
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
        )

    def test_rejects_asymmetric(self):
        for head_dim, v_head_dim in ASYM_DIM_PAIRS:
            with self.subTest(head_dim=head_dim, v_head_dim=v_head_dim):
                pool = _build_pool(head_dim, v_head_dim)
                with self.assertRaises(NotImplementedError):
                    self._call_prefix_valid(pool, head_dim, v_head_dim)

    def test_accepts_symmetric(self):
        # The guard must not tighten the equal-width path it already served.
        pool = _build_pool(128, 128)
        self._call_prefix_valid(pool, 128, 128)
        expected = torch.arange(1, 3, device="cuda")
        self.assertTrue(torch.any(pool.k_buffer[0][expected] != 0))
        self.assertTrue(torch.any(pool.v_buffer[0][expected] != 0))


@unittest.skipUnless(_HAS_CUDA and torch.version.hip is None, "CUDA FP8 fusion")
class TestFp8MHAPoolStore(CustomTestCase):
    @unittest.skipUnless(_HAS_NATIVE_FP8_CUDA, "native FP8 writer requires SM89+")
    def test_rounding_scales_and_asymmetric_rows(self):
        """Fusion must retain div_'s source rounding for each scale kind and width."""
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for k_width, v_width in ((64, 64), (65, 33), (33, 65)):
                for kind in ("tensor", "host", "mixed", "none"):
                    with self.subTest(
                        dtype=dtype, widths=(k_width, v_width), kind=kind
                    ):
                        pool = _build_pool(k_width, v_width, torch.float8_e4m3fn)
                        scale = torch.tensor(0.1, device="cuda")
                        k_scale = scale if kind in ("tensor", "mixed") else 0.1
                        v_scale = scale if kind == "tensor" else 0.3
                        if kind == "none":
                            k_scale = v_scale = None
                        loc = torch.arange(0, 12, device="cuda")[::2]
                        inputs = [
                            torch.linspace(-2, 2, 6 * HEAD_NUM * width, device="cuda")
                            .to(dtype)
                            .reshape(6, HEAD_NUM, width)
                            for width in (k_width, v_width)
                        ]
                        # Values near FP8 ties expose removing source-dtype rounding.
                        inputs[0][:, :, 0] = 0.0966796875
                        expected = []
                        for value, s in zip(inputs, (k_scale, v_scale)):
                            ref = value.clone()
                            if s is not None:
                                ref.div_(s)
                            expected.append(ref.clamp(-448, 448).to(pool.dtype))
                        originals = [x.clone() for x in inputs]
                        for buffer in (pool.k_buffer[0], pool.v_buffer[0]):
                            buffer.zero_()
                        pool.set_kv_buffer(
                            SimpleNamespace(layer_id=0), loc, *inputs, k_scale, v_scale
                        )
                        for buffer, ref, value, original in zip(
                            (pool.get_key_buffer(0), pool.get_value_buffer(0)),
                            expected,
                            inputs,
                            originals,
                        ):
                            self.assertTrue(
                                torch.equal(
                                    buffer[loc[1:]].view(torch.uint8),
                                    ref[1:].view(torch.uint8),
                                )
                            )
                            self.assertTrue(torch.equal(value, original))
                            untouched = torch.ones(
                                buffer.shape[0], device="cuda", dtype=torch.bool
                            )
                            untouched[loc[1:]] = False
                            self.assertEqual(
                                torch.count_nonzero(
                                    buffer[untouched].view(torch.uint8)
                                ),
                                0,
                            )

    def test_fallback_scale_forms_and_strided_sources(self):
        """Ineligible scalar forms and inner strides retain the eager value path."""
        cases = (
            (torch.tensor(0.1), False),
            (torch.tensor([0.1], device="cuda"), False),
            (0.1, True),
        )
        for scale, inner_strided in cases:
            pool = _build_pool(64, 64, torch.float8_e4m3fn)
            k = torch.linspace(-2, 2, 512, device="cuda").to(torch.bfloat16)
            k = k.reshape(2, 2, 128)[:, :, ::2]
            if not inner_strided:
                k = k.contiguous()
            v = -k.clone()
            refs, scaled = [], []
            for value in (k, v):
                ref = value.clone()
                ref.div_(scale)
                scaled.append(ref)
                refs.append(ref.clamp(-448, 448).to(pool.dtype))
            loc = torch.tensor([2, 4], device="cuda")
            pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, k, v, scale, scale)
            for buf, ref, value, divided in zip(
                (pool.get_key_buffer(0), pool.get_value_buffer(0)), refs, (k, v), scaled
            ):
                self.assertTrue(
                    torch.equal(buf[loc].view(torch.uint8), ref.view(torch.uint8))
                )
                self.assertTrue(torch.equal(value, divided))

    @unittest.skipUnless(_HAS_NATIVE_FP8_CUDA, "native FP8 writer requires SM89+")
    def test_independent_token_strides_and_storage_offsets(self):
        """QKV views can have different padded rows and unaligned source bases."""
        pool = _build_pool(65, 33, torch.float8_e4m3fn)
        inputs = []
        for width, padding, offset in ((65, 13, 1), (33, 17, 3)):
            storage = torch.linspace(-2, 2, 6 * (2 * width + padding), device="cuda")
            value = storage.to(torch.bfloat16).as_strided(
                (6, 2, width), (2 * width + padding, width, 1), offset
            )
            inputs.append(value)
        originals = [v.clone() for v in inputs]
        loc = torch.arange(1, 7, device="cuda", dtype=torch.int32)
        scale = torch.tensor(0.1, device="cuda")
        pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, *inputs, scale, scale)
        for buffer, value, original in zip(
            (pool.get_key_buffer(0), pool.get_value_buffer(0)), inputs, originals
        ):
            ref = original.clone().div_(scale).clamp(-448, 448).to(pool.dtype)
            self.assertTrue(
                torch.equal(buffer[loc].view(torch.uint8), ref.view(torch.uint8))
            )
            self.assertTrue(torch.equal(value, original))

    @unittest.skipUnless(_HAS_NATIVE_FP8_CUDA, "native FP8 writer requires SM89+")
    def test_graph_replay_reads_updated_device_scales(self):
        """Captured writes must read live tensor scales, not stale host shadows."""
        pool = _build_pool(64, 64, torch.float8_e4m3fn)
        k = torch.full((4, 2, 64), 0.0966796875, dtype=torch.bfloat16, device="cuda")
        v = -k.clone()
        scale = torch.tensor(0.1, device="cuda")
        loc = torch.tensor([0, 1, 0, 3], device="cuda")
        layer = SimpleNamespace(layer_id=0)
        for buffer in (pool.k_buffer[0], pool.v_buffer[0]):
            buffer.fill_(17)
        pool.set_kv_buffer(layer, loc, k, v, scale, scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.set_kv_buffer(layer, loc, k, v, scale, scale)
        scale.fill_(0.3)
        loc.copy_(torch.tensor([4, 0, 6, 0], device="cuda"))
        graph.replay()
        for buf, value in ((pool.get_key_buffer(0), k), (pool.get_value_buffer(0), v)):
            ref = value.clone()
            ref.div_(scale)
            ref = ref.clamp(-448, 448).to(pool.dtype)
            self.assertTrue(
                torch.equal(
                    buf[loc[[0, 2]]].view(torch.uint8), ref[[0, 2]].view(torch.uint8)
                )
            )
            self.assertTrue(torch.all(buf[0].view(torch.uint8) == 17))

    def test_pre_sm89_devices_keep_eager_quantization(self):
        """Triton rejects native E4M3 on SM80; capability must gate fusion."""
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            pool = _build_pool(64, 64, torch.float8_e4m3fn)
        k = torch.full((1, 2, 64), 0.0966796875, dtype=torch.bfloat16, device="cuda")
        v = -k.clone()
        divided = [value.clone().div_(0.1) for value in (k, v)]
        loc = torch.tensor([1], device="cuda")
        pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, k, v, 0.1, 0.1)
        for buffer, value, ref in zip(
            (pool.get_key_buffer(0), pool.get_value_buffer(0)), (k, v), divided
        ):
            self.assertTrue(torch.equal(value, ref))
            self.assertTrue(
                torch.equal(
                    buffer[loc].view(torch.uint8), ref.to(pool.dtype).view(torch.uint8)
                )
            )

    def test_dcp_mask_preserves_unowned_slots(self):
        """DCP-owned rows must use the masked writer, not the dense fast path."""
        pool = _build_pool(64, 64, torch.float8_e4m3fn)
        for buffer in (pool.k_buffer[0], pool.v_buffer[0]):
            buffer.fill_(17)
        k = torch.full((4, 2, 64), 0.0966796875, dtype=torch.bfloat16, device="cuda")
        v = -k.clone()
        loc = torch.tensor([1, 2, 3, 4], device="cuda")
        mask = torch.tensor([True, False, True, False], device="cuda")
        refs = [
            value.clone().div_(0.1).clamp(-448, 448).to(pool.dtype) for value in (k, v)
        ]
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=0), loc, k, v, 0.1, 0.1, dcp_kv_mask=mask
        )
        for buffer, ref in zip(
            (pool.get_key_buffer(0), pool.get_value_buffer(0)), refs
        ):
            self.assertTrue(
                torch.equal(
                    buffer[loc[mask]].view(torch.uint8), ref[mask].view(torch.uint8)
                )
            )
            untouched = torch.ones(buffer.shape[0], dtype=torch.bool, device="cuda")
            untouched[loc[mask]] = False
            self.assertTrue(torch.all(buffer[untouched].view(torch.uint8) == 17))

    def test_hnd_page_layout_keeps_its_physical_writer(self):
        """Page size greater than one makes HND writes noncontiguous slot rows."""
        with envs.SGLANG_USE_HND_KVCACHE.override(True):
            pool = MHATokenToKVPool(
                size=60,
                page_size=4,
                dtype=torch.float8_e4m3fn,
                head_num=2,
                head_dim=64,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
                enable_alt_stream=False,
            )
        k = torch.full((2, 2, 64), 0.0966796875, dtype=torch.bfloat16, device="cuda")
        v = -k.clone()
        loc = torch.tensor([1, 6], device="cuda")
        refs = [
            value.clone().div_(0.1).clamp(-448, 448).to(pool.dtype) for value in (k, v)
        ]
        for buffer in (pool.k_buffer[0], pool.v_buffer[0]):
            buffer.fill_(17)
        pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, k, v, 0.1, 0.1)
        for buffer, ref in zip(
            (pool.get_key_buffer(0), pool.get_value_buffer(0)), refs
        ):
            expected = torch.full_like(buffer.view(torch.uint8), 17)
            expected[loc // 4, :, loc % 4, :] = ref.view(torch.uint8)
            self.assertTrue(torch.equal(buffer.view(torch.uint8), expected))

    def test_overflow_and_nan_are_not_conflated(self):
        """Finite overflow saturates; NaNs must not turn into a finite bound."""
        pool = _build_pool(64, 64, torch.float8_e4m3fn)
        for scale in (0.5, torch.tensor(0.5, device="cuda")):
            values = torch.tensor([8192, -8192, float("nan"), 1], device="cuda")
            k = values.to(torch.bfloat16).repeat(32).reshape(1, 2, 64)
            v = k.clone()
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=0),
                torch.tensor([1], device="cuda"),
                k,
                v,
                scale,
                scale,
            )
            for buffer in (pool.get_key_buffer(0), pool.get_value_buffer(0)):
                actual = buffer[1, 0, :4].float()
                self.assertEqual(actual[0].item(), 448.0)
                self.assertEqual(actual[1].item(), -448.0)
                self.assertTrue(torch.isnan(actual[2]))
                self.assertEqual(actual[3].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
