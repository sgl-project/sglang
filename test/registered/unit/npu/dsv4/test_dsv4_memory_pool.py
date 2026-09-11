"""Unit tests for dsv4_memory_pool classes and the npu_state_pool_size helper."""

import enum
import sys
import unittest
from collections import namedtuple
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.mem_cache",
    "sgl_kernel_npu.mem_cache.allocator",
    "sglang.srt.utils.hf_transformers_patches",
    "sglang.global_config",
    "sglang.lang.api",
    "sglang.lang.backend.runtime_endpoint",
    "sglang.lang.choices",
    "sglang.utils",
    "sglang.srt.configs",
    "sglang.srt.configs.model_config",
    "sglang.srt.dllm.config",
    "sglang.srt.layers.attention.base_attn_backend",
    "sglang.srt.layers.radix_attention",
    "sglang.srt.layers.utils.cp_utils",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.swa_memory_pool",
    "sglang.srt.mem_cache.allocation",
    "sglang.srt.mem_cache.allocator",
    "sglang.srt.model_executor",
    "sglang.srt.hardware_backend.npu.allocator_npu",
    # NOTE: dsv4_memory_pool is NOT mocked — it is the module under test.
    "sglang.srt.runtime_context",
    "sglang.srt.speculative.spec_info",
):
    sys.modules.setdefault(_, MagicMock())
# triton may be unavailable on CPU/Windows; use the real one if importable,
# otherwise mock it as a package so triton.backends / triton.language resolve.
try:
    import triton  # noqa: F401
except ImportError:
    import types as _types

    _triton = _types.ModuleType("triton")
    _triton.__path__ = []
    sys.modules["triton"] = _triton
    sys.modules.setdefault("triton.backends", MagicMock())
    sys.modules.setdefault("triton.language", MagicMock())


_TORCH_NPU = sys.modules["torch_npu"]

import torch  # noqa: E402

# --- Stub parent classes / helpers (real, injected into synthetic modules so
#     dsv4_memory_pool can subclass them without pulling the heavy base chain). ---


class _KVAndScore:
    """Minimal faithful stub of mem_cache.deepseek_v4_compress_state.KVAndScore.

    Holds a flat ``kv_score`` tensor plus ``kv`` / ``score`` views over its
    first / second half, matching the real dataclass's ``[kv | score]`` layout.
    """

    def __init__(self, size, last_dim, dtype, device):
        self.kv_score = torch.zeros(size, last_dim, dtype=dtype, device=device)
        half = last_dim // 2
        self.kv = self.kv_score[:, :half]
        self.score = self.kv_score[:, half:]

    def clear(self):
        """kv -> 0, score -> -inf (matches the real KVAndScore.clear)."""
        self.kv.zero_()
        self.score.fill_(float("-inf"))


def _lcm(a, b):
    import math

    return a // math.gcd(a, b) * b


class CompressStatePool:
    """Stub parent for NPUCompressStatePool — mirrors the shared base sizing /
    dummy-row-sentinel logic that the NPU subclass relies on."""

    def __init__(
        self,
        *,
        size,
        ring_size,
        overlap,
        head_dim,
        dtype,
        device,
        enable_memory_saver,
        ratio,
        online=False,
        swa_page_size=0,
        online_mtp_max_draft_tokens=0,
        state_cache_page_size=1,
    ):
        self.ratio = ratio
        self.ring_size = ring_size
        self.swa_page_size = swa_page_size
        self.page_size = state_cache_page_size
        self.enable_memory_saver = enable_memory_saver
        self.online = online
        self.online_mtp_state_slot_offset = 0
        self.online_mtp_max_draft_tokens = 0
        if online:
            assert ring_size == 1, "online compress requires ring_size=1"
            self._logical_size = size + self.ring_size + 1
            if online_mtp_max_draft_tokens > 0:
                self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
                self.online_mtp_state_slot_offset = self._logical_size
            self._size = self._logical_size * (1 + self.online_mtp_max_draft_tokens)
            last_dim = 3 * head_dim
        else:
            self._size = size + self.ring_size + 1
            pad_to = ratio
            if state_cache_page_size > 1:
                pad_to = _lcm(pad_to, state_cache_page_size)
            self._size = (self._size + pad_to - 1) // pad_to * pad_to
            self._logical_size = self._size
            last_dim = 2 * (1 + overlap) * head_dim
        self.last_dim = last_dim
        self._alloc_kv_score_buffer(
            dtype=dtype, device=device, enable_memory_saver=enable_memory_saver
        )
        if not online:
            # The shared pool clears the last row as its dummy / sentinel row.
            self.kv_score_buffer.score[-1].fill_(float("-inf"))

    def _alloc_kv_score_buffer(self, *, dtype, device, enable_memory_saver):
        self.kv_score_buffer = _KVAndScore(self._size, self.last_dim, dtype, device)

    @property
    def state_cache_3d(self):
        assert not self.online, (
            "state_cache_3d is for the fused compressor path; "
            "online (3*head_dim) buffer is indexer-only."
        )
        assert self.page_size > 1, (
            "state_cache_3d requires page_size>1; pool was constructed "
            "with the default page_size=1 (flat 2D layout)."
        )
        return self.kv_score_buffer.kv_score.view(-1, self.page_size, self.last_dim)


class DeepSeekV4SingleKVPool:
    """Stub parent for NPUDeepSeekV4SingleKVPool — sets the buffer-creation
    fields and runs the same ``_create_buffers`` loop the NPU override plugs
    into, without the CUDA fp8 packing path."""

    def __init__(
        self,
        size,
        page_size,
        dtype,
        qk_nope_head_dim,
        qk_rope_head_dim,
        layer_num,
        device,
        enable_memory_saver,
        start_layer=None,
        end_layer=None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = dtype
        self.device = device
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num
        self.kv_cache_total_dim = None
        self.bytes_per_page_padded = None
        self._create_buffers()

    def _create_buffers(self):
        self.kv_buffer = [
            self.create_buffer(
                num_pages=(self.size + self.page_size + 1) // self.page_size
            )
            for _ in range(self.layer_num)
        ]

    def create_buffer(self, *, num_pages):
        kv_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_cache_total_dim = kv_dim
        self.bytes_per_page_padded = kv_dim
        return torch.zeros(
            num_pages, kv_dim, dtype=self.store_dtype, device=self.device
        )


class _NullMemorySaver:
    """No-op memory-saver stand-in whose ``region`` is an empty context."""

    def region(self, tag):
        return nullcontext()


class DeepSeekV4IndexerPool:
    """Stub parent for NPUDeepSeekV4IndexerPool — keeps the packed CUDA buffer
    allocation so ``super()._create_buffer()`` works, plus the
    ``memory_saver_adapter`` the NPU override needs."""

    def __init__(
        self,
        size,
        page_size,
        dtype,
        index_head_dim,
        layer_num,
        device,
        enable_memory_saver,
        start_layer=None,
        end_layer=None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = dtype
        self.device = device
        self.index_head_dim = index_head_dim
        self.layer_num = layer_num
        self.memory_saver_adapter = _NullMemorySaver()
        self._create_buffer()

    def _create_buffer(self):
        self.index_k_with_scale_buffer = [
            torch.zeros(1, dtype=torch.uint8, device=self.device)
            for _ in range(self.layer_num)
        ]


class DeepSeekV4TokenToKVPool:
    """Stub parent for DSV4NPUTokenToKVPool — the NPU subclass overrides every
    method under test; the base just needs to be a real class to subclass."""

    pass


ONLINE_C128 = False

_cs_mod = type(sys)("sglang.srt.mem_cache.deepseek_v4_compress_state")
_cs_mod.CompressStatePool = CompressStatePool
sys.modules["sglang.srt.mem_cache.deepseek_v4_compress_state"] = _cs_mod

_mp_mod = type(sys)("sglang.srt.mem_cache.deepseek_v4_memory_pool")
_mp_mod.ONLINE_C128 = ONLINE_C128
_mp_mod.DeepSeekV4SingleKVPool = DeepSeekV4SingleKVPool
_mp_mod.DeepSeekV4IndexerPool = DeepSeekV4IndexerPool
_mp_mod.DeepSeekV4TokenToKVPool = DeepSeekV4TokenToKVPool
sys.modules["sglang.srt.mem_cache.deepseek_v4_memory_pool"] = _mp_mod

# sglang.version
_ver = type(sys)("sglang.version")
_ver.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _ver


# --- AscendStateType stub + synthetic disaggregation chain. get_pd_state_components
#     lazy-imports sglang.srt.disaggregation.ascend.conn.AscendStateType; we provide a
#     real (tiny) enum here so assertions are readable and the heavy real package
#     __init__ is never pulled. ---
class AscendStateType(str, enum.Enum):
    DSV4_SWA = "dsv4_swa"
    DSV4_C4 = "dsv4_c4"
    DSV4_C128 = "dsv4_c128"
    DSV4_INDEXER = "dsv4_indexer"
    DSV4_C4_STATE = "dsv4_c4_state"
    DSV4_C128_STATE = "dsv4_c128_state"


for _pkg in ("sglang.srt.disaggregation", "sglang.srt.disaggregation.ascend"):
    _pkg_mod = type(sys)(_pkg)
    _pkg_mod.__path__ = []
    sys.modules[_pkg] = _pkg_mod
_conn_mod = type(sys)("sglang.srt.disaggregation.ascend.conn")
_conn_mod.AscendStateType = AscendStateType
sys.modules["sglang.srt.disaggregation.ascend.conn"] = _conn_mod

from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (  # noqa: E402
    DSV4NPUTokenToKVPool,
    NPUCompressStatePool,
    NPUDeepSeekV4IndexerPool,
    NPUDeepSeekV4SingleKVPool,
)

_MOD = "sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool"

_LayerItem = namedtuple(
    "_LayerItem", ["compress_ratio", "compress_layer_id", "compress_kv_pool"]
)


from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3, suite="base-a-test-1-npu-a2")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_state_pool(
    *,
    size=8,
    ring_size=2,
    swa_page_size=16,
    overlap=True,
    head_dim=4,
    dtype=torch.float32,
    device="cpu",
    ratio=4,
):
    """Real NPUCompressStatePool with tiny dims for accessor tests.

    ``ring_size`` doubles as the 3-D view page size (the base stores
    ``page_size = state_cache_page_size`` and the NPU adapter passes
    ``state_cache_page_size=ring_size``).
    """
    return NPUCompressStatePool(
        size=size,
        overlap=overlap,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        enable_memory_saver=False,
        ratio=ratio,
        ring_size=ring_size,
        swa_page_size=swa_page_size,
    )


def _make_subpool(
    layer_num, num_pages, page_size, dim, dtype=torch.bfloat16, device="cpu"
):
    """Fake single-KV pool: kv_buffer is a list of real PA_ND tensors."""
    sp = SimpleNamespace()
    sp.kv_buffer = [
        torch.zeros(num_pages, page_size, 1, dim, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    sp.set_key_buffer_fused = MagicMock()
    return sp


def _make_kvpool(device="cpu", page_size=128, swa_page_size=128):
    """Create a DSV4NPUTokenToKVPool bypassing __init__.

    Sub-pools are real tensors (for PA_ND read/write tests); the indexer pool
    carries real int8 K + fp16 scale buffers plus MagicMock accessors; the
    compress-state pools are real NPUCompressStatePool instances.
    """
    pool = object.__new__(DSV4NPUTokenToKVPool)
    pool.device = device
    pool.page_size = page_size
    pool.swa_page_size = swa_page_size
    pool.qk_nope_head_dim = 128
    pool.qk_rope_head_dim = 64
    pool.indexer_head_dim = 256
    pool.c4_state_dtype = torch.float32
    pool.c128_state_dtype = torch.float32
    pool.c4_state_pool_size = 64
    pool.c128_state_pool_size = 32
    pool._state_pool_size = MagicMock(return_value=64)
    pool.get_ring_size = MagicMock(
        side_effect=lambda ratio: 8 if ratio == 4 else 128
    )
    pool.swa_kv_pool = _make_subpool(3, 2, page_size, 192)
    pool.c4_kv_pool = _make_subpool(2, 2, page_size // 4, 192)
    pool.c128_kv_pool = _make_subpool(1, 2, 1, 192)

    # Indexer pool: real int8 K + fp16 scale buffers (for get_pd_state_components)
    # plus MagicMock accessors (for set/get hooks).
    idx = SimpleNamespace()
    idx.index_k_buffer = [
        torch.zeros(
            2, page_size, 1, pool.indexer_head_dim, dtype=torch.int8, device=device
        )
        for _ in range(2)
    ]
    idx.index_scale_buffer = [
        torch.zeros(2, page_size, 1, 1, dtype=torch.float16, device=device)
        for _ in range(2)
    ]
    idx.has_npu_storage = True
    idx.set_index_k_scale = MagicMock()
    idx.set_index_fused = MagicMock()
    idx.set_index_k_scale_buffer = MagicMock()
    idx.get_index_k = MagicMock()
    idx.get_index_scale = MagicMock()
    pool.c4_indexer_kv_pool = idx

    pool.layer_mapping = {
        0: _LayerItem(0, 0, "swa"),
        1: _LayerItem(4, 0, "c4"),
        2: _LayerItem(128, 0, "c128"),
    }
    pool.get_attention_compress_states = MagicMock()
    pool.get_indexer_compress_states = MagicMock()

    # Compress-state pools (real NPUCompressStatePool) for get_pd_state_components.
    # compress_state_pools is aligned 1:1 with compression_ratios.
    pool.compression_ratios = [4, 128]
    pool.compress_state_pools = [
        _make_state_pool(ratio=4),
        _make_state_pool(ratio=128, overlap=False),
    ]
    pool.indexer_compress_state_pools = [_make_state_pool(ratio=4)]
    return pool


# ---------------------------------------------------------------------------
#  npu_state_pool_size
# ---------------------------------------------------------------------------


@unittest.skip("removed in upstream refactor")
class TestNpuStatePoolSize(unittest.TestCase):
    """Tests for the module-level npu_state_pool_size helper."""

    def test_formula_cases(self):
        """Explicit ceil/+1/clamp cases across ratio/page_size combos."""
        self.assertEqual(npu_state_pool_size(ratio=4, page_size=1, max_num_reqs=1), 9)
        self.assertEqual(
            npu_state_pool_size(ratio=128, page_size=1, max_num_reqs=2), 464
        )
        self.assertEqual(npu_state_pool_size(ratio=4, page_size=8, max_num_reqs=4), 64)
        self.assertEqual(
            npu_state_pool_size(ratio=128, page_size=128, max_num_reqs=1), 384
        )
        self.assertEqual(
            npu_state_pool_size(ratio=4, page_size=128, max_num_reqs=1), 256
        )

    def test_properties(self):
        """Result divisible by page_size, linear in reqs, zero for no reqs."""
        for r, ps, mn in [(4, 1, 7), (128, 1, 3), (4, 8, 5), (128, 128, 2)]:
            self.assertEqual(
                npu_state_pool_size(ratio=r, page_size=ps, max_num_reqs=mn) % ps, 0
            )
        base = npu_state_pool_size(ratio=4, page_size=1, max_num_reqs=1)
        self.assertEqual(
            npu_state_pool_size(ratio=4, page_size=1, max_num_reqs=5), base * 5
        )
        self.assertEqual(npu_state_pool_size(ratio=4, page_size=1, max_num_reqs=0), 0)


# ---------------------------------------------------------------------------
#  NPUCompressStatePool
# ---------------------------------------------------------------------------


class TestNPUCompressStatePool(unittest.TestCase):
    """Tests for NPUCompressStatePool construction and layout."""

    def test_valid_ratios_construct(self):
        """ratio 4/128 (overlap True/False) construct without error."""
        for ratio, overlap in [(4, True), (128, False)]:
            sp = _make_state_pool(ratio=ratio, overlap=overlap, head_dim=4, ring_size=2)
            self.assertIsInstance(sp, NPUCompressStatePool)

    def test_invalid_ratio_raises(self):
        """ratio not in (4, 128) -> AssertionError."""
        with self.assertRaises(AssertionError):
            NPUCompressStatePool(
                size=8,
                overlap=True,
                head_dim=4,
                dtype=torch.float32,
                device="cpu",
                enable_memory_saver=False,
                ratio=8,
                ring_size=2,
                swa_page_size=16,
            )

    def test_ring_size_one_state_cache_3d_raises(self):
        """ring_size=1 -> page_size=1 -> the 3-D kernel view raises."""
        sp = _make_state_pool(ring_size=1)
        self.assertEqual(sp.page_size, 1)
        with self.assertRaises(AssertionError):
            _ = sp.state_cache_3d

    def test_size_is_padded_to_full_buffer_pages(self):
        """_size = (size + ring_size + 1) padded to lcm(ratio, page_size)."""
        sp = _make_state_pool(size=8, ring_size=2)
        # _size=8+2+1=11; pad_to=lcm(4,2)=4 -> 12
        self.assertEqual(sp._size, 12)

    def test_ring_size_stored_and_online_false(self):
        """ring_size stored on the pool; paged pool has no online mode."""
        sp = _make_state_pool()
        self.assertEqual(sp.ring_size, 2)
        self.assertFalse(sp.online)

    def test_page_size_stored(self):
        """page_size (=state_cache_page_size=ring_size) stored for the 3D view."""
        sp = _make_state_pool(ring_size=4)
        self.assertEqual(sp.page_size, 4)

    def test_ratio_stored(self):
        """ratio is stored on the pool (block-0 sentinel path uses it)."""
        self.assertEqual(_make_state_pool(ratio=4).ratio, 4)
        self.assertEqual(_make_state_pool(ratio=128, overlap=False).ratio, 128)

    def test_last_dim_scales_with_overlap(self):
        """last_dim = 2 * (1 + overlap) * head_dim."""
        self.assertEqual(_make_state_pool(overlap=True, head_dim=4).last_dim, 4 * 4)
        self.assertEqual(
            _make_state_pool(overlap=False, head_dim=4, ratio=128).last_dim, 2 * 4
        )

    def test_dummy_row_kv_zeroed_score_neg_inf(self):
        """kv all-zero; last dummy row score -inf, other rows 0."""
        sp = _make_state_pool(size=8, ring_size=2, head_dim=4)
        self.assertTrue(torch.all(sp.kv_score_buffer.kv == 0))
        score = sp.kv_score_buffer.score
        self.assertTrue(torch.all(score[-1] == float("-inf")))
        self.assertTrue(torch.all(score[:-1] == 0))

    def test_ratio128_all_rows_cleared(self):
        """ratio=128 clears every row (cold C128 request banks start cleared)."""
        sp = _make_state_pool(
            size=8, ring_size=2, head_dim=4, ratio=128, overlap=False
        )
        score = sp.kv_score_buffer.score
        self.assertTrue(torch.all(score == float("-inf")))
        self.assertTrue(torch.all(sp.kv_score_buffer.kv == 0))

    def test_state_cache_3d_shape_and_sentinel(self):
        """3D view shape (npages, page_size, last_dim); last row is the dummy."""
        sp = _make_state_pool(size=8, ring_size=2, head_dim=4)
        # _size=12, page_size=2, last_dim=16 -> (6, 2, 16)
        self.assertEqual(
            sp.state_cache_3d.shape, (sp._size // sp.page_size, 2, sp.last_dim)
        )
        last_row = sp.state_cache_3d.reshape(-1, sp.last_dim)[-1]
        half = sp.last_dim // 2
        self.assertTrue(torch.all(last_row[:half] == 0))
        self.assertTrue(torch.all(last_row[half:] == float("-inf")))


# ---------------------------------------------------------------------------
#  NPUDeepSeekV4SingleKVPool
# ---------------------------------------------------------------------------


class TestNPUDeepSeekV4SingleKVPool(unittest.TestCase):
    """Tests for NPUDeepSeekV4SingleKVPool buffer creation."""

    def _make(self, **over):
        defaults = dict(
            size=128,
            page_size=1,
            dtype=torch.bfloat16,
            qk_nope=128,
            qk_rope=64,
            layer_num=2,
            device="cpu",
            kernel_page_size=16,
        )
        defaults.update(over)
        d = defaults
        return NPUDeepSeekV4SingleKVPool(
            d["size"],
            d["page_size"],
            d["dtype"],
            d["qk_nope"],
            d["qk_rope"],
            d["layer_num"],
            d["device"],
            False,
            kernel_page_size=d["kernel_page_size"],
        )

    def test_bf16_creates_pa_nd_buffers(self):
        """bf16 store -> kv_buffer entries are PA_ND (npages, kp, 1, kv_dim)."""
        pool = self._make(kernel_page_size=16)
        self.assertEqual(pool.kernel_page_size, 16)
        self.assertEqual(pool.kv_cache_total_dim, 128 + 64)  # qk_nope + qk_rope
        self.assertEqual(len(pool.kv_buffer), 2)
        buf = pool.kv_buffer[0]
        npu_pages = (128 + 16 + 1) // 16  # 9
        self.assertEqual(buf.shape, (npu_pages, 16, 1, 192))
        self.assertEqual(buf.dtype, torch.bfloat16)

    def test_npu_num_pages_formula(self):
        """npu_num_pages = (size + kernel_page_size + 1) // kernel_page_size."""
        pool = self._make(size=200, kernel_page_size=16)
        expected = (200 + 16 + 1) // 16
        self.assertEqual(pool.kv_buffer[0].shape[0], expected)

    def test_non_bf16_falls_back_to_base(self):
        """Non-bf16 store_dtype -> super().create_buffer (2D packed buffer)."""
        pool = self._make(dtype=torch.float32)
        buf = pool.kv_buffer[0]
        num_pages = (128 + 1 + 1) // 1  # base _create_buffers formula
        self.assertEqual(buf.shape, (num_pages, 192))
        self.assertEqual(buf.dtype, torch.float32)

    def test_layer_num_drives_buffer_count(self):
        """kv_buffer length matches layer_num."""
        pool = self._make(layer_num=3)
        self.assertEqual(len(pool.kv_buffer), 3)


# ---------------------------------------------------------------------------
#  NPUDeepSeekV4IndexerPool
# ---------------------------------------------------------------------------


class TestNPUDeepSeekV4IndexerPool(unittest.TestCase):
    """Tests for NPUDeepSeekV4IndexerPool buffers and accessors."""

    def _make(self, **over):
        defaults = dict(
            size=128,
            page_size=1,
            dtype=torch.bfloat16,
            index_head_dim=256,
            layer_num=2,
            device="cpu",
            kernel_page_size=16,
        )
        defaults.update(over)
        d = defaults
        return NPUDeepSeekV4IndexerPool(
            d["size"],
            d["page_size"],
            d["dtype"],
            d["index_head_dim"],
            d["layer_num"],
            d["device"],
            False,
            kernel_page_size=d["kernel_page_size"],
        )

    def test_kernel_page_size_set_before_super(self):
        """_kernel_page_size is available right after construction."""
        pool = self._make(kernel_page_size=16)
        self.assertEqual(pool._kernel_page_size, 16)

    def test_buffer_layouts(self):
        """k (int8) and scale (fp16) buffers are layer_num PA_ND tensors."""
        pool = self._make()
        npu_pages = (128 + 16 + 1) // 16  # 9
        self.assertEqual(len(pool.index_k_buffer), 2)
        self.assertEqual(pool.index_k_buffer[0].shape, (npu_pages, 16, 1, 256))
        self.assertEqual(pool.index_k_buffer[0].dtype, torch.int8)
        self.assertEqual(len(pool.index_scale_buffer), 2)
        self.assertEqual(pool.index_scale_buffer[0].shape, (npu_pages, 16, 1, 1))
        self.assertEqual(pool.index_scale_buffer[0].dtype, torch.float16)
        self.assertTrue(pool.has_npu_storage)

    def test_get_buffers_returns_layer(self):
        """get_index_k / get_index_scale return the layer's buffer."""
        pool = self._make()
        self.assertIs(pool.get_index_k(0), pool.index_k_buffer[0])
        self.assertIs(pool.get_index_k(1), pool.index_k_buffer[1])
        self.assertIs(pool.get_index_scale(0), pool.index_scale_buffer[0])

    def test_set_index_k_scale_with_scale_two_scatters(self):
        """index_k_scale given -> scatter into both k and scale buffers."""
        pool = self._make()
        _TORCH_NPU.npu_scatter_nd_update_.reset_mock()
        loc = torch.tensor([0, 1])
        index_k = torch.zeros(2, 256, dtype=torch.int8)
        index_k_scale = torch.zeros(2, 1, dtype=torch.float16)
        pool.set_index_k_scale(0, loc, index_k, index_k_scale)
        self.assertEqual(_TORCH_NPU.npu_scatter_nd_update_.call_count, 2)
        # First scatter source = index_k.to(int8).view(-1, 1, d)
        k_src = _TORCH_NPU.npu_scatter_nd_update_.call_args_list[0].args[2]
        self.assertEqual(k_src.shape, (2, 1, 256))
        self.assertEqual(k_src.dtype, torch.int8)

    def test_set_index_k_scale_without_scale_one_scatter(self):
        """index_k_scale None -> only the k buffer is scattered."""
        pool = self._make()
        _TORCH_NPU.npu_scatter_nd_update_.reset_mock()
        loc = torch.tensor([0, 1])
        index_k = torch.zeros(2, 256, dtype=torch.int8)
        pool.set_index_k_scale(0, loc, index_k, None)
        self.assertEqual(_TORCH_NPU.npu_scatter_nd_update_.call_count, 1)

    def test_set_index_k_scale_target_is_k_buffer(self):
        """First scatter target is a (-1, 1, d) view of index_k_buffer[layer]."""
        pool = self._make()
        _TORCH_NPU.npu_scatter_nd_update_.reset_mock()
        loc = torch.tensor([0])
        index_k = torch.zeros(1, 256, dtype=torch.int8)
        pool.set_index_k_scale(1, loc, index_k, None)
        target = _TORCH_NPU.npu_scatter_nd_update_.call_args.args[0]
        base = pool.index_k_buffer[1]
        # Same underlying storage (a view), reshaped to (-1, 1, index_head_dim).
        self.assertEqual(target.data_ptr(), base.data_ptr())
        self.assertEqual(target.shape, (base.numel() // 256, 1, 256))
        self.assertEqual(target.dtype, torch.int8)


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool._make_kv_pool
# ---------------------------------------------------------------------------


class TestMakeKvPool(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool._make_kv_pool."""

    def test_returns_npu_single_kv_pool(self):
        """Returns an NPUDeepSeekV4SingleKVPool with kernel_page_size set."""
        pool = _make_kvpool()
        kv = pool._make_kv_pool(
            size=128,
            page_size=128,
            dtype=torch.bfloat16,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            global_page_size=128,
        )
        self.assertIsInstance(kv, NPUDeepSeekV4SingleKVPool)
        self.assertEqual(kv.kernel_page_size, 128)
        self.assertEqual(len(kv.kv_buffer), 2)

    def test_non_singleton_cls_raises(self):
        """cls != DeepSeekV4SingleKVPool -> AssertionError (no hisparse)."""
        pool = _make_kvpool()
        with self.assertRaises(AssertionError):
            pool._make_kv_pool(
                size=128,
                page_size=128,
                dtype=torch.bfloat16,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                global_page_size=128,
                cls=DeepSeekV4IndexerPool,
            )

    def test_passes_head_dims_from_self(self):
        """qk_nope/rope head dims come from self, not kwargs."""
        pool = _make_kvpool()
        kv = pool._make_kv_pool(
            size=128,
            page_size=128,
            dtype=torch.bfloat16,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            global_page_size=128,
        )
        self.assertEqual(kv.qk_nope_head_dim, pool.qk_nope_head_dim)
        self.assertEqual(kv.qk_rope_head_dim, pool.qk_rope_head_dim)


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool._make_attn_state_pool / _make_indexer_state_pool
# ---------------------------------------------------------------------------


class TestMakeAttnStatePool(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool._make_attn_state_pool."""

    def test_ratios_build_pool(self):
        """ratio 4/128 -> NPUCompressStatePool with per-ratio overlap/last_dim."""
        pool = _make_kvpool()
        head_dim = pool.qk_nope_head_dim + pool.qk_rope_head_dim
        for ratio, mult in [(4, 4), (128, 2)]:
            sp = pool._make_attn_state_pool(ratio=ratio, enable_memory_saver=False)
            self.assertIsInstance(sp, NPUCompressStatePool)
            # page_size = state_cache_page_size = ring_size from get_ring_size(ratio)
            self.assertEqual(sp.page_size, pool.get_ring_size(ratio))
            self.assertEqual(sp.ring_size, pool.get_ring_size(ratio))
            self.assertEqual(sp.last_dim, mult * head_dim)

    def test_state_pool_size_consulted_for_size(self):
        """_state_pool_size is called with the ratio and sizes the pool."""
        pool = _make_kvpool()
        pool._state_pool_size.return_value = 64
        sp = pool._make_attn_state_pool(ratio=4, enable_memory_saver=False)
        pool._state_pool_size.assert_called_once_with(4)
        # _size = 64 + ring_size(8) + 1 = 73 -> pad to lcm(4, 8) = 80
        self.assertEqual(sp._size, 80)

    @patch(_MOD + ".ONLINE_C128", True)
    def test_online_c128_only_rejects_ratio128(self):
        """ONLINE_C128 forbids ratio 128 but allows ratio 4 (no online on NPU)."""
        pool = _make_kvpool()
        with self.assertRaises(AssertionError):
            pool._make_attn_state_pool(ratio=128, enable_memory_saver=False)
        sp = pool._make_attn_state_pool(ratio=4, enable_memory_saver=False)
        self.assertIsInstance(sp, NPUCompressStatePool)


class TestMakeIndexerStatePool(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool._make_indexer_state_pool."""

    def test_builds_pool_with_indexer_head_dim(self):
        """Uses indexer_head_dim and c4_state_pool_size budget."""
        pool = _make_kvpool()
        pool.c4_state_pool_size = 64
        sp = pool._make_indexer_state_pool(ratio=4, enable_memory_saver=False)
        self.assertIsInstance(sp, NPUCompressStatePool)
        # overlap True -> last_dim = 4 * indexer_head_dim
        self.assertEqual(sp.last_dim, 4 * pool.indexer_head_dim)
        # page_size = state_cache_page_size = ring_size from get_ring_size(4)
        self.assertEqual(sp.page_size, pool.get_ring_size(4))
        # _size = 64 + ring_size(8) + 1 = 73 -> pad to lcm(4, 8) = 80
        self.assertEqual(sp._size, 80)


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool._make_indexer_pool / clear_unaccepted_c128_draft_states
# ---------------------------------------------------------------------------


class TestMakeIndexerPool(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool._make_indexer_pool."""

    def test_returns_npu_indexer_pool(self):
        """Returns NPUDeepSeekV4IndexerPool; kernel_page_size = page_size, head dim flows."""
        pool = _make_kvpool()
        ip = pool._make_indexer_pool(
            size=128,
            page_size=128,
            dtype=torch.bfloat16,
            index_head_dim=512,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self.assertIsInstance(ip, NPUDeepSeekV4IndexerPool)
        self.assertEqual(ip._kernel_page_size, pool.page_size)
        self.assertEqual(ip.index_head_dim, 512)
        self.assertEqual(len(ip.index_k_buffer), 2)


@unittest.skip("clear_unaccepted_c128_draft_states moved to shared base (out of NPU scope)")
class TestClearUnacceptedC128DraftStates(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.clear_unaccepted_c128_draft_states."""

    def test_is_noop(self):
        """Method body is a bare pass -> returns None, no side effects."""
        pool = _make_kvpool()
        result = pool.clear_unaccepted_c128_draft_states(
            torch.tensor([0]), torch.tensor([8]), torch.tensor([1]), 4
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.get_state_cache
# ---------------------------------------------------------------------------


class TestGetStateCache(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.get_state_cache."""

    def test_routes_to_attention_pool(self):
        """from_indexer=False -> get_attention_compress_states(layer_id)."""
        pool = _make_kvpool()
        sp = _make_state_pool()
        pool.get_attention_compress_states.return_value = sp
        result = pool.get_state_cache(1, from_indexer=False)
        expected = sp.kv_score_buffer.kv_score.view(-1, sp.page_size, sp.last_dim)
        self.assertTrue(torch.equal(result, expected))
        pool.get_attention_compress_states.assert_called_once_with(1)

    def test_routes_to_indexer_pool(self):
        """from_indexer=True -> get_indexer_compress_states(layer_id)."""
        pool = _make_kvpool()
        sp = _make_state_pool()
        pool.get_indexer_compress_states.return_value = sp
        result = pool.get_state_cache(2, from_indexer=True)
        expected = sp.kv_score_buffer.kv_score.view(-1, sp.page_size, sp.last_dim)
        self.assertTrue(torch.equal(result, expected))
        pool.get_indexer_compress_states.assert_called_once_with(2)


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.get_key_buffer / get_value_buffer / get_kv_buffer
# ---------------------------------------------------------------------------


class TestGetKeyBuffer(unittest.TestCase):
    """Tests for get_key_buffer routing by compression ratio."""

    def test_ratio_routes_to_subpool(self):
        """ratio 0/4/128 -> swa/c4/c128 kv_buffer[compress_layer_id]."""
        pool = _make_kvpool()
        for layer_id, buf in [
            (0, pool.swa_kv_pool.kv_buffer[0]),
            (1, pool.c4_kv_pool.kv_buffer[0]),
            (2, pool.c128_kv_pool.kv_buffer[0]),
        ]:
            self.assertIs(pool.get_key_buffer(layer_id), buf)

    def test_unknown_ratio_raises(self):
        """Unsupported ratio -> ValueError."""
        pool = _make_kvpool()
        pool.layer_mapping[3] = _LayerItem(7, 0, "weird")
        with self.assertRaises(ValueError):
            pool.get_key_buffer(3)

    def test_uses_compress_layer_id_not_raw(self):
        """compress_layer_id indexes the sub-pool, not raw layer_id."""
        pool = _make_kvpool()
        pool.layer_mapping[1] = _LayerItem(4, 1, "c4")
        self.assertIs(pool.get_key_buffer(1), pool.c4_kv_pool.kv_buffer[1])

    def test_value_equals_key_and_kv_returns_pair(self):
        """V4 MQA: V buffer == K buffer; get_kv_buffer returns (k, v) with v == k."""
        pool = _make_kvpool()
        self.assertIs(pool.get_value_buffer(1), pool.get_key_buffer(1))
        k, v = pool.get_kv_buffer(1)
        self.assertIs(k, pool.c4_kv_pool.kv_buffer[0])
        self.assertIs(v, pool.c4_kv_pool.kv_buffer[0])


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.get_swa_buffer
# ---------------------------------------------------------------------------


class TestGetSwaBuffer(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.get_swa_buffer."""

    def test_returns_raw_buffer(self):
        """No loc -> raw swa buffer; indexed by raw layer_id, not compress_layer_id."""
        pool = _make_kvpool()
        self.assertIs(pool.get_swa_buffer(0), pool.swa_kv_pool.kv_buffer[0])
        self.assertIs(pool.get_swa_buffer(2), pool.swa_kv_pool.kv_buffer[2])

    def test_with_loc_gathers_flat(self):
        """loc -> flatten(num_pages, page_size) then gather by loc."""
        pool = _make_kvpool()
        loc = torch.tensor([0, 128])
        result = pool.get_swa_buffer(0, loc)
        flat = pool.swa_kv_pool.kv_buffer[0].flatten(0, 1)
        self.assertTrue(torch.equal(result, flat[loc]))
        self.assertEqual(result.shape, (2, 1, 192))


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.get_compress_buffer
# ---------------------------------------------------------------------------


class TestGetCompressBuffer(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.get_compress_buffer."""

    def test_ratio0_returns_none(self):
        """ratio 0 (SWA) has no compress KV -> None."""
        pool = _make_kvpool()
        self.assertIsNone(pool.get_compress_buffer(0))

    def test_ratio4_from_indexer(self):
        """ratio 4 from_indexer -> c4_indexer_kv_pool.get_index_k."""
        pool = _make_kvpool()
        pool.c4_indexer_kv_pool.get_index_k.return_value = "idx_k"
        self.assertEqual(pool.get_compress_buffer(1, from_indexer=True), "idx_k")
        pool.c4_indexer_kv_pool.get_index_k.assert_called_once_with(0)

    def test_non_indexer_returns_kv_buffer(self):
        """ratio 4/128 not indexer -> corresponding compress kv_buffer."""
        pool = _make_kvpool()
        self.assertIs(
            pool.get_compress_buffer(1, from_indexer=False),
            pool.c4_kv_pool.kv_buffer[0],
        )
        self.assertIs(
            pool.get_compress_buffer(2, from_indexer=False),
            pool.c128_kv_pool.kv_buffer[0],
        )

    def test_ratio128_from_indexer_raises(self):
        """c128 has no indexer pool -> AssertionError."""
        pool = _make_kvpool()
        with self.assertRaises(AssertionError):
            pool.get_compress_buffer(2, from_indexer=True)

    def test_with_loc_flattens(self):
        """loc -> flatten(num_pages, page_size) then gather."""
        pool = _make_kvpool()
        loc = torch.tensor([0, 1])
        result = pool.get_compress_buffer(1, from_indexer=False, loc=loc)
        flat = pool.c4_kv_pool.kv_buffer[0].flatten(0, 1)
        self.assertTrue(torch.equal(result, flat[loc]))


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.set_swa_buffer
# ---------------------------------------------------------------------------


class TestSetSwaBuffer(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.set_swa_buffer."""

    def test_writes_cache_with_dim_axis(self):
        """2D cache -> unsqueeze(1) before index_put; 3D cache written as-is."""
        pool = _make_kvpool()
        loc = torch.tensor([0, 1])
        flat = pool.swa_kv_pool.kv_buffer[0].flatten(0, 1)
        c2 = torch.full((2, 192), 1, dtype=torch.bfloat16)
        pool.set_swa_buffer(0, loc, c2)
        self.assertTrue(torch.equal(flat[loc], c2.unsqueeze(1)))
        flat[loc] = 0  # reset so the 3D write is observable
        c3 = torch.full((2, 1, 192), 2, dtype=torch.bfloat16)
        pool.set_swa_buffer(0, loc, c3)
        self.assertTrue(torch.equal(flat[loc], c3))

    def test_casts_to_buffer_dtype(self):
        """cache dtype differs from buffer -> cast on write."""
        pool = _make_kvpool()
        loc = torch.tensor([0])
        cache = torch.ones(1, 1, 192, dtype=torch.float32)
        pool.set_swa_buffer(0, loc, cache)
        flat = pool.swa_kv_pool.kv_buffer[0].flatten(0, 1)
        self.assertEqual(flat.dtype, torch.bfloat16)
        self.assertTrue(torch.all(flat[loc] == 1))


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.set_state_buffer
# ---------------------------------------------------------------------------


@unittest.skip("set_state_buffer removed in upstream refactor")
class TestSetStateBuffer(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.set_state_buffer."""

    def test_writes_kv_and_score_halves(self):
        """kv -> first half, score -> second half at rows [loc]; sentinels intact."""
        pool = _make_kvpool()
        sp = _make_state_pool(head_dim=4, page_size=2, size=8)
        pool.get_attention_compress_states.return_value = sp
        loc = torch.tensor([2, 3])
        kv = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
        score = torch.arange(100, 116, dtype=torch.float32).reshape(2, 8)
        pool.set_state_buffer(1, loc, kv, score, from_indexer=False)
        ks = sp.kv_score_buffer.kv_score
        self.assertTrue(torch.equal(ks[2, :8], kv[0]))
        self.assertTrue(torch.equal(ks[3, :8], kv[1]))
        self.assertTrue(torch.equal(ks[2, 8:], score[0]))
        self.assertTrue(torch.equal(ks[3, 8:], score[1]))
        self.assertTrue(torch.all(ks[0, 8:] == float("-inf")))  # sentinel row

    def test_from_indexer_route_and_cast(self):
        """from_indexer routes to indexer pool; kv/score cast on write."""
        pool = _make_kvpool()
        sp = _make_state_pool(head_dim=4, page_size=2, size=8, dtype=torch.bfloat16)
        pool.get_indexer_compress_states.return_value = sp
        loc = torch.tensor([2])
        kv = torch.ones(1, 8, dtype=torch.float32)
        score = torch.ones(1, 8, dtype=torch.float32)
        pool.set_state_buffer(1, loc, kv, score, from_indexer=True)
        pool.get_indexer_compress_states.assert_called_once_with(1)
        self.assertTrue(torch.equal(sp.kv_score_buffer.kv_score[2, :8], kv[0]))
        self.assertEqual(sp.kv_score_buffer.kv_score.dtype, torch.bfloat16)


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.get_state_buffer
# ---------------------------------------------------------------------------


@unittest.skip("get_state_buffer removed in upstream refactor")
class TestGetStateBuffer(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.get_state_buffer."""

    def test_splits_kv_and_score(self):
        """Split + unsqueeze(-2) num_kv_heads axis; optional kv_indices gather."""
        pool = _make_kvpool()
        sp = _make_state_pool(head_dim=4, page_size=2, size=8)
        pool.get_attention_compress_states.return_value = sp
        ks = sp.kv_score_buffer.kv_score
        half = sp.last_dim // 2
        kv, score = pool.get_state_buffer(1, from_indexer=False)
        self.assertEqual(kv.shape, (sp._size, 1, half))
        self.assertEqual(kv.shape[-2], 1)
        self.assertTrue(torch.equal(kv.squeeze(1), ks[:, :half]))
        self.assertTrue(torch.equal(score.squeeze(1), ks[:, half:]))
        idx = torch.tensor([2, 3])
        kv, score = pool.get_state_buffer(1, from_indexer=False, kv_indices=idx)
        self.assertEqual(kv.shape, (2, 1, half))
        self.assertTrue(torch.equal(kv.squeeze(1), ks[idx, :half]))
        self.assertTrue(torch.equal(score.squeeze(1), ks[idx, half:]))


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.set_compress_buffer
# ---------------------------------------------------------------------------


class TestSetCompressBuffer(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.set_compress_buffer."""

    def test_indexer_npu_calls_set_index_k_scale(self):
        """from_indexer + npu device -> c4_indexer.set_index_k_scale."""
        pool = _make_kvpool()
        kv = MagicMock()
        kv.device.type = "npu"
        scale = torch.tensor([1.0])
        pool.set_compress_buffer(1, torch.tensor([0]), kv, scale, from_indexer=True)
        call = pool.c4_indexer_kv_pool.set_index_k_scale.call_args
        self.assertEqual(call.args[0], 0)
        self.assertIs(call.args[2], kv)
        self.assertIs(call.args[3], scale)

    def test_indexer_invalid_conditions_raise(self):
        """from_indexer path raises for missing NPU storage / non-c4 layer."""
        pool = _make_kvpool()
        kv = MagicMock()
        kv.device.type = "npu"
        pool.c4_indexer_kv_pool.has_npu_storage = False
        with self.assertRaises(AssertionError):
            pool.set_compress_buffer(1, torch.tensor([0]), kv, None, from_indexer=True)
        pool.c4_indexer_kv_pool.has_npu_storage = True
        with self.assertRaises(AssertionError):
            pool.set_compress_buffer(2, torch.tensor([0]), kv, None, from_indexer=True)

    def test_indexer_cpu_no_scale_calls_set_index_fused(self):
        """from_indexer + cpu + kv_scale None -> set_index_fused."""
        pool = _make_kvpool()
        kv = torch.zeros(2, 256, dtype=torch.int8)
        loc = torch.tensor([0, 1])
        pool.set_compress_buffer(1, loc, kv, None, from_indexer=True)
        call = pool.c4_indexer_kv_pool.set_index_fused.call_args
        self.assertEqual(call.args[0], 0)
        self.assertTrue(torch.equal(call.args[1], loc))
        self.assertIs(call.args[2], kv)

    def test_indexer_cpu_with_scale_calls_set_index_k_scale_buffer(self):
        """from_indexer + cpu + kv_scale given -> set_index_k_scale_buffer."""
        pool = _make_kvpool()
        kv = torch.zeros(2, 256, dtype=torch.int8)
        scale = torch.zeros(2, 1, dtype=torch.float16)
        loc = torch.tensor([0, 1])
        pool.set_compress_buffer(1, loc, kv, scale, from_indexer=True)
        call = pool.c4_indexer_kv_pool.set_index_k_scale_buffer.call_args
        self.assertEqual(call.args[0], 0)
        self.assertIs(call.args[3], scale)

    def test_non_indexer_npu_writes_pa_nd(self):
        """npu device: 3D src written directly, 2D src unsqueezed first."""
        pool = _make_kvpool()
        flat = pool.c4_kv_pool.kv_buffer[0].flatten(0, 1)
        loc = torch.tensor([0, 1])
        kv = MagicMock()
        kv.device.type = "npu"
        kv.to.return_value = torch.ones(2, 1, 192, dtype=torch.bfloat16)
        pool.set_compress_buffer(1, loc, kv, None, from_indexer=False)
        self.assertTrue(torch.equal(flat[loc], kv.to.return_value))
        flat[loc] = 0  # reset so the 2D write is observable
        kv.to.return_value = torch.ones(2, 192, dtype=torch.bfloat16)
        pool.set_compress_buffer(1, loc, kv, None, from_indexer=False)
        self.assertTrue(torch.equal(flat[loc], kv.to.return_value.unsqueeze(1)))

    def test_non_indexer_cpu_calls_set_key_buffer_fused(self):
        """from_indexer False + cpu device -> set_key_buffer_fused."""
        pool = _make_kvpool()
        kv = torch.zeros(2, 192, dtype=torch.bfloat16)
        loc = torch.tensor([0, 1])
        pool.set_compress_buffer(1, loc, kv, None, from_indexer=False)
        call = pool.c4_kv_pool.set_key_buffer_fused.call_args
        self.assertEqual(call.args[0], 0)
        self.assertIs(call.args[2], kv)

    def test_non_indexer_c128_routes_to_c128_pool(self):
        """ratio 128 -> writes to c128_kv_pool, not c4_kv_pool."""
        pool = _make_kvpool()
        kv = MagicMock()
        kv.device.type = "npu"
        src = torch.ones(1, 1, 192, dtype=torch.bfloat16)
        kv.to.return_value = src
        loc = torch.tensor([0])
        pool.set_compress_buffer(2, loc, kv, None, from_indexer=False)
        flat = pool.c128_kv_pool.kv_buffer[0].flatten(0, 1)
        self.assertTrue(torch.equal(flat[loc], src))


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.get_compress_dequant_scale_buffer
# ---------------------------------------------------------------------------


class TestGetCompressDequantScaleBuffer(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.get_compress_dequant_scale_buffer."""

    def test_from_indexer_false_raises(self):
        """Only the indexer compress pool has a dequant scale."""
        pool = _make_kvpool()
        with self.assertRaises(AssertionError):
            pool.get_compress_dequant_scale_buffer(1, from_indexer=False)

    def test_from_indexer_true_returns_scale(self):
        """from_indexer=True -> c4_indexer_kv_pool.get_index_scale."""
        pool = _make_kvpool()
        pool.c4_indexer_kv_pool.get_index_scale.return_value = "scale"
        self.assertEqual(
            pool.get_compress_dequant_scale_buffer(1, from_indexer=True), "scale"
        )
        pool.c4_indexer_kv_pool.get_index_scale.assert_called_once_with(0)


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.translate_kv_loc_to_compress_state_loc
# ---------------------------------------------------------------------------


@unittest.skip("translate_kv_loc_to_compress_state_loc removed in upstream refactor")
class TestTranslateKvLocToCompressStateLoc(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.translate_kv_loc_to_compress_state_loc."""

    def test_raises_runtime_error(self):
        """Paged kernel cannot ring-hash -> RuntimeError with paged/ring message."""
        pool = _make_kvpool()
        for ratio in (4, 128):
            try:
                pool.translate_kv_loc_to_compress_state_loc(torch.tensor([0]), ratio)
                self.fail("expected RuntimeError")
            except RuntimeError as e:
                msg = str(e).lower()
                self.assertIn("paged", msg)
                self.assertIn("ring", msg)
                self.assertIn("out_cache_loc_dsv4", msg)


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.get_contiguous_buf_infos
# ---------------------------------------------------------------------------


class TestGetContiguousBufInfos(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.get_contiguous_buf_infos."""

    def test_returns_real_buffers_in_fixed_order(self):
        """c4 kv + indexer K + indexer scale buffers, one entry per buffer."""
        pool = _make_kvpool()
        ptrs, lens, ilens = pool.get_contiguous_buf_infos()
        bufs = (
            list(pool.c4_kv_pool.kv_buffer)
            + list(pool.c4_indexer_kv_pool.index_k_buffer)
            + list(pool.c4_indexer_kv_pool.index_scale_buffer)
        )
        self.assertEqual(len(ptrs), len(bufs))  # 2 + 2 + 2 = 6
        self.assertEqual(ptrs, [b.data_ptr() for b in bufs])
        self.assertEqual(lens, [b.nbytes for b in bufs])
        self.assertEqual(ilens, [b[0].nbytes for b in bufs])


# ---------------------------------------------------------------------------
#  DSV4NPUTokenToKVPool.get_pd_state_components
# ---------------------------------------------------------------------------


@unittest.skip("get_pd_state_components removed in upstream refactor")
class TestGetPdStateComponents(unittest.TestCase):
    """Tests for DSV4NPUTokenToKVPool.get_pd_state_components."""

    def test_components_fixed_order_and_aligned(self):
        """All pools present -> 6 components in fixed order, aligned int lists."""
        pool = _make_kvpool()
        comps = pool.get_pd_state_components()
        self.assertEqual(
            [c[0] for c in comps],
            [
                AscendStateType.DSV4_SWA,
                AscendStateType.DSV4_C4,
                AscendStateType.DSV4_C128,
                AscendStateType.DSV4_INDEXER,
                AscendStateType.DSV4_C4_STATE,
                AscendStateType.DSV4_C128_STATE,
            ],
        )
        for _type, ptrs, lens, ilens in comps:
            self.assertEqual(len(ptrs), len(lens))
            self.assertEqual(len(lens), len(ilens))
            self.assertGreater(len(ptrs), 0)
            self.assertTrue(all(isinstance(p, int) for p in ptrs))
            self.assertTrue(all(isinstance(n, int) for n in lens))
            self.assertTrue(all(isinstance(i, int) for i in ilens))

    def test_buffer_components_match_pools(self):
        """SWA ptrs come from swa pool; INDEXER bundles k + scale buffers."""
        pool = _make_kvpool()
        by_type = {c[0]: c for c in pool.get_pd_state_components()}
        swa = by_type[AscendStateType.DSV4_SWA]
        bufs = pool.swa_kv_pool.kv_buffer
        self.assertEqual(swa[1], [b.data_ptr() for b in bufs])
        self.assertEqual(swa[2], [b.nbytes for b in bufs])
        self.assertEqual(swa[3], [b[0].nbytes for b in bufs])
        idx = by_type[AscendStateType.DSV4_INDEXER]
        idx_pool = pool.c4_indexer_kv_pool
        ibufs = list(idx_pool.index_k_buffer) + list(idx_pool.index_scale_buffer)
        self.assertEqual(len(idx[1]), len(ibufs))  # 2 + 2 = 4
        self.assertEqual(idx[1], [b.data_ptr() for b in ibufs])

    def test_state_components(self):
        """c4 bundles attn+indexer pools; c128 excludes indexer; ilens use page_size."""
        pool = _make_kvpool()
        by_type = {c[0]: c for c in pool.get_pd_state_components()}
        attn_c4 = pool.compress_state_pools[0]
        indexer_sp = pool.indexer_compress_state_pools[0]
        c4s = by_type[AscendStateType.DSV4_C4_STATE]
        self.assertEqual(
            c4s[1],
            [
                attn_c4.kv_score_buffer.kv_score.data_ptr(),
                indexer_sp.kv_score_buffer.kv_score.data_ptr(),
            ],
        )
        c128s = by_type[AscendStateType.DSV4_C128_STATE]
        c128_sp = pool.compress_state_pools[1]
        self.assertEqual(len(c128s[1]), 1)
        self.assertEqual(c128s[1], [c128_sp.kv_score_buffer.kv_score.data_ptr()])
        self.assertEqual(c128s[3], [c128_sp.kv_score_buffer.kv_score[0].nbytes * c128_sp.page_size])

    def test_none_and_empty_pools_dropped(self):
        """None/empty pools drop their component instead of crashing."""
        pool = _make_kvpool()
        pool.swa_kv_pool = None
        pool.compress_state_pools[0] = None
        pool.c128_kv_pool.kv_buffer = []
        comps = pool.get_pd_state_components()
        self.assertFalse(any(c[0] is AscendStateType.DSV4_SWA for c in comps))
        self.assertFalse(any(c[0] is AscendStateType.DSV4_C128 for c in comps))
        c4s = next(c for c in comps if c[0] is AscendStateType.DSV4_C4_STATE)
        self.assertEqual(len(c4s[1]), 1)  # only the indexer pool
        self.assertEqual(len(comps), 4)  # C4, INDEXER, C4_STATE, C128_STATE


if __name__ == "__main__":
    unittest.main()
