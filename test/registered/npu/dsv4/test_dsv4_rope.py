"""Unit tests for dsv4_rope Dsv4NpuRoPE (interleaved cos/sin cache)."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Ensure the workspace-local sglang copy (D:\sglang\python) is imported,
# not a stale editable-install copy that may differ. Four ".." climb from
# test/registered/npu/dsv4/ up to the repo root, then into python/.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "python",
    ),
)

# ---- Mock modules unavailable on CPU/Windows before any sglang import ----
_mock = MagicMock()
for _mod in (
    "torch_npu",
    "triton",
    "triton.language",
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
    # NOTE: dsv4_rope is NOT mocked — it is the module under test.
    "sglang.srt.runtime_context",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.utils",
    "aiohttp",
    "sglang.test.ci.ci_register",
    "sglang.test.test_utils",
):
    sys.modules[_mod] = _mock

# sglang.version
_ver = type(sys)("sglang.version")
_ver.__version__ = "0.0.0.dev0"
sys.modules["sglang.version"] = _ver

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE  # noqa: E402


def register_npu_ci(est_time, suite=None, nightly=False, disabled=None):
    def decorator(cls):
        return cls

    return decorator


class CustomTestCase(unittest.TestCase):
    pass


register_npu_ci(est_time=3, suite="stage-a-unit-test-npu")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_freqs_cis(max_pos=4, rope_dim_half=3, dtype=torch.complex64, device="cpu"):
    """Build a complex freqs_cis [max_pos, rope_dim/2] (deterministic values)."""
    torch.manual_seed(0)
    return torch.randn(max_pos, rope_dim_half, dtype=dtype, device=device)


def _make_rope(freqs_cis=None, rotary_emb=None):
    """Construct a Dsv4NpuRoPE directly via __init__ (bypasses the singleton)."""
    if freqs_cis is None:
        freqs_cis = _make_freqs_cis()
    return Dsv4NpuRoPE(freqs_cis, rotary_emb=rotary_emb)


def _make_rotary_module():
    """A real nn.Module with register_buffer/_buffers for the buffer path."""
    return nn.Module()


# ---------------------------------------------------------------------------
#  __init__
# ---------------------------------------------------------------------------


class TestInit(CustomTestCase):
    """Tests for Dsv4NpuRoPE.__init__."""

    def test_stores_freqs_cis(self):
        """freqs_cis is retained by reference on the instance."""
        fc = _make_freqs_cis()
        rope = _make_rope(fc)
        self.assertIs(rope.freqs_cis, fc)

    def test_rotary_emb_defaults_none(self):
        """rotary_emb is None when not supplied."""
        rope = _make_rope()
        self.assertIsNone(rope.rotary_emb)

    def test_rotary_emb_stored_when_given(self):
        """rotary_emb is retained by reference when supplied."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        self.assertIs(rope.rotary_emb, rot)

    def test_real_imag_starts_none(self):
        """_real_imag is None until _contig_real_imag materializes it."""
        rope = _make_rope()
        self.assertIsNone(rope._real_imag)

    def test_tables_dict_starts_empty(self):
        """_tables cache is empty at construction."""
        rope = _make_rope()
        self.assertEqual(rope._tables, {})


# ---------------------------------------------------------------------------
#  for_freqs (singleton by id(freqs_cis))
# ---------------------------------------------------------------------------


class TestForFreqs(CustomTestCase):
    """Tests for Dsv4NpuRoPE.for_freqs classmethod."""

    def setUp(self):
        # _instances is a class-level shared dict; clear for isolation.
        Dsv4NpuRoPE._instances.clear()

    def test_same_freqs_cis_returns_same_instance(self):
        """for_freqs is a singleton keyed by id(freqs_cis)."""
        fc = _make_freqs_cis()
        a = Dsv4NpuRoPE.for_freqs(fc)
        b = Dsv4NpuRoPE.for_freqs(fc)
        self.assertIs(a, b)

    def test_different_freqs_cis_returns_different_instance(self):
        """Distinct freqs_cis tensors map to distinct instances."""
        a = Dsv4NpuRoPE.for_freqs(_make_freqs_cis())
        b = Dsv4NpuRoPE.for_freqs(_make_freqs_cis())
        self.assertIsNot(a, b)

    def test_instance_registered_in_instances_dict(self):
        """The created instance is stored under id(freqs_cis)."""
        fc = _make_freqs_cis()
        inst = Dsv4NpuRoPE.for_freqs(fc)
        self.assertIs(Dsv4NpuRoPE._instances.get(id(fc)), inst)

    def test_passes_rotary_emb_on_creation(self):
        """rotary_emb is only used at creation; it is stored on the instance."""
        rot = _make_rotary_module()
        fc = _make_freqs_cis()
        inst = Dsv4NpuRoPE.for_freqs(fc, rotary_emb=rot)
        self.assertIs(inst.rotary_emb, rot)

    def test_omitting_rotary_emb_on_warm_freqs_returns_existing(self):
        """Callers sharing a warmed-up freqs_cis may omit rotary_emb."""
        fc = _make_freqs_cis()
        first = Dsv4NpuRoPE.for_freqs(fc, _make_rotary_module())
        second = Dsv4NpuRoPE.for_freqs(fc)
        self.assertIs(first, second)
        self.assertIs(first.rotary_emb, second.rotary_emb)

    def test_reused_id_with_new_object_recreates(self):
        """If id() is reused but the stored freqs_cis is no longer the same
        object, for_freqs builds a fresh instance and re-points the dict."""
        fc = _make_freqs_cis()
        first = Dsv4NpuRoPE.for_freqs(fc)
        # Force the dict to keep the id but point at a stale instance whose
        # freqs_cis is a different tensor than fc.
        stale = Dsv4NpuRoPE(_make_freqs_cis())
        Dsv4NpuRoPE._instances[id(fc)] = stale
        second = Dsv4NpuRoPE.for_freqs(fc)
        self.assertIsNot(second, stale)
        self.assertIsNot(second, first)
        self.assertIs(second.freqs_cis, fc)
        self.assertIs(Dsv4NpuRoPE._instances[id(fc)], second)


# ---------------------------------------------------------------------------
#  _contig_real_imag
# ---------------------------------------------------------------------------


class TestContigRealImag(CustomTestCase):
    """Tests for Dsv4NpuRoPE._contig_real_imag."""

    def test_real_is_contiguous(self):
        """Materialized real half is a contiguous tensor."""
        rope = _make_rope()
        real, _ = rope._contig_real_imag()
        self.assertTrue(real.is_contiguous())

    def test_imag_is_contiguous(self):
        """Materialized imag half is a contiguous tensor."""
        rope = _make_rope()
        _, imag = rope._contig_real_imag()
        self.assertTrue(imag.is_contiguous())

    def test_real_matches_freqs_cis_real(self):
        """Real half equals freqs_cis.real (materialized, not a strided view)."""
        fc = _make_freqs_cis()
        rope = _make_rope(fc)
        real, _ = rope._contig_real_imag()
        self.assertTrue(torch.equal(real, fc.real.contiguous()))

    def test_imag_matches_freqs_cis_imag(self):
        """Imag half equals freqs_cis.imag (materialized, not a strided view)."""
        fc = _make_freqs_cis()
        rope = _make_rope(fc)
        _, imag = rope._contig_real_imag()
        self.assertTrue(torch.equal(imag, fc.imag.contiguous()))

    def test_shape_preserved(self):
        """Real/imag halves keep the [max_pos, rope_dim/2] shape."""
        fc = _make_freqs_cis(max_pos=8, rope_dim_half=5)
        rope = _make_rope(fc)
        real, imag = rope._contig_real_imag()
        self.assertEqual(real.shape, (8, 5))
        self.assertEqual(imag.shape, (8, 5))

    def test_caches_in_real_imag(self):
        """First call populates the _real_imag cache."""
        rope = _make_rope()
        self.assertIsNone(rope._real_imag)
        rope._contig_real_imag()
        self.assertIsNotNone(rope._real_imag)

    def test_second_call_returns_same_objects(self):
        """Repeat calls return the cached (identical) tensors."""
        rope = _make_rope()
        r1, i1 = rope._contig_real_imag()
        r2, i2 = rope._contig_real_imag()
        self.assertIs(r1, r2)
        self.assertIs(i1, i2)


# ---------------------------------------------------------------------------
#  _buffer_names
# ---------------------------------------------------------------------------


class TestBufferNames(CustomTestCase):
    """Tests for Dsv4NpuRoPE._buffer_names staticmethod."""

    def test_bfloat16_names(self):
        """bfloat16 produces the _bfloat16 suffix."""
        cos, sin = Dsv4NpuRoPE._buffer_names(torch.bfloat16)
        self.assertEqual(cos, "_npu_interleaved_rope_cos_cache_bfloat16")
        self.assertEqual(sin, "_npu_interleaved_rope_sin_cache_bfloat16")

    def test_float16_names(self):
        """float16 produces the _float16 suffix."""
        cos, sin = Dsv4NpuRoPE._buffer_names(torch.float16)
        self.assertEqual(cos, "_npu_interleaved_rope_cos_cache_float16")
        self.assertEqual(sin, "_npu_interleaved_rope_sin_cache_float16")

    def test_float32_names(self):
        """float32 produces the _float32 suffix."""
        cos, sin = Dsv4NpuRoPE._buffer_names(torch.float32)
        self.assertEqual(cos, "_npu_interleaved_rope_cos_cache_float32")
        self.assertEqual(sin, "_npu_interleaved_rope_sin_cache_float32")

    def test_cos_and_sin_names_are_distinct(self):
        """cos and sin buffer names differ only in the cos/sin segment."""
        cos, sin = Dsv4NpuRoPE._buffer_names(torch.float32)
        self.assertNotEqual(cos, sin)
        self.assertIn("cos", cos)
        self.assertIn("sin", sin)

    def test_suffix_strips_torch_prefix_and_dots(self):
        """Suffix = str(dtype) with 'torch.' removed and '.' -> '_'."""
        cos, _ = Dsv4NpuRoPE._buffer_names(torch.float64)
        self.assertEqual(cos, "_npu_interleaved_rope_cos_cache_float64")


# ---------------------------------------------------------------------------
#  _register_or_set_buffer
# ---------------------------------------------------------------------------


class TestRegisterOrSetBuffer(CustomTestCase):
    """Tests for Dsv4NpuRoPE._register_or_set_buffer."""

    def test_registers_buffer_when_absent(self):
        """On an nn.Module, a new buffer is registered via register_buffer."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        t = torch.zeros(2, 2)
        rope._register_or_set_buffer("my_buf", t)
        self.assertIn("my_buf", rot._buffers)
        self.assertIs(rot._buffers["my_buf"], t)

    def test_buffer_registered_non_persistent(self):
        """Registered buffers are non-persistent (excluded from state_dict)."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        rope._register_or_set_buffer("my_buf", torch.zeros(1))
        self.assertNotIn("my_buf", rot.state_dict())

    def test_replaces_existing_buffer_via_setattr(self):
        """If the buffer already exists, setattr replaces it in place."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        old = torch.zeros(2, 2)
        rope._register_or_set_buffer("my_buf", old)
        new = torch.ones(2, 2)
        rope._register_or_set_buffer("my_buf", new)
        self.assertIs(rot._buffers["my_buf"], new)

    def test_falls_back_to_setattr_when_no_register_buffer(self):
        """A plain object without register_buffer gets the tensor setattr'd."""
        rot = SimpleNamespace()
        rope = Dsv4NpuRoPE(_make_freqs_cis(), rotary_emb=rot)
        t = torch.zeros(3)
        rope._register_or_set_buffer("my_buf", t)
        self.assertIs(getattr(rot, "my_buf", None), t)


# ---------------------------------------------------------------------------
#  ensure_tables
# ---------------------------------------------------------------------------


class TestEnsureTablesTablesPath(CustomTestCase):
    """Tests for ensure_tables on the _tables-dict path (rotary_emb is None)."""

    def test_returns_tables_with_expected_shape(self):
        """Tables have shape [max_pos, rope_dim] = [max_pos, rope_dim/2 * 2]."""
        fc = _make_freqs_cis(max_pos=4, rope_dim_half=3)
        rope = _make_rope(fc)
        cos, sin = rope.ensure_tables(torch.float32)
        self.assertEqual(tuple(cos.shape), (4, 6))
        self.assertEqual(tuple(sin.shape), (4, 6))

    def test_cos_is_real_interleaved(self):
        """cos row equals freqs_cis.real repeat_interleaved by 2."""
        fc = _make_freqs_cis()
        rope = _make_rope(fc)
        cos, _ = rope.ensure_tables(torch.float32)
        expected = fc.real.contiguous().repeat_interleave(2, dim=-1)
        self.assertTrue(torch.equal(cos, expected))

    def test_sin_is_imag_interleaved(self):
        """sin row equals freqs_cis.imag repeat_interleaved by 2."""
        fc = _make_freqs_cis()
        rope = _make_rope(fc)
        _, sin = rope.ensure_tables(torch.float32)
        expected = fc.imag.contiguous().repeat_interleave(2, dim=-1)
        self.assertTrue(torch.equal(sin, expected))

    def test_interleaved_layout_pairs(self):
        """Layout is [c0,c0,c1,c1,...]: each value appears twice in a row."""
        fc = _make_freqs_cis(max_pos=1, rope_dim_half=3)
        rope = _make_rope(fc)
        cos, sin = rope.ensure_tables(torch.float32)
        self.assertEqual(cos[0, 0].item(), cos[0, 1].item())
        self.assertEqual(cos[0, 2].item(), cos[0, 3].item())
        self.assertEqual(sin[0, 0].item(), sin[0, 1].item())

    def test_tables_are_contiguous(self):
        """Built tables are contiguous (stable for aclgraph capture)."""
        rope = _make_rope()
        cos, sin = rope.ensure_tables(torch.float32)
        self.assertTrue(cos.is_contiguous())
        self.assertTrue(sin.is_contiguous())

    def test_dtype_matches_requested(self):
        """Tables are cast to the requested dtype."""
        rope = _make_rope()
        cos, sin = rope.ensure_tables(torch.float16)
        self.assertEqual(cos.dtype, torch.float16)
        self.assertEqual(sin.dtype, torch.float16)

    def test_device_matches_freqs_cis(self):
        """Tables live on the same device as freqs_cis."""
        fc = _make_freqs_cis(device="cpu")
        rope = _make_rope(fc)
        cos, sin = rope.ensure_tables(torch.float32)
        self.assertEqual(cos.device, fc.device)
        self.assertEqual(sin.device, fc.device)

    def test_caches_in_tables_dict(self):
        """First build stores the pair in _tables under (dtype, device)."""
        rope = _make_rope()
        cos, sin = rope.ensure_tables(torch.float32)
        cached = rope._tables.get((torch.float32, rope.freqs_cis.device))
        self.assertIsNotNone(cached)
        self.assertIs(cached[0], cos)
        self.assertIs(cached[1], sin)

    def test_second_call_returns_same_objects(self):
        """ensure_tables is idempotent: returns the cached pair."""
        rope = _make_rope()
        cos1, sin1 = rope.ensure_tables(torch.float32)
        cos2, sin2 = rope.ensure_tables(torch.float32)
        self.assertIs(cos1, cos2)
        self.assertIs(sin1, sin2)

    def test_different_dtype_separate_entries(self):
        """Each dtype gets its own _tables entry and distinct tensors."""
        rope = _make_rope()
        cos_f32, _ = rope.ensure_tables(torch.float32)
        cos_f16, _ = rope.ensure_tables(torch.float16)
        self.assertEqual(len(rope._tables), 2)
        self.assertIsNot(cos_f32, cos_f16)
        self.assertEqual(cos_f32.dtype, torch.float32)
        self.assertEqual(cos_f16.dtype, torch.float16)

    def test_allow_build_false_raises_when_missing(self):
        """allow_build=False with no cached tables raises RuntimeError."""
        rope = _make_rope()
        with self.assertRaises(RuntimeError):
            rope.ensure_tables(torch.float32, allow_build=False)

    def test_allow_build_false_returns_when_present(self):
        """allow_build=False returns the cached pair once built."""
        rope = _make_rope()
        cos, sin = rope.ensure_tables(torch.float32)
        cos2, sin2 = rope.ensure_tables(torch.float32, allow_build=False)
        self.assertIs(cos, cos2)
        self.assertIs(sin, sin2)


class TestEnsureTablesBufferPath(CustomTestCase):
    """Tests for ensure_tables on the rotary_emb buffer path."""

    def test_registers_buffers_on_rotary_emb(self):
        """Built cos/sin are registered as buffers on rotary_emb."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        rope.ensure_tables(torch.float32)
        cos_name, sin_name = Dsv4NpuRoPE._buffer_names(torch.float32)
        self.assertIn(cos_name, rot._buffers)
        self.assertIn(sin_name, rot._buffers)

    def test_buffers_non_persistent(self):
        """Registered RoPE buffers are non-persistent (not in state_dict)."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        rope.ensure_tables(torch.float32)
        cos_name, _ = Dsv4NpuRoPE._buffer_names(torch.float32)
        self.assertNotIn(cos_name, rot.state_dict())

    def test_returns_registered_buffers(self):
        """Returned tensors are the same objects as the registered buffers."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        cos, sin = rope.ensure_tables(torch.float32)
        cos_name, sin_name = Dsv4NpuRoPE._buffer_names(torch.float32)
        self.assertIs(rot._buffers[cos_name], cos)
        self.assertIs(rot._buffers[sin_name], sin)

    def test_second_call_returns_registered_buffers(self):
        """A second ensure_tables returns the already-registered buffers."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        cos1, sin1 = rope.ensure_tables(torch.float32)
        cos2, sin2 = rope.ensure_tables(torch.float32)
        self.assertIs(cos1, cos2)
        self.assertIs(sin1, sin2)

    def test_dtype_mismatch_rebuilds(self):
        """A different dtype builds and registers a fresh buffer pair."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        cos_f32, _ = rope.ensure_tables(torch.float32)
        cos_f16, _ = rope.ensure_tables(torch.float16)
        self.assertIsNot(cos_f32, cos_f16)
        self.assertEqual(cos_f32.dtype, torch.float32)
        self.assertEqual(cos_f16.dtype, torch.float16)
        cos_name16, _ = Dsv4NpuRoPE._buffer_names(torch.float16)
        self.assertIn(cos_name16, rot._buffers)

    def test_shape_mismatch_rebuilds(self):
        """If a stored buffer has a wrong shape, it is rebuilt."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        cos, _ = rope.ensure_tables(torch.float32)
        # Corrupt the shape so the validation rejects it and rebuilds.
        cos_name, _ = Dsv4NpuRoPE._buffer_names(torch.float32)
        rot._buffers[cos_name] = torch.zeros(1, 1, dtype=torch.float32)
        cos2, _ = rope.ensure_tables(torch.float32)
        self.assertEqual(tuple(cos2.shape), (rope.freqs_cis.shape[0], 6))
        self.assertIs(rot._buffers[cos_name], cos2)

    def test_dtype_mismatch_rebuilds_buffer(self):
        """A stored buffer with the wrong dtype triggers a rebuild."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        rope.ensure_tables(torch.float32)
        cos_name, _ = Dsv4NpuRoPE._buffer_names(torch.float32)
        # Replace with a right-shape, wrong-dtype tensor.
        rot._buffers[cos_name] = torch.zeros(
            rope.freqs_cis.shape[0], 6, dtype=torch.float16
        )
        cos2, _ = rope.ensure_tables(torch.float32)
        self.assertEqual(cos2.dtype, torch.float32)

    def test_allow_build_false_raises_when_missing(self):
        """allow_build=False with no registered buffers raises RuntimeError."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        with self.assertRaises(RuntimeError):
            rope.ensure_tables(torch.float32, allow_build=False)

    def test_allow_build_false_returns_when_present(self):
        """allow_build=False returns the registered buffers once built."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        cos, sin = rope.ensure_tables(torch.float32)
        cos2, sin2 = rope.ensure_tables(torch.float32, allow_build=False)
        self.assertIs(cos, cos2)
        self.assertIs(sin, sin2)

    def test_does_not_populate_tables_dict(self):
        """On the buffer path, _tables stays empty (ownership is on rotary)."""
        rot = _make_rotary_module()
        rope = _make_rope(rotary_emb=rot)
        rope.ensure_tables(torch.float32)
        self.assertEqual(rope._tables, {})


# ---------------------------------------------------------------------------
#  get_cos_sin
# ---------------------------------------------------------------------------


class TestGetCosSin(CustomTestCase):
    """Tests for Dsv4NpuRoPE.get_cos_sin."""

    def test_returns_position_gathered_cos_sin(self):
        """cos/sin equal index_select over the cached tables by positions."""
        rope = _make_rope()
        positions = torch.tensor([0, 2])
        cos, sin = rope.get_cos_sin(positions, torch.float32)
        tables = rope._tables[(torch.float32, rope.freqs_cis.device)]
        self.assertTrue(torch.equal(cos, tables[0].index_select(0, positions)))
        self.assertTrue(torch.equal(sin, tables[1].index_select(0, positions)))

    def test_shape_is_T_rope_dim(self):
        """Output shape is [T, rope_dim] with no view_4d."""
        fc = _make_freqs_cis(max_pos=8, rope_dim_half=4)
        rope = _make_rope(fc)
        positions = torch.tensor([1, 3, 5])
        cos, sin = rope.get_cos_sin(positions, torch.float32)
        self.assertEqual(tuple(cos.shape), (3, 8))
        self.assertEqual(tuple(sin.shape), (3, 8))

    def test_view_4d_reshapes_to_4d(self):
        """view_4d reshapes output to [T, 1, 1, rope_dim]."""
        rope = _make_rope()
        positions = torch.tensor([0, 1])
        cos, sin = rope.get_cos_sin(positions, torch.float32, view_4d=True)
        self.assertEqual(tuple(cos.shape), (2, 1, 1, 6))
        self.assertEqual(tuple(sin.shape), (2, 1, 1, 6))

    def test_view_4d_values_match_flat(self):
        """view_4d output is a reshape of the flat [T, rope_dim] gather."""
        rope = _make_rope()
        positions = torch.tensor([0, 2])
        flat_cos, _ = rope.get_cos_sin(positions, torch.float32)
        cos4, _ = rope.get_cos_sin(positions, torch.float32, view_4d=True)
        self.assertTrue(torch.equal(cos4, flat_cos.unsqueeze(1).unsqueeze(2)))

    def test_inverse_negates_sin(self):
        """inverse=True flips the sign of sin; cos is unchanged."""
        rope = _make_rope()
        positions = torch.tensor([0, 1])
        cos, sin = rope.get_cos_sin(positions, torch.float32)
        cos_inv, sin_inv = rope.get_cos_sin(positions, torch.float32, inverse=True)
        self.assertTrue(torch.equal(cos_inv, cos))
        self.assertTrue(torch.equal(sin_inv, -sin))

    def test_dtype_casts_when_differs_from_cache(self):
        """cache_dtype builds in fp32 but output is cast to bf16."""
        rope = _make_rope()
        positions = torch.tensor([0, 1])
        cos, sin = rope.get_cos_sin(
            positions, torch.bfloat16, cache_dtype=torch.float32
        )
        self.assertEqual(cos.dtype, torch.bfloat16)
        self.assertEqual(sin.dtype, torch.bfloat16)
        # cache built in fp32
        self.assertIn((torch.float32, rope.freqs_cis.device), rope._tables)

    def test_cache_dtype_defaults_to_dtype(self):
        """Omitting cache_dtype builds tables in the same dtype as dtype."""
        rope = _make_rope()
        positions = torch.tensor([0])
        rope.get_cos_sin(positions, torch.float32)
        self.assertIn((torch.float32, rope.freqs_cis.device), rope._tables)

    def test_allow_build_propagated(self):
        """allow_build=False is forwarded to ensure_tables and raises if missing."""
        rope = _make_rope()
        positions = torch.tensor([0])
        with self.assertRaises(RuntimeError):
            rope.get_cos_sin(positions, torch.float32, allow_build=False)

    def test_allow_build_false_returns_when_present(self):
        """allow_build=False still gathers once tables exist."""
        rope = _make_rope()
        positions = torch.tensor([0, 1])
        rope.get_cos_sin(positions, torch.float32)
        cos, sin = rope.get_cos_sin(positions, torch.float32, allow_build=False)
        self.assertEqual(tuple(cos.shape), (2, 6))

    def test_different_positions_gather_distinct_rows(self):
        """get_cos_sin does not cache across calls: new positions -> new rows."""
        rope = _make_rope()
        cos_a, _ = rope.get_cos_sin(torch.tensor([0]), torch.float32)
        cos_b, _ = rope.get_cos_sin(torch.tensor([1]), torch.float32)
        self.assertFalse(torch.equal(cos_a, cos_b))


# ---------------------------------------------------------------------------
#  apply_rotary_mul_inplace
# ---------------------------------------------------------------------------


class TestApplyRotaryMulInplace(CustomTestCase):
    """Tests for Dsv4NpuRoPE.apply_rotary_mul_inplace staticmethod.

    The NPU custom op torch.ops.custom.inplace_partial_rotary_mul is absent on
    CPU; each test patches it (create=True) with a fresh MagicMock.
    """

    @staticmethod
    def _make_q(t=3, n_heads=2, head_dim=8):
        return torch.randn(t, n_heads, head_dim)

    @staticmethod
    def _make_cos_sin4(t=3, rope_dim=4):
        return torch.randn(t, 1, 1, rope_dim), torch.randn(t, 1, 1, rope_dim)

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_calls_op_once_when_kv_none(self, mock_op):
        """kv_rope None -> only the q call is made."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4()
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, None, cos4, sin4)
        self.assertEqual(mock_op.call_count, 1)

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_calls_op_twice_when_kv_given(self, mock_op):
        """kv_rope supplied -> q call + kv call = 2 calls."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4()
        kv = torch.randn(3, 1, 8)
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, kv, cos4, sin4)
        self.assertEqual(mock_op.call_count, 2)

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_q_arg_is_unsqueezed(self, mock_op):
        """The q tensor passed to the op is q.unsqueeze(1) (adds head dim)."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4()
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, None, cos4, sin4)
        passed_q = mock_op.call_args_list[0].args[0]
        self.assertTrue(torch.equal(passed_q, q.unsqueeze(1)))

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_cos_and_sin_passed_positionally(self, mock_op):
        """cos4 and sin4 are the 2nd and 3rd positional args."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4()
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, None, cos4, sin4)
        args = mock_op.call_args_list[0].args
        self.assertTrue(torch.equal(args[1], cos4))
        self.assertTrue(torch.equal(args[2], sin4))

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_rotary_mode_is_interleave(self, mock_op):
        """rotary_mode kwarg is always 'interleave'."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4()
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, None, cos4, sin4)
        self.assertEqual(mock_op.call_args_list[0].kwargs["rotary_mode"], "interleave")

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_partial_slice_default_qk_nope_dim_zero(self, mock_op):
        """Default qk_nope_dim=0 -> partial_slice = [0, rope_dim]."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4(rope_dim=4)
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, None, cos4, sin4)
        self.assertEqual(mock_op.call_args_list[0].kwargs["partial_slice"], [0, 4])

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_partial_slice_with_qk_nope_dim(self, mock_op):
        """qk_nope_dim=2, rope_dim=4 -> partial_slice = [2, 6]."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4(rope_dim=4)
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, None, cos4, sin4, qk_nope_dim=2)
        self.assertEqual(mock_op.call_args_list[0].kwargs["partial_slice"], [2, 6])

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_kv_3d_is_unsqueezed(self, mock_op):
        """A 3D kv_rope [T,1,head_dim] is passed as kv.unsqueeze(1)."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4(rope_dim=4)
        kv = torch.randn(3, 1, 8)
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, kv, cos4, sin4)
        passed_kv = mock_op.call_args_list[1].args[0]
        self.assertTrue(torch.equal(passed_kv, kv.unsqueeze(1)))

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_kv_4d_is_viewed(self, mock_op):
        """A 4D kv_rope is passed as kv.view(-1, 1, 1, rope_dim)."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4(rope_dim=4)
        kv = torch.randn(3, 1, 1, 4)
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, kv, cos4, sin4)
        passed_kv = mock_op.call_args_list[1].args[0]
        self.assertTrue(torch.equal(passed_kv, kv.view(-1, 1, 1, 4)))

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_kv_uses_same_cos_sin_and_slice_as_q(self, mock_op):
        """The kv call reuses the same cos4/sin4 and partial_slice as q."""
        q = self._make_q()
        cos4, sin4 = self._make_cos_sin4(rope_dim=4)
        kv = torch.randn(3, 1, 8)
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, kv, cos4, sin4, qk_nope_dim=2)
        kv_call = mock_op.call_args_list[1]
        self.assertTrue(torch.equal(kv_call.args[1], cos4))
        self.assertTrue(torch.equal(kv_call.args[2], sin4))
        self.assertEqual(kv_call.kwargs["rotary_mode"], "interleave")
        self.assertEqual(kv_call.kwargs["partial_slice"], [2, 6])

    @patch("torch.ops.custom.inplace_partial_rotary_mul", create=True)
    def test_rope_dim_taken_from_cos4(self, mock_op):
        """rope_dim is derived from cos4.shape[-1], not from the tensors."""
        q = self._make_q(head_dim=12)
        cos4, sin4 = self._make_cos_sin4(rope_dim=6)
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, None, cos4, sin4, qk_nope_dim=4)
        self.assertEqual(mock_op.call_args_list[0].kwargs["partial_slice"], [4, 10])


if __name__ == "__main__":
    unittest.main()
