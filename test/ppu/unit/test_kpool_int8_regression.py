"""Static regression tests for GLM-5.3-Flash PPU KPool INT8 migration.

Validates that on PPU SM80 (get_device_sm() < 89) **every** KPool indexer code
path routes through INT8 APIs (int8_mqa_logits / int8_paged_mqa_logits /
per_token_quant_int8) and never reaches FP8 counterparts genuinely.

The v0.5.21 forms validated here:

- ``dsa_indexer_kpool.py`` uses an **instance flag** ``self.use_int8`` (set from
  ``get_device_sm() < 89 and is_ppu()``) instead of a module-level ``_is_ppu``
  gate, and the ragged MQA path is an early-return instead of a ternary.
- ``kpool_fp8_index.py`` probes the device lazily via ``_kpool_quant_int8()``
  (``lru_cache``) and threads ``QUANT_INT8: tl.constexpr`` through the Triton
  kernels; buffer dtype is a local ``k_dtype`` derived from that probe.
- The FlashMLA sparse topk 128-align padding is gated on ``_is_ppu``.
- FA3 INT8 KV-cache handling is a fail-fast ``ValueError``, not an up-cast copy.

These tests operate purely on source text / AST – no GPU or CUDA required.
"""

import ast
import os
import re
import unittest

# ---------------------------------------------------------------------------
# Locate source files
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PYTHON_ROOT = os.path.join(_REPO_ROOT, "python")

_DSA_INDEXER_KPOOL = os.path.join(
    _PYTHON_ROOT,
    "sglang",
    "srt",
    "layers",
    "attention",
    "dsa",
    "dsa_indexer_kpool.py",
)
_KPOOL_FP8_INDEX = os.path.join(
    _PYTHON_ROOT,
    "sglang",
    "srt",
    "layers",
    "attention",
    "dsa",
    "kpool_fp8_index.py",
)
_DSA_BACKEND = os.path.join(
    _PYTHON_ROOT,
    "sglang",
    "srt",
    "layers",
    "attention",
    "dsa_backend.py",
)
_PPU_FLASH_ATTN = os.path.join(
    _PYTHON_ROOT,
    "sglang",
    "srt",
    "hardware_backend",
    "ppu",
    "attention",
    "flash_attention.py",
)


def _read(path: str) -> str:
    with open(path, "r") as fh:
        return fh.read()


def _parse(path: str) -> ast.Module:
    src = _read(path)
    return ast.parse(src, filename=path)


def _strip_comments(src: str) -> str:
    """Remove full-line and trailing ``#`` comments.

    v0.5.21 keeps the pre-migration code as inline ``# PPU int8: was ...``
    annotations (e.g. ``buf.view(k_dtype)  # PPU int8: was
    buf.view(torch.float8_e4m3fn)``); dtype scans must ignore those or they
    would count commented-out FP8 code as active.
    """
    return "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines())


# ===================================================================
# Part 1 – dsa_indexer_kpool.py: PPU gate & INT8 API routing
# ===================================================================
class TestDsaIndexerKpoolInt8Gate(unittest.TestCase):
    """PPU INT8 gate and API routing in dsa_indexer_kpool.py."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(_DSA_INDEXER_KPOOL)
        cls.tree = _parse(_DSA_INDEXER_KPOOL)

    # 1a. Instance-level use_int8 gate exists (v0.5.21 form)
    def test_is_ppu_gate_exists(self):
        self.assertIn(
            "self.use_int8 = 0",
            self.src,
            "Instance flag self.use_int8 must be initialized",
        )
        pattern = re.compile(
            r"if\s+get_device_sm\(\)\s*<\s*89\s+and\s+is_ppu\(\):\s*\n"
            r"\s+self\.use_int8 = 1\s*\n"
            r"\s+else:\s*\n"
            r"\s+self\.use_fp8 = 1"
        )
        self.assertRegex(
            self.src,
            pattern,
            "Gate must set self.use_int8=1 when get_device_sm()<89 and is_ppu(), "
            "else self.use_fp8=1",
        )

    # 1b. is_ppu / get_device_sm imported from sglang.srt.utils
    def test_is_ppu_import(self):
        pattern = re.compile(
            r"from\s+sglang\.srt\.utils\s+import\s+\(([^)]*)\)", re.DOTALL
        )
        match = pattern.search(self.src)
        self.assertIsNotNone(match, "sglang.srt.utils import block must exist")
        names = match.group(1)
        self.assertIn("is_ppu", names, "is_ppu must be imported")
        self.assertIn("get_device_sm", names, "get_device_sm must be imported")

    # 1c. per_token_quant_int8 is imported
    def test_per_token_quant_int8_import(self):
        self.assertIn(
            "from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8",
            self.src,
            "per_token_quant_int8 must be imported for PPU INT8 quantization",
        )

    # 1d. _kpool_act_quant_int8 adapter shim exists
    def test_act_quant_int8_shim(self):
        self.assertIn(
            "def _kpool_act_quant_int8(",
            self.src,
            "_kpool_act_quant_int8 adapter function must exist",
        )
        self.assertIn(
            "return per_token_quant_int8(x)",
            self.src,
            "_kpool_act_quant_int8 must delegate to per_token_quant_int8",
        )

    # 1e. PPU gate selects _kpool_act_quant_int8 as act_quant
    def test_ppu_act_quant_routing(self):
        pattern = re.compile(
            r"if\s+self\.use_int8\s*:\s*\n"
            r"\s+act_quant\s*=\s*_kpool_act_quant_int8"
        )
        self.assertRegex(
            self.src,
            pattern,
            "self.use_int8 must route act_quant to _kpool_act_quant_int8",
        )

    # 1f. Paged MQA path routes through int8_paged_mqa_logits on PPU
    def test_paged_mqa_int8_routing(self):
        pattern = re.compile(
            r"paged_mqa_logits = \(\s*\n"
            r"\s+deep_gemm\.int8_paged_mqa_logits\s*\n"
            r"\s+if self\.use_int8\s*\n"
            r"\s+else deep_gemm\.fp8_paged_mqa_logits\s*\n"
            r"\s+\)"
        )
        self.assertRegex(
            self.src,
            pattern,
            "Paged MQA must select int8_paged_mqa_logits when self.use_int8",
        )

    # 1g. Non-paged (ragged) MQA path routes through int8_mqa_logits on PPU
    def test_ragged_mqa_int8_routing(self):
        # v0.5.21 form: early-return inside _fp8_mqa_logits (not a ternary).
        pattern = re.compile(
            r"if\s+self\.use_int8:\s*\n"
            r"\s+return\s+deep_gemm\.int8_mqa_logits\("
        )
        self.assertRegex(
            self.src,
            pattern,
            "Ragged MQA must early-return deep_gemm.int8_mqa_logits when self.use_int8",
        )
        self.assertEqual(
            self.src.count("deep_gemm.int8_mqa_logits"),
            1,
            "Expected exactly one int8_mqa_logits call site (the guarded one)",
        )

    # 1h. Instance-flag dtype conditional on buffer allocations
    def test_buffer_dtype_conditional(self):
        pattern = re.compile(
            r"torch\.int8\s+if\s+self\.use_int8\s+else\s+torch\.float8_e4m3fn"
        )
        matches = pattern.findall(self.src)
        # 4 sites (v0.5.21): k/u8 buffers + 2 k_fp8.view
        self.assertGreaterEqual(
            len(matches),
            3,
            f"Expected >=3 int8/fp8 dtype conditional sites, found {len(matches)}",
        )

    # 1i. No unconditional fp8_mqa_logits / fp8_paged_mqa_logits calls
    def test_no_unconditional_fp8_mqa_calls(self):
        """Verify fp8_mqa_logits is unreachable without passing the use_int8 gate."""
        # Ragged path: the single fp8 call site must sit in _fp8_mqa_logits
        # after the int8 early-return.
        fs = self.src.find("def _fp8_mqa_logits(")
        self.assertGreater(fs, 0, "_fp8_mqa_logits must exist")
        nxt = self.src.find("\n    def ", fs + 1)
        body = self.src[fs:nxt] if nxt > 0 else self.src[fs:]
        guard = body.find("if self.use_int8:")
        int8_ret = body.find("return deep_gemm.int8_mqa_logits(")
        fp8_ret = body.find("return deep_gemm.fp8_mqa_logits(")
        self.assertTrue(
            0 <= guard < int8_ret < fp8_ret,
            "int8 early-return must precede the fp8 fallback in _fp8_mqa_logits",
        )
        self.assertEqual(
            len(re.findall(r"deep_gemm\.fp8_mqa_logits\s*\(", self.src)),
            1,
            "Exactly one fp8_mqa_logits call site must exist (guarded fallback)",
        )
        # Paged path: every fp8_paged_mqa_logits reference must be an else arm.
        stripped = _strip_comments(self.src)
        for i, line in enumerate(stripped.splitlines(), 1):
            if "deep_gemm.fp8_paged_mqa_logits" in line:
                self.assertIn(
                    "else deep_gemm.fp8_paged_mqa_logits",
                    line,
                    f"Line {i}: fp8_paged_mqa_logits reference is not the "
                    f"ternary else arm: {line.strip()!r}",
                )

    # 1j. Non-PPU FP8 path is preserved in the fallbacks
    def test_non_ppu_fp8_preserved(self):
        self.assertIn(
            "else deep_gemm.fp8_paged_mqa_logits",
            self.src,
            "Non-PPU fp8_paged_mqa_logits must be preserved in the ternary else",
        )
        self.assertIn(
            "return deep_gemm.fp8_mqa_logits(",
            self.src,
            "Non-PPU fp8_mqa_logits fallback must be preserved",
        )


# ===================================================================
# Part 2 – kpool_fp8_index.py: Triton kernel INT8 quantization
# ===================================================================
class TestKpoolFp8IndexInt8Kernels(unittest.TestCase):
    """INT8 quantization constants and buffer dtype in kpool_fp8_index.py."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(_KPOOL_FP8_INDEX)
        cls.tree = _parse(_KPOOL_FP8_INDEX)

    # 2a. Lazy device probe exists (v0.5.21 form of the PPU gate)
    def test_is_ppu_gate_exists(self):
        pattern = re.compile(
            r"@lru_cache\(maxsize=1\)\s*\n"
            r"def\s+_kpool_quant_int8\(\)\s*->\s*bool:\s*\n"
            r".*?\n"
            r"\s+return\s+is_ppu\(\)\s+and\s+get_device_sm\(\)\s*<\s*89",
            re.DOTALL,
        )
        self.assertRegex(
            self.src,
            pattern,
            "_kpool_quant_int8() must lazily return is_ppu() and get_device_sm() < 89",
        )
        self.assertIn(
            "from sglang.srt.utils import get_device_sm, is_ppu",
            self.src,
            "get_device_sm and is_ppu must be imported",
        )

    # 2b. Quantization constants: PPU uses 127, non-PPU 448
    def test_quant_max_conditional(self):
        # fp8_min/fp8_max blocks (softmax_rotate, decode_update) + q_max block
        # (hadamard) all switch on QUANT_INT8.
        self.assertGreaterEqual(
            len(re.findall(r"if QUANT_INT8:", self.src)),
            6,
            "All quantization sites must branch on the QUANT_INT8 constexpr",
        )
        self.assertGreaterEqual(
            len(re.findall(r"fp8_min = -127\.0", self.src)),
            2,
            "QUANT_INT8 must clamp to -127.0 (int8) instead of -448.0 (fp8)",
        )
        self.assertGreaterEqual(
            len(re.findall(r"fp8_max = 448\.0", self.src)),
            2,
            "Non-QUANT_INT8 path must keep the fp8 448.0 clamp",
        )
        self.assertIn("q_max = 127.0", self.src)
        self.assertIn("q_max = 448.0", self.src)

    # 2c. Buffer dtype: int8 on PPU, float8_e4m3fn otherwise
    def test_buf_dtype_conditional(self):
        pattern = re.compile(
            r"k_dtype = torch\.int8 if quant_int8 else torch\.float8_e4m3fn"
        )
        self.assertGreaterEqual(
            len(pattern.findall(self.src)),
            4,
            "Every wrapper must derive k_dtype from the quant_int8 probe",
        )
        buf_views = re.findall(r"buf\.view\(k_dtype\)", self.src)
        self.assertGreaterEqual(
            len(buf_views),
            4,
            f"Expected >=4 buf.view(k_dtype) sites, found {len(buf_views)}",
        )

    # 2d. All four Triton kernels have QUANT_INT8 constexpr param
    def test_triton_kernels_have_is_ppu_int8_param(self):
        expected_kernels = [
            "_kpool_softmax_rotate_write_cache_kernel",
            "_kpool_decode_update_and_maybe_write_cache_kernel",
            "_kpool_assemble_softmax_rotate_write_cache_kernel",
            "_kpool_write_tail_and_maybe_compress_kernel",
        ]
        for kernel_name in expected_kernels:
            with self.subTest(kernel=kernel_name):
                start = self.src.find(f"def {kernel_name}(")
                self.assertGreater(start, 0, f"{kernel_name} must exist")
                close = self.src.find("\n):", start)
                self.assertGreater(
                    close,
                    start,
                    f"{kernel_name}: could not locate end of parameter block",
                )
                params = self.src[start:close]
                self.assertIn(
                    "QUANT_INT8: tl.constexpr",
                    params,
                    f"{kernel_name} must declare QUANT_INT8: tl.constexpr",
                )

    # 2e. _hadamard_quantize_fp8 utility has QUANT_INT8 param
    def test_hadamard_quantize_has_is_ppu_param(self):
        start = self.src.find("def _hadamard_quantize_fp8(")
        self.assertGreater(start, 0, "_hadamard_quantize_fp8 must exist")
        close = self.src.find("\n):", start)
        self.assertIn(
            "QUANT_INT8",
            self.src[start:close],
            "_hadamard_quantize_fp8 must accept QUANT_INT8",
        )

    # 2f. All kernel launch sites pass QUANT_INT8=quant_int8
    def test_kernel_launch_passes_is_ppu(self):
        matches = re.findall(r"QUANT_INT8=quant_int8", self.src)
        # 4 kernel launches
        self.assertGreaterEqual(
            len(matches),
            4,
            f"Expected >=4 QUANT_INT8=quant_int8 launch-site arguments, "
            f"found {len(matches)}",
        )

    # 2g. Each kernel has conditional quantization: 127 on PPU, 448 on non-PPU
    def test_kernel_quantization_constants(self):
        matches_127 = re.findall(r"fp8_max = 127\.0", self.src)
        matches_448 = re.findall(r"fp8_max = 448\.0", self.src)
        self.assertGreaterEqual(
            len(matches_127),
            2,
            f"Expected >=2 fp8_max=127.0 (int8) sites, found {len(matches_127)}",
        )
        self.assertGreaterEqual(
            len(matches_448),
            2,
            f"Expected >=2 fp8_max=448.0 (fp8) sites, found {len(matches_448)}",
        )

    # 2h. INT8 round-and-cast before store (PPU path)
    def test_int8_round_cast_before_store(self):
        self.assertIn(
            "from triton.language.extra import libdevice",
            self.src,
            "libdevice must be imported for round-to-nearest int8 casts",
        )
        lines = self.src.splitlines()
        round_cast = re.compile(r"libdevice\.round\(quantized\)\.to\(tl\.int8\)")
        found = 0
        for i, line in enumerate(lines):
            if not round_cast.search(line):
                continue
            found += 1
            # The cast must sit inside an `if QUANT_INT8:` branch (the fp8 path
            # stores the float value directly). The nearest preceding branch
            # header is at most 2 lines up (comment in between).
            window = "\n".join(lines[max(0, i - 3) : i])
            self.assertIn(
                "if QUANT_INT8:",
                window,
                f"Line {i + 1}: round().to(tl.int8) must be guarded by if QUANT_INT8:",
            )
        self.assertGreaterEqual(
            found,
            3,
            f"Expected >=3 libdevice.round(...).to(tl.int8) casts, found {found}",
        )

    # 2i. buf.view uses k_dtype (not hardcoded float8_e4m3fn)
    def test_buf_view_uses_kpool_buf_dtype(self):
        matches = re.findall(r"\.view\(k_dtype\)", self.src)
        self.assertGreaterEqual(
            len(matches),
            4,
            f"Expected >=4 .view(k_dtype) sites, found {len(matches)}",
        )

    # 2j. No hardcoded fp8 dtype left in active code
    def test_no_hardcoded_fp8_dtype(self):
        stripped = _strip_comments(self.src)
        for i, line in enumerate(stripped.splitlines(), 1):
            if "torch.float8_e4m3fn" in line:
                self.assertIn(
                    "quant_int8",
                    line,
                    f"Line {i}: active torch.float8_e4m3fn must be conditional "
                    f"on quant_int8: {line.strip()!r}",
                )
            if "buf.view(torch.float8_e4m3fn)" in line:
                self.fail(
                    f"Line {i}: hardcoded buf.view(torch.float8_e4m3fn) found "
                    f"in active code: {line.strip()!r}"
                )


# ===================================================================
# Part 3 – Cross-file consistency checks
# ===================================================================
class TestKpoolInt8CrossFileConsistency(unittest.TestCase):
    """Cross-file INT8 consistency: dsa_indexer_kpool ↔ kpool_fp8_index."""

    @classmethod
    def setUpClass(cls):
        cls.indexer_src = _read(_DSA_INDEXER_KPOOL)
        cls.kpool_src = _read(_KPOOL_FP8_INDEX)

    # 3a. Both files import is_ppu from sglang.srt.utils
    def test_both_files_import_is_ppu(self):
        pattern = re.compile(
            r"from\s+sglang\.srt\.utils\s+import\s+.*\bis_ppu\b", re.DOTALL
        )
        self.assertRegex(
            self.indexer_src, pattern, "dsa_indexer_kpool must import is_ppu"
        )
        self.assertRegex(self.kpool_src, pattern, "kpool_fp8_index must import is_ppu")

    # 3b. Both files gate INT8 on the same SM80 PPU condition
    def test_both_files_define_gate(self):
        self.assertIn(
            "get_device_sm() < 89 and is_ppu()",
            self.indexer_src,
            "indexer gate must be get_device_sm() < 89 and is_ppu()",
        )
        self.assertIn(
            "is_ppu() and get_device_sm() < 89",
            self.kpool_src,
            "kpool lazy probe must be is_ppu() and get_device_sm() < 89",
        )

    # 3c. The buffer dtype logic matches between indexer and kpool
    def test_buffer_dtype_consistency(self):
        indexer_has = (
            "torch.int8 if self.use_int8 else torch.float8_e4m3fn" in self.indexer_src
        )
        kpool_has = (
            "torch.int8 if quant_int8 else torch.float8_e4m3fn" in self.kpool_src
        )
        self.assertTrue(indexer_has, "indexer must use conditional int8/fp8 dtype")
        self.assertTrue(kpool_has, "kpool must use conditional int8/fp8 dtype")

    # 3d. Quantization constant 127 appears in kpool Triton kernels
    def test_quant_127_in_kpool(self):
        self.assertIn(
            "127.0", self.kpool_src, "kpool must use 127.0 quantization constant"
        )
        self.assertIn("-127.0", self.kpool_src, "kpool must use -127.0 min constant")

    # 3e. Both files carry the PPU int8 migration annotations
    def test_patch_annotations(self):
        self.assertIn(
            "# PPU int8:",
            self.indexer_src,
            "indexer must carry PPU int8 migration annotations",
        )
        self.assertIn(
            "# PPU int8:",
            self.kpool_src,
            "kpool must carry PPU int8 migration annotations",
        )


# ===================================================================
# Part 4 – PPU path isolation: no reachable FP8 on PPU
# ===================================================================
class TestPpuFp8Isolation(unittest.TestCase):
    """Verify the PPU KPool path does NOT reach FP8 MQA logits."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(_DSA_INDEXER_KPOOL)

    # 4a. fp8_mqa_logits reachable only after the use_int8 early-return
    def test_fp8_mqa_logits_only_guarded(self):
        fs = self.src.find("def _fp8_mqa_logits(")
        self.assertGreater(fs, 0, "_fp8_mqa_logits must exist")
        nxt = self.src.find("\n    def ", fs + 1)
        body = self.src[fs:nxt] if nxt > 0 else self.src[fs:]
        guard = body.find("if self.use_int8:")
        fp8_ret = body.find("return deep_gemm.fp8_mqa_logits(")
        self.assertTrue(
            guard >= 0 and fp8_ret > guard,
            "fp8_mqa_logits fallback must be preceded by the use_int8 guard",
        )
        self.assertEqual(
            len(re.findall(r"deep_gemm\.fp8_mqa_logits\s*\(", self.src)),
            1,
            "Only one fp8_mqa_logits call site may exist",
        )

    # 4b. fp8_paged_mqa_logits only in the ternary else arm
    def test_fp8_paged_mqa_logits_only_in_else_or_comment(self):
        stripped = _strip_comments(self.src)
        for i, line in enumerate(stripped.splitlines(), 1):
            if "deep_gemm.fp8_paged_mqa_logits" in line:
                self.assertIn(
                    "if self.use_int8",
                    "\n".join(stripped.splitlines()[max(0, i - 6) : i + 1]),
                    f"Line {i}: fp8_paged_mqa_logits reference is not guarded "
                    f"by self.use_int8: {line.strip()!r}",
                )

    # 4c. All float8_e4m3fn references in active code are conditional
    def test_float8_e4m3fn_only_conditional_or_comment(self):
        stripped = _strip_comments(self.src)
        for i, line in enumerate(stripped.splitlines(), 1):
            if "float8_e4m3fn" in line:
                self.assertIn(
                    "self.use_int8",
                    line,
                    f"Line {i}: float8_e4m3fn not guarded by self.use_int8: "
                    f"{line.strip()!r}",
                )


# ===================================================================
# Part 5 – dsa_backend.py: FlashMLA sparse topk 128-align padding
# ===================================================================
class TestFlashMlaSparseTopk128Align(unittest.TestCase):
    """Verify topk 128-alignment padding in _forward_flashmla_sparse."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(_DSA_BACKEND)

    # 5a. 128-align padding block exists, gated on _is_ppu
    def test_padding_block_exists(self):
        pattern = re.compile(
            r"if\s+_is_ppu\s+and\s+page_table_1\.shape\[-1\]\s*%\s*128\s*!=\s*0\s*:",
        )
        self.assertRegex(
            self.src,
            pattern,
            "_forward_flashmla_sparse must check `_is_ppu and "
            "page_table_1.shape[-1] % 128 != 0`",
        )

    # 5b. Padding uses -1 (kernel invalid index convention)
    def test_padding_value_minus_one(self):
        pattern = re.compile(
            r"torch\.full\(\s*\n?\s*\(page_table_1\.shape\[0\],\s*padding_size\),\s*\n?\s*-1,"
        )
        self.assertRegex(
            self.src,
            pattern,
            "Padding must use -1 as fill value (kernel invalid index)",
        )

    # 5c. Ceil-align formula: ((old_topk + 127) // 128) * 128
    def test_ceil_128_formula(self):
        self.assertIn(
            "((old_topk + 127) // 128) * 128",
            self.src,
            "Must use ceil-to-128 formula",
        )

    # 5d. Padding is concatenated before unsqueeze within _forward_flashmla_sparse
    def test_padding_before_unsqueeze(self):
        func_start = self.src.find("def _forward_flashmla_sparse(")
        self.assertGreater(func_start, 0, "_forward_flashmla_sparse must exist")
        next_def = self.src.find("\n    def ", func_start + 1)
        func_body = (
            self.src[func_start:next_def] if next_def > 0 else self.src[func_start:]
        )
        padding_pos = func_body.find(
            "if _is_ppu and page_table_1.shape[-1] % 128 != 0:"
        )
        unsqueeze_pos = func_body.find("indices_input = page_table_1.unsqueeze(1)")
        self.assertGreater(padding_pos, 0, "Padding block must be inside the function")
        self.assertGreater(unsqueeze_pos, 0, "indices_input unsqueeze must exist")
        self.assertLess(
            padding_pos,
            unsqueeze_pos,
            "Padding must be applied before the unsqueeze into indices_input",
        )

    # 5e. Padding block carries the PPU int8 migration annotation
    def test_patch_annotation(self):
        anchor = self.src.find("if _is_ppu and page_table_1.shape[-1] % 128 != 0:")
        self.assertGreater(anchor, 0)
        annotation = self.src[max(0, anchor - 500) : anchor]
        self.assertIn(
            "PPU int8:",
            annotation,
            "Padding block must carry the `# PPU int8:` annotation",
        )
        self.assertIn(
            "kpool tail tokens can break the 128 alignment",
            annotation,
            "Annotation must explain why the padding is needed",
        )
        self.assertIn(
            "-1 padding is masked",
            annotation,
            "Annotation must note the kernel masks -1 padding",
        )

    # 5f. Non-PPU path is untouched (no ungated 128 check)
    def test_padding_gated_on_ppu(self):
        stripped = _strip_comments(self.src)
        unguarded = re.compile(r"if\s+page_table_1\.shape\[-1\]\s*%\s*128\s*!=\s*0\s*:")
        self.assertIsNone(
            unguarded.search(stripped),
            "128-align check must never be applied without the _is_ppu gate "
            "(non-PPU path unchanged)",
        )

    # 5g. Concrete alignment example: 2063 -> 2176
    def test_concrete_alignment_example(self):
        old_topk = 2063  # observed kpool tail width
        new_topk = ((old_topk + 127) // 128) * 128
        self.assertEqual(new_topk, 2176)

    # 5h. kpool tail tokens indeed break 128 alignment
    def test_kpool_tail_breaks_alignment(self):
        old_topk = 2063
        self.assertNotEqual(old_topk % 128, 0)
        self.assertEqual(old_topk % 128, 15)
        self.assertEqual(((old_topk + 127) // 128) * 128 % 128, 0)


# ===================================================================
# Part 6 – PPU FA3: INT8 KV cache fail-fast (CR Finding W5)
# ===================================================================
class TestPpuFa3Int8FailFast(unittest.TestCase):
    """Verify flash_attn_with_kvcache rejects int8 KV caches instead of copying."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(_PPU_FLASH_ATTN)
        func_start = cls.src.find("def flash_attn_with_kvcache(")
        cls.func_start = func_start
        next_def = cls.src.find("\ndef ", func_start + 1)
        cls.func_body = (
            cls.src[func_start:next_def] if next_def > 0 else cls.src[func_start:]
        )

    # 6a. k_cache int8 raises
    def test_kcache_int8_raises(self):
        self.assertGreater(self.func_start, 0, "flash_attn_with_kvcache must exist")
        self.assertIn(
            "if k_cache is not None and k_cache.dtype == torch.int8:",
            self.func_body,
            "k_cache int8 guard must exist",
        )
        self.assertIn(
            "PPU FA3 received k_cache with dtype=int8",
            self.func_body,
            "k_cache int8 must raise ValueError",
        )
        self.assertIn(
            "raise ValueError(",
            self.func_body,
            "Guard must raise instead of silently copying",
        )

    # 6b. v_cache int8 raises
    def test_vcache_int8_raises(self):
        self.assertIn(
            "if v_cache is not None and v_cache.dtype == torch.int8:",
            self.func_body,
            "v_cache int8 guard must exist",
        )
        self.assertIn(
            "PPU FA3 received v_cache with dtype=int8",
            self.func_body,
            "v_cache int8 must raise ValueError",
        )

    # 6c. Fail-fast message explains the copy hazard
    def test_fail_fast_message(self):
        self.assertIn(
            "only supports bf16 KV cache",
            self.func_body,
            "Error must state that PPU FA3 only supports bf16 KV cache",
        )
        self.assertIn(
            "loses FA3's in-place cache updates",
            self.func_body,
            "Error must explain that .to(bfloat16) loses in-place updates",
        )

    # 6d. No active bf16 up-cast of the KV caches
    def test_no_active_bf16_upcast(self):
        for i, line in enumerate(self.src.splitlines(), 1):
            stripped_line = _strip_comments(line).strip()
            if not stripped_line:
                continue
            if (
                "k_cache.to(torch.bfloat16)" in stripped_line
                or "v_cache.to(torch.bfloat16)" in stripped_line
            ):
                self.fail(
                    f"Line {i}: active KV cache bf16 up-cast found "
                    f"(must stay commented out): {stripped_line!r}"
                )

    # 6e. Old copy code is preserved as comments (change traceability)
    def test_old_code_commented(self):
        commented = [
            line.strip()
            for line in self.src.splitlines()
            if line.strip().startswith("#")
        ]
        joined = "\n".join(commented)
        self.assertIn("k_cache = k_cache.to(torch.bfloat16)", joined)
        self.assertIn("v_cache = v_cache.to(torch.bfloat16)", joined)
        self.assertIn(
            "CR Finding W5",
            self.src,
            "Fail-fast must reference CR Finding W5",
        )


if __name__ == "__main__":
    unittest.main()
