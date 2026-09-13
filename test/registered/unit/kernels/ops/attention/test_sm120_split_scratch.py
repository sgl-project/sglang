"""The SM120 sparse-MLA page-split scratch is zero-initialized.

``_split_kv_pages_to_64`` in ``sglang/kernels/ops/attention/flash_mla_sm120.py``
re-pages the 256-token SWA pool into a persistent, grow-only 64-token scratch
and copies only the source pages a step references. FlashInfer <= 0.6.18's
SM120 sparse-MLA prefill and decode kernels clamp every masked (-1) candidate
index to slot 0 and gather that slot's bytes with only the score masked, so a
``torch.empty`` scratch whose page 0 is never written hands the kernel whatever
the caching allocator recycled there: NaN-encoded fp8 bytes turn into
``P(0) * V(NaN) = NaN`` for every query row with -1 padding (DeepSeek-V4 on
RTX PRO 6000: cold prompts of 65+ tokens returned garbage, 64 were correct).

These tests pin the ``torch.zeros`` allocation without a GPU, torch or triton:
the module is loaded by file path with a recording ``torch`` stub and
passthrough ``triton`` / ``sglang.srt`` stubs (the way ``unit/tools/`` loads
``ci_register``), and ``_split_kv_pages_to_64`` is driven directly with its two
Triton kernels replaced by launch recorders.

    python test/registered/unit/kernels/ops/attention/test_sm120_split_scratch.py
"""

import ast
import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[6]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
MODULE_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "kernels"
    / "ops"
    / "attention"
    / "flash_mla_sm120.py"
)
MODULE_NAME = "flash_mla_sm120_under_test"

# The DSv4 row and the 64-token split page the module hard-codes.
BYTES_PER_TOKEN = 584
SPLIT_PAGE_BYTES = 37440
SRC_PBS = 256
DST_PBS = 64
RATIO = SRC_PBS // DST_PBS


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StubModule(ModuleType):
    """A module whose every attribute is a permissive callable.

    Covers ``@triton.jit`` (returns the decorated function), ``tl.constexpr``
    in kernel annotations, ``triton.cdiv`` and ``torch.*`` names in type
    annotations.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def _passthrough(*args, **kwargs):
            return args[0] if args else _passthrough

        return _passthrough


class _FakeTensor:
    """The slice of tensor surface ``_split_kv_pages_to_64`` touches."""

    def __init__(
        self,
        shape,
        dtype="uint8",
        device="cuda:0",
        strides=None,
        fill=None,
        origin=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._strides = strides
        self.fill = fill  # 0 for torch.zeros, None for torch.empty
        self.origin = origin  # the tensor this one is a view of
        self.zeroed = 0

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def stride(self, dim):
        if self._strides is not None:
            return self._strides[dim]
        strides, acc = [], 1
        for d in reversed(self.shape):
            strides.append(acc)
            acc *= d
        return list(reversed(strides))[dim]

    def __getitem__(self, item):
        assert isinstance(item, slice) and item.start is None and item.step is None
        n = self.shape[0] if item.stop is None else item.stop
        return _FakeTensor(
            (n,) + self.shape[1:], self.dtype, self.device, fill=self.fill, origin=self
        )

    def as_strided(self, size, stride):
        return _FakeTensor(
            size, self.dtype, self.device, strides=stride, fill=self.fill, origin=self
        )

    def reshape(self, *shape):
        return self

    def contiguous(self):
        return self

    def to(self, dtype):
        return _FakeTensor(self.shape, dtype, self.device, fill=self.fill, origin=self)

    def zero_(self):
        self.zeroed += 1
        self.fill = 0
        return self


class _FakeTorch(_StubModule):
    """Permissive torch stub that records ``zeros`` / ``empty`` allocations."""

    uint8 = "uint8"
    int8 = "int8"
    int32 = "int32"
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __init__(self):
        super().__init__("torch")
        self.zeros_calls = []
        self.empty_calls = []

    def zeros(self, *shape, dtype=None, device=None):
        self.zeros_calls.append((shape, {"dtype": dtype, "device": device}))
        return _FakeTensor(shape, dtype, device, fill=0)

    def empty(self, *shape, dtype=None, device=None):
        self.empty_calls.append((shape, {"dtype": dtype, "device": device}))
        return _FakeTensor(shape, dtype, device, fill=None)

    def inference_mode(self, enabled=True):
        return contextlib.nullcontext()

    def as_strided(self, tensor, size, stride):
        return tensor.as_strided(size, stride)


class _Launchable:
    """Stands in for a ``@triton.jit`` kernel: ``kernel[grid](*args)``."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def _launch(*args):
            self.launches.append((grid, args))

        return _launch


class _Knob:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _package(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


@contextlib.contextmanager
def _loaded():
    """Load flash_mla_sm120.py against stubs; yield (module, torch_stub, buffers).

    ``buffers`` is the process buffer registry (``get_resources().buffers``)
    the split scratch is stored in. The two Triton kernels are replaced by
    launch recorders so the function can run without a GPU.
    """
    environ = ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace(SGLANG_SM120_FLASHMLA_BACKEND=_Knob("flashinfer"))
    utils = ModuleType("sglang.srt.utils")
    utils.is_hip = lambda: False
    buffers = {}
    runtime_context = ModuleType("sglang.srt.runtime_context")
    runtime_context.get_resources = lambda: SimpleNamespace(buffers=buffers)

    torch_stub = _FakeTorch()
    triton_stub = _StubModule("triton")
    triton_language = _StubModule("triton.language")
    # ``import triton.language as tl`` binds ``tl`` via getattr on the parent.
    triton_stub.language = triton_language

    stubs = {
        "torch": torch_stub,
        "triton": triton_stub,
        "triton.language": triton_language,
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.environ": environ,
        "sglang.srt.utils": utils,
        "sglang.srt.runtime_context": runtime_context,
    }
    with patch.dict(sys.modules, stubs):
        module = _load_module(MODULE_NAME, MODULE_PATH)
        module._page_split_kernel = _Launchable()
        module._page_mark_kernel = _Launchable()
        yield module, torch_stub, buffers


def _split(module, num_src_pages, num_rows=65, device="cuda:0"):
    # A pbs=256 SWA pool view (N, 256, 1, 584) and the (rows, topk) int32
    # index tensor the backend passes, -1 padded (values are not read here).
    kv_u8 = _FakeTensor((num_src_pages, SRC_PBS, 1, BYTES_PER_TOKEN), device=device)
    idx = _FakeTensor((num_rows, 2048), dtype="int32", device=device)
    return module._split_kv_pages_to_64(kv_u8, SRC_PBS, touched_indices=idx)


def _uint8_empty_calls(torch_stub):
    return [c for c in torch_stub.empty_calls if c[1]["dtype"] == "uint8"]


def _split_key(device="cuda:0"):
    return f"flash_mla_sm120_split:{device}"


class TestSplitScratchIsZeroInitialized(unittest.TestCase):
    def test_first_allocation_is_torch_zeros_and_is_registered(self):
        with _loaded() as (module, torch_stub, buffers):
            out = _split(module, num_src_pages=3)
            num_dst_pages = 3 * RATIO
            self.assertEqual(
                torch_stub.zeros_calls,
                [
                    (
                        (num_dst_pages, SPLIT_PAGE_BYTES),
                        {"dtype": "uint8", "device": "cuda:0"},
                    )
                ],
            )
            # The only torch.empty is the int8 page mask, which is zeroed per call.
            self.assertEqual(_uint8_empty_calls(torch_stub), [])
            self.assertEqual([c[1]["dtype"] for c in torch_stub.empty_calls], ["int8"])
            buf = buffers[_split_key()]
            self.assertEqual(buf.fill, 0)
            self.assertEqual(buf.shape, (num_dst_pages, SPLIT_PAGE_BYTES))
            # The split kernel writes into a slice of that zeroed buffer, masked
            # to the touched pages only (which is why page 0 needs the zeros).
            ((grid, args),) = module._page_split_kernel.launches
            self.assertEqual(grid, (num_dst_pages,))
            dst = args[1]
            self.assertIs(dst.origin, buf)
            self.assertEqual(dst.shape, (num_dst_pages, SPLIT_PAGE_BYTES))
            self.assertIs(args[-1], True)  # HAS_MASK
            # The returned view addresses the same zeroed buffer as 64-token pages.
            self.assertIs(out.origin.origin, buf)
            self.assertEqual(out.shape, (num_dst_pages, DST_PBS, 1, BYTES_PER_TOKEN))
            self.assertEqual(
                out._strides, (SPLIT_PAGE_BYTES, BYTES_PER_TOKEN, BYTES_PER_TOKEN, 1)
            )

    def test_same_or_smaller_pool_reuses_the_buffer(self):
        with _loaded() as (module, torch_stub, buffers):
            _split(module, num_src_pages=3)
            first = buffers[_split_key()]
            _split(module, num_src_pages=3)
            out_small = _split(module, num_src_pages=2)
            self.assertEqual(len(torch_stub.zeros_calls), 1)
            self.assertEqual(_uint8_empty_calls(torch_stub), [])
            self.assertIs(buffers[_split_key()], first)
            # The smaller call is served from a prefix slice of the same buffer.
            self.assertIs(out_small.origin.origin, first)
            self.assertEqual(out_small.shape[0], 2 * RATIO)

    def test_grow_reallocates_with_torch_zeros(self):
        with _loaded() as (module, torch_stub, buffers):
            _split(module, num_src_pages=3)
            first = buffers[_split_key()]
            _split(module, num_src_pages=5)
            grown = buffers[_split_key()]
            self.assertIsNot(grown, first)
            self.assertEqual(grown.fill, 0)
            self.assertEqual(
                torch_stub.zeros_calls[1],
                ((5 * RATIO, SPLIT_PAGE_BYTES), {"dtype": "uint8", "device": "cuda:0"}),
            )
            self.assertEqual(len(torch_stub.zeros_calls), 2)
            self.assertEqual(_uint8_empty_calls(torch_stub), [])
            # The launch after the grow writes into the new zeroed buffer.
            self.assertIs(module._page_split_kernel.launches[-1][1][1].origin, grown)

    def test_one_scratch_per_device(self):
        with _loaded() as (module, torch_stub, buffers):
            _split(module, num_src_pages=3, device="cuda:0")
            _split(module, num_src_pages=3, device="cuda:1")
            devices = [c[1]["device"] for c in torch_stub.zeros_calls]
            self.assertEqual(devices, ["cuda:0", "cuda:1"])
            self.assertEqual(
                sorted(k for k in buffers if k.startswith("flash_mla_sm120_split:")),
                [_split_key("cuda:0"), _split_key("cuda:1")],
            )

    def test_page_mask_is_zeroed_on_every_call(self):
        with _loaded() as (module, _, buffers):
            _split(module, num_src_pages=3)
            _split(module, num_src_pages=3)
            mbuf = buffers["flash_mla_sm120_mask:cuda:0"]
            self.assertEqual(mbuf.dtype, "int8")
            self.assertEqual(len(module._page_mark_kernel.launches), 2)
            for _, args in module._page_mark_kernel.launches:
                mask = args[1]
                self.assertIs(mask.origin, mbuf)
                self.assertEqual(mask.zeroed, 1)
                self.assertEqual(args[3], SRC_PBS)  # SRC_PBS constexpr

    def test_64_token_pool_needs_no_scratch(self):
        with _loaded() as (module, torch_stub, buffers):
            kv_u8 = _FakeTensor((3, DST_PBS, 1, BYTES_PER_TOKEN))
            idx = _FakeTensor((65, 2048), dtype="int32")
            self.assertIs(module._split_kv_pages_to_64(kv_u8, DST_PBS, idx), kv_u8)
            self.assertEqual(torch_stub.zeros_calls, [])
            self.assertEqual(torch_stub.empty_calls, [])
            self.assertEqual(buffers, {})


def _calls_named(tree, dotted):
    """All ``ast.Call`` nodes whose callee unparses to ``dotted``."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted
    ]


def _kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return ast.unparse(kw.value)
    return None


class TestSplitScratchSourcePins(unittest.TestCase):
    """AST pins so the fix survives a stub-independent read of the module."""

    @classmethod
    def setUpClass(cls):
        cls.source = MODULE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        (cls.split_fn,) = [
            node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_split_kv_pages_to_64"
        ]

    def test_split_scratch_is_the_only_uint8_allocation_and_uses_zeros(self):
        zeros_u8 = [
            c
            for c in _calls_named(self.tree, "torch.zeros")
            if _kwarg(c, "dtype") == "torch.uint8"
        ]
        empty_u8 = [
            c
            for c in _calls_named(self.tree, "torch.empty")
            if _kwarg(c, "dtype") == "torch.uint8"
        ]
        self.assertEqual(len(zeros_u8), 1)
        self.assertEqual(
            empty_u8, [], "a torch.empty uint8 buffer would re-expose slot 0"
        )
        # ... and that one allocation is the split scratch inside
        # _split_kv_pages_to_64, sized (num_dst_pages, _BYTES_PER_DST_PAGE_PADDED)
        # on the split's device.
        (call,) = zeros_u8
        self.assertEqual(_calls_named(self.split_fn, "torch.zeros"), [call])
        self.assertEqual(
            [ast.unparse(a) for a in call.args],
            ["num_dst_pages", "_BYTES_PER_DST_PAGE_PADDED"],
        )
        self.assertEqual(_kwarg(call, "device"), "dev")
        # No torch.empty inside the split function except the int8 page mask.
        empties = _calls_named(self.split_fn, "torch.empty")
        self.assertEqual([_kwarg(c, "dtype") for c in empties], ["torch.int8"])

    def test_allocation_comment_documents_the_slot_0_gather(self):
        segment = ast.get_source_segment(self.source, self.split_fn)
        for needle in ("torch.zeros(", "slot 0", "masked", "FlashInfer"):
            self.assertIn(needle, segment)
        self.assertNotIn("stale data", segment)


if __name__ == "__main__":
    unittest.main()
