"""Opt-in SGL DeepGEMM score/histogram + exact physical TopK, B200 only.

One eagerly created plan owns one ordered execution sequence. Q/cache/weights
keep their addresses; their contents and the supplied lengths/table may change.
Each plan has an actual batch size in 1..128, including non-powers of two. This
API does not change active batch size inside an already captured CUDA graph.
Importing this module neither loads CUDA libraries nor modifies SGLang dispatch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading


HERE = Path(__file__).resolve().parent
MAIN_COMMIT = "1853080cf74589228fcd5dc333197c2ca6df49cf"
PATCH_SHA256 = "4397806ac79b5d03e69761832cc7b55d1ae1035f078460309e47091e57c17548"
ENVELOPE = 1_048_576
TOPK = 2048
_IMPORT_LOCK = threading.RLock()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified(root, relative, digest):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or _sha(path) != digest:
        raise ValueError(f"artifact hash/path mismatch: {path}")
    return path


def load_deepgemm(directory=None):
    """Load a privately built package without replacing installed deep_gemm."""
    with _IMPORT_LOCK:
        return _load_deepgemm(directory)


def _load_deepgemm(directory):
    root = Path(directory or HERE / "build/fused/deepgemm").resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "litetopk-deepgemm-main-fusion-v1"
            or manifest.get("main_commit") != MAIN_COMMIT
            or manifest.get("patch_sha256") != PATCH_SHA256
            or _sha(HERE / "deepgemm/opt_in_histogram.patch") != PATCH_SHA256):
        raise ValueError("expected the pinned main fusion build manifest")
    required = ("deep_gemm/_C.so", "deep_gemm/__init__.py",
                "deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh",
                "deep_gemm/include/deep_gemm/epilogue/coarse_histogram.cuh")
    if any(name not in manifest["staged_files"] for name in required):
        raise ValueError("manifest does not cover required fused package files")
    for name in required[1:]:
        if manifest["staged_files"][name] != manifest["patched_source_sha256"].get(name):
            raise ValueError("staged package differs from the patched source manifest")
    for relative, digest in manifest["staged_files"].items():
        _verified(root, relative, digest)
    library = _verified(root, manifest["library"], manifest["library_sha256"])
    package = (root / manifest["package"]).resolve()
    if package != library.parent:
        raise ValueError("package/library manifest mismatch")
    name = "_litetopk_deepgemm_" + hashlib.sha256(str(package).encode()).hexdigest()[:16]
    if name in sys.modules:
        module = sys.modules[name]
        if getattr(module, "_litetopk_library_sha256", None) != manifest["library_sha256"]:
            raise ValueError("private library changed after import; start a fresh process")
        return module
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ImportError(package)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module._litetopk_library_sha256 = manifest["library_sha256"]
    except Exception:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        raise
    return module


def _tensor(tensor, shape, dtype, device, alignment=4):
    return (tuple(tensor.shape) == tuple(shape) and tensor.dtype == dtype
            and tensor.device == device and tensor.is_contiguous()
            and tensor.data_ptr() % alignment == 0)


class FusedDecodePlan:
    """Reusable buffers for an explicitly selected score + histogram path.

    Instantiate before graph capture, call metadata(lengths) outside timed work,
    then call plan(table, lengths, schedule). Results are physical KV slots in
    an int32[B,2048] tensor owned by this plan and overwritten on the next call.
    Native valid lengths/page IDs/schedule preconditions apply. No host read of
    device tensor values is performed in the execution path.

    Candidate capacity is fixed by the selector build (default 32768 per row).
    Candidate overflow, or a histogram total that differs from a row length
    (e.g. a NaN score), raises a CUDA device error; rebuild with a larger
    --candidate-capacity for inputs that exceed the limit.
    """
    def __init__(self, q, cache, weights, *, deepgemm_package=None,
                 selector_dir=None, max_context_len=ENVELOPE):
        import torch
        import tvm_ffi
        import ctypes
        from cutlass.cute import runtime
        if not q.is_cuda or torch.cuda.is_current_stream_capturing():
            raise ValueError("create the CUDA plan eagerly before capture")
        self.device = q.device
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("select Q's CUDA device before creating a plan")
        self.rows = q.shape[0]
        if (not 1 <= self.rows <= 128
                or not _tensor(q, (self.rows, 1, 32, 128), torch.float8_e4m3fn, self.device, 16)
                or not _tensor(weights, (self.rows, 32), torch.float32, self.device, 16)):
            raise ValueError("expected contiguous FP8 Q[B,1,32,128], FP32 weights[B,32], B1..128")
        if (cache.ndim != 4 or cache.shape[0] < 1 or tuple(cache.shape[1:]) != (64, 1, 132)
                or not _tensor(cache, cache.shape, torch.uint8, self.device, 16)):
            raise ValueError("expected native packed uint8 KV[pages,64,1,132]")
        if type(max_context_len) is not int or not 1 <= max_context_len <= ENVELOPE:
            raise ValueError("max_context_len must be in 1..1048576")
        properties = torch.cuda.get_device_properties(self.device)
        if properties.major != 10 or properties.minor != 0 or properties.multi_processor_count != 148:
            raise ValueError("this selector ABI is qualified for 148-SM B200 only")
        self.q, self.cache, self.weights = q, cache, weights
        self.max_context_len = max_context_len
        with _IMPORT_LOCK, torch.cuda.device(self.device):
            self.deepgemm = load_deepgemm(deepgemm_package)
            bound_device = getattr(self.deepgemm, "_litetopk_device", self.device.index)
            if bound_device != self.device.index:
                raise ValueError("use a separate process for a different CUDA device")
            if self.deepgemm.get_num_sms() != 148:
                raise ValueError("the private DeepGEMM instance must use all 148 SMs")
            self.deepgemm._litetopk_device = self.device.index
        root = Path(selector_dir or HERE / "build/fused").resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        capacity = manifest.get("candidate_capacity")
        if capacity not in (16384, 32768, 65536, 131072):
            raise ValueError("unsupported selector candidate capacity")
        expected = dict(schema="sglang-litetopk-fused-selectors-v4", cutoff=8,
                        histogram_mapping="hybrid1024-fp16rn16-unit-overflow-v1",
                        workspace_layout="fixed-header-pair-v1", workspace_candidate_offset=4096,
                        workspace_tail_offset=2048, workspace_bytes=4096 + 160 * capacity * 8,
                        certificate_miss="error", overflow_policy="error", device_assertions=True)
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("expected the qualified selector pair")
        # The 512-thread route serves B <= cutoff; the 1024-thread route serves larger batches.
        record = manifest["small" if self.rows <= 8 else "large"]
        _verified(root, record["source"], record["source_sha256"])
        library = _verified(root, record["library"], record["library_sha256"])
        self._ffi_libraries = [ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                               for path in runtime.find_runtime_libraries(enable_tvm_ffi=True)]
        self.module = tvm_ffi.load_module(str(library))
        self.fn = self.module.select
        self.candidate_capacity = capacity
        self.histogram = torch.zeros((self.rows, 1024), dtype=torch.int32, device=self.device)
        # 4 KiB of per-row counters, then one slab of `capacity` 8-byte candidates per row.
        self.workspace = torch.zeros((4096 + self.rows * capacity * 8) // 4,
                                     dtype=torch.int32, device=self.device)
        self.output = torch.empty((self.rows, TOPK), dtype=torch.int32, device=self.device)
        self.selector_diag = torch.empty((self.rows, 6), dtype=torch.int32, device=self.device)
        self._active = torch.full((1,), self.rows, dtype=torch.int32, device=self.device)
        self._owners = {}
        self.dense = None

    def metadata(self, lengths):
        """Rebuild outside capture/timing whenever lengths change."""
        import torch
        if not _tensor(lengths, (self.rows,), torch.int32, self.device):
            raise ValueError("expected contiguous int32 lengths[B] on the plan device")
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("select the plan's CUDA device before building metadata")
        return self.deepgemm.get_paged_mqa_logits_metadata(
            lengths.reshape(self.rows, 1), 64, 148, indices=None)

    def __call__(self, table, lengths, schedule):
        import torch
        if (not _tensor(lengths, (self.rows,), torch.int32, self.device)
                or table.ndim != 2 or table.shape[0] != self.rows
                or table.shape[1] < (self.max_context_len + 63) // 64
                or not _tensor(table, table.shape, torch.int32, self.device)):
            raise ValueError("length/page-table shape, device or alignment mismatch")
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("select the plan's CUDA device before calling it")
        # A reused graph requires in-place metadata updates, not new pointers.
        # Hold capture inputs and returned dense buffers for the graph lifetime.
        dense = self.deepgemm.fp8_fp4_paged_mqa_logits(
            (self.q, None), self.cache, self.weights, lengths.reshape(self.rows, 1), table, schedule,
            self.max_context_len, clean_logits=False, indices=None, histogram=self.histogram)
        self.dense = dense
        # Native output is a logical slice of a 256-aligned allocation. Expose
        # its full row stride to the compact-tensor selector ABI without a copy.
        selector_scores = dense.as_strided((self.rows, dense.stride(0)), dense.stride())
        self.fn(selector_scores, self.output, self.workspace, self.max_context_len, lengths,
                self.histogram.reshape(-1), self.selector_diag, table.reshape(-1), table.shape[1],
                self._active)
        if torch.cuda.is_current_stream_capturing():
            self._owners[(table.data_ptr(), lengths.data_ptr(), schedule.data_ptr(),
                          dense.data_ptr())] = (table, lengths, schedule, dense)
        return self.output
