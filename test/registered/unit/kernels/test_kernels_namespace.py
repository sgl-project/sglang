"""GPU-free import / registry / selector tests for ``sglang.kernels`` (RFC #29630)."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import sglang.kernels as K
import sglang.kernels.fused_op as fo
import sglang.kernels.selector as sel
from sglang.kernels import KernelBackend, PlatformInfo
from sglang.kernels.spec import CapabilityRequirement as Cap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=24, suite="base-a-test-cpu")

_CPU = PlatformInfo(device_type="cpu")
_SM90 = PlatformInfo(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0)
_SM100 = PlatformInfo(device_type="cuda", cuda_arch_major=10, cuda_arch_minor=0)
_HIP = PlatformInfo(device_type="hip")


def test_single_backend_resolves_without_backend():
    assert (
        K.select_kernel("kvcache.reshape_and_cache_flash").backend
        is KernelBackend.TRITON
    )


def test_fp8_scaled_mm_requires_explicit_registry_backend(monkeypatch):
    monkeypatch.setattr(sel, "_platform", lambda: _SM90)
    with pytest.raises(ValueError, match="multiple backends"):
        K.select_kernel("gemm.fp8_scaled_mm")
    assert (
        K.select_kernel("gemm.fp8_scaled_mm", backend=KernelBackend.AOT).backend
        is KernelBackend.AOT
    )


def test_unknown_op_or_backend_raises():
    with pytest.raises(KeyError):
        K.select_kernel("does_not.exist")
    with pytest.raises(KeyError):
        K.select_kernel("gemm.fp8_scaled_mm", backend=KernelBackend.TRITON)


@pytest.mark.parametrize(
    "platform, eligible",
    [
        (_SM90, True),
        (_SM100, True),
        (PlatformInfo(device_type="cuda", cuda_arch_major=8, cuda_arch_minor=0), False),
        (_CPU, False),
        (_HIP, False),
    ],
)
def test_qwen_qkv_registry_accepts_hopper(platform, eligible):
    spec = K.select_kernel("diffusion.qwen_qkv_epilogue")
    assert spec.backend is KernelBackend.JIT
    assert K.capabilities_satisfied(spec.capabilities, platform) is eligible


def test_multi_backend_requires_explicit_backend(monkeypatch):
    # Device is a hard eligibility filter, not a ranking: >1 usable backend on
    # the current device means selection must name one.
    monkeypatch.setattr(sel, "_platform", lambda: _SM90)
    with pytest.raises(ValueError):
        K.select_kernel("layernorm.rmsnorm")
    spec = K.select_kernel("layernorm.rmsnorm", backend=KernelBackend.JIT)
    assert spec.backend is KernelBackend.JIT
    assert spec.target == "sglang.kernels.ops.layernorm:_RMSNORM.forward_jit"


@pytest.mark.parametrize("device, expect", [("cuda", "jit"), ("hip", "aot")])
def test_activation_default_backend(monkeypatch, device, expect):
    # silu_and_mul default matches production: jit on CUDA, aot (sgl_kernel) on HIP.
    from sglang.kernels.ops.activation import _SILU_AND_MUL

    monkeypatch.setattr(fo, "_platform", lambda: PlatformInfo(device_type=device))
    assert _SILU_AND_MUL.auto_selected_backend().value == expect


@pytest.mark.parametrize(
    "op_attr, device, expect",
    [
        ("_RMSNORM", "cuda", "aot"),
        ("_RMSNORM", "hip", "aiter"),
        ("_RMSNORM", "npu", "torch_npu"),
        ("_GEMMA_RMSNORM", "cuda", "aot"),
        ("_GEMMA_RMSNORM", "hip", "jit"),  # rocm-triton JIT pinned to HIP
        ("_GEMMA_RMSNORM", "npu", "torch_npu"),
    ],
)
def test_layernorm_default_backend(monkeypatch, op_attr, device, expect):
    # Same AOT provenance, different device coverage per op: rmsnorm's AOT is
    # CUDA-only, so HIP falls to aiter and NPU to torch_npu.
    ln = importlib.import_module("sglang.kernels.ops.layernorm")
    monkeypatch.setattr(fo, "_platform", lambda: PlatformInfo(device_type=device))
    assert getattr(ln, op_attr).auto_selected_backend().value == expect


@pytest.mark.parametrize(
    "req, plat, ok",
    [
        (Cap.CUDA, _CPU, False),
        (Cap.CUDA, _SM90, True),
        (Cap.CUDA, _HIP, False),
        (Cap.HIP, _HIP, True),
        (Cap.cuda(min_sm=(10, 0)), _SM90, False),
        (Cap.cuda(min_sm=(10, 0)), _SM100, True),
        (Cap.cuda(max_sm=(9, 0)), _SM100, False),
    ],
)
def test_capability_is_satisfied_by(req, plat, ok):
    assert req.is_satisfied_by(plat) is ok


def test_capabilities_or_semantics():
    both = {Cap.CUDA, Cap.HIP}
    assert K.capabilities_satisfied(both, _SM90)
    assert K.capabilities_satisfied(both, _HIP)
    assert not K.capabilities_satisfied(both, _CPU)
    assert K.capabilities_satisfied((), _CPU)  # empty = unrestricted
    assert K.capabilities_satisfied(Cap.CUDA, _SM90)  # single tolerated


@pytest.mark.parametrize(
    "relative_path",
    (
        "srt/utils/common.py",
        "multimodal_gen/runtime/utils/common.py",
    ),
)
def test_amx_backend_probe_is_lazy(relative_path):
    loader = (
        "package = importlib.util.find_spec('sglang'); "
        "path = pathlib.Path(next(iter(package.submodule_search_locations))) / "
        f"{relative_path!r}; "
        "spec = importlib.util.spec_from_file_location('_common_under_test', path); "
        "module = importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name] = module; "
        "spec.loader.exec_module(module)"
    )
    code = "; ".join(
        (
            "import builtins, importlib.util, pathlib, sys",
            "from unittest import mock",
            "real_import = builtins.__import__",
            "import_mock = mock.Mock(wraps=real_import)",
            "builtins.__import__ = import_mock",
            loader,
            "builtins.__import__ = real_import",
            "attempted = any(call.args and call.args[0] == 'sgl_kernel' "
            "for call in import_mock.call_args_list)",
            "print('DIRTY' if attempted else 'CLEAN')",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_import_stays_metadata_only():
    # Importing the namespace must not pull in the AOT backend (sgl_kernel) or
    # the JIT compilation infra (sglang.kernels.jit), which import torch / nvcc.
    code = (
        "import sys, sglang.kernels.ops; "
        "print('DIRTY' if any(m in sys.modules for m in "
        "('sgl_kernel', 'cutlass', 'flydsl', 'aiter', "
        "'sglang.kernels.ops.gemm.kimi_k3')) or any("
        "m.startswith('sglang.kernels.jit') for m in sys.modules) else 'CLEAN')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "CLEAN" in r.stdout


@pytest.mark.parametrize(
    "op, backend, device",
    [
        ("activation.situ_and_mul", KernelBackend.JIT, "cuda"),
        ("moe.situ_and_mul_masked_post_quant", KernelBackend.JIT, "cuda"),
        ("attention.attn_res_fused_tma", KernelBackend.JIT, "cuda"),
        ("attention.attn_res_hip", KernelBackend.TRITON, "hip"),
        ("attention.kimi_k3_mla_output_gate", KernelBackend.JIT, "cuda"),
        ("attention.fused_kda_decode_mtp_dspark", KernelBackend.CUTE_DSL, "cuda"),
        ("attention.flydsl_kimi_k3_kda_decode", KernelBackend.FLYDSL, "hip"),
        ("attention.flydsl_kimi_k3_kda_decode_with_f_b", KernelBackend.FLYDSL, "hip"),
        ("communication.all_reduce_push_res", KernelBackend.JIT, "cuda"),
        ("communication.gemm_ag_up_proj", KernelBackend.JIT, "cuda"),
        ("communication.reduce_scatter_res", KernelBackend.JIT, "cuda"),
    ],
)
def test_kimi_k3_kernels_are_inventoried_by_operator(op, backend, device):
    spec = K.select_kernel(op, backend=backend)
    group = op.split(".")[0]
    assert spec.target.startswith(f"sglang.kernels.ops.{group}.")
    assert K.capabilities_satisfied(spec.capabilities, PlatformInfo(device_type=device))
    assert not K.capabilities_satisfied(spec.capabilities, _CPU)


def test_kimi_k3_model_namespace_is_retired():
    assert importlib.util.find_spec("sglang.kernels.ops.kimi_k3") is None


def test_operator_and_test_groups_agree():
    """A new root-level model bundle must not bypass logical op grouping."""
    root = Path(K.__file__).resolve().parents[3]
    ops = root / "python/sglang/kernels/ops"
    groups = set(K.ops.__all__)
    assert {p.name for p in ops.glob("*.py")} == {"__init__.py"}
    assert {p.name for p in ops.iterdir() if (p / "__init__.py").is_file()} == groups
    for kind in ("ops", "benchmark"):
        tests = root / "test/registered/kernels" / kind
        actual = {p.relative_to(tests).parts[0] for p in tests.rglob("*.py")}
        assert actual <= groups, actual - groups


def test_inventory_targets_survive_module_moves():
    """Moving a module without its lazy target otherwise breaks only on first call."""
    root = Path(K.__file__).resolve().parents[3] / "python"
    for spec in K.registry.all_specs():
        module, _ = spec.target.split(":", 1)
        if not module.startswith("sglang.kernels.ops."):
            continue
        path = root.joinpath(*module.split("."))
        assert path.with_suffix(".py").is_file() or (path / "__init__.py").is_file(), (
            spec
        )


def test_reclassified_public_entry_points_are_inventoried():
    """Public compute functions in these formerly unregistered modules need metadata."""
    root = Path(K.__file__).resolve().parent / "ops"
    targets = {spec.target for spec in K.registry.all_specs()}
    modules = (
        "attention.minicpm_sala.get_block_table",
        "attention.fast_topk",
        "attention.dsa.kpool_topk_transform",
        "attention.dsa.indexer_k",
        "attention.dsv4.wo_a",
        "embeddings.qwen4_ngram",
        "elementwise.qwen4_gate",
        "elementwise.row_scale",
        "mamba.qwen4_short_conv",
        "mamba.lfm_short_conv",
        "gemm.dsv4_wo_a",
        "gemm.inkling_rel_proj",
        "gemm.hopper_bf16_gemv",
        "gemm.sm120_fp8_gemv",
        "gemm.fp8_blockwise_gemm",
        "gemm.gptq_marlin",
        "layernorm.rmsnorm_hf",
        "layernorm.grouped_gemma_rmsnorm",
        "moe.dsv4",
        "moe.gemma4_routing",
        "memory.adler32",
        "memory.row_compact",
        "mm.process.image",
    )
    for module in modules:
        tree = ast.parse(
            root.joinpath(*module.split(".")).with_suffix(".py").read_text()
        )
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if (
                node.name.startswith(("_", "can_", "is_", "use_"))
                or node.name == "make_name"
            ):
                continue
            assert f"sglang.kernels.ops.{module}:{node.name}" in targets


@pytest.mark.parametrize(
    "op, sm, expected",
    [
        ("gemm.hopper_bf16_gemv", (9, 0), True),
        ("gemm.hopper_bf16_gemv", (10, 0), False),
        ("gemm.fp8_blockwise_scaled_mm", (12, 0), True),
        ("gemm.fp8_blockwise_scaled_mm", (12, 1), True),
        ("gemm.fp8_blockwise_scaled_mm", (10, 0), False),
        ("attention.fused_rope_wo_a_bf16", (9, 0), False),
        ("attention.fused_rope_wo_a_bf16", (10, 0), True),
        ("attention.fused_rope_wo_a_bf16", (10, 3), True),
        ("attention.fused_rope_wo_a_bf16", (12, 0), False),
        ("attention.deep_select_topk", (8, 0), False),
        ("attention.deep_select_topk", (9, 0), True),
        ("attention.deep_select_topk", (10, 0), True),
        ("attention.deep_select_topk", (10, 1), False),
        ("attention.deep_select_topk", (10, 3), True),
        ("attention.deep_select_topk", (12, 0), False),
    ],
)
def test_registered_architecture_boundaries(op, sm, expected):
    spec = K.registry.get_backend(op, KernelBackend.JIT)
    platform = PlatformInfo(
        device_type="cuda", cuda_arch_major=sm[0], cuda_arch_minor=sm[1]
    )
    assert K.capabilities_satisfied(spec.capabilities, platform) is expected
    assert not K.capabilities_satisfied(spec.capabilities, _CPU)


def test_deep_select_spec_matches_wrapper_architectures():
    """The group cannot import the wrapper, so this is the only link between the two SM lists."""
    from sglang.kernels.ops.attention import deep_select

    spec = K.registry.get_backend("attention.deep_select_topk", KernelBackend.JIT)
    assert spec.capabilities == frozenset(
        Cap.cuda(min_sm=sm, max_sm=sm) for sm in deep_select._SUPPORTED_CAPABILITIES
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
