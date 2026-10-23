"""GPU-free import / registry / selector tests for ``sglang.kernels`` (RFC #29630)."""

import ast
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
_MUSA = PlatformInfo(device_type="musa")


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


def test_gguf_vector_registry_is_cuda_local_jit():
    spec = K.select_kernel("moe.gguf_moe_a8_vec")
    assert spec.backend is KernelBackend.JIT
    assert spec.target == "sglang.kernels.ops.moe.gguf_moe_vec:gguf_moe_a8_vec"
    assert K.capabilities_satisfied(spec.capabilities, _SM90)
    assert not K.capabilities_satisfied(spec.capabilities, _MUSA)
    assert not K.capabilities_satisfied(spec.capabilities, _CPU)


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
        (_MUSA, False),
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
        K.select_kernel("activation.silu_and_mul")
    spec = K.select_kernel("activation.silu_and_mul", backend=KernelBackend.JIT)
    assert spec.backend is KernelBackend.JIT
    assert spec.target == "sglang.kernels.ops.activation:_SILU_AND_MUL.forward_jit"


@pytest.mark.parametrize("device, expect", [("cuda", "jit"), ("hip", "aot")])
def test_activation_default_backend(monkeypatch, device, expect):
    # silu_and_mul default matches production: jit on CUDA, aot (sgl_kernel) on HIP.
    from sglang.kernels.ops.activation import _SILU_AND_MUL

    monkeypatch.setattr(fo, "_platform", lambda: PlatformInfo(device_type=device))
    assert _SILU_AND_MUL.auto_selected_backend().value == expect


@pytest.mark.parametrize(
    "req, plat, ok",
    [
        (Cap.CUDA, _CPU, False),
        (Cap.CUDA, _SM90, True),
        (Cap.CUDA, _HIP, False),
        (Cap.HIP, _HIP, True),
        (Cap.MUSA, _MUSA, True),
        (Cap.MUSA, _CPU, False),
        (Cap.MUSA, _SM90, False),
        (Cap.MUSA, _HIP, False),
        (Cap.CUDA, _MUSA, False),
        (Cap.HIP, _MUSA, False),
        (Cap.cuda(min_sm=(10, 0)), _SM90, False),
        (Cap.cuda(min_sm=(10, 0)), _SM100, True),
        (Cap.cuda(max_sm=(9, 0)), _SM100, False),
    ],
)
def test_capability_is_satisfied_by(req, plat, ok):
    assert req.is_satisfied_by(plat) is ok


@pytest.mark.parametrize(
    "musa_version, cuda_available, musa_available, expected_device",
    [
        ("4.0", True, None, "musa"),
        (None, True, None, "cuda"),
        ("4.0", False, None, "cpu"),
        ("4.0", False, True, "musa"),
    ],
)
def test_platform_detection_distinguishes_musa_cuda_alias(
    monkeypatch, musa_version, cuda_available, musa_available, expected_device
):
    """Native MUSA remains eligible before the CUDA compatibility alias is ready."""
    torch = SimpleNamespace(
        version=SimpleNamespace(hip=None, musa=musa_version),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_capability=lambda: (9, 0),
        ),
    )
    if musa_available is not None:
        torch.musa = SimpleNamespace(is_available=lambda: musa_available)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert PlatformInfo.detect().device_type == expected_device


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
        "'sglang.kernels.ops.gemm.kimi_k3', "
        "'sglang.kernels.ops.activation.softcap', "
        "'sglang.kernels.ops.attention.dllm_kv_pack', "
        "'sglang.kernels.ops.attention.flash_attention_v4', "
        "'sglang.kernels.ops.attention.gemma_qkv_norm_rope', "
        "'sglang.kernels.ops.layernorm.rmsnorm_fanout', "
        "'sglang.kernels.ops.speculative.row_argmax')) or any("
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
        ("elementwise.kimi_k3_mla_output_gate", KernelBackend.JIT, "cuda"),
        ("attention.fused_kda_decode_mtp_dspark", KernelBackend.CUTE_DSL, "cuda"),
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


def _readme() -> str:
    return (Path(K.__file__).resolve().parent / "README.md").read_text()


def test_readme_operator_groups_match_registry():
    """The README group list is the documented contract for ``ops._GROUPS``."""
    paragraph = _readme().split("Operator groups", 1)[1].split("\n\n", 1)[0]
    documented = re.findall(r"`([a-z_0-9]+)`", paragraph)
    assert len(documented) == len(set(documented)), documented
    assert set(documented) == set(K.ops.__all__) == set(K.ops._GROUPS)


def test_readme_layout_lists_every_top_level_directory():
    """A new directory directly under ``kernels/`` must be documented first."""
    layout = _readme().split("## Layout", 1)[1].split("```", 2)[1]
    documented = set(re.findall(r"^  ([a-z_0-9]+)/", layout, re.MULTILINE))
    root = Path(K.__file__).resolve().parent
    actual = {p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert documented == actual


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


def test_agent_kernels_register_their_backend():
    """Agent-generated kernels are listed by backend, so directory and backend must agree."""
    agent_backends = {
        "sglang.kernels.kda_kernels.": KernelBackend.KDA,
        "sglang.kernels.cake_kernels.": KernelBackend.CAKE,
    }
    for spec in K.registry.all_specs():
        expected = next(
            (b for p, b in agent_backends.items() if spec.target.startswith(p)), None
        )
        if expected is not None or spec.backend in agent_backends.values():
            assert spec.backend is expected, spec


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
        "attention.dllm_kv_pack",
        "attention.gemma_qkv_norm_rope",
        "attention.minicpm_sala.get_block_table",
        "attention.fast_topk",
        "attention.dsa.kpool_topk_transform",
        "attention.dsa.indexer_k",
        "attention.dsv4.wo_a",
        "attention.helion.kda_replayssm",
        "attention.helion.kda_verify",
        "attention.minimax_sparse.common.index",
        "attention.minimax_sparse.decode.indexer_cp",
        "attention.minimax_sparse.decode.topk_sparse",
        "attention.minimax_sparse.prefill.gluon_prefill",
        "attention.minimax_sparse.prefill.topk_sparse",
        "embeddings.qwen4_ngram",
        "elementwise.qwen4_gate",
        "elementwise.row_scale",
        "elementwise.hc_combine_decode",
        "mamba.qwen4_short_conv",
        "mamba.lfm_short_conv",
        "mamba.kda_prefill_track",
        "gemm.dsv4_wo_a",
        "gemm.inkling_rel_proj",
        "gemm.hopper_bf16_gemv",
        "gemm.sm120_fp8_gemv",
        "gemm.fp8_blockwise_gemm",
        "gemm.gptq_marlin",
        "layernorm.rmsnorm_hf",
        "layernorm.rmsnorm_fanout",
        "layernorm.grouped_gemma_rmsnorm",
        "moe.dsv4",
        "activation.silu_and_mul_clamp",
        "moe.gemma4_routing",
        "moe.shared_expert_gate",
        "memory.adler32",
        "memory.row_compact",
        "kv_canary.plan.api",
        "kv_canary.write",
        "kv_canary.verify",
        "kv_canary.scatter_req_token_ids",
        "mm.process.image",
        "speculative.row_argmax",
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


def _module_defines(module: str, attr: str) -> bool:
    """AST check that ``module`` defines ``attr`` (``name`` or ``name.method``).

    For ``instance.method`` the instance's class must define ``method``.
    """
    root = Path(K.__file__).resolve().parents[2]
    tree = ast.parse(root.joinpath(*module.split(".")).with_suffix(".py").read_text())
    top = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top[target.id] = node.value
    name, _, method = attr.partition(".")
    node = top.get(name)
    if node is None or not method:
        return node is not None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        node = top.get(node.func.id)
    return isinstance(node, ast.ClassDef) and any(
        isinstance(item, ast.FunctionDef) and item.name == method for item in node.body
    )


@pytest.mark.parametrize(
    "op, backends",
    [
        ("attention.muse_spark_qk_norm_rope_store", {"torch", "triton", "helion"}),
        ("communication.muse_spark_all_gather", {"jit"}),
        ("communication.muse_spark_all_gather_norm_residual", {"jit"}),
        (
            "communication.muse_spark_moe_finalize_all_reduce",
            {"torch", "cute_dsl", "helion"},
        ),
        ("communication.all_gather_matmul", {"cute_dsl"}),
        ("communication.matmul_reduce_scatter", {"cute_dsl"}),
        ("elementwise.row_sum", {"jit"}),
        ("gemm.muse_spark_router_gemm", {"jit"}),
        ("gemm.muse_spark_router_gemv", {"cute_dsl"}),
        ("layernorm.muse_spark_norm_gate", {"torch", "jit", "helion", "triton"}),
        ("layernorm.muse_spark_norm_residual", {"torch", "jit", "triton"}),
        ("layernorm.muse_spark_rmsnorm_nvfp4_quant", {"torch", "jit", "helion"}),
        ("moe.muse_spark_routing", {"torch", "jit", "triton"}),
        ("moe.muse_spark_post_expert_finalize", {"triton"}),
    ],
)
def test_muse_spark_kernels_are_inventoried_by_operator(op, backends):
    specs = [s for s in K.registry.all_specs() if s.op == op]
    assert {s.backend.value for s in specs} == backends
    group = op.split(".")[0]
    for spec in specs:
        module, attr = spec.target.split(":", 1)
        assert module.startswith(f"sglang.kernels.ops.{group}."), spec
        assert _module_defines(module, attr), spec
        if spec.backend is not KernelBackend.TORCH:
            assert not K.capabilities_satisfied(spec.capabilities, _CPU), spec


@pytest.mark.parametrize(
    "op, sm, expected",
    [
        ("gemm.hopper_bf16_gemv", (9, 0), True),
        ("gemm.hopper_bf16_gemv", (10, 0), False),
        ("elementwise.hc_combine_apply_norm", (8, 9), False),
        ("elementwise.hc_combine_apply_norm", (9, 0), True),
        ("communication.moe_finalize_shared_gate_all_reduce", (8, 9), False),
        ("communication.moe_finalize_shared_gate_all_reduce", (10, 3), True),
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
        ("attention.chunk_kda_fwd", (9, 0), False),
        ("attention.chunk_kda_fwd", (10, 0), True),
        ("attention.chunk_kda_fwd", (10, 1), False),
        ("attention.chunk_kda_fwd", (10, 3), True),
        ("attention.chunk_kda_fwd", (12, 0), False),
        ("attention.absorbed_bmm_concat_cast_q_fp8", (9, 0), True),
        ("attention.absorbed_bmm_concat_cast_q_fp8", (10, 0), False),
        ("gemm.convrot_int8_fused_linear", (8, 9), False),
        ("gemm.convrot_int8_fused_linear", (9, 0), True),
        ("gemm.convrot_int8_fused_linear", (10, 0), True),
        ("gemm.convrot_int8_fused_linear", (10, 3), False),
        ("gemm.convrot_int8_fused_linear", (12, 0), True),
        ("gemm.convrot_int8_fused_linear", (12, 1), True),
        ("gemm.convrot_int8_fused_linear", (12, 2), False),
    ],
)
def test_registered_architecture_boundaries(op, sm, expected):
    spec = K.registry.get_backend(op, KernelBackend.JIT)
    platform = PlatformInfo(
        device_type="cuda", cuda_arch_major=sm[0], cuda_arch_minor=sm[1]
    )
    assert K.capabilities_satisfied(spec.capabilities, platform) is expected
    assert not K.capabilities_satisfied(spec.capabilities, _CPU)


def test_convrot_int8_specs_match_wrapper_capabilities():
    """The gemm group cannot import the wrapper, so this pins its three registry
    entries to the wrapper's SUPPORTED_CAPABILITIES table."""
    from sglang.kernels.ops.gemm import convrot_int8

    expected = frozenset(
        Cap.cuda(min_sm=sm, max_sm=sm) for sm in convrot_int8.SUPPORTED_CAPABILITIES
    )
    for op in (
        "gemm.convrot_rotate_quantize_activation",
        "gemm.convrot_int8_fused_linear",
        "gemm.convrot_int8_linear_prequant",
    ):
        spec = K.registry.get_backend(op, KernelBackend.JIT)
        assert spec.capabilities == expected, op


def test_deep_select_spec_matches_wrapper_architectures():
    """The group cannot import the wrapper, so this is the only link between the two SM lists."""
    from sglang.kernels.ops.attention import deep_select

    spec = K.registry.get_backend("attention.deep_select_topk", KernelBackend.JIT)
    assert spec.capabilities == frozenset(
        Cap.cuda(min_sm=sm, max_sm=sm) for sm in deep_select.SUPPORTED_CUDA_ARCHS
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
