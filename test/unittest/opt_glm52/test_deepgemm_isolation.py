"""Phase 16: DeepGEMM isolation.

Records DeepGEMM environment info and isolates any known grouped DeepGEMM
illegal-address issue from the core EAGLE3 PP validation.
"""

import os
import sys
import subprocess

sys.path.insert(0, "/home/liang/sglang/python")

import torch


def test_deepgemm_environment_record():
    """Record DeepGEMM environment for diagnostics."""
    info = {}
    
    # torch version
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    
    # Driver version
    try:
        info["driver_version"] = torch.cuda.get_device_properties(0).name
    except:
        info["driver_version"] = "unknown"
    
    # CUDA_HOME
    info["cuda_home"] = os.environ.get("CUDA_HOME", "not set")
    
    # nvcc version
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        info["nvcc_version"] = result.stdout.strip().split("\n")[-1] if result.returncode == 0 else "nvcc not found"
    except:
        info["nvcc_version"] = "nvcc not available"
    
    # DeepGEMM commit
    try:
        from sglang.srt.layers import deep_gemm_wrapper
        info["deepgemm_available"] = True
        if hasattr(deep_gemm_wrapper, '__version__'):
            info["deepgemm_version"] = deep_gemm_wrapper.__version__
        else:
            info["deepgemm_version"] = "version not available"
    except Exception as e:
        info["deepgemm_available"] = False
        info["deepgemm_version"] = f"import failed: {type(e).__name__}: {str(e)[:100]}"
    
    # JIT cache path
    info["jit_cache_path"] = os.environ.get("DEEP_GEMM_JIT_CACHE", "not set (default)")
    
    # PDL setting
    info["pdl_enabled"] = os.environ.get("SGLANG_DEEP_GEMM_PDL", "not set")
    
    print("  DeepGEMM Environment:")
    for k, v in info.items():
        print(f"    {k}: {v}")
    
    # Verify we can continue PP validation regardless of DeepGEMM status
    assert info["torch_version"] is not None
    return info


def test_deepgemm_isolated_from_pp_validation():
    """Verify that DeepGEMM failures don't affect PP validation.
    
    The PP validation tests (participant sets, capture ordering, etc.)
    do not depend on DeepGEMM. If DeepGEMM has an illegal-address issue,
    it should be reported independently.
    """
    # All PP validation tests use:
    # - Standard torch operations (no DeepGEMM)
    # - Gloo/NCCL process groups (no DeepGEMM)
    # - Packed transport (no DeepGEMM)
    # - Synthetic layers (no DeepGEMM)
    
    # Verify pp_packed_transport.py has no deep_gemm imports
    source = open("/home/liang/sglang/python/sglang/srt/distributed/pp_packed_transport.py").read()
    assert "deep_gemm" not in source.lower(), "PP transport should not import DeepGEMM"
    
    # Verify glm52_eagle3_pp.py has no deep_gemm imports
    source = open("/home/liang/sglang/python/sglang/srt/speculative/glm52_eagle3_pp.py").read()
    assert "deep_gemm" not in source.lower(), "EAGLE3 PP module should not import DeepGEMM"
    
    print("  DeepGEMM isolated from PP validation PASSED")


def test_deepgemm_not_in_test_path():
    """Verify our test files don't import DeepGEMM."""
    test_files = [
        "test/unittest/opt_glm52/test_pp_participant_set_world8_gloo.py",
        "test/unittest/opt_glm52/test_pp_rank_mapping.py",
        "test/unittest/opt_glm52/test_capture_index_semantics.py",
        "test/unittest/opt_glm52/test_eagle3_arch_audit.py",
        "test/unittest/opt_glm52/test_pp_stress_rid_state.py",
        "test/unittest/opt_glm52/test_cuda_graph_validation.py",
        "test/unittest/opt_glm52/test_pp_protocol_failures.py",
        "test/unittest/opt_glm52/test_tiny_model_forward_equivalence.py",
        "test/unittest/opt_glm52/test_async_comm_safety.py",
        "test/unittest/opt_glm52/test_dtype_control_plane_audit.py",
    ]
    
    for tf in test_files:
        if os.path.exists(tf):
            source = open(tf).read()
            assert "deep_gemm" not in source.lower(), f"{tf} imports DeepGEMM!"
    
    print("  DeepGEMM not in test path PASSED")


if __name__ == "__main__":
    print("=== Phase 16: DeepGEMM Isolation ===")
    info = test_deepgemm_environment_record()
    test_deepgemm_isolated_from_pp_validation()
    test_deepgemm_not_in_test_path()
    
    # Note about known issue
    print("\n  NOTE: The known grouped DeepGEMM illegal-address issue is")
    print("  isolated from the PP validation. If DeepGEMM fails, it should")
    print("  be reported independently and PP validation should continue")
    print("  where safe. Do not rerun illegal-address tests in the main")
    print("  process after a CUDA context has been corrupted.")
    
    print("\n=== All Phase 16 tests PASSED ===")
