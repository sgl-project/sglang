from pathlib import Path


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def gen_sgl_trtllm_gen_fused_moe_sm100_module():
    import flashinfer

    from flashinfer.artifacts import ArtifactPath, CheckSumHash
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec
    from flashinfer.jit.cubin_loader import (
        ensure_symlink,
        get_artifact,
        get_meta_hash,
        verify_symlinked_headers,
    )
    from flashinfer.jit.fused_moe import BMM_EXPORT_HEADERS

    overlay_data_dir = _data_dir()
    overlay_csrc_dir = overlay_data_dir / "csrc"
    overlay_include_dir = overlay_data_dir / "include"
    flashinfer_data_dir = Path(flashinfer.__file__).resolve().parent / "data"
    flashinfer_csrc_dir = flashinfer_data_dir / "csrc"
    flashinfer_include_dir = flashinfer_data_dir / "include"

    include_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/include"
    header_name = "flashinferMetaInfo"
    checksum_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/checksums.txt"
    checksum = get_artifact(checksum_path, CheckSumHash.TRTLLM_GEN_BMM)
    assert checksum, f"Failed to get checksums.txt from {checksum_path}"
    meta_hash = get_meta_hash(checksum)

    metainfo = get_artifact(f"{include_path}/{header_name}.h", meta_hash)
    assert metainfo, f"{header_name}.h not found"

    bmm_export_path = f"{include_path}/trtllmGen_bmm_export"
    for header in BMM_EXPORT_HEADERS:
        h = get_artifact(f"{bmm_export_path}/{header}", get_meta_hash(checksum, header))
        assert h, f"{header} not found"

    symlink_path = (
        jit_env.FLASHINFER_CUBIN_DIR
        / "flashinfer"
        / "trtllm"
        / "batched_gemm"
        / "trtllmGen_bmm_export"
    )
    ensure_symlink(symlink_path, jit_env.FLASHINFER_CUBIN_DIR / bmm_export_path)
    verify_symlinked_headers(symlink_path, BMM_EXPORT_HEADERS, checksum)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 12]
    )

    return gen_jit_spec(
        "sgl_fused_moe_trtllm_sm100",
        [
            flashinfer_csrc_dir / "nv_internal/cpp/kernels/quantization.cu",
            flashinfer_csrc_dir / "nv_internal/cpp/common/envUtils.cpp",
            flashinfer_csrc_dir / "nv_internal/cpp/common/logger.cpp",
            flashinfer_csrc_dir / "nv_internal/cpp/common/stringUtils.cpp",
            flashinfer_csrc_dir / "nv_internal/cpp/common/tllmException.cpp",
            flashinfer_csrc_dir / "nv_internal/cpp/common/memoryUtils.cu",
            overlay_csrc_dir / "trtllm_fused_moe_kernel_launcher.cu",
            overlay_csrc_dir / "trtllm_fused_moe_runner.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_llama4.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu",
            overlay_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu",
            flashinfer_csrc_dir / "trtllm_batched_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_GEN_EXPORT_FLASHINFER",
            "-DTLLM_ENABLE_CUDA",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_BMM}\\"',
        ]
        + nvcc_flags,
        extra_include_paths=[
            overlay_include_dir,
            overlay_csrc_dir,
            flashinfer_include_dir,
            flashinfer_csrc_dir,
            flashinfer_csrc_dir / "nv_internal",
            flashinfer_csrc_dir / "nv_internal/include",
            jit_env.FLASHINFER_CUBIN_DIR,
            jit_env.FLASHINFER_CUBIN_DIR / include_path,
        ],
    )
