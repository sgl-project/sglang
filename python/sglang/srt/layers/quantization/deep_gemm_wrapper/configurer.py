def _compute_enable_deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        logger.warning("Failed to import deep_gemm, disable ENABLE_JIT_DEEPGEMM.")
        return False

    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    return get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true")


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()
