"""Bounded actual-dispatch hybrid policy; no universal same-arithmetic claim."""
def select_ksplit(unquant,shape):
    m,n,k=shape
    if n!=512 or k!=2560 or not unquant.get_bf16_gemm_backend().is_cutedsl():
        return None
    split=unquant._enable_bf16_splitk_gemm and unquant.use_bf16_splitk_gemm(m,n,k)
    if m in (4,8) and split:
        tactic=unquant._BF16_SPLITK_TUNED_TACTICS.get(shape)
        if (tactic is not None and tactic[2]==4
            and shape not in unquant._BF16_WARP_SPLITK_TUNED_TACTICS
            and not unquant._prefer_direct(m,n,k)):
            return 4
        return None
    if m==16 and not split:
        if (unquant._use_hopper_bf16_gemv is not None and unquant._use_hopper_bf16_gemv(m,n,k)):
            return None
        if (unquant._use_cutedsl_bf16_gemm is not None and unquant._use_cutedsl_bf16_gemm(m,n,k)):
            return None
        return 2
    return None
