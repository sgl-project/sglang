import triton
import triton.language as tl

# -----------------------------------------------------------------------------
#                                  Utilities
# -----------------------------------------------------------------------------


@triton.constexpr_function
def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


@triton.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@triton.jit
def swizzle2d(pid, grid_m, grid_n, GROUP_M: tl.constexpr):
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    tl.assume(group_size >= 0)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n


@triton.jit
def _load_tile_attrs(
    tile_id,
    num_tiles,
    unpadded_m,
    grid_n,
    M,
    K,
    ExptData,
    ExptHist,
    ExptOffs,
    ExptTileOffs,
    EXPT_IS_INNER: tl.constexpr,
    X_IS_PADDED: tl.constexpr,
    W_IS_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACKED_BLOCK_K_W: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    SWIZZLE_MX_VALUE: tl.constexpr,
):
    # unpack and swizzle program ids
    pid_emnk = tile_id
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, num_tiles, XCD_SWIZZLE)
    pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    if SPLIT_K > 1:
        pid_k = pid_mnk % SPLIT_K
        pid_mn = pid_mnk // SPLIT_K
    else:
        pid_k: tl.constexpr = 0
        pid_mn = pid_mnk
    pid_m, pid_n = swizzle2d(pid_mn, unpadded_m, grid_n, GROUP_M)

    # unpack expert data
    if EXPT_IS_INNER:
        # pid_e indicates expert ID: experts are laid sequentially along the K dimension
        # (i.e., we have columns for expert 0, and then expert 1, and then so on).
        # pid_k is meaningless (always zero).
        tl.static_assert(X_IS_PADDED or W_IS_PADDED, "At least one input must be padded!")
        tl.static_assert(SPLIT_K == 1, "Not supported yet")
        tl.static_assert(M is not None)
        expt_id, pid_z, pid_z_out, start_m, block_id, eM = 0, 0, pid_e, 0, pid_m, M
        k_tiles = tl.cdiv(tl.load(ExptHist + pid_e), BLOCK_K)
        padded_start_off_raw = tl.load(ExptTileOffs + pid_e)
        padded_start_off = padded_start_off_raw * BLOCK_K
        unpadded_start_off = tl.load(ExptOffs + pid_e)
        off_k_x = padded_start_off if X_IS_PADDED else unpadded_start_off
        # K_W is only used for non-TMA kernel (W bound is handled by TMA on TMA kernel).
        if W_IS_PADDED:
            off_k_w = padded_start_off_raw * PACKED_BLOCK_K_W
            K_W = tl.load(ExptTileOffs + pid_e + 1) * PACKED_BLOCK_K_W
        else:
            off_k_w = unpadded_start_off
            K_W = tl.load(ExptOffs + pid_e + 1)
    else:
        off_k_x = pid_k * BLOCK_K
        off_k_w = pid_k * PACKED_BLOCK_K_W
        if PACKED_BLOCK_K_W >= BLOCK_K:
            K_W = K * (PACKED_BLOCK_K_W // BLOCK_K)
        else:
            K_W = K // (BLOCK_K // PACKED_BLOCK_K_W)
        if SWIZZLE_MX_VALUE == "HOPPER_VALUE":
            K_W = tl.cdiv(K_W, 128) * 128
        k_tiles = tl.cdiv(K - off_k_x, BLOCK_K * SPLIT_K)
        if ExptData is None:
            tl.static_assert(M is not None)
            expt_id, pid_z, pid_z_out, start_m, block_id, eM = pid_e, pid_e, pid_e, 0, pid_m, M
        else:
            tl.static_assert(M is None)
            expt_data = tl.load(ExptData + pid_m)
            expt_id = expt_data & 0x0000FFFF
            block_id = expt_data >> 16
            eM = tl.load(ExptHist + expt_id)
            start_m = tl.load(ExptOffs + expt_id)
            pid_z, pid_z_out = 0, 0

    off_m = BLOCK_M * block_id

    return (
        expt_id,
        pid_z,
        pid_z_out,
        start_m,
        eM,
        off_m,
        pid_n,
        k_tiles,
        pid_k,
        off_k_x,
        off_k_w,
        K_W,
    )


def make_matmul_repr(base_name, order):

    def matmul_repr(specialization):
        signature = specialization.signature
        constants = specialization.constants
        reorder = lambda L: [L[i] for i in order]
        layout = lambda stride: "N" if stride in constants else "T"

        def convert_dtype(dtype):
            if "tensordesc" in dtype:
                ret = convert_dtype(dtype.split("<")[1].split("[")[0])
                return ret
            elif "u8" in dtype:
                return "mxfp4"
            elif dtype[0] == "*":
                return dtype[1:]
            else:
                return dtype

        dtypes = "x".join([convert_dtype(f"{signature[i]}") for i in reorder(["Y", "X", "W"])])
        layouts = "".join([f"{layout(i)}" for i in reorder(["stride_y_n", "stride_x_k", "stride_w_n"])])
        blocks = "x".join([f"{constants[i]}" for i in ["BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K"]])
        suffix = "_acc" if "OutAcc" in signature and "OutAcc" not in constants else ""
        # mode = []
        # if "GatherIndx" not in constants:
        #     mode += ['g']
        # if "ScatterSrcIndx" not in constants:
        #     mode += ['s']
        # suffix = "" if not mode else "_o" + (''.join(mode))
        # if base_name.startswith("_p"):
        #     suffix += "_ptma"
        return f"{base_name}{suffix}_{layouts}_{dtypes}_{blocks}"

    return matmul_repr


def matmul_launch_metadata(grid, kernel, args):
    from ..proton_opts import launch_metadata_allow_sync

    ret = dict()
    M, N, K = args["M"], args["N"], args["K"]
    Y, X, W = args["YPtr"], args["XPtr"], args["WPtr"]
    tokens_per_expt = args.get("TOKENS_PER_EXPT_FOR_ANNOTATION")
    hist = args["ExptHist"]
    batch_size = args.get("batch_size", 1)
    expt_is_inner = args["EXPT_IS_INNER"]
    if hist is not None:
        # If annotation is given, use that to generate name for profiling.
        if tokens_per_expt is not None:
            n_rows = f"{tokens_per_expt}*"
        elif launch_metadata_allow_sync():
            n_rows = int(hist.float().mean())
        else:
            n_rows = "unknown"

        if launch_metadata_allow_sync():
            n_tokens = float(hist.sum())
            n_w_bytes = (W.numel() * W.element_size() // hist.numel()) * (hist > 0).sum()
        elif tokens_per_expt is not None:
            n_tokens = tokens_per_expt * args["N_EXPTS_TOT"]
            # This may not be totally correct (e.g., we might not be using all experts)
            # but it's better than nothing.
            n_w_bytes = W.numel() * W.element_size()
        else:
            n_tokens = None
            n_w_bytes = 0

        # If annotation is given, use that to generate name for profiling.
        tokens_per_expt = args.get("TOKENS_PER_EXPT_FOR_ANNOTATION")
        n_rows = f"{tokens_per_expt}*" if tokens_per_expt is not None else n_rows
    else:
        n_tokens = None
        n_w_bytes = W.numel() * W.element_size()
    if expt_is_inner:
        K = None if n_tokens is None else int(n_tokens)
    repr = lambda s, x: f"{s} = {x}" if x is not None else f"E_{len(hist)}({s}) = {n_rows}"
    nbits = X.dtype.itemsize * 8
    batch_repr = ""
    if batch_size > 1:
        batch_repr = repr("B", args["batch_size"]) + ", "
    ret["name"] = f"{kernel.name} [{batch_repr}{repr('M', M)}, {repr('N', N)}, {repr('K', K)}] stg{kernel.num_stages}"
    ep_subtile = args["EPILOGUE_SUBTILE"]
    if ep_subtile is not None and ep_subtile > 1:
        ret["name"] += f" ep/{ep_subtile}"

    if hist is not None and n_tokens is None:
        return ret  # Don't fill metadata because we can't compute them properly.

    fM = M if M is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * K * (1 if expt_is_inner else batch_size)

    # sindx = args.get("WriteBackIndx", None)
    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    if hist is not None:
        assert n_tokens is not None
        n_read_rows = n_tokens

        if expt_is_inner:
            n_x_bytes = n_read_rows * X.shape[-2] * X.element_size()
            # Here, we're computing dW = X.T@dY, so "W" is actually dY and "Y" is actually dW.
            n_y_bytes = Y.numel() * Y.element_size() * (2 if args["OutAcc"] is not None else 1)
            n_w_bytes = n_read_rows * W.shape[-1] * W.element_size()
        else:
            n_x_bytes = n_read_rows * X.shape[-1] * X.element_size()
            n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()

    ret["bytes"] = int(n_x_bytes + n_y_bytes + n_w_bytes)

    return ret


@triton.jit
def threadfence_system():
    tl.inline_asm_elementwise("mov.u32 $0, 0x0; fence.sc.sys;", args=(), dtype=(tl.int32, ), is_pure=False, pack=1,
                              constraints="=r")
