MiB = 1024 * 1024

TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    9: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
    10: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
}

KiB = 1024

# Largest payload the Intel XPU one-shot all-reduce takes, by world size: the
# largest size where it clearly beat oneCCL under XPU graph replay on two 8x Arc
# Pro B60 (PCIe Gen5) hosts, one of them dual-socket. World sizes not listed
# were not measured and keep the path off.
XPU_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    2: 32 * KiB,
    4: 16 * KiB,
    6: 16 * KiB,
    8: 8 * KiB,
}
