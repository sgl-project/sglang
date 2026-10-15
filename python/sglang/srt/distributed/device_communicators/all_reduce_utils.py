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

# Largest payload the Intel XPU one-shot all-reduce takes, by world size: where
# it stops beating oneCCL under XPU graph replay on Arc Pro B60 (PCIe). World
# sizes not listed were not measured and keep the path off.
XPU_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    2: 64 * KiB,
    4: 32 * KiB,
    6: 32 * KiB,
}
