MiB = 1024 * 1024

SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    9: {
        2: 64 * MiB,  # 64 MB
        4: 32 * MiB,  # 32 MB
        6: 64 * MiB,  # 64 MB
        8: 64 * MiB,  # 64 MB
    },
    10: {
        2: 8 * MiB,  # 8 MB
        4: 32 * MiB,  # 32 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
}
