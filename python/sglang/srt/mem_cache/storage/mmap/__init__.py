# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Mmap allocator storage backend helpers for SGLang HiCache."""

from .mmap_allocator import alloc_mmap, alloc_shm, hugetlb_pool_free_bytes

__all__ = [
    "alloc_mmap",
    "alloc_shm",
    "hugetlb_pool_free_bytes",
]
