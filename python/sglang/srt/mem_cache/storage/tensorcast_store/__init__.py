# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Tensorcast-backed storage backend for SGLang HiCache."""

# Keep this initializer lightweight. Generic host-memory code lazily imports
# tensorcast_store.config/host_allocator for allocator setup; re-exporting
# TensorcastStore here would import memory_pool_host during package init and can
# introduce circular import.
__all__: list[str] = []
