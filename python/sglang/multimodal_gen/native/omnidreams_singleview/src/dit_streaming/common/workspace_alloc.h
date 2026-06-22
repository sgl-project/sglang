// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace ts {

// ============================================================================
// WorkspaceAllocator — bounds-checked linear allocator for GPU workspace memory
// ============================================================================
//
// Replaces raw pointer arithmetic (ptr += N * K * sizeof(half)) with a typed,
// bounds-checked bump allocator. Every alloc() call verifies that the request
// fits within the pre-allocated memory pool, catching overflow/overlap bugs at
// initialization time rather than at kernel crash time.
//
// Usage:
//   WorkspaceAllocator ws(pool_ptr, pool_bytes, "model_workspace");
//   auto* hidden  = ws.alloc<half>(N * M * K);        // aligned, bounds-checked
//   auto* scratch = ws.alloc<float>(N * M * K);       // auto-aligned to alignof(float)
//   ws.align_to(256);                                  // explicit alignment bump
//   auto* int8_buf = ws.alloc<int8_t>(N * M * FF);
//
// Debug mode (compile with -DOMNIDREAMS_SINGLEVIEW_DEBUG_WORKSPACE):
//   Prints every allocation with label, offset, size, and remaining capacity.
//
class WorkspaceAllocator {
public:
    WorkspaceAllocator() : base_(nullptr), total_(0), used_(0), label_("") {}

    WorkspaceAllocator(void* base, size_t total_bytes, const char* label = "workspace")
        : base_(static_cast<uint8_t*>(base))
        , total_(total_bytes)
        , used_(0)
        , label_(label) {}

    // Allocate `count` elements of type T, with optional alignment (defaults to alignof(T)).
    // Returns a typed device pointer. Aborts on overflow in debug builds.
    template <typename T>
    T* alloc(size_t count, size_t align = 0) {
        if (align == 0) align = alignof(T);
        align_to(align);

        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            std::fprintf(stderr,
                "[FATAL] WorkspaceAllocator '%s' size overflow: requested %zu elements of %zu bytes\n",
                label_, count, sizeof(T));
#ifndef OMNIDREAMS_SINGLEVIEW_WORKSPACE_NO_ABORT
            std::abort();
#else
            return nullptr;
#endif
        }
        size_t bytes = count * sizeof(T);

#ifdef OMNIDREAMS_SINGLEVIEW_DEBUG_WORKSPACE
        const size_t remaining = used_ <= total_ ? total_ - used_ : 0;
        std::printf("[WorkspaceAlloc][%s] +%zu bytes (%zu x %zu) at offset %zu / %zu (%.1f%% used)\n",
                    label_, bytes, count, sizeof(T), used_, total_,
                    total_ > 0 ? 100.0 * (total_ - remaining + std::min(bytes, remaining)) / total_ : 0.0);
#endif

        if (used_ > total_ || bytes > (total_ - used_)) {
            const size_t remaining = used_ <= total_ ? total_ - used_ : 0;
            const size_t shortfall = bytes > remaining ? bytes - remaining : bytes;
            std::fprintf(stderr,
                "[FATAL] WorkspaceAllocator '%s' overflow: requested %zu bytes at offset %zu, "
                "but total capacity is %zu bytes (%zu bytes short)\n",
                label_, bytes, used_, total_, shortfall);
#ifndef OMNIDREAMS_SINGLEVIEW_WORKSPACE_NO_ABORT
            std::abort();
#else
            return nullptr;
#endif
        }

        T* ptr = reinterpret_cast<T*>(base_ + used_);
        used_ += bytes;
        return ptr;
    }

    // Advance the internal pointer to satisfy the given alignment.
    void align_to(size_t alignment) {
        if (alignment <= 1) return;
        if ((alignment & (alignment - 1)) != 0 ||
            used_ > total_ ||
            reinterpret_cast<uintptr_t>(base_) > std::numeric_limits<uintptr_t>::max() - used_) {
            std::fprintf(stderr,
                "[FATAL] WorkspaceAllocator '%s' invalid alignment request: align=%zu, offset=%zu, total=%zu\n",
                label_, alignment, used_, total_);
#ifndef OMNIDREAMS_SINGLEVIEW_WORKSPACE_NO_ABORT
            std::abort();
#else
            return;
#endif
        }
        uintptr_t addr = reinterpret_cast<uintptr_t>(base_) + used_;
        uintptr_t mask = alignment - 1;
        if (addr > std::numeric_limits<uintptr_t>::max() - mask) {
            std::fprintf(stderr,
                "[FATAL] WorkspaceAllocator '%s' alignment address overflow: align=%zu, offset=%zu\n",
                label_, alignment, used_);
#ifndef OMNIDREAMS_SINGLEVIEW_WORKSPACE_NO_ABORT
            std::abort();
#else
            return;
#endif
        }
        uintptr_t aligned = (addr + mask) & ~mask;
        size_t padding = aligned - addr;
        if (padding > total_ - used_) {
            std::fprintf(stderr,
                "[FATAL] WorkspaceAllocator '%s' alignment overflow: requested %zu bytes at offset %zu, "
                "but total capacity is %zu bytes\n",
                label_, padding, used_, total_);
#ifndef OMNIDREAMS_SINGLEVIEW_WORKSPACE_NO_ABORT
            std::abort();
#else
            return;
#endif
        }
        used_ += padding;
    }

    // Reset used counter to zero (reuse the same memory pool from the start).
    void reset() { used_ = 0; }

    // Current write position (raw byte pointer).
    uint8_t* current() const { return base_ + used_; }

    size_t used() const { return used_; }
    size_t remaining() const { return total_ > used_ ? total_ - used_ : 0; }
    size_t total() const { return total_; }

    // Create a sub-allocator from the current position with a fixed byte budget.
    // Advances this allocator by `budget_bytes` (bounds-checked).
    WorkspaceAllocator sub(size_t budget_bytes, const char* sub_label = "sub") {
        if (used_ > total_ || budget_bytes > (total_ - used_)) {
            std::fprintf(stderr,
                "[FATAL] WorkspaceAllocator '%s' sub-alloc overflow: requested %zu bytes at offset %zu, "
                "but total capacity is %zu bytes\n",
                label_, budget_bytes, used_, total_);
#ifndef OMNIDREAMS_SINGLEVIEW_WORKSPACE_NO_ABORT
            std::abort();
#else
            return WorkspaceAllocator(nullptr, 0, sub_label);
#endif
        }
        WorkspaceAllocator child(base_ + used_, budget_bytes, sub_label);
        used_ += budget_bytes;
        return child;
    }

private:
    uint8_t* base_;
    size_t total_;
    size_t used_;
    const char* label_;
};

} // namespace ts
