// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend selector for native CUDA kernels
// Usage:
// - Define OMNIDREAMS_SINGLEVIEW_BACKEND at compile time to switch implementations.
// - Defaults to CUTLASS to preserve existing behavior.

// Backend constants
#define OMNIDREAMS_SINGLEVIEW_BACKEND_CUTLASS 1
#define OMNIDREAMS_SINGLEVIEW_BACKEND_TIN     2

// Default backend: CUTLASS
#ifndef OMNIDREAMS_SINGLEVIEW_BACKEND
#define OMNIDREAMS_SINGLEVIEW_BACKEND OMNIDREAMS_SINGLEVIEW_BACKEND_CUTLASS
#endif

// Convenience feature macros
#if OMNIDREAMS_SINGLEVIEW_BACKEND == OMNIDREAMS_SINGLEVIEW_BACKEND_CUTLASS
#ifndef OMNIDREAMS_SINGLEVIEW_USE_CUTLASS
#define OMNIDREAMS_SINGLEVIEW_USE_CUTLASS
#endif
#endif

// When using PyTorch extensions, these macros are often defined to disable half operators.
// We re-enable them project-wide before including any CUDA half headers or third-party libs.
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
