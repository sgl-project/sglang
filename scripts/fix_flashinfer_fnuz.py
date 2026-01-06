#!/usr/bin/env python3
"""
Fix FlashInfer FP8 vec_t for AMD MI300X (gfx942).

MI300X uses FNUZ FP8 format, not OCP format. The OCP FP8 types have 
host-only default constructors on gfx942, causing device code compilation failures.

This script patches the installed FlashInfer to:
1. Use FNUZ types for gfx942
2. Add explicit default constructors using storage types

Usage:
    python fix_flashinfer_fp8_hip.py [--dry-run] [--restore]
"""

import argparse
import os
import shutil
import sys


def find_flashinfer_vec_dtypes():
    """Find the FlashInfer vec_dtypes.cuh file."""
    try:
        import flashinfer
        base_path = os.path.dirname(flashinfer.__file__)
        vec_dtypes_path = os.path.join(base_path, "data", "include", "flashinfer", "vec_dtypes.cuh")
        if os.path.exists(vec_dtypes_path):
            return vec_dtypes_path
    except ImportError:
        pass
    
    # Try common paths
    common_paths = [
        "/opt/venv/lib/python3.10/site-packages/flashinfer/data/include/flashinfer/vec_dtypes.cuh",
        "/usr/local/lib/python3.10/dist-packages/flashinfer/data/include/flashinfer/vec_dtypes.cuh",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


# The fix for FP8 type aliases
FP8_TYPE_ALIAS_OLD = '''#if __has_include(<hip/hip_fp8.h>)
#include <hip/hip_fp8.h>
// FP8 type aliases at global namespace (CUDA-compatible names)
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2;
using __nv_fp8_storage_t = __hip_fp8_storage_t;
using __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;
using __nv_fp8x4_storage_t = __hip_fp8x4_storage_t;'''

FP8_TYPE_ALIAS_NEW = '''#if __has_include(<hip/hip_fp8.h>)
#include <hip/hip_fp8.h>

// FP8 type aliases - use FNUZ types for gfx942 (MI300X), OCP for newer GPUs
// gfx942: HIP_FP8_TYPE_FNUZ=1, HIP_FP8_TYPE_OCP=0 (OCP ctors are host-only)
// gfx950: HIP_FP8_TYPE_FNUZ=0, HIP_FP8_TYPE_OCP=1 (OCP ctors are host+device)
#if defined(__gfx942__) || (defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ && !defined(HIP_FP8_TYPE_OCP))
// MI300X and FNUZ-primary architectures - use FNUZ types for device compatibility
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
using __nv_fp8_e5m2 = __hip_fp8_e5m2_fnuz;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2_fnuz;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3_fnuz;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2_fnuz;
#elif defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP
// OCP-primary architectures (MI350/gfx950, etc.) - use OCP types
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2;
#else
// Fallback: try FNUZ first (safer for older HIP versions)
#if defined(__hip_fp8_e4m3_fnuz)
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
using __nv_fp8_e5m2 = __hip_fp8_e5m2_fnuz;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2_fnuz;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3_fnuz;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2_fnuz;
#else
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2;
#endif
#endif

// Storage types are always the same
using __nv_fp8_storage_t = __hip_fp8_storage_t;
using __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;
using __nv_fp8x4_storage_t = __hip_fp8x4_storage_t;'''


# Fix for vec_t<__nv_fp8_e4m3, 2> - add default constructor
VEC_T_E4M3_2_OLD = '''// __nv_fp8_e4m3 x 2
template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  __nv_fp8x2_e4m3 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i)'''

VEC_T_E4M3_2_NEW = '''// __nv_fp8_e4m3 x 2
template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  // Use union with storage type to ensure default constructor works on device
  union {
    __nv_fp8x2_e4m3 data;
    __nv_fp8x2_storage_t data_storage;
  };

  // Explicit default constructor - always device-compatible
  FLASHINFER_INLINE vec_t() : data_storage(0) {}

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i)'''


# Fix for vec_t<__nv_fp8_e4m3, 4>
VEC_T_E4M3_4_OLD = '''// __nv_fp8_e4m3 x 4

template <>
struct vec_t<__nv_fp8_e4m3, 4> {
  __nv_fp8x4_e4m3 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i)'''

VEC_T_E4M3_4_NEW = '''// __nv_fp8_e4m3 x 4

template <>
struct vec_t<__nv_fp8_e4m3, 4> {
  // Use union with storage type to ensure default constructor works on device
  union {
    __nv_fp8x4_e4m3 data;
    __nv_fp8x4_storage_t data_storage;
  };

  // Explicit default constructor - always device-compatible
  FLASHINFER_INLINE vec_t() : data_storage(0) {}

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i)'''


# Fix for vec_t<__nv_fp8_e4m3, 8>
VEC_T_E4M3_8_OLD = '''// __nv_fp8_e4m3 x 8

template <>
struct vec_t<__nv_fp8_e4m3, 8> {
  uint2 data;

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i)'''

VEC_T_E4M3_8_NEW = '''// __nv_fp8_e4m3 x 8

template <>
struct vec_t<__nv_fp8_e4m3, 8> {
  uint2 data;

  // Explicit default constructor
  FLASHINFER_INLINE vec_t() : data{0, 0} {}

  FLASHINFER_INLINE __nv_fp8_e4m3& operator[](size_t i)'''


# Fix for vec_t<__nv_fp8_e5m2, 2>
VEC_T_E5M2_2_OLD = '''// __nv_fp8_e5m2 x 2
template <>
struct vec_t<__nv_fp8_e5m2, 2> {
  __nv_fp8x2_e5m2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i)'''

VEC_T_E5M2_2_NEW = '''// __nv_fp8_e5m2 x 2
template <>
struct vec_t<__nv_fp8_e5m2, 2> {
  // Use union with storage type to ensure default constructor works on device
  union {
    __nv_fp8x2_e5m2 data;
    __nv_fp8x2_storage_t data_storage;
  };

  // Explicit default constructor - always device-compatible
  FLASHINFER_INLINE vec_t() : data_storage(0) {}

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i)'''


# Fix for vec_t<__nv_fp8_e5m2, 4>
VEC_T_E5M2_4_OLD = '''// __nv_fp8_e5m2 x 4

template <>
struct vec_t<__nv_fp8_e5m2, 4> {
  __nv_fp8x4_e5m2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i)'''

VEC_T_E5M2_4_NEW = '''// __nv_fp8_e5m2 x 4

template <>
struct vec_t<__nv_fp8_e5m2, 4> {
  // Use union with storage type to ensure default constructor works on device
  union {
    __nv_fp8x4_e5m2 data;
    __nv_fp8x4_storage_t data_storage;
  };

  // Explicit default constructor - always device-compatible
  FLASHINFER_INLINE vec_t() : data_storage(0) {}

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i)'''


# Fix for vec_t<__nv_fp8_e5m2, 8>
VEC_T_E5M2_8_OLD = '''// __nv_fp8_e5m2 x 8

template <>
struct vec_t<__nv_fp8_e5m2, 8> {
  uint2 data;

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i)'''

VEC_T_E5M2_8_NEW = '''// __nv_fp8_e5m2 x 8

template <>
struct vec_t<__nv_fp8_e5m2, 8> {
  uint2 data;

  // Explicit default constructor
  FLASHINFER_INLINE vec_t() : data{0, 0} {}

  FLASHINFER_INLINE __nv_fp8_e5m2& operator[](size_t i)'''


# Also fix the fill() functions to use data_storage
FILL_E4M3_2_OLD = '''FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}'''

FILL_E4M3_2_NEW = '''FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 2>::fill(__nv_fp8_e4m3 val) {
  data_storage = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}'''

FILL_E4M3_4_OLD = '''FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}'''

FILL_E4M3_4_NEW = '''FLASHINFER_INLINE void vec_t<__nv_fp8_e4m3, 4>::fill(__nv_fp8_e4m3 val) {
  data_storage = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}'''

FILL_E5M2_2_OLD = '''FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}'''

FILL_E5M2_2_NEW = '''FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 2>::fill(__nv_fp8_e5m2 val) {
  data_storage = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}'''

FILL_E5M2_4_OLD = '''FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}'''

FILL_E5M2_4_NEW = '''FLASHINFER_INLINE void vec_t<__nv_fp8_e5m2, 4>::fill(__nv_fp8_e5m2 val) {
  data_storage = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}'''


# Fix bfloat16 type mismatch - hip_bfloat16 vs __hip_bfloat16
# Also add include guard to prevent duplicate definition with cudaBf16Fallbacks.cuh
MAKE_BFLOAT162_OLD = '''__device__ __forceinline__ __hip_bfloat162 make_bfloat162(hip_bfloat16 x, hip_bfloat16 y) {
  __hip_bfloat162 result;
  result.x = x;
  result.y = y;
  return result;
}'''

MAKE_BFLOAT162_NEW = '''#ifndef __FLASHINFER_MAKE_BFLOAT162_DEFINED__
#define __FLASHINFER_MAKE_BFLOAT162_DEFINED__
__device__ __forceinline__ __hip_bfloat162 make_bfloat162(__hip_bfloat16 x, __hip_bfloat16 y) {
  __hip_bfloat162 result;
  result.x = x;
  result.y = y;
  return result;
}
#endif'''


REPLACEMENTS = [
    (FP8_TYPE_ALIAS_OLD, FP8_TYPE_ALIAS_NEW, "FP8 type aliases (use FNUZ for gfx942)"),
    (VEC_T_E4M3_2_OLD, VEC_T_E4M3_2_NEW, "vec_t<__nv_fp8_e4m3, 2> default constructor"),
    (VEC_T_E4M3_4_OLD, VEC_T_E4M3_4_NEW, "vec_t<__nv_fp8_e4m3, 4> default constructor"),
    (VEC_T_E4M3_8_OLD, VEC_T_E4M3_8_NEW, "vec_t<__nv_fp8_e4m3, 8> default constructor"),
    (VEC_T_E5M2_2_OLD, VEC_T_E5M2_2_NEW, "vec_t<__nv_fp8_e5m2, 2> default constructor"),
    (VEC_T_E5M2_4_OLD, VEC_T_E5M2_4_NEW, "vec_t<__nv_fp8_e5m2, 4> default constructor"),
    (VEC_T_E5M2_8_OLD, VEC_T_E5M2_8_NEW, "vec_t<__nv_fp8_e5m2, 8> default constructor"),
    (FILL_E4M3_2_OLD, FILL_E4M3_2_NEW, "vec_t<__nv_fp8_e4m3, 2>::fill() use data_storage"),
    (FILL_E4M3_4_OLD, FILL_E4M3_4_NEW, "vec_t<__nv_fp8_e4m3, 4>::fill() use data_storage"),
    (FILL_E5M2_2_OLD, FILL_E5M2_2_NEW, "vec_t<__nv_fp8_e5m2, 2>::fill() use data_storage"),
    (FILL_E5M2_4_OLD, FILL_E5M2_4_NEW, "vec_t<__nv_fp8_e5m2, 4>::fill() use data_storage"),
    (MAKE_BFLOAT162_OLD, MAKE_BFLOAT162_NEW, "make_bfloat162 use __hip_bfloat16"),
]


def apply_fix(file_path, dry_run=False):
    """Apply the FP8 fix to vec_dtypes.cuh."""
    print(f"Reading {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    applied = []
    skipped = []
    
    for old, new, desc in REPLACEMENTS:
        if old in content:
            content = content.replace(old, new)
            applied.append(desc)
            print(f"  ✓ Applied: {desc}")
        else:
            skipped.append(desc)
            print(f"  - Skipped (not found or already applied): {desc}")
    
    if content == original_content:
        print("\nNo changes needed - file may already be patched.")
        return False
    
    if dry_run:
        print(f"\n[DRY RUN] Would apply {len(applied)} changes to {file_path}")
        return True
    
    # Create backup
    backup_path = file_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"\nCreated backup: {backup_path}")
    
    # Write patched content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Applied {len(applied)} fixes to {file_path}")
    return True


def restore_backup(file_path):
    """Restore from backup."""
    backup_path = file_path + ".backup"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        print(f"✓ Restored {file_path} from backup")
        return True
    else:
        print(f"✗ No backup found at {backup_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fix FlashInfer FP8 for AMD MI300X")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--restore", action="store_true", help="Restore from backup")
    parser.add_argument("--path", type=str, help="Path to vec_dtypes.cuh (auto-detected if not specified)")
    args = parser.parse_args()
    
    # Find the file
    if args.path:
        file_path = args.path
    else:
        file_path = find_flashinfer_vec_dtypes()
    
    if not file_path or not os.path.exists(file_path):
        print("Error: Could not find FlashInfer vec_dtypes.cuh")
        print("Please specify the path with --path")
        sys.exit(1)
    
    print(f"FlashInfer vec_dtypes.cuh: {file_path}")
    
    if args.restore:
        if restore_backup(file_path):
            sys.exit(0)
        else:
            sys.exit(1)
    
    if apply_fix(file_path, dry_run=args.dry_run):
        if not args.dry_run:
            print("\n" + "="*60)
            print("FP8 fix applied successfully!")
            print("Please clear the FlashInfer JIT cache and test:")
            print("  rm -rf ~/.cache/flashinfer_jit")
            print("  python -c 'from flashinfer.norm import rmsnorm; print(\"FP8 works!\")'")
            print("="*60)
    

if __name__ == "__main__":
    main()
