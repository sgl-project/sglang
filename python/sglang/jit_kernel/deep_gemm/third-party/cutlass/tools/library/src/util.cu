/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <iosfwd>
#include <complex>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/complex.h"
#include "cutlass/blas3.h"

#include "cutlass/layout/matrix.h"

#include "cutlass/library/library.h"
#include "cutlass/library/util.h"

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  Provider enumerant;
}
Provider_enumerants[] = {
  {"none", "None", Provider::kNone},
  {"cutlass", "CUTLASS", Provider::kCUTLASS},
  {"host", "reference_host", Provider::kReferenceHost},
  {"device", "reference_device", Provider::kReferenceDevice},
  {"cublas", "cuBLAS", Provider::kCUBLAS},
  {"cudnn", "cuDNN", Provider::kCUDNN},                           
};

/// Converts a Provider enumerant to a string
char const *to_string(Provider provider, bool pretty) {

  for (auto const & possible : Provider_enumerants) {
    if (provider == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }
  
  return pretty ? "Invalid" : "invalid";
}

/// Parses a Provider enumerant from a string
template <>
Provider from_string<Provider>(std::string const &str) {

  for (auto const & possible : Provider_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return Provider::kInvalid;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  GemmKind enumerant;
}
GemmKind_enumerants[] = {
  {"gemm", "<Gemm>", GemmKind::kGemm},
  {"spgemm", "<Sparse>", GemmKind::kSparse},
  {"universal", "<Universal>", GemmKind::kUniversal},
  {"planar_complex", "<PlanarComplex>", GemmKind::kPlanarComplex},
  {"planar_complex_array", "<PlanarComplexArray>", GemmKind::kPlanarComplexArray},
  {"grouped", "<Grouped>", GemmKind::kGrouped},
};

/// Converts a GemmKind enumerant to a string
char const *to_string(GemmKind type, bool pretty) {

  for (auto const & possible : GemmKind_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  RankKKind enumerant;
}
RankKKind_enumerants[] = {
  {"universal", "<Universal>", RankKKind::kUniversal},
};

/// Converts a SyrkKind enumerant to a string
char const *to_string(RankKKind type, bool pretty) {

  for (auto const & possible :RankKKind_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  TrmmKind enumerant;
}
TrmmKind_enumerants[] = {
  {"universal", "<Universal>", TrmmKind::kUniversal},
};

/// Converts a TrmmKind enumerant to a string
char const *to_string(TrmmKind type, bool pretty) {

  for (auto const & possible :TrmmKind_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  SymmKind enumerant;
}
SymmKind_enumerants[] = {
  {"universal", "<Universal>", SymmKind::kUniversal},
};

/// Converts a SymmKind enumerant to a string
char const *to_string(SymmKind type, bool pretty) {

  for (auto const & possible :SymmKind_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}
///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  SideMode enumerant;
}
SideMode_enumerants[] = {
  {"left", "Left", SideMode::kLeft},
  {"right", "Right", SideMode::kRight}
};

/// Converts a SideMode enumerant to a string
char const *to_string(SideMode type, bool pretty) {

  for (auto const & possible :SideMode_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  FillMode enumerant;
}
FillMode_enumerants[] = {
  {"lower", "Lower", FillMode::kLower},
  {"upper", "Upper", FillMode::kUpper}
};

/// Converts a FillMode enumerant to a string
char const *to_string(FillMode type, bool pretty) {

  for (auto const & possible :FillMode_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  BlasMode enumerant;
}
BlasMode_enumerants[] = {
  {"symmetric", "Symmetric", BlasMode::kSymmetric},
  {"hermitian", "Hermitian", BlasMode::kHermitian}
};

/// Converts a BlasMode enumerant to a string
char const *to_string(BlasMode type, bool pretty) {

  for (auto const & possible :BlasMode_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  DiagType enumerant;
}
DiagType_enumerants[] = {
  {"nonunit", "NonUnit", DiagType::kNonUnit},
  {"unit", "Unit", DiagType::kUnit}
};

/// Converts a DiagType enumerant to a string
char const *to_string(DiagType type, bool pretty) {

  for (auto const & possible :DiagType_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  OperationKind enumerant;
} OperationKind_enumerants[] = {
  {"eq_gemm", "EqGemm", OperationKind::kEqGemm},
  {"gemm", "Gemm", OperationKind::kGemm},
  {"block_scaled_gemm", "blockScaledGemm", OperationKind::kBlockScaledGemm}, 
  {"blockwise_gemm", "blockwiseGemm", OperationKind::kBlockwiseGemm}, 
  {"rank_k", "RankK", OperationKind::kRankK},
  {"rank_2k", "Rank2K", OperationKind::kRank2K},
  {"trmm", "Trmm", OperationKind::kTrmm},
  {"symm", "Symm", OperationKind::kSymm},
  {"conv2d", "Conv2d", OperationKind::kConv2d},
  {"conv3d", "Conv3d", OperationKind::kConv3d},
  {"spgemm", "SparseGemm", OperationKind::kSparseGemm},
  {"grouped_gemm", "GroupedGemm", OperationKind::kGroupedGemm},
};

/// Converts a Status enumerant to a string
char const *to_string(OperationKind enumerant, bool pretty) {

  for (auto const & possible : OperationKind_enumerants) {
    if (enumerant == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Converts a Status enumerant from a string
template <>
OperationKind from_string<OperationKind>(std::string const &str) {

  for (auto const & possible : OperationKind_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return OperationKind::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  Status enumerant;
}
Status_enumerants[] = {
  {"success", "Success", Status::kSuccess},
  {"misaligned_operand", "Error: misaligned operand", Status::kErrorMisalignedOperand},
  {"invalid_problem", "Error: invalid problem", Status::kErrorInvalidProblem},
  {"not_supported", "Error: not supported", Status::kErrorNotSupported},
  {"internal", "Error: internal", Status::kErrorInternal}
};

/// Converts a Status enumerant to a string
char const *to_string(Status status, bool pretty) {

  for (auto const & possible : Status_enumerants) {
    if (status == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Converts a Status enumerant from a string
template <>
Status from_string<Status>(std::string const &str) {

  for (auto const & possible : Status_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return Status::kInvalid;
}

///////////////////////////////////////////////////////////////////////////////////////////////////


static struct {
  char const *text;
  char const *pretty;
  RuntimeDatatype enumerant;
}
RuntimeDatatype_enumerants[] = {
  {"e4m3", "<e4m3>", RuntimeDatatype::kE4M3},
  {"e5m2", "<e5m2>", RuntimeDatatype::kE5M2},
  {"e3m2", "<e3m2>", RuntimeDatatype::kE3M2},
  {"e2m3", "<e2m3>", RuntimeDatatype::kE2M3},
  {"e2m1", "<e2m1>", RuntimeDatatype::kE2M1}
};

/// Converts a RuntimeDatatype enumerant to a string
char const *to_string(RuntimeDatatype type, bool pretty) {

  for (auto const & possible : RuntimeDatatype_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}


/// Converts a RuntimeDatatype enumerant from a string
template <>
RuntimeDatatype from_string<RuntimeDatatype>(std::string const &str) {

  for (auto const & possible : RuntimeDatatype_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return RuntimeDatatype::kInvalid;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  NumericTypeID enumerant;
}
NumericTypeID_enumerants[] = {
  {"unknown", "<unknown>", NumericTypeID::kUnknown},
  {"void", "Void", NumericTypeID::kVoid},
  {"b1", "B1", NumericTypeID::kB1},
  {"u2", "U2", NumericTypeID::kU2},
  {"u4", "U4", NumericTypeID::kU4},
  {"u8", "U8", NumericTypeID::kU8},
  {"u16", "U16", NumericTypeID::kU16},
  {"u32", "U32", NumericTypeID::kU32},
  {"u64", "U64", NumericTypeID::kU64},
  {"s2", "S2", NumericTypeID::kS2},
  {"s4", "S4", NumericTypeID::kS4},
  {"s8", "S8", NumericTypeID::kS8},
  {"s16", "S16", NumericTypeID::kS16},
  {"s32", "S32", NumericTypeID::kS32},
  {"s64", "S64", NumericTypeID::kS64},
  {"fe4m3", "FE4M3", NumericTypeID::kFE4M3},
  {"fe5m2", "FE5M2", NumericTypeID::kFE5M2},
  
  {"f8", "F8", NumericTypeID::kF8},
  {"f6", "F6", NumericTypeID::kF6},
  {"f4", "F4", NumericTypeID::kF4},
  {"fe2m3", "FE2M3", NumericTypeID::kFE2M3},
  {"fe3m2", "FE3M2", NumericTypeID::kFE3M2},
  {"fe2m1", "FE2M1", NumericTypeID::kFE2M1},
  {"fue8m0", "FUE8M0", NumericTypeID::kFUE8M0},
  {"fue4m3", "FUE4M3", NumericTypeID::kFUE4M3},
  {"f16", "F16", NumericTypeID::kF16},
  {"bf16", "BF16", NumericTypeID::kBF16},
  {"f32", "F32", NumericTypeID::kF32},
  {"tf32", "TF32", NumericTypeID::kTF32},
  {"f64", "F64", NumericTypeID::kF64},
  {"cf16", "CF16", NumericTypeID::kCF16},
  {"cbf16", "CBF16", NumericTypeID::kCBF16},
  {"cf32", "CF32", NumericTypeID::kCF32},
  {"ctf32", "CTF32", NumericTypeID::kCTF32},
  {"cf64", "CF64", NumericTypeID::kCF64},
  {"cu2", "CU2", NumericTypeID::kCU2},
  {"cu4", "CU4", NumericTypeID::kCU4},
  {"cu8", "CU8", NumericTypeID::kCU8},
  {"cu16", "CU16", NumericTypeID::kCU16},
  {"cu32", "CU32", NumericTypeID::kCU32},
  {"cu64", "CU64", NumericTypeID::kCU64},  
  {"cs2", "CS2", NumericTypeID::kCS2},
  {"cs4", "CS4", NumericTypeID::kCS4},
  {"cs8", "CS8", NumericTypeID::kCS8},
  {"cs16", "CS16", NumericTypeID::kCS16},
  {"cs32", "CS32", NumericTypeID::kCS32},
  {"cs64", "CS64", NumericTypeID::kCS64},
  {"*", "<unknown/enumerate all>", NumericTypeID::kUnknown}
};

/// Converts a NumericTypeID enumerant to a string
char const *to_string(NumericTypeID type, bool pretty) {

  for (auto const & possible : NumericTypeID_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Parses a NumericTypeID enumerant from a string
template <>
NumericTypeID from_string<NumericTypeID>(std::string const &str) {

  for (auto const & possible : NumericTypeID_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return NumericTypeID::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the size of a data type in bits
int sizeof_bits(NumericTypeID type) {
  switch (type) {
    case NumericTypeID::kFE4M3: return 8;
    case NumericTypeID::kFE5M2: return 8;
    
    case NumericTypeID::kF8: return 8;
    case NumericTypeID::kF6: return 6;
    case NumericTypeID::kF4: return 4;
    case NumericTypeID::kFE2M3: return 6;
    case NumericTypeID::kFE3M2: return 6;
    case NumericTypeID::kFE2M1: return 4;
    case NumericTypeID::kFUE8M0: return 8;
    case NumericTypeID::kFUE4M3: return 8;
    case NumericTypeID::kF16: return 16;
    case NumericTypeID::kBF16: return 16;
    case NumericTypeID::kTF32: return 32;
    case NumericTypeID::kF32: return 32;
    case NumericTypeID::kF64: return 64;
    case NumericTypeID::kCF16: return 32;
    case NumericTypeID::kCBF16: return 32;
    case NumericTypeID::kCF32: return 64;
    case NumericTypeID::kCTF32: return 64;
    case NumericTypeID::kCF64: return 128;
    case NumericTypeID::kS2: return 2;
    case NumericTypeID::kS4: return 4;
    case NumericTypeID::kS8: return 8;
    case NumericTypeID::kS16: return 16;
    case NumericTypeID::kS32: return 32;
    case NumericTypeID::kS64: return 64;
    case NumericTypeID::kU2: return 2;
    case NumericTypeID::kU4: return 4;
    case NumericTypeID::kU8: return 8;
    case NumericTypeID::kU16: return 16;
    case NumericTypeID::kU32: return 32;
    case NumericTypeID::kU64: return 64;
    case NumericTypeID::kB1:  return 1;
    default: break;
  }
  return 0;
}

/// Returns true if the numeric type is a complex data type or false if real-valued.
bool is_complex_type(NumericTypeID type) {
  switch (type) {
    case NumericTypeID::kCF16: return true;
    case NumericTypeID::kCF32: return true;
    case NumericTypeID::kCF64: return true;
    case NumericTypeID::kCBF16: return true;
    case NumericTypeID::kCTF32: return true;
    default: break;
  }
  return false;
}

/// Returns the field underlying a complex valued type
NumericTypeID get_real_type(NumericTypeID type) {
  switch (type) {
    case NumericTypeID::kCF16: return NumericTypeID::kF16;
    case NumericTypeID::kCF32: return NumericTypeID::kF32;
    case NumericTypeID::kCF64: return NumericTypeID::kF64;
    case NumericTypeID::kCBF16: return NumericTypeID::kBF16;
    case NumericTypeID::kCTF32: return NumericTypeID::kTF32;
    default: break;
  }
  return type;
}

/// Returns true if numeric type is integer
bool is_integer_type(NumericTypeID type) {
  switch (type) {
    case NumericTypeID::kS2: return true;
    case NumericTypeID::kS4: return true;
    case NumericTypeID::kS8: return true;
    case NumericTypeID::kS16: return true;
    case NumericTypeID::kS32: return true;
    case NumericTypeID::kS64: return true;
    case NumericTypeID::kU2: return true;
    case NumericTypeID::kU4: return true;
    case NumericTypeID::kU8: return true;
    case NumericTypeID::kU16: return true;
    case NumericTypeID::kU32: return true;
    case NumericTypeID::kU64: return true;
    default: break;
  }
  return false;
}

/// Returns true if numeric type is signed
bool is_signed_type(NumericTypeID type) {
  switch (type) {
    case NumericTypeID::kFE4M3: return true;
    case NumericTypeID::kFE5M2: return true;
    
    case NumericTypeID::kF8: return true;
    case NumericTypeID::kF6: return true;
    case NumericTypeID::kF4: return true;
    case NumericTypeID::kFE2M3: return true;
    case NumericTypeID::kFE3M2: return true;
    case NumericTypeID::kFE2M1: return true;
    case NumericTypeID::kFUE8M0: return false;
    case NumericTypeID::kFUE4M3: return false;
    case NumericTypeID::kF16: return true;
    case NumericTypeID::kBF16: return true;
    case NumericTypeID::kTF32: return true;
    case NumericTypeID::kF32: return true;
    case NumericTypeID::kF64: return true;
    case NumericTypeID::kS2: return true;
    case NumericTypeID::kS4: return true;
    case NumericTypeID::kS8: return true;
    case NumericTypeID::kS16: return true;
    case NumericTypeID::kS32: return true;
    case NumericTypeID::kS64: return true;
    default: break;
  }
  return false;
}

/// Returns true if numeric type is a signed integer
bool is_signed_integer(NumericTypeID type) {
  return is_integer_type(type) && is_signed_type(type);
}

/// returns true if numeric type is an unsigned integer
bool is_unsigned_integer(NumericTypeID type) {
  return is_integer_type(type) && !is_signed_type(type);
}

/// Returns true if numeric type is floating-point type
bool is_float_type(NumericTypeID type) {
  switch (type) {
  case NumericTypeID::kFE4M3: return true;
  case NumericTypeID::kFE5M2: return true;
  
  case NumericTypeID::kF8: return true;
  case NumericTypeID::kF6: return true;
  case NumericTypeID::kF4: return true;
  case NumericTypeID::kFE2M3: return true;
  case NumericTypeID::kFE3M2: return true;
  case NumericTypeID::kFE2M1: return true;
  case NumericTypeID::kFUE8M0: return true;
  case NumericTypeID::kFUE4M3: return true;
  case NumericTypeID::kF16: return true;
  case NumericTypeID::kBF16: return true;
  case NumericTypeID::kTF32: return true;
  case NumericTypeID::kF32: return true;
  case NumericTypeID::kF64: return true;
  case NumericTypeID::kCF16: return true;
  case NumericTypeID::kCBF16: return true;
  case NumericTypeID::kCTF32: return true;
  case NumericTypeID::kCF32: return true;
  case NumericTypeID::kCF64: return true;
  default: break;
  }
  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  LayoutTypeID layout;
  char const *alias;
}
layout_aliases[] = {
  {LayoutTypeID::kUnknown, "unknown"},
  {LayoutTypeID::kRowMajor, "row"},
  {LayoutTypeID::kRowMajor, "t"},
  {LayoutTypeID::kColumnMajor, "column"},
  {LayoutTypeID::kColumnMajor, "col"},
  {LayoutTypeID::kColumnMajor, "n"},
 
  {LayoutTypeID::kColumnMajorInterleavedK2, "nk2"},
  {LayoutTypeID::kRowMajorInterleavedK2, "tk2"},
 
  {LayoutTypeID::kColumnMajorInterleavedK4, "nk4"},
  {LayoutTypeID::kRowMajorInterleavedK4, "tk4"},
  
  {LayoutTypeID::kColumnMajorInterleavedK16, "nk16"},
  {LayoutTypeID::kRowMajorInterleavedK16, "tk16"},
  
  {LayoutTypeID::kColumnMajorInterleavedK32, "nk32"},
  {LayoutTypeID::kRowMajorInterleavedK32, "tk32"},

  {LayoutTypeID::kColumnMajorInterleavedK64, "nk64"},
  {LayoutTypeID::kRowMajorInterleavedK64, "tk64"},

  {LayoutTypeID::kTensorNCHW, "nchw"},
  {LayoutTypeID::kTensorNCDHW, "ncdhw"},
  {LayoutTypeID::kTensorNHWC, "nhwc"},
  {LayoutTypeID::kTensorNDHWC, "ndhwc"},
  {LayoutTypeID::kTensorNC32HW32, "nc32hw32"},
  {LayoutTypeID::kTensorNC64HW64, "nc64hw64"},
  {LayoutTypeID::kTensorC32RSK32, "c32rsk32"},
  {LayoutTypeID::kTensorC64RSK64, "c64rsk64"},

  {LayoutTypeID::kUnknown, "*"},
  {LayoutTypeID::kInvalid, nullptr}
};

/// Converts a LayoutTypeID enumerant to a string
char const *to_string(LayoutTypeID layout, bool pretty) {
  for (auto const & alias : layout_aliases) {
    if (alias.layout == layout) {
      return alias.alias;
    }
  }
  return pretty ? "Invalid" : "invalid";
}

/// Parses a LayoutTypeID enumerant from a string
template <>
LayoutTypeID from_string<LayoutTypeID>(std::string const &str) {
  for (auto const & alias : layout_aliases) {
    if (str.compare(alias.alias) == 0) {
      return alias.layout;
    }
  }
  return LayoutTypeID::kInvalid;
}

/// Gets stride rank for the layout_id (static function)
int get_layout_stride_rank(LayoutTypeID layout_id) {
  switch (layout_id) {
    case LayoutTypeID::kColumnMajor:
      return cutlass::layout::ColumnMajor::kStrideRank;
    case LayoutTypeID::kRowMajor:
      return cutlass::layout::RowMajor::kStrideRank;
    case LayoutTypeID::kColumnMajorInterleavedK2:
      return cutlass::layout::ColumnMajorInterleaved<2>::kStrideRank;
    case LayoutTypeID::kRowMajorInterleavedK2:
      return cutlass::layout::RowMajorInterleaved<2>::kStrideRank;
    case LayoutTypeID::kColumnMajorInterleavedK4:
      return cutlass::layout::ColumnMajorInterleaved<4>::kStrideRank;
    case LayoutTypeID::kRowMajorInterleavedK4:
      return cutlass::layout::RowMajorInterleaved<4>::kStrideRank;
    case LayoutTypeID::kColumnMajorInterleavedK16:
      return cutlass::layout::ColumnMajorInterleaved<16>::kStrideRank;
    case LayoutTypeID::kRowMajorInterleavedK16:
      return cutlass::layout::RowMajorInterleaved<16>::kStrideRank;
    case LayoutTypeID::kColumnMajorInterleavedK32:
      return cutlass::layout::ColumnMajorInterleaved<32>::kStrideRank;
    case LayoutTypeID::kRowMajorInterleavedK32:
      return cutlass::layout::RowMajorInterleaved<32>::kStrideRank;
    case LayoutTypeID::kColumnMajorInterleavedK64:
      return cutlass::layout::ColumnMajorInterleaved<64>::kStrideRank;
    case LayoutTypeID::kRowMajorInterleavedK64:
      return cutlass::layout::RowMajorInterleaved<64>::kStrideRank;
    case LayoutTypeID::kTensorNCHW:
      return cutlass::layout::TensorNCHW::kStrideRank;
    case LayoutTypeID::kTensorNHWC:
      return cutlass::layout::TensorNHWC::kStrideRank;
    case LayoutTypeID::kTensorNDHWC:
      return cutlass::layout::TensorNDHWC::kStrideRank;
    case LayoutTypeID::kTensorNC32HW32:
      return cutlass::layout::TensorNCxHWx<32>::kStrideRank;
    case LayoutTypeID::kTensorNC64HW64:
      return cutlass::layout::TensorNCxHWx<64>::kStrideRank;
    case LayoutTypeID::kTensorC32RSK32:
      return cutlass::layout::TensorCxRSKx<32>::kStrideRank;
    case LayoutTypeID::kTensorC64RSK64:
      return cutlass::layout::TensorCxRSKx<64>::kStrideRank;
    default:
      throw std::runtime_error("Unsupported LayoutTypeID in LayoutType::get_stride_rank");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  OpcodeClassID enumerant;
}
OpcodeClassID_enumerants[] = {
  {"simt", "<simt>", OpcodeClassID::kSimt},
  {"tensorop", "<tensorop>", OpcodeClassID::kTensorOp},
  {"wmmatensorop", "<wmmatensorop>", OpcodeClassID::kWmmaTensorOp},
  {"wmma", "<wmma>", OpcodeClassID::kWmmaTensorOp},
  {"sptensorop", "<sptensorop>", OpcodeClassID::kSparseTensorOp}
};

/// Converts a OpcodeClassID enumerant to a string
char const *to_string(OpcodeClassID type, bool pretty) {

  for (auto const & possible : OpcodeClassID_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Converts a OpcodeClassID enumerant from a string
template <>
OpcodeClassID from_string<OpcodeClassID>(std::string const &str) {

  for (auto const & possible : OpcodeClassID_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return OpcodeClassID::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  ComplexTransform enumerant;
}
ComplexTransform_enumerants[] = {
  {"n", "none", ComplexTransform::kNone},
  {"c", "conj", ComplexTransform::kConjugate}
};

/// Converts a ComplexTransform enumerant to a string
char const *to_string(ComplexTransform type, bool pretty) {

  for (auto const & possible : ComplexTransform_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Converts a ComplexTransform enumerant from a string
template <>
ComplexTransform from_string<ComplexTransform>(std::string const &str) {

  for (auto const & possible : ComplexTransform_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return ComplexTransform::kInvalid;
}


static struct {
  char const *text;
  char const *pretty;
  SplitKMode enumerant;
}
SplitKMode_enumerants[] = {
  {"serial", "<serial>", SplitKMode::kSerial},
  {"parallel", "<parallel>", SplitKMode::kParallel},
};

/// Converts a SplitKMode enumerant to a string
char const *to_string(SplitKMode type, bool pretty) {

  for (auto const & possible : SplitKMode_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Converts a SplitKMode enumerant from a string
template <>
SplitKMode from_string<SplitKMode>(std::string const &str) {

  for (auto const & possible : SplitKMode_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return SplitKMode::kInvalid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
static struct {
  char const *text;
  char const *pretty;
  ConvModeID enumerant;
}
ConvModeID_enumerants[] = {
  {"cross", "<cross>", ConvModeID::kCrossCorrelation},
  {"conv", "<conv>", ConvModeID::kConvolution},
};

/// Converts a ConvModeID enumerant to a string
char const *to_string(ConvModeID type, bool pretty) {

  for (auto const & possible : ConvModeID_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Converts a ConvModeID enumerant from a string
template <>
ConvModeID from_string<ConvModeID>(std::string const &str) {

  for (auto const & possible : ConvModeID_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return ConvModeID::kInvalid;
}


static struct {
  char const *text;
  char const *pretty;
  IteratorAlgorithmID enumerant;
}
IteratorAlgorithmID_enumerants[] = {
  {"none", "<none>", IteratorAlgorithmID::kNone},
  {"analytic", "<analytic>", IteratorAlgorithmID::kAnalytic},
  {"optimized", "<optimized>", IteratorAlgorithmID::kOptimized},
  {"fixed_channels", "<fixed_channels>", IteratorAlgorithmID::kFixedChannels},
  {"few_channels", "<few_channels>", IteratorAlgorithmID::kFewChannels},
};

/// Converts a ConvModeID enumerant to a string
char const *to_string(IteratorAlgorithmID type, bool pretty) {

  for (auto const & possible : IteratorAlgorithmID_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}

/// Converts a ConvModeID enumerant from a string
template <>
IteratorAlgorithmID from_string<IteratorAlgorithmID>(std::string const &str) {

  for (auto const & possible : IteratorAlgorithmID_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return IteratorAlgorithmID::kInvalid;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  ConvKind enumerant;
}
ConvKind_enumerants[] = {
  {"unknown", "<unknown>", ConvKind::kUnknown},
  {"fprop", "<fprop>", ConvKind::kFprop},
  {"dgrad", "<dgrad>", ConvKind::kDgrad},
  {"wgrad", "<wgrad>", ConvKind::kWgrad},
};

/// Converts a ConvKind enumerant to a string
char const *to_string(ConvKind type, bool pretty) {

  for (auto const & possible : ConvKind_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}


/// Converts a ConvKind enumerant from a string
template <>
ConvKind from_string<ConvKind>(std::string const &str) {

  for (auto const & possible : ConvKind_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0)) {
      return possible.enumerant;
    }
  }

  return ConvKind::kInvalid;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  char const *character;
  RasterOrder enumerant;
}
RasterOrder_enumerants[] = {
  {"along_n", "<along_n>", "N", RasterOrder::kAlongN},
  {"along_m", "<along_m>", "M", RasterOrder::kAlongM},
  {"heuristic", "<heuristic>", "H", RasterOrder::kHeuristic},
};

/// Converts a RasterOrder enumerant to a string
char const *to_string(RasterOrder type, bool pretty) {

  for (auto const & possible : RasterOrder_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}


/// Converts a RasterOrder enumerant from a string
template <>
RasterOrder from_string<RasterOrder>(std::string const &str) {

  for (auto const & possible : RasterOrder_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0) ||
        (str.compare(possible.character) == 0)) {
      return possible.enumerant;
    }
  }

  return RasterOrder::kInvalid;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const *text;
  char const *pretty;
  char const *character;
  bool enumerant;
}
Bool_enumerants[] = {
  {"true", "<true>", "t", true},
  {"false", "<false>", "f", false},
};

/// Converts a RasterOrder enumerant to a string
char const *to_string(bool type, bool pretty) {

  for (auto const & possible : Bool_enumerants) {
    if (type == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      }
      else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}


/// Converts a RasterOrder enumerant from a string
template <>
bool from_string<bool>(std::string const &str) {

  for (auto const & possible : Bool_enumerants) {
    if ((str.compare(possible.text) == 0) ||
        (str.compare(possible.pretty) == 0) ||
        (str.compare(possible.character) == 0)) {
      return possible.enumerant;
    }
  }

  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Lexical cast a string to a byte array. Returns true if cast is successful or false if invalid.
bool lexical_cast(std::vector<uint8_t> &bytes, NumericTypeID type, std::string const &str) {
  int size_bytes = sizeof_bits(type) / 8;
  if (!size_bytes) {
    return false;
  }

  bytes.resize(size_bytes, 0);

  std::stringstream ss;
  ss << str;

  switch (type) {
  case NumericTypeID::kU8:
  {
    ss >> *reinterpret_cast<uint8_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kU16:
  {
    ss >> *reinterpret_cast<uint16_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kU32:
  {
    ss >> *reinterpret_cast<uint32_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kU64:
  {
    ss >> *reinterpret_cast<uint64_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kS8:
  {
    ss >> *reinterpret_cast<int8_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kS16:
  {
    ss >> *reinterpret_cast<int16_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kS32:
  {
    ss >> *reinterpret_cast<int32_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kS64:
  {
    ss >> *reinterpret_cast<int64_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kFE4M3:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<float_e4m3_t *>(bytes.data()) = static_cast<float_e4m3_t>(tmp);
  }
    break;
  case NumericTypeID::kFE5M2:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<float_e5m2_t *>(bytes.data()) = static_cast<float_e5m2_t>(tmp);
  }
    break;
  case NumericTypeID::kFE2M3:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<float_e2m3_t *>(bytes.data()) = static_cast<float_e2m3_t>(tmp);
  }
    break;
  case NumericTypeID::kFE3M2:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<float_e3m2_t *>(bytes.data()) = static_cast<float_e3m2_t>(tmp);
  }
    break;
  case NumericTypeID::kFE2M1:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<float_e2m1_t *>(bytes.data()) = static_cast<float_e2m1_t>(tmp);
  }
    break;
  case NumericTypeID::kFUE8M0:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<float_ue8m0_t *>(bytes.data()) = static_cast<float_ue8m0_t>(tmp);
  }
    break;
  case NumericTypeID::kFUE4M3:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<float_ue4m3_t *>(bytes.data()) = static_cast<float_ue4m3_t>(tmp);
  }
    break;
  case NumericTypeID::kF16:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<half_t *>(bytes.data()) = static_cast<half_t>(tmp);
  }
    break;
  case NumericTypeID::kBF16:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<bfloat16_t *>(bytes.data()) = static_cast<bfloat16_t>(tmp);
  }
    break;
  case NumericTypeID::kTF32:
  {
    float tmp;
    ss >> tmp;
    *reinterpret_cast<tfloat32_t *>(bytes.data()) = static_cast<tfloat32_t>(tmp);
  }
    break;
  case NumericTypeID::kF32:
  {
    ss >> *reinterpret_cast<float *>(bytes.data());
  }
    break;
  case NumericTypeID::kF64:
  {
    ss >> *reinterpret_cast<double *>(bytes.data());
  }
    break;
  case NumericTypeID::kCF16:
  {
    std::complex<float> tmp;
    ss >> tmp;
    cutlass::complex<cutlass::half_t> *x = reinterpret_cast<cutlass::complex<half_t> *>(bytes.data());
    x->real() = static_cast<half_t>(std::real(tmp));
    x->imag() = static_cast<half_t>(std::imag(tmp));
  }
    break;
  case NumericTypeID::kCBF16:
  {
    std::complex<float> tmp;
    ss >> tmp;
    cutlass::complex<cutlass::bfloat16_t> *x = reinterpret_cast<cutlass::complex<bfloat16_t> *>(bytes.data());
    x->real() = static_cast<bfloat16_t>(std::real(tmp));
    x->imag() = static_cast<bfloat16_t>(std::imag(tmp));
  }
    break;
  case NumericTypeID::kCF32:
  {
    ss >> *reinterpret_cast<std::complex<float>*>(bytes.data());
  }
    break;
  case NumericTypeID::kCTF32:
  {
    std::complex<float> tmp;
    ss >> tmp;
    cutlass::complex<cutlass::tfloat32_t> *x = reinterpret_cast<cutlass::complex<tfloat32_t> *>(bytes.data());
    x->real() = static_cast<tfloat32_t>(std::real(tmp));
    x->imag() = static_cast<tfloat32_t>(std::imag(tmp));
  }
    break;
  case NumericTypeID::kCF64:
  {
    ss >> *reinterpret_cast<std::complex<double>*>(bytes.data());
  }
    break;
  default:
    return false;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::string lexical_cast(int64_t int_value) {
  std::stringstream ss;
  ss << int_value;
  return ss.str();
}

/// Lexical cast TO a string FROM a byte array. Returns true if cast is successful or false if invalid.
std::string lexical_cast(std::vector<uint8_t> &bytes, NumericTypeID type) {

  size_t size_bytes = sizeof_bits(type) / 8;

  if (!size_bytes || size_bytes != bytes.size()) {
    return "<invalid>";
  }

  bytes.resize(size_bytes, 0);

  std::stringstream ss;

  switch (type) {
  case NumericTypeID::kU8:
  {
    ss << *reinterpret_cast<uint8_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kU16:
  {
    ss << *reinterpret_cast<uint16_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kU32:
  {
    ss << *reinterpret_cast<uint32_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kU64:
  {
    ss << *reinterpret_cast<uint64_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kS8:
  {
    ss << *reinterpret_cast<int8_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kS16:
  {
    ss << *reinterpret_cast<int16_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kS32:
  {
    ss << *reinterpret_cast<int32_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kS64:
  {
    ss << *reinterpret_cast<int64_t *>(bytes.data());
  }
    break;
  case NumericTypeID::kFE4M3:
  {
    float tmp = *reinterpret_cast<float_e4m3_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kFE5M2:
  {
    float tmp = *reinterpret_cast<float_e5m2_t *>(bytes.data());
    ss << tmp;
  }
    break;
  
  case NumericTypeID::kFE2M3:
  {
    float tmp = *reinterpret_cast<float_e2m3_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kFE3M2:
  {
    float tmp = *reinterpret_cast<float_e3m2_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kFE2M1:
  {
    float tmp = *reinterpret_cast<float_e2m1_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kFUE8M0:
  {
    float tmp = *reinterpret_cast<float_ue8m0_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kFUE4M3:
  {
    float tmp = *reinterpret_cast<float_ue4m3_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kF16:
  {
    float tmp = *reinterpret_cast<half_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kBF16:
  {
    float tmp = *reinterpret_cast<bfloat16_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kTF32:
  {
    float tmp = *reinterpret_cast<tfloat32_t *>(bytes.data());
    ss << tmp;
  }
    break;
  case NumericTypeID::kF32:
  {
    ss << *reinterpret_cast<float *>(bytes.data());
  }
    break;
  case NumericTypeID::kF64:
  {
    ss << *reinterpret_cast<double *>(bytes.data());
  }
    break;
  case NumericTypeID::kCF16:
  {
    cutlass::complex<half_t> const *x = 
      reinterpret_cast<cutlass::complex<half_t> const *>(bytes.data());

    ss << float(x->real());

    if (x->imag() != cutlass::half_t()) {
      ss << "+i" << float(x->imag());
    }
  }
    break;
  case NumericTypeID::kCBF16:
  {
    cutlass::complex<bfloat16_t> const *x = 
      reinterpret_cast<cutlass::complex<bfloat16_t> const *>(bytes.data());

    ss << float(x->real());

    if (x->imag() != cutlass::bfloat16_t()) {
      ss << "+i" << float(x->imag());
    }
  }
    break;
  case NumericTypeID::kCF32:
  {
    cutlass::complex<float> const * x = reinterpret_cast<cutlass::complex<float> const *>(bytes.data());

    ss << x->real();

    if (x->imag() != float()) {
      ss << "+i" << x->imag();
    }
  }
    break;
  case NumericTypeID::kCTF32:
  {
    cutlass::complex<tfloat32_t> const * x = reinterpret_cast<cutlass::complex<tfloat32_t> const *>(bytes.data());

    ss << float(x->real());

    if (x->imag() != tfloat32_t()) {
      ss << "+i" << float(x->imag());
    }
  }
    break;
  case NumericTypeID::kCF64:
  {
    cutlass::complex<double> const * x = reinterpret_cast<cutlass::complex<double> const *>(bytes.data());
    
    ss << x->real();

    if (x->imag() != double()) {
      ss << "+i" << x->imag();
    }
  }
    break;
  default:
    return "<unknown>";
  }

  return ss.str();
}

/// Casts from a signed int64 to the destination type. Returns true if successful.
bool cast_from_int64(std::vector<uint8_t> &bytes, NumericTypeID type, int64_t src) {
  int size_bytes = sizeof_bits(type) / 8;
  if (!size_bytes) {
    return false;
  }

  bytes.resize(size_bytes, 0);

  switch (type) {
  case NumericTypeID::kU8:
  {
    *reinterpret_cast<uint8_t *>(bytes.data()) = static_cast<uint8_t>(src);
  }
    break;
  case NumericTypeID::kU16:
  {
    *reinterpret_cast<uint16_t *>(bytes.data()) = static_cast<uint16_t>(src);
  }
    break;
  case NumericTypeID::kU32:
  {
    *reinterpret_cast<uint32_t *>(bytes.data()) = static_cast<uint32_t>(src);
  }
    break;
  case NumericTypeID::kU64:
  {
    *reinterpret_cast<uint64_t *>(bytes.data()) = static_cast<uint64_t>(src);
  }
    break;
  case NumericTypeID::kS8:
  {
    *reinterpret_cast<int8_t *>(bytes.data()) = static_cast<int8_t>(src);
  }
    break;
  case NumericTypeID::kS16:
  {
    *reinterpret_cast<int16_t *>(bytes.data()) = static_cast<int16_t>(src);
  }
    break;
  case NumericTypeID::kS32:
  {
    *reinterpret_cast<int32_t *>(bytes.data()) = static_cast<int32_t>(src);
  }
    break;
  case NumericTypeID::kS64:
  {
    *reinterpret_cast<int64_t *>(bytes.data()) = static_cast<int64_t>(src);
  }
    break;
  case NumericTypeID::kFE4M3:
  {
    *reinterpret_cast<float_e4m3_t *>(bytes.data()) = static_cast<float_e4m3_t>(float(src));
  }
    break;
  case NumericTypeID::kFE5M2:
  {
    *reinterpret_cast<float_e5m2_t *>(bytes.data()) = static_cast<float_e5m2_t>(float(src));
  }
    break;
  
  case NumericTypeID::kFE2M3:
  {
    *reinterpret_cast<float_e2m3_t *>(bytes.data()) = static_cast<float_e2m3_t>(float(src));
  }
    break;
  case NumericTypeID::kFE3M2:
  {
    *reinterpret_cast<float_e3m2_t *>(bytes.data()) = static_cast<float_e3m2_t>(float(src));
  }
    break;
  case NumericTypeID::kFE2M1:
  {
    *reinterpret_cast<float_e2m1_t *>(bytes.data()) = static_cast<float_e2m1_t>(float(src));
  }
    break;
  case NumericTypeID::kFUE8M0:
  {
    *reinterpret_cast<float_ue8m0_t *>(bytes.data()) = static_cast<float_ue8m0_t>(float(src));
  }
    break;
  case NumericTypeID::kFUE4M3:
  {
    *reinterpret_cast<float_ue4m3_t *>(bytes.data()) = static_cast<float_ue4m3_t>(float(src));
  }
    break;
  case NumericTypeID::kF16:
  {
    *reinterpret_cast<half_t *>(bytes.data()) = static_cast<half_t>(float(src));
  }
    break;
  case NumericTypeID::kBF16:
  {
    *reinterpret_cast<bfloat16_t *>(bytes.data()) = static_cast<bfloat16_t>(float(src));
  }
    break;
  case NumericTypeID::kTF32:
  {
    *reinterpret_cast<tfloat32_t *>(bytes.data()) = static_cast<tfloat32_t>(float(src));
  }
    break;
  case NumericTypeID::kF32:
  {
    *reinterpret_cast<float *>(bytes.data()) = static_cast<float>(src);
  }
    break;
  case NumericTypeID::kF64:
  {
    *reinterpret_cast<double *>(bytes.data()) = double(src);
  }
    break;
  case NumericTypeID::kCF16:
  {
    cutlass::complex<cutlass::half_t> *x = reinterpret_cast<cutlass::complex<half_t> *>(bytes.data());
    x->real() = static_cast<half_t>(float(src));
    x->imag() = static_cast<half_t>(float(0));
  }
    break;
  case NumericTypeID::kCF32:
  {
    *reinterpret_cast<cutlass::complex<float>*>(bytes.data()) = cutlass::complex<float>(float(src), float(0));
  }
    break;
  case NumericTypeID::kCF64:
  {
    *reinterpret_cast<cutlass::complex<double>*>(bytes.data()) = cutlass::complex<double>(double(src), double(0));
  }
    break;
  default:
    return false;
  }

  return true;

}

/// Casts from an unsigned int64 to the destination type. Returns true if successful.
bool cast_from_uint64(std::vector<uint8_t> &bytes, NumericTypeID type, uint64_t src) {
  int size_bytes = sizeof_bits(type) / 8;
  if (!size_bytes) {
    return false;
  }

  bytes.resize(size_bytes, 0);

  switch (type) {
  case NumericTypeID::kU8:
  {
    *reinterpret_cast<uint8_t *>(bytes.data()) = static_cast<uint8_t>(src);
  }
    break;
  case NumericTypeID::kU16:
  {
    *reinterpret_cast<uint16_t *>(bytes.data()) = static_cast<uint16_t>(src);
  }
    break;
  case NumericTypeID::kU32:
  {
    *reinterpret_cast<uint32_t *>(bytes.data()) = static_cast<uint32_t>(src);
  }
    break;
  case NumericTypeID::kU64:
  {
    *reinterpret_cast<uint64_t *>(bytes.data()) = static_cast<uint64_t>(src);
  }
    break;
  case NumericTypeID::kS8:
  {
    *reinterpret_cast<int8_t *>(bytes.data()) = static_cast<int8_t>(src);
  }
    break;
  case NumericTypeID::kS16:
  {
    *reinterpret_cast<int16_t *>(bytes.data()) = static_cast<int16_t>(src);
  }
    break;
  case NumericTypeID::kS32:
  {
    *reinterpret_cast<int32_t *>(bytes.data()) = static_cast<int32_t>(src);
  }
    break;
  case NumericTypeID::kS64:
  {
    *reinterpret_cast<int64_t *>(bytes.data()) = static_cast<int64_t>(src);
  }
    break;
  case NumericTypeID::kFE4M3:
  {
    *reinterpret_cast<float_e4m3_t *>(bytes.data()) = static_cast<float_e4m3_t>(float(src));
  }
    break;
  case NumericTypeID::kFE5M2:
  {
    *reinterpret_cast<float_e5m2_t *>(bytes.data()) = static_cast<float_e5m2_t>(float(src));
  }
    break;
  
  case NumericTypeID::kFE2M3:
  {
    *reinterpret_cast<float_e2m3_t *>(bytes.data()) = static_cast<float_e2m3_t>(float(src));
  }
    break;
  case NumericTypeID::kFE3M2:
  {
    *reinterpret_cast<float_e3m2_t *>(bytes.data()) = static_cast<float_e3m2_t>(float(src));
  }
    break;
  case NumericTypeID::kFE2M1:
  {
    *reinterpret_cast<float_e2m1_t *>(bytes.data()) = static_cast<float_e2m1_t>(float(src));
  }
    break;
  case NumericTypeID::kFUE8M0:
  {
    *reinterpret_cast<float_ue8m0_t *>(bytes.data()) = static_cast<float_ue8m0_t>(float(src));
  }
    break;
  case NumericTypeID::kFUE4M3:
  {
    *reinterpret_cast<float_ue4m3_t *>(bytes.data()) = static_cast<float_ue4m3_t>(float(src));
  }
    break;
  case NumericTypeID::kF16:
  {
    *reinterpret_cast<half_t *>(bytes.data()) = static_cast<half_t>(float(src));
  }
    break;
  case NumericTypeID::kBF16:
  {
    *reinterpret_cast<bfloat16_t *>(bytes.data()) = static_cast<bfloat16_t>(float(src));
  }
    break;
  case NumericTypeID::kTF32:
  {
    *reinterpret_cast<tfloat32_t *>(bytes.data()) = static_cast<tfloat32_t>(float(src));
  }
    break;
  case NumericTypeID::kF32:
  {
    *reinterpret_cast<float *>(bytes.data()) = static_cast<float>(src);
  }
    break;
  case NumericTypeID::kF64:
  {
    *reinterpret_cast<double *>(bytes.data()) = double(src);
  }
    break;
  case NumericTypeID::kCF16:
  {
    cutlass::complex<cutlass::half_t> *x = reinterpret_cast<cutlass::complex<half_t> *>(bytes.data());
    x->real() = static_cast<half_t>(float(src));
    x->imag() = static_cast<half_t>(float(0));
  }
    break;
  case NumericTypeID::kCF32:
  {
    *reinterpret_cast<std::complex<float>*>(bytes.data()) = std::complex<float>(float(src), float(0));
  }
    break;
  case NumericTypeID::kCF64:
  {
    *reinterpret_cast<std::complex<double>*>(bytes.data()) = std::complex<double>(double(src), double(0));
  }
    break;
  default:
    return false;
  }

  return true;

}

/// Lexical cast a string to a byte array. Returns true if cast is successful or false if invalid.
bool cast_from_double(std::vector<uint8_t> &bytes, NumericTypeID type, double src) {

  int size_bytes = sizeof_bits(type) / 8;
  if (!size_bytes) {
    return false;
  }

  bytes.resize(size_bytes, 0);

  switch (type) {
  case NumericTypeID::kU8:
  {
    *reinterpret_cast<uint8_t *>(bytes.data()) = static_cast<uint8_t>(src);
  }
    break;
  case NumericTypeID::kU16:
  {
    *reinterpret_cast<uint16_t *>(bytes.data()) = static_cast<uint16_t>(src);
  }
    break;
  case NumericTypeID::kU32:
  {
    *reinterpret_cast<uint32_t *>(bytes.data()) = static_cast<uint32_t>(src);
  }
    break;
  case NumericTypeID::kU64:
  {
    *reinterpret_cast<uint64_t *>(bytes.data()) = static_cast<uint64_t>(src);
  }
    break;
  case NumericTypeID::kS8:
  {
    *reinterpret_cast<int8_t *>(bytes.data()) = static_cast<int8_t>(src);
  }
    break;
  case NumericTypeID::kS16:
  {
    *reinterpret_cast<int16_t *>(bytes.data()) = static_cast<int16_t>(src);
  }
    break;
  case NumericTypeID::kS32:
  {
    *reinterpret_cast<int32_t *>(bytes.data()) = static_cast<int32_t>(src);
  }
    break;
  case NumericTypeID::kS64:
  {
    *reinterpret_cast<int64_t *>(bytes.data()) = static_cast<int64_t>(src);
  }
    break;
  case NumericTypeID::kFE4M3:
  {
    *reinterpret_cast<float_e4m3_t *>(bytes.data()) = static_cast<float_e4m3_t>(float(src));
  }
    break;
  case NumericTypeID::kFE5M2:
  {
    *reinterpret_cast<float_e5m2_t *>(bytes.data()) = static_cast<float_e5m2_t>(float(src));
  }
    break;
  
  case NumericTypeID::kFE2M3:
  {
    *reinterpret_cast<float_e2m3_t *>(bytes.data()) = static_cast<float_e2m3_t>(float(src));
  }
    break;
  case NumericTypeID::kFE3M2:
  {
    *reinterpret_cast<float_e3m2_t *>(bytes.data()) = static_cast<float_e3m2_t>(float(src));
  }
    break;
  case NumericTypeID::kFE2M1:
  {
    *reinterpret_cast<float_e2m1_t *>(bytes.data()) = static_cast<float_e2m1_t>(float(src));
  }
    break;
  case NumericTypeID::kFUE8M0:
  {
    *reinterpret_cast<float_ue8m0_t *>(bytes.data()) = static_cast<float_ue8m0_t>(float(src));
  }
    break;
  case NumericTypeID::kFUE4M3:
  {
    *reinterpret_cast<float_ue4m3_t *>(bytes.data()) = static_cast<float_ue4m3_t>(float(src));
  }
    break;
  case NumericTypeID::kF16:
  {
    *reinterpret_cast<half_t *>(bytes.data()) = static_cast<half_t>(float(src));
  }
    break;
  case NumericTypeID::kBF16:
  {
    *reinterpret_cast<bfloat16_t *>(bytes.data()) = static_cast<bfloat16_t>(float(src));
  }
    break;
  case NumericTypeID::kTF32:
  {
    *reinterpret_cast<tfloat32_t *>(bytes.data()) = static_cast<tfloat32_t>(float(src));
  }
    break;
  case NumericTypeID::kF32:
  {
    *reinterpret_cast<float *>(bytes.data()) = static_cast<float>(src);
  }
    break;
  case NumericTypeID::kF64:
  {
    *reinterpret_cast<double *>(bytes.data()) = src;
  }
    break;
  case NumericTypeID::kCF16:
  {
    cutlass::complex<cutlass::half_t> *x = reinterpret_cast<cutlass::complex<half_t> *>(bytes.data());
    x->real() = static_cast<half_t>(float(src));
    x->imag() = static_cast<half_t>(float(0));
  }
    break;
  case NumericTypeID::kCBF16:
  {
    cutlass::complex<cutlass::bfloat16_t> *x = reinterpret_cast<cutlass::complex<bfloat16_t> *>(bytes.data());
    x->real() = static_cast<bfloat16_t>(bfloat16_t(src));
    x->imag() = static_cast<bfloat16_t>(bfloat16_t(0));
  }
    break;
  case NumericTypeID::kCF32:
  {
    *reinterpret_cast<cutlass::complex<float>*>(bytes.data()) = cutlass::complex<float>(float(src), float());
  }
    break;
  case NumericTypeID::kCTF32:
  {
    *reinterpret_cast<cutlass::complex<tfloat32_t>*>(bytes.data()) = cutlass::complex<tfloat32_t>(tfloat32_t(src), tfloat32_t());
  }
    break;
  case NumericTypeID::kCF64:
  {
    *reinterpret_cast<cutlass::complex<double>*>(bytes.data()) = cutlass::complex<double>(src, double());
  }
    break;
  default:
    return false;
  }

  return true;
}


NumericTypeID dynamic_datatype_to_id(RuntimeDatatype type) {
  NumericTypeID element{};
  switch (type) {
    case RuntimeDatatype::kE4M3:
      element = NumericTypeID::kFE4M3;
      break;
    case RuntimeDatatype::kE5M2:
      element = NumericTypeID::kFE5M2;
      break;
    
    case RuntimeDatatype::kE2M3:
      element = NumericTypeID::kFE2M3;
      break;
    case RuntimeDatatype::kE3M2:
      element = NumericTypeID::kFE3M2;
      break;
    case RuntimeDatatype::kE2M1:
      element = NumericTypeID::kFE2M1;
      break;
    
    default:
      assert("illegal runtime datatype!");
      break;
  }
  return element;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
