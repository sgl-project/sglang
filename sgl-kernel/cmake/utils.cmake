# Adapt from: https://github.com/neuralmagic/vllm-flash-attention/blob/main/cmake/utils.cmake
#
# Clear all `-gencode` flags from `CMAKE_CUDA_FLAGS` and store them in
# `CUDA_ARCH_FLAGS`.
#
# Example:
#   CMAKE_CUDA_FLAGS="-Wall -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"
#   clear_cuda_arches(CUDA_ARCH_FLAGS)
#   CUDA_ARCH_FLAGS="-gencode arch=compute_70,code=sm_70;-gencode arch=compute_75,code=sm_75"
#   CMAKE_CUDA_FLAGS="-Wall"
#
macro(clear_cuda_arches CUDA_ARCH_FLAGS)
    # Extract all `-gencode` flags from `CMAKE_CUDA_FLAGS`
    string(REGEX MATCHALL "-gencode arch=[^ ]+" CUDA_ARCH_FLAGS
      ${CMAKE_CUDA_FLAGS})

    # Remove all `-gencode` flags from `CMAKE_CUDA_FLAGS` since they will be modified
    # and passed back via the `CUDA_ARCHITECTURES` property.
    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
      ${CMAKE_CUDA_FLAGS})
endmacro()


# Macro for converting a `gencode` version number to a cmake version number.
macro(string_to_ver OUT_VER IN_STR)
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
endmacro()

#
# Extract unique CUDA architectures from a list of compute capabilities codes in
# the form `<major><minor>[<letter>]`, convert them to the form sort
# `<major>.<minor>`, dedupes them and then sorts them in ascending order and
# stores them in `OUT_ARCHES`.
#
# Example:
#   CUDA_ARCH_FLAGS="-gencode arch=compute_75,code=sm_75;...;-gencode arch=compute_90a,code=sm_90a"
#   extract_unique_cuda_archs_ascending(OUT_ARCHES CUDA_ARCH_FLAGS)
#   OUT_ARCHES="7.5;...;9.0"
function(extract_unique_cuda_archs_ascending OUT_ARCHES CUDA_ARCH_FLAGS)
  set(_CUDA_ARCHES)
  foreach(_ARCH ${CUDA_ARCH_FLAGS})
    string(REGEX MATCH "arch=compute_\([0-9]+a?\)" _COMPUTE ${_ARCH})
    if (_COMPUTE)
      set(_COMPUTE ${CMAKE_MATCH_1})
    endif()

    if (_COMPUTE)
      string_to_ver(_COMPUTE_VER ${_COMPUTE})
      list(APPEND _CUDA_ARCHES ${_COMPUTE_VER})
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _CUDA_ARCHES)
  list(SORT _CUDA_ARCHES COMPARE NATURAL ORDER ASCENDING)
  set(${OUT_ARCHES} ${_CUDA_ARCHES} PARENT_SCOPE)
endfunction()

#
# For the given `SRC_CUDA_ARCHS` list of gencode versions in the form
#  `<major>.<minor>[letter]` compute the "loose intersection" with the
#  `TGT_CUDA_ARCHS` list of gencodes. We also support the `+PTX` suffix in
#  `SRC_CUDA_ARCHS` which indicates that the PTX code should be built when there
#  is a CUDA_ARCH in `TGT_CUDA_ARCHS` that is equal to or larger than the
#  architecture in `SRC_CUDA_ARCHS`.
# The loose intersection is defined as:
#   { max{ x \in tgt | x <= y } | y \in src, { x \in tgt | x <= y } != {} }
#  where `<=` is the version comparison operator.
# In other words, for each version in `TGT_CUDA_ARCHS` find the highest version
#  in `SRC_CUDA_ARCHS` that is less or equal to the version in `TGT_CUDA_ARCHS`.
# We have special handling for x.0a, if x.0a is in `SRC_CUDA_ARCHS` and x.0 is
#  in `TGT_CUDA_ARCHS` then we should remove x.0a from `SRC_CUDA_ARCHS` and add
#  x.0a to the result (and remove x.0 from TGT_CUDA_ARCHS).
# The result is stored in `OUT_CUDA_ARCHS`.
#
# Example:
#   SRC_CUDA_ARCHS="7.5;8.0;8.6;9.0;9.0a"
#   TGT_CUDA_ARCHS="8.0;8.9;9.0"
#   cuda_archs_loose_intersection(OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
#   OUT_CUDA_ARCHS="8.0;8.6;9.0;9.0a"
#
# Example With PTX:
#   SRC_CUDA_ARCHS="8.0+PTX"
#   TGT_CUDA_ARCHS="9.0"
#   cuda_archs_loose_intersection(OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
#   OUT_CUDA_ARCHS="8.0+PTX"
#
function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
    foreach(arch IN LISTS SRC_CUDA_ARCHS)
        if(arch MATCHES "(.*)\\+PTX$")
            list(APPEND _PTX_ARCHS "${CMAKE_MATCH_1}")
            list(APPEND clean_src "${CMAKE_MATCH_1}")
        else()
            list(APPEND clean_src "${arch}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES clean_src)
    list(REMOVE_DUPLICATES _PTX_ARCHS)

    set(modified_tgt "${TGT_CUDA_ARCHS}")
    foreach(special_ver IN ITEMS 9.0 10.0)
        if("${special_ver}a" IN_LIST clean_src AND "${special_ver}" IN_LIST modified_tgt)
            list(APPEND _CUDA_ARCHS "${special_ver}a")
            list(REMOVE_ITEM modified_tgt "${special_ver}")
            list(REMOVE_ITEM clean_src "${special_ver}a")
        endif()
    endforeach()

    list(SORT clean_src COMPARE NATURAL)
    foreach(target_arch IN LISTS modified_tgt)
        string(REGEX MATCH "^([0-9]+)" tgt_major "${target_arch}")
        set(best_match)
        foreach(src_arch IN LISTS clean_src)
            string(REGEX MATCH "^([0-9]+)" src_major "${src_arch}")
            if(src_arch VERSION_GREATER target_arch)
                break()
            endif()
            if((src_arch IN_LIST _PTX_ARCHS OR src_major STREQUAL tgt_major)
               AND src_arch VERSION_LESS_EQUAL target_arch)
                set(best_match "${src_arch}")
            endif()
        endforeach()
        if(best_match)
            list(APPEND _CUDA_ARCHS "${best_match}")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _CUDA_ARCHS)
    set(final_archs)
    foreach(arch IN LISTS _CUDA_ARCHS)
        list(APPEND final_archs "${arch}$<$IN_LIST:${arch},_PTX_ARCHS>:+PTX>")
    endforeach()

    set(${OUT_CUDA_ARCHS} ${final_archs} PARENT_SCOPE)
endfunction()
