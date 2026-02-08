get_filename_component(NvidiaCutlass_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

include(CMakeFindDependencyMacro)

if(TARGET nvidia::cutlass::CUTLASS)
  return()
endif()

include("${NvidiaCutlass_CMAKE_DIR}/NvidiaCutlassTargets.cmake")
