set(ONEMKL_FOUND FALSE)

set(ONEMKL_LIBRARIES)

# In order to be compatible with various situations of Pytorch development
# bundle setup, ENV{MKLROOT} and SYCL_ROOT will be checked sequentially to get
# the root directory of oneMKL.
if(DEFINED ENV{MKLROOT})
  # Directly get the root directory of oneMKL if ENV{MKLROOT} exists.
  set(ONEMKL_ROOT $ENV{MKLROOT})
elseif(SYCL_FOUND)
  # oneMKL configuration may not be imported into the build system. Get the root
  # directory of oneMKL based on the root directory of compiler relatively.
  get_filename_component(ONEMKL_ROOT "${SYCL_ROOT}/../../mkl/latest" REALPATH)
endif()

if(NOT DEFINED ONEMKL_ROOT)
  message(
    WARNING
      "Cannot find either ENV{MKLROOT} or SYCL_ROOT, please setup oneAPI environment before building!!"
  )
  return()
endif()

if(NOT EXISTS ${ONEMKL_ROOT})
  message(
    WARNING
      "${ONEMKL_ROOT} not found, please setup oneAPI environment before building!!"
  )
  return()
endif()

find_file(
  ONEMKL_INCLUDE_DIR
  NAMES include
  HINTS ${ONEMKL_ROOT}
  NO_DEFAULT_PATH)

find_file(
  ONEMKL_LIB_DIR
  NAMES lib
  HINTS ${ONEMKL_ROOT}
  NO_DEFAULT_PATH)

if((ONEMKL_INCLUDE_DIR STREQUAL "ONEMKL_INCLUDE_DIR-NOTFOUND")
   OR(ONEMKL_LIB_DIR STREQUAL "ONEMKL_LIB_DIR-NOTFOUND"))
  message(WARNING "oneMKL sdk is incomplete!!")
  return()
endif()

if(WIN32)
  set(MKL_LIB_NAMES "mkl_sycl_blas" "mkl_sycl_dft" "mkl_sycl_lapack"
                    "mkl_intel_lp64" "mkl_intel_thread" "mkl_core")
  list(TRANSFORM MKL_LIB_NAMES APPEND "_dll.lib")
else()
  set(MKL_LIB_NAMES "mkl_sycl_blas" "mkl_sycl_dft" "mkl_sycl_lapack"
                    "mkl_intel_lp64" "mkl_gnu_thread" "mkl_core")
  list(TRANSFORM MKL_LIB_NAMES PREPEND "lib")
  list(TRANSFORM MKL_LIB_NAMES APPEND ".so")
endif()

foreach(LIB_NAME IN LISTS MKL_LIB_NAMES)
  find_library(
    ${LIB_NAME}_library
    NAMES ${LIB_NAME}
    HINTS ${ONEMKL_LIB_DIR}
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
  list(APPEND ONEMKL_LIBRARIES ${${LIB_NAME}_library})
endforeach()

set(ONEMKL_FOUND TRUE)
