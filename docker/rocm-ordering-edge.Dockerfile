#
# NATIVE ROCm-10 device-resident ordering edges (ROCm/rocm-systems#11212).
#
# Unlike prs/ordering_edge/Dockerfile (which checked out the PR-branch head and thereby
# regressed ROCr to 1.18.0 / HIP 7.2 -- ABI-incompatible with the ROCm-10 image's rocminfo),
# this recipe starts from the image's EXACT ROCm-10.0 base commit (ROCr 1.21.0 / HIP 7.15) and
# applies a cumulative patch that is the #11212 series cherry-picked onto that base.  ROCr thus
# stays at native 1.21.0 (ABI-compatible) while gaining hsa_amd_signal_create_v2 +
# HSA_AMD_SIGNAL_CREATE_DEVICE_MEM_VALUE_WORD and the CLR-side ordering-edge naming.
#
# It rebuilds ROCr (libhsa-runtime64) + CLR/HIP (libamdhip64), swaps those two shared libraries
# into /opt/rocm, and sets ROCPROFILER_QUEUE_INTERPOSITION=0.
#
# Build (BuildKit may be unavailable in this environment; legacy builder works fine):
#   DOCKER_BUILDKIT=0 docker build -f docker/rocm-ordering-edge.Dockerfile \
#     -t lmsysorg/sglang-rocm:v0.5.19-rocm10-mi35x-20260909-edge docker
#
# Revert switch at runtime (no rebuild): DEBUG_CLR_DISABLE_ORDERING_EDGE=1

ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.19-rocm10-mi35x-20260909

# Exact rocm-systems source the ROCm-10.0 image was built from (ROCr 1.21.0 / HIP 7.15).
ARG ROCM_SYSTEMS_REPO=https://github.com/ROCm/rocm-systems.git
ARG ROCM_RUNTIME_COMMIT=6b0e43f341195e203754e08f850e437ff2fc09f9
# Cumulative patch: #11212 (commits 72ee0a5a..25e14349) cherry-picked onto ROCM_RUNTIME_COMMIT.
ARG ORDERING_EDGE_PATCH=ordering_edge_11212_on_rocm10.patch

###
### ROCr + CLR build (device-resident ordering edges)
###
FROM ${BASE_IMAGE} AS build_rocm_runtime
ARG ROCM_SYSTEMS_REPO
ARG ROCM_RUNTIME_COMMIT
ARG ORDERING_EDGE_PATCH

# Build deps for rocr-runtime + clr. cmake/ninja/hipcc/clang already ship in the
# SGLang image; xxd and the -dev libs do not.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
       libelf-dev libdrm-dev libnuma-dev libdw-dev xxd \
       libglvnd-dev libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir CppHeaderParser

# Bring the prebuilt cumulative patch into the image (build context is prs/ordering_edge).
COPY ${ORDERING_EDGE_PATCH} /tmp/ordering_edge.patch

# Sparse, blobless, shallow checkout of just the pieces the patch touches / we build.
# rocprofiler-sdk is included ONLY so the patch applies cleanly; it is not compiled below.
RUN git init -q /src && cd /src \
    && git remote add origin ${ROCM_SYSTEMS_REPO} \
    && git config core.sparseCheckout true \
    && git config remote.origin.promisor true \
    && git config remote.origin.partialclonefilter blob:none \
    && git sparse-checkout init --cone \
    && git sparse-checkout set projects/rocr-runtime projects/clr projects/hip projects/rocprofiler-sdk shared cmake \
    && git fetch --filter=blob:none --depth 1 origin ${ROCM_RUNTIME_COMMIT} \
    && git checkout -q FETCH_HEAD

# Apply the cherry-picked #11212 series. --check first so a bad patch fails loudly and early,
# before the (long) compile, rather than half-applying.
RUN cd /src \
    && git apply --check --verbose /tmp/ordering_edge.patch \
    && git apply --whitespace=nowarn /tmp/ordering_edge.patch \
    && echo "ordering-edge patch applied" \
    && grep -q 'get_version("1.21.0")' projects/rocr-runtime/CMakeLists.txt \
    && grep -q 'hsa_amd_signal_create_v2' projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h

# ROCr's trap_handler/blit_shaders call find_package(Clang/LLVM REQUIRED) only to
# get IMPORTED executables. The ROCm image ships the binaries but no CMake package
# config, so supply a shim pointing at them.
RUN mkdir -p /opt/rocm-cmake-shim \
    && printf 'if(NOT TARGET clang)\n  add_executable(clang IMPORTED GLOBAL)\n  set_target_properties(clang PROPERTIES IMPORTED_LOCATION "/opt/rocm/llvm/bin/clang")\nendif()\nset(Clang_PACKAGE_VERSION "rocm-image-shim")\n' > /opt/rocm-cmake-shim/ClangConfig.cmake \
    && printf 'if(NOT TARGET llvm-objcopy)\n  add_executable(llvm-objcopy IMPORTED GLOBAL)\n  set_target_properties(llvm-objcopy PROPERTIES IMPORTED_LOCATION "/opt/rocm/llvm/bin/llvm-objcopy")\nendif()\nset(LLVM_FOUND TRUE)\nset(LLVM_PACKAGE_VERSION "rocm-image-shim")\n' > /opt/rocm-cmake-shim/LLVMConfig.cmake

# ROCr first, installed into /opt/rocm because the clr build below needs its headers.
RUN cd /src/projects/rocr-runtime \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DClang_DIR=/opt/rocm-cmake-shim \
        -DLLVM_DIR=/opt/rocm-cmake-shim \
    && cmake --build build --parallel "$(nproc)" \
    && cmake --install build \
    && cmake --install build --prefix /rocr-install --strip

# CRITICAL: ROCM_KPACK_ENABLED=ON. The stock image libamdhip64 links librocm_kpack.so.0 and
# parses fat binaries / registered kernels through it; the image's device code is a kpack archive
# (_rocm_sdk_*/.kpack/rand_lib_gfx950.kpack). Building CLR with the default ROCM_KPACK_ENABLED=OFF
# produces a libamdhip64 whose getDeviceKernel() map lookup is incompatible with that device code,
# which SIGSEGVs on the very first kernel launch (independent of the ordering-edge patch). Enabling
# kpack + pointing find_package(rocm-kpack) at the SDK cmake config is what makes the rebuild run.
RUN KPACK_CMAKE=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/cmake/rocm-kpack \
    && cd /src/projects/clr \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLR_BUILD_HIP=ON \
        -DCLR_BUILD_OCL=OFF \
        -DHIP_COMMON_DIR=/src/projects/hip \
        -DROCM_PATH=/opt/rocm \
        -DCMAKE_PREFIX_PATH="/opt/rocm;$KPACK_CMAKE" \
        -Drocm-kpack_DIR="$KPACK_CMAKE" \
        -DROCM_KPACK_ENABLED=ON \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DLLVM_DIR=/opt/rocm-cmake-shim \
        -DHIP_LLVM_ROOT=/opt/rocm/llvm \
    && cmake --build build --parallel "$(nproc)" \
    && cmake --install build --prefix /clr-install --strip \
    && readelf -d /clr-install/lib/libamdhip64.so.7.* | grep -q librocm_kpack

RUN mkdir -p /staging/lib \
    && cp -P /rocr-install/lib/libhsa-runtime64.so* /staging/lib/ \
    && cp -P /clr-install/lib/libamdhip64.so* /staging/lib/ \
    && ls -l /staging/lib

###
### Final: swap the patched runtimes into the SGLang image
###
FROM ${BASE_IMAGE} AS final
ARG ROCM_SYSTEMS_REPO
ARG ROCM_RUNTIME_COMMIT
ARG ORDERING_EDGE_PATCH

COPY --from=build_rocm_runtime /staging/ /staging/

# Swap the patched runtimes into EVERY ROCm-SDK lib dir, not just /opt/rocm.
#
# This image ships ROCm as two pip packages: `_rocm_sdk_devel` (which /opt/rocm
# symlinks to, and which rocminfo loads via LD_LIBRARY_PATH) and `_rocm_sdk_core`
# (which torch/HIP load via RPATH -- this is the ACTUAL inference runtime path).
# Patching only /opt/rocm (== _rocm_sdk_devel) fixes rocminfo but leaves torch on
# the stock unpatched libs, so the ordering-edge feature would be inert at runtime.
# Overwrite the real file behind every libhsa-runtime64.so.1* / libamdhip64.so.7*
# in both packages (and anywhere else under the venv, e.g. torch/lib) so the SONAME
# lookup resolves to the rebuilt library everywhere.
RUN set -eux; \
    hsa_src="$(readlink -f /staging/lib/libhsa-runtime64.so.1)"; \
    hip_src="$(readlink -f /staging/lib/libamdhip64.so.7)"; \
    test -s "$hsa_src"; test -s "$hip_src"; \
    found_hsa=0; found_hip=0; \
    for d in $(find /opt/venv /opt/rocm/lib -xdev -type f \
                 \( -name 'libhsa-runtime64.so.1' -o -name 'libhsa-runtime64.so.1.*' \
                    -o -name 'libamdhip64.so.7' -o -name 'libamdhip64.so.7.*' \) 2>/dev/null); do \
        case "$(basename "$d")" in \
            libhsa*)      cp -f --remove-destination "$hsa_src" "$d"; found_hsa=$((found_hsa+1));; \
            libamdhip64*) cp -f --remove-destination "$hip_src" "$d"; found_hip=$((found_hip+1));; \
        esac; \
        echo "patched $d"; \
    done; \
    echo "swapped libhsa copies=$found_hsa libamdhip64 copies=$found_hip"; \
    [ "$found_hsa" -ge 2 ] && [ "$found_hip" -ge 2 ]

RUN ldconfig

# Matches the vLLM #55099 mitigation for torch-profiler queue-interposition hangs.
ENV ROCPROFILER_QUEUE_INTERPOSITION=0

RUN mkdir -p /app \
    && printf 'ORDERING_EDGE_PATCH: ROCm/rocm-systems#11212 (device-resident ordering edges)\nSTRATEGY: cherry-pick onto image ROCm-10.0 base (ROCr stays 1.21.0)\nROCM_SYSTEMS_REPO: %s\nROCM_RUNTIME_COMMIT: %s\nPATCH_FILE: %s\n' \
        "${ROCM_SYSTEMS_REPO}" "${ROCM_RUNTIME_COMMIT}" "${ORDERING_EDGE_PATCH}" > /app/ordering_edge_patch.txt \
    && cat /app/ordering_edge_patch.txt
