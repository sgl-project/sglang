# Qwen38 serving image (CUDA 13 / GB300 aarch64: sm_90a + sm_100a + sm_103a).
#
# Base ships stock SGLang (editable at /sgl-workspace/sglang), a DeepEP the
# recipe below replaces, a released FlashInfer trio, the Rust toolchain at
# /root/.cargo, and the CUDA 13 toolchain (nvcc +
# /usr/local/cuda/include/cccl). How DeepEP arrives in the base is not assumed:
# v0.5.17 shipped a source tree at /sgl-workspace/DeepEP plus a `deep_ep`
# distribution, the dev images ship a packaged `sgl-deep-ep` instead, and
# apply_deepep_v2_patch.sh handles either.
#
# This image adds the four Qwen38-specific pieces that stock lacks:
#   1. DeepEP v2 (deepseek-ai@01dc3aa) patched and built from source; see
#      apply_deepep_v2_patch.sh for what the patch does and why
#   2. NCCL pinned to the version the DeepEP v2 wheel requires, staged at
#      /opt/nccl-2.30.7/lib so the srt-slurm configs can LD_PRELOAD it without
#      a host library mount
#   3. FlashInfer, replacing the base's release: flashinfer-python from a pinned
#      git commit (PR #4358, the Blackwell MNNVL CuTe DSL all-reduce backend),
#      with the nightly's prebuilt flashinfer-cubin and flashinfer-jit-cache
#      alongside it, since neither is published per commit.
#      nvidia-cutlass-dsl is floored at 4.7.0 because the CuTe DSL kernels call
#      PipelineTmaAsync.create(enable_multicast_signaling=..)
#   4. this repo's SGLang code, copied from the build context and editable-installed
#
# The build also refuses to start unless the base's torch is the one
# python/pyproject.toml pins, reinstalls the CUDA-tagged wheels this tree pins
# above what the base ships, and asserts those pins before finishing. All three
# live in docker/qwen38/cuda_pins.sh, which explains why each is needed.
#
# Build (no GPU needed; nvcc cross-compiles the DeepEP cubins). The context must
# be the repo root -- the SGLang source is taken from it rather than cloned:
#   docker build -f docker/qwen38/qwen38_cu13.Dockerfile -t qwen38-cu130 .

# The nightly, not a release tag: v0.5.17 ships torch 2.11.0 / sglang-kernel
# 0.4.5, which the gate below rejects against this tree's pins. `dev` is the
# CUDA 13 nightly and carries torch 2.13.0+cu130, sglang-kernel 0.4.6.post1 and
# sgl-deep-gemm 0.1.5.post2 -- exactly what python/pyproject.toml asks for.
#
# `dev` MOVES every night. Builds are therefore not reproducible from this tag
# alone; pin `dev@sha256:<digest>` when a build has to be repeatable.
FROM lmsysorg/sglang:dev AS base

# --- 0. Base/tree compatibility gate, FIRST ---
# Whether this base can host this tree is decided entirely by the base's torch,
# and it is knowable before any work happens. That matters most here: the DeepEP
# v2 nvcc compile below is the longest step in either recipe, and running it
# against a base that cannot host this tree is pure waste.
#
# pyproject.toml is copied to its own path purely to make that possible: the tree
# itself only lands at COPY below, far too late to gate anything. Both copies
# come from the same build context, so they cannot disagree.
COPY docker/qwen38/cuda_pins.sh /opt/qwen38/cuda_pins.sh
COPY python/pyproject.toml /opt/qwen38/pyproject.toml

RUN bash /opt/qwen38/cuda_pins.sh check-torch /opt/qwen38/pyproject.toml

# --- 1. DeepEP v2: patch + build, replacing the base's stock tree and wheel ---
ARG DEEPEP_V2_COMMIT=01dc3aaac82068020353dce2c302e38153c0bfaa
ARG DEEPEP_CUDA_ARCH_LIST="9.0;10.0;10.3"
ARG BUILD_AND_DOWNLOAD_PARALLEL=8

COPY docker/qwen38/apply_deepep_v2_patch.sh /opt/qwen38/

# DeepEP v2's NCCL backend uses the GIN API (ncclCommProperties,
# ncclGinRequest_t, NCCL_GIN_*), which first shipped in NCCL 2.30 -- and the base
# is below that (2.29.7 on `dev` as of 2026-08-10), so its headers do not compile
# it. Install the pinned NCCL BEFORE the DeepEP build so the extension compiles
# and links against the same version that gets preloaded at run time.
ARG NCCL_PIN_VERSION=2.30.7
RUN python3 -m pip install "nvidia-nccl-cu13==${NCCL_PIN_VERSION}" && \
    rm -rf /root/.cache/pip

RUN DEEPEP_V2_COMMIT="${DEEPEP_V2_COMMIT}" \
    TORCH_CUDA_ARCH_LIST="${DEEPEP_CUDA_ARCH_LIST}" \
    MAX_JOBS="${BUILD_AND_DOWNLOAD_PARALLEL}" \
    bash /opt/qwen38/apply_deepep_v2_patch.sh

# Assert the pin was still live for the DeepEP compile above. `pip install X==v`
# does not re-resolve the graph, so nothing should have moved it here -- but the
# headers DeepEP just compiled against are only correct if this holds.
RUN NCCL_EXPECTED="${NCCL_PIN_VERSION}" python3 -c 'import os; from importlib.metadata import version; e = os.environ["NCCL_EXPECTED"]; v = version("nvidia-nccl-cu13"); assert v == e, (v, e)'

# --- 2. NCCL preload path: staged at the END of the build, see below ---

# --- 3. FlashInfer: nightly cubin/jit-cache, python from a pinned commit ---
# Named apart from the base's ENV FLASHINFER_VERSION, which would otherwise
# shadow a same-named ARG and silently resolve to the base's 0.6.15.post1.
#
# The nightly version below no longer covers flashinfer-python: that comes from
# FLASHINFER_GIT_COMMIT instead. cubin and jit-cache stay on the nightly because
# they are prebuilt artifacts published per nightly date and per release only --
# no wheel of either exists for an arbitrary commit. The nightly named here must
# therefore be the one whose main the pinned commit merges, so the prebuilt
# kernels correspond to the source they were compiled from.
ARG FLASHINFER_NIGHTLY_VERSION=0.6.18.dev20260807
ARG FLASHINFER_JIT_CACHE_CUDA_TAG=cu130
ARG CUTLASS_DSL_MIN_VERSION=4.7.0

# flashinfer-ai/flashinfer#4358, "feat(comm): add Blackwell MNNVL CuTe DSL
# all-reduce fusion backend" -- the MNNVL path this image's GB300 target needs.
# Pinned to the PR's merge commit on main, which is what keeps it fetchable. An
# earlier pin of a PR-branch commit (23922f9a) built fine until that branch was
# deleted, after which git refused to serve the object at all -- "upload-pack:
# not our ref", because no remaining ref reaches it. A commit on main cannot rot
# that way. It also carries the PR's two later review fixes, which the branch
# snapshot predated.
ARG FLASHINFER_GIT_REPO=https://github.com/flashinfer-ai/flashinfer.git
ARG FLASHINFER_GIT_COMMIT=906181e3f4cf4bcc81835fb480db4011bbd80b62

# Uninstall first: a mixed python/cubin/jit-cache installation fails at import,
# and pip would otherwise leave the base's jit-cache shadowing JIT compilation.
# Installed with dependency resolution so apache-tvm-ffi lands at whatever the
# jit-cache wheel's metadata requires -- an exact pin here could contradict it.
#
# NOTE: this resolution DOWNGRADES nvidia-nccl-cu13 back off the 2.30.7 pin,
# because torch declares its own hard `nvidia-nccl-cu13==` (2.29.7 on the `dev`
# base) and flashinfer pulls in nccl4py, dragging NCCL into the resolve. The
# exact version torch drags in tracks the base and is not worth hardcoding here;
# what matters is that it is below 2.30 and therefore lacks the GIN API. The pin
# is re-applied after all pip work completes -- see the final stage, which
# asserts the version rather than trusting it.
#
# The git clone needs its submodules, and not as an optimisation: the released
# wheel packages the cccl, cutlass and spdlog headers into flashinfer/data for
# runtime JIT compilation, so a non-recursive clone yields a package that
# imports fine and then cannot compile a kernel. Shallow, blob-filtered, and
# deleted in the same layer because cutlass and cccl are large.
RUN python3 -m pip uninstall -y \
      flashinfer-python flashinfer-cubin flashinfer-jit-cache && \
    rm -rf /root/.cache/flashinfer && \
    python3 -m pip install \
      "flashinfer-cubin==${FLASHINFER_NIGHTLY_VERSION}" \
      "flashinfer-jit-cache==${FLASHINFER_NIGHTLY_VERSION}+${FLASHINFER_JIT_CACHE_CUDA_TAG}" \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
      --extra-index-url "https://flashinfer.ai/whl/nightly/${FLASHINFER_JIT_CACHE_CUDA_TAG}/" && \
    git clone --filter=blob:none "${FLASHINFER_GIT_REPO}" /tmp/flashinfer && \
    git -C /tmp/flashinfer checkout --detach "${FLASHINFER_GIT_COMMIT}" && \
    git -C /tmp/flashinfer submodule update --init --recursive --depth 1 && \
    python3 -m pip install --no-deps /tmp/flashinfer && \
    python3 -m pip install "nvidia-cutlass-dsl[cu13]>=${CUTLASS_DSL_MIN_VERSION}" && \
    # Assert the prebuilt pair is the nightly this commit was matched against.
    # flashinfer-python is deliberately NOT compared -- see the opt-out below.
    FLASHINFER_EXPECTED="${FLASHINFER_NIGHTLY_VERSION}" \
    FLASHINFER_CUDA_TAG="${FLASHINFER_JIT_CACHE_CUDA_TAG}" \
    python3 -c 'import os; from importlib.metadata import version; e = os.environ["FLASHINFER_EXPECTED"]; tag = os.environ["FLASHINFER_CUDA_TAG"]; got = {p: version(p) for p in ("flashinfer-cubin", "flashinfer-jit-cache")}; assert got["flashinfer-cubin"].split("+")[0] == e, got; assert got["flashinfer-jit-cache"].startswith(e + "+" + tag), got' && \
    rm -rf /tmp/flashinfer /root/.cache/pip

# flashinfer-python now reports the pinned commit's version while cubin and
# jit-cache report the nightly's. flashinfer/jit/env.py raises RuntimeError at
# import on exactly that mismatch, for both packages, and this variable is the
# opt-out its own error message tells you to use. The alternative -- installing
# no cubin and no jit-cache, which makes the check skip itself -- would send
# every existing kernel through runtime JIT on first use, so the mismatch is
# accepted knowingly instead.
#
# The cost is that the check is now off for good, including for a mismatch
# nobody intended, which is what the assertions above and below are for.
ENV FLASHINFER_DISABLE_VERSION_CHECK=1

# FLASHINFER_VERSION describes the prebuilt artifacts, which is what a consumer
# reading it wants to know; the python tree is recorded separately because the
# two genuinely differ in this image.
ENV FLASHINFER_VERSION=${FLASHINFER_NIGHTLY_VERSION}
ENV FLASHINFER_PYTHON_GIT_COMMIT=${FLASHINFER_GIT_COMMIT}

LABEL ai.radixark.flashinfer.python_git_commit="${FLASHINFER_GIT_COMMIT}" \
      ai.radixark.flashinfer.prebuilt_nightly="${FLASHINFER_NIGHTLY_VERSION}"

# This tree's BF16 Split-K GEMM loads FlashInfer PR #4266's standalone direct
# kernel by file path out of SGLANG_FLASHINFER_PR4266_SOURCE (see
# python/sglang/srt/layers/quantization/unquant.py). On SM100/SM103 that path is
# on by DEFAULT -- bf16_gemm_backend=auto resolves to cutedsl, and
# SGLANG_ENABLE_BF16_SPLITK_GEMM defaults to True -- so an unset variable makes
# the server raise at startup rather than degrade. Point it at the installed
# FlashInfer through a stable symlink instead of a hardcoded dist-packages path,
# and fail the build now if the kernel is missing: a GB300 image that cannot
# start is worse than a build that stops here.
#
# `import flashinfer` here is load-bearing beyond resolving the path: it runs
# flashinfer/jit/env.py, so it is where a failed version-check opt-out would
# surface. The two file checks then confirm the pinned commit is the tree that
# actually got installed -- checked as files rather than imports because these
# modules pull in CuTe DSL, which cannot load on a CPU build host.
RUN FI_ROOT="$(python3 -c 'import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent.parent)')" && \
    ln -sfn "${FI_ROOT}" /opt/flashinfer-src && \
    if [ ! -f /opt/flashinfer-src/flashinfer/gemm/kernels/dense_bf16_gemm_direct.py ]; then \
        echo "ERROR: flashinfer at ${FLASHINFER_GIT_COMMIT} does not ship flashinfer/gemm/kernels/dense_bf16_gemm_direct.py (PR #4266)." >&2; \
        echo "       Pin a FlashInfer that carries it, or serve with SGLANG_ENABLE_BF16_SPLITK_GEMM=0." >&2; \
        exit 1; \
    fi && \
    if [ ! -f /opt/flashinfer-src/flashinfer/comm/mnnvl_cutedsl/__init__.py ]; then \
        echo "ERROR: flashinfer at ${FLASHINFER_GIT_COMMIT} does not ship flashinfer/comm/mnnvl_cutedsl (PR #4358)." >&2; \
        echo "       That commit is the reason this image builds flashinfer-python from git;" >&2; \
        echo "       if it is absent, the wrong ref was installed." >&2; \
        exit 1; \
    fi

ENV SGLANG_FLASHINFER_PR4266_SOURCE=/opt/flashinfer-src

# --- 4. Qwen38 SGLang code (replaces the base's stock sglang, editable) ---
# rm first: COPY merges into an existing directory, so files the stock release
# has and this tree does not would otherwise survive.
RUN rm -rf /sgl-workspace/sglang

COPY . /sgl-workspace/sglang

# .git is discarded, so setuptools-scm cannot derive a version and would fall
# back to 0.0.0.dev0; pass SGLANG_VERSION to label the build.
# Keep the installed extension modules, but discard Rust and pip build
# artifacts that are not used at runtime.
ARG SGLANG_VERSION=0.0.0.dev0
# Which +<tag> build of the SGLang wheels to pull. Kept separate from the
# FlashInfer jit-cache tag even though both read "cu130" today: they index two
# unrelated wheel sets, and silently reusing one for the other would make a
# FlashInfer retag quietly change which kernel ABI gets installed.
ARG SGL_WHL_CUDA_TAG=cu130
RUN cd /sgl-workspace/sglang && \
    rm -rf .git && \
    test ! -e .git && \
    SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_VERSION}" \
      pip install -e python --no-deps && \
    # --no-deps protects the base's CUDA-tagged wheels from being replaced by
    # untagged PyPI builds, but it also drops any pin this tree raised above
    # what the base ships. Reinstall those from SGLang's index at the pinned
    # version, read out of pyproject.toml so a future bump needs no edit here.
    bash docker/qwen38/cuda_pins.sh reconcile \
      python/pyproject.toml "${SGL_WHL_CUDA_TAG}" \
      sglang-kernel sgl-deep-gemm && \
    kernels lock python && \
    ( success=0; \
      if [ "$(uname -m)" = "aarch64" ]; then \
          echo "Skipping sgl-flash-attn3 cubin download on aarch64; kernels will be JIT-compiled at runtime"; \
          success=1; \
      else \
          for i in 1 2 3; do \
              echo "Attempt $i/3: downloading sgl-kernel cubins..."; \
              if kernels download python; then success=1; break; fi; \
              [ "$i" = "3" ] || { echo "sgl-kernel cubin download failed, retrying in 30s..."; sleep 30; }; \
          done; \
      fi; \
      [ "$success" = "1" ] || \
        echo "WARNING: no matching sgl-flash-attn3 cubin variant; kernels will be JIT-compiled at runtime" ) && \
    mkdir -p /root/.cache/huggingface /root/.cache/sglang && \
    ( if [ -f python/kernels.lock ]; then mv python/kernels.lock /root/.cache/sglang/; fi ) && \
    rm -rf \
      rust/target \
      rust/sglang-grpc/target \
      rust/sglang-mm/target \
      rust/sglang-server/target \
      /root/.cargo/registry \
      /root/.cache/pip

# --- 2 (deferred). Re-pin NCCL and stage the preload path, LAST ---
# torch declares `nvidia-nccl-cu13==2.28.9`, so every pip install above that
# resolves dependencies drags NCCL back down to 2.28.9 -- silently undoing the
# pin that DeepEP v2 was compiled against, and leaving deep_ep._C with an
# unresolved `ncclGetLsaDevicePointer`. Re-pinning here, after all pip work, is
# what makes the runtime match the compile. --no-deps so this install cannot
# itself re-resolve; pip will warn that torch's pin is now unsatisfied, which is
# the intended override.
#
# The link target is resolved from the installed package rather than hardcoded
# to /usr/local/lib/python3.12/dist-packages: a base that moves to another
# interpreter or to /usr/lib/python3 would otherwise leave a dangling link, and
# LD_PRELOAD of a dangling path fails silently at runtime.
# nvidia.nccl is a namespace package: __file__ is None, only __path__ is set.
RUN python3 -m pip install --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NCCL_PIN_VERSION}" && \
    rm -rf /root/.cache/pip && \
    NCCL_LIB="$(python3 -c 'import os, nvidia.nccl; print(os.path.join(list(nvidia.nccl.__path__)[0], "lib"))')" && \
    test -d "${NCCL_LIB}" && \
    mkdir -p /opt/nccl-${NCCL_PIN_VERSION} && \
    ln -sfn "${NCCL_LIB}" /opt/nccl-${NCCL_PIN_VERSION}/lib && \
    test -e /opt/nccl-${NCCL_PIN_VERSION}/lib/libnccl.so.2 && \
    # Assert the VERSION, not just the path: the old `test -e` guard passed
    # happily against the downgraded 2.28.9 sitting at the same location.
    NCCL_EXPECTED="${NCCL_PIN_VERSION}" python3 -c 'import os; from importlib.metadata import version; e = os.environ["NCCL_EXPECTED"]; v = version("nvidia-nccl-cu13"); assert v == e, (v, e)' && \
    # The check that actually matters, and the one whose absence let a broken
    # image build green: DeepEP v2 must load against the NCCL now installed.
    python3 -c 'import deep_ep; print("deep_ep import OK")'

# --- 5. Assert the dependency pins the tree declares, LAST ---
# `import sglang` proves nothing about dependency versions: assert_pkg_version
# lives in srt/entrypoints/engine.py and only runs once a server starts. A tree
# that outgrew the base's sglang-kernel therefore builds green here, ships, and
# dies on every rank at launch -- the same shape as the deep_ep bug above, and it
# has already happened once: an image built on a base carrying sglang-kernel
# 0.4.5 shipped while engine.py asserts a 0.4.6.post1 floor.
#
# Runs after every pip step in the file, so nothing below can quietly move a
# version back down.
RUN cd /sgl-workspace/sglang && \
    python3 -c 'import sglang; print("sglang", sglang.__version__)' && \
    bash docker/qwen38/cuda_pins.sh verify \
      python/pyproject.toml sglang-kernel sgl-deep-gemm

WORKDIR /sgl-workspace/sglang
