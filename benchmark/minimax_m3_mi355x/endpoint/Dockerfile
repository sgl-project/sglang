# MiniMax-M3 on MI350X/MI355X: SGLang M3-perf overlay on the pinned ROCm 7.2.4 gfx950 nightly.
# Base = lmsysorg/sglang-rocm:v0.5.19-rocm724-mi35x-20260911 (sglang 822e73ccdd, aiter 4ad99832, sgl-kernel 0.4.6.post1),
# the exact compiled stack the M3-perf branch was validated against (822e73ccdd is an ancestor of M3-perf; the branch
# does not touch python/sglang/kernels/aot, so no kernel rebuild is needed).
FROM lmsysorg/sglang-rocm@sha256:e62c69db892add25bb6a82f50a61726895b0814014c360e8acdd61fabc93a5c6

ARG SGLANG_COMMIT
COPY sglang_python.tar.gz /tmp/sglang_python.tar.gz
COPY aiter.patch /tmp/aiter.patch
COPY minimax_m3_gfx950_perf_tuned_fmoe.csv /sgl-workspace/aiter/aiter/configs/model_configs/minimax_m3_gfx950_perf_tuned_fmoe.csv
COPY benchmark/minimax_m3_mi355x /sgl-workspace/sglang/benchmark/minimax_m3_mi355x

RUN set -eux; \
    cd /sgl-workspace/sglang; \
    python -c "import sglang, os; assert os.path.realpath(os.path.dirname(sglang.__file__)) == '/sgl-workspace/sglang/python/sglang', sglang.__file__"; \
    # Overlay, do not replace: python/sglang/srt/rust_extensions/*.so is compiled into the tree by the base build.
    tar -xzf /tmp/sglang_python.tar.gz -C /sgl-workspace/sglang; \
    test -f python/sglang/srt/rust_extensions/_multimodal.cpython-312-x86_64-linux-gnu.so; \
    rm -f /tmp/sglang_python.tar.gz; \
    python -m compileall -q python/sglang/srt; \
    cd /sgl-workspace/aiter; \
    git rev-parse HEAD | grep -q ^4ad99832; \
    git apply --check /tmp/aiter.patch && git apply /tmp/aiter.patch; \
    rm -f /tmp/aiter.patch; \
    python -c "import aiter.ops.flydsl.moe_kernels" || true

ENV SGLANG_USE_AITER=1 HIP_FORCE_DEV_KERNARG=1
LABEL ai.radixark.base.image="lmsysorg/sglang-rocm:v0.5.19-rocm724-mi35x-20260911" \
      ai.radixark.base.digest="sha256:e62c69db892add25bb6a82f50a61726895b0814014c360e8acdd61fabc93a5c6" \
      ai.radixark.base.sglang_commit="822e73ccdd" \
      ai.radixark.overlay.branch="kevin-mii/sglang@M3-perf" \
      ai.radixark.overlay.sglang_commit="${SGLANG_COMMIT}" \
      ai.radixark.overlay.prs="sgl-project/sglang#36546,#36549,#36559,#36560,#36574,#36575,#36576,#39466(umbrella)"
WORKDIR /sgl-workspace
