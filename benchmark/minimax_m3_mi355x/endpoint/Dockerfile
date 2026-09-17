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
    # The base image already ships 2 of this patch's 4 files pre-modified in its working tree
    # (pa_mqa_logits_fp4_prefill.py, torch_utils.py -- byte-identical to our post-image blobs).
    # A whole-patch `git apply --check` therefore fails, and as the LEFT side of an `&&` list its
    # failure does NOT trip `set -e`, so the previous recipe silently skipped all four files --
    # including the two that carry the XCD-swizzle fix. Apply per file and classify each one.
    # Do NOT use --3way here: it implies --index, which stages the result and empties the
    # `git diff` that the endpoint's documented verification (and the assert below) reads.
    for f in aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py \
             aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py \
             aiter/ops/flydsl/moe_kernels.py \
             csrc/cpp_itfs/torch_utils.py; do \
      if git apply --check --include="$f" /tmp/aiter.patch 2>/dev/null; then \
        git apply --include="$f" /tmp/aiter.patch; echo "APPLIED         $f"; \
      elif git apply -R --check --include="$f" /tmp/aiter.patch 2>/dev/null; then \
        echo "ALREADY-IN-BASE $f"; \
      else \
        echo "FAILED          $f"; git apply --check --include="$f" /tmp/aiter.patch || true; exit 1; \
      fi; \
    done; \
    rm -f /tmp/aiter.patch; \
    # Assert the END STATE, not the apply: every hunk must be present however it got there.
    test "$(grep -c 'XCD remap is a bijection' aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py)" = 2; \
    grep -q 'XCD-contiguous, M-block-major' aiter/ops/flydsl/moe_kernels.py; \
    grep -q 'k_batches = \[1, 2, 3, 4, 6, 7, 8, 12, 14, 24\]' aiter/ops/flydsl/moe_kernels.py; \
    grep -q '^from functools import cache$' aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py; \
    grep -q 'isinstance(arg, torch.Stream)' csrc/cpp_itfs/torch_utils.py; \
    python -m compileall -q aiter/ops/flydsl/moe_kernels.py \
        aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py \
        aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py \
        csrc/cpp_itfs/torch_utils.py; \
    git diff --stat | tail -1 | grep -qE '^ 4 files changed'; \
    test -f aiter/configs/model_configs/minimax_m3_gfx950_perf_tuned_fmoe.csv; \
    python -c "import aiter.ops.flydsl.moe_kernels" || true

ENV SGLANG_USE_AITER=1 HIP_FORCE_DEV_KERNARG=1
LABEL ai.radixark.base.image="lmsysorg/sglang-rocm:v0.5.19-rocm724-mi35x-20260911" \
      ai.radixark.base.digest="sha256:e62c69db892add25bb6a82f50a61726895b0814014c360e8acdd61fabc93a5c6" \
      ai.radixark.base.sglang_commit="822e73ccdd" \
      ai.radixark.overlay.branch="kevin-mii/sglang@M3-perf" \
      ai.radixark.overlay.sglang_commit="${SGLANG_COMMIT}" \
      ai.radixark.overlay.prs="sgl-project/sglang#36546,#36549,#36559,#36560,#36574,#36575,#36576,#39466(umbrella)"
WORKDIR /sgl-workspace
