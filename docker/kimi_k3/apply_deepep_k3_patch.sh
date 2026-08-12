#!/bin/bash
# Kimi-K3 DeepEP patch + rebuild -- IMAGE-BUILD variant of the runtime patcher.
#
# Stock DeepEP (deepseek-ai@d28bd67, shipped in the base image at
# /sgl-workspace/DeepEP) does not serve Kimi-K3:
#   - LL topk capped at 11; K3 routes top-16  -> assert at internode_ll.cu
#   - SWITCH_HIDDEN lacks 3584 (K3 latent-MoE dispatch dim)
#   - normal-internode issues an unaligned 64-bit SourceMeta access for packed
#     K3 FP8 scales when dispatch crosses the 8-rank NVL domain (EP > 8)
#   - CUDA 13 relocated the cccl headers -> the stock build cannot find them
#
# Differences vs the devbox runtime script (why a separate copy):
#   1. arch comes from $TORCH_CUDA_ARCH_LIST (no GPU at image-build time; the
#      runtime script probes torch.cuda.get_device_capability which needs a GPU)
#   2. adds the setup.py cccl include-dir fix (required to COMPILE on CUDA 13;
#      the devbox had it applied manually)
#   3. adds the configs.cuh CPU/cycle timeout bump (cross-node init headroom)
#
# Idempotent (grep/count-guarded). The default wheel contains native cubins for
# Hopper sm_90, B200 sm_100a, and GB300 sm_103a.
set -euo pipefail
: "${TORCH_CUDA_ARCH_LIST:=9.0;10.0a;10.3a}"
DEEPEP_DIR="${DEEPEP_DIR:-/sgl-workspace/DeepEP}"
DEEPEP_COMMIT="${DEEPEP_COMMIT:-d28bd676c2120573c9f1425f0c16c39faa4117e6}"

# PyTorch accepts CUDA architecture lists separated by spaces or semicolons.
read -r -a CUDA_ARCHES <<< "${TORCH_CUDA_ARCH_LIST//;/ }"
[ "${#CUDA_ARCHES[@]}" -gt 0 ] || { echo "ERROR: TORCH_CUDA_ARCH_LIST is empty"; exit 1; }
for CUDA_ARCH in "${CUDA_ARCHES[@]}"; do
  [[ "$CUDA_ARCH" =~ ^[0-9]+\.[0-9]+a?$ ]] || {
    echo "ERROR: unsupported CUDA architecture '$CUDA_ARCH' in TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST'"
    exit 1
  }
done

# The base image is expected to ship the DeepEP source here; clone-pin if not.
if [ ! -d "$DEEPEP_DIR/csrc" ]; then
  echo "== $DEEPEP_DIR missing — cloning deepseek-ai/DeepEP @ $DEEPEP_COMMIT"
  git clone --recursive https://github.com/deepseek-ai/DeepEP.git "$DEEPEP_DIR"
  git -C "$DEEPEP_DIR" checkout "$DEEPEP_COMMIT"
  git -C "$DEEPEP_DIR" submodule update --init --recursive
fi
cd "$DEEPEP_DIR"

echo "== [1/6] internode_ll.cu: LL topk caps 9/11 -> 16"
TOPK_CAPS_EXPECTED=$(grep -Eci "kNumMaxTop[Kk] = (9|11|16)" csrc/kernels/internode_ll.cu || true)
[ "$TOPK_CAPS_EXPECTED" -ge 2 ] || {
  echo "ERROR: expected >=2 recognized topk caps, found $TOPK_CAPS_EXPECTED"
  exit 1
}
sed -i 's/constexpr int kNumMaxTopK = 11;/constexpr int kNumMaxTopK = 16;/' csrc/kernels/internode_ll.cu
sed -i 's/constexpr int kNumMaxTopK = 9;/constexpr int kNumMaxTopK = 16;/'  csrc/kernels/internode_ll.cu
sed -i 's/constexpr int kNumMaxTopk = 9;/constexpr int kNumMaxTopk = 16;/'  csrc/kernels/internode_ll.cu
sed -i 's/constexpr int kNumMaxTopk = 11;/constexpr int kNumMaxTopk = 16;/' csrc/kernels/internode_ll.cu
TOPK_CAPS_PATCHED=$(grep -ci "kNumMaxTop[Kk] = 16" csrc/kernels/internode_ll.cu || true)
TOPK_CAPS_UNPATCHED=$(grep -Eci "kNumMaxTop[Kk] = (9|11)" csrc/kernels/internode_ll.cu || true)
if [ "$TOPK_CAPS_PATCHED" -ne "$TOPK_CAPS_EXPECTED" ] || [ "$TOPK_CAPS_UNPATCHED" -ne 0 ]; then
  echo "ERROR: expected $TOPK_CAPS_EXPECTED topk caps patched, found $TOPK_CAPS_PATCHED patched and $TOPK_CAPS_UNPATCHED unpatched"
  exit 1
fi

echo "== [2/6] launch.cuh: SWITCH_HIDDEN += case 3584 (K3 latent MoE)"
grep -q "case_macro(3584)" csrc/kernels/launch.cuh || \
  sed -i 's/case 4096: case_macro(4096);/case 3584: case_macro(3584); case 4096: case_macro(4096);/' csrc/kernels/launch.cuh
grep -q "case_macro(3584)" csrc/kernels/launch.cuh || { echo "ERROR: hidden 3584 case not applied"; exit 1; }

echo "== [3/6] configs.cuh: raise CPU/cycle timeouts 100s -> 1000s (cross-node init headroom)"
# '@' delimiter (pattern contains '#'); '$' anchor avoids the ENABLE_FAST_DEBUG '...10' line.
sed -i 's@^#define NUM_CPU_TIMEOUT_SECS 100$@#define NUM_CPU_TIMEOUT_SECS 1000@' csrc/kernels/configs.cuh
sed -i 's@^#define NUM_TIMEOUT_CYCLES 200000000000ull@#define NUM_TIMEOUT_CYCLES 2000000000000ull@' csrc/kernels/configs.cuh
grep -q "NUM_CPU_TIMEOUT_SECS 1000$" csrc/kernels/configs.cuh || echo "   WARN: CPU timeout not raised (layout changed?) — non-fatal"

echo "== [4/6] tests/test_low_latency.py: nvfp4 tolerance 0.007 -> 0.008 (topk16 noise)"
python3 - <<'EOF'
p = "tests/test_low_latency.py"
s = open(p).read()
old = "elif dispatch_use_nvfp4:\n                                    diff_threshold = 0.007"
new = "elif dispatch_use_nvfp4:\n                                    diff_threshold = 0.008"
if old in s:
    open(p, "w").write(s.replace(old, new)); print("   patched")
else:
    print("   already patched or layout changed (skipped)")
EOF

echo "== [5/6] internode.cu: 4-byte-aligned SourceMeta scalar access (EP>8 normal internode)"
python3 - <<'EOF'
from pathlib import Path
path = Path("csrc/kernels/internode.cu")
source = path.read_text()
changed = False
abi_old = 'EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");'
abi_new = 'EP_STATIC_ASSERT(sizeof(SourceMeta) == 2 * sizeof(int), "SourceMeta scalar access requires exactly two int fields");'
if source.count(abi_old) == 2 and source.count(abi_new) == 0:
    source = source.replace(abi_old, abi_new); changed = True; print("   tightened both SourceMeta ABI assertions")
elif source.count(abi_old) == 0 and source.count(abi_new) == 2:
    print("   ABI assertions already tightened")
else:
    raise SystemExit(f"ERROR: unexpected SourceMeta ABI layout: old={source.count(abi_old)}, new={source.count(abi_new)}")
replacements = (
    ("sender store",
     """            // Copy source metadata into symmetric send buffer
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
""",
     """            // SourceMeta may be only 4-byte aligned after packed scales.
            // Store its two int fields separately to avoid an unaligned 64-bit store.
            if (lane_id < num_topk_ranks) {
                auto meta_values = reinterpret_cast<int*>(dst_send_buffers[lane_id]);
                st_na_global(meta_values, src_meta.src_rdma_rank);
                st_na_global(meta_values + 1, src_meta.is_token_in_nvl_rank_bits);
            }
"""),
    ("forwarder load",
     """                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
""",
     """                auto src_meta_values = reinterpret_cast<const int*>(shifted + hidden_bytes + scale_bytes);
                SourceMeta src_meta;
                src_meta.src_rdma_rank = ld_nc_global(src_meta_values);
                src_meta.is_token_in_nvl_rank_bits = ld_nc_global(src_meta_values + 1);
"""),
    ("receiver load",
     """                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
""",
     """                auto meta_values = reinterpret_cast<const int*>(shifted + hidden_bytes + scale_bytes);
                SourceMeta meta;
                meta.src_rdma_rank = ld_nc_global(meta_values);
                meta.is_token_in_nvl_rank_bits = ld_nc_global(meta_values + 1);
"""),
)
for label, old, new in replacements:
    oc, nc = source.count(old), source.count(new)
    if oc == 1 and nc == 0:
        source = source.replace(old, new); changed = True; print(f"   patched {label}")
    elif oc == 0 and nc == 1:
        print(f"   {label} already patched")
    else:
        raise SystemExit(f"ERROR: unexpected {label} layout: old={oc}, new={nc}")
for unsafe in ("st_na_global(reinterpret_cast<SourceMeta*>", "ld_nc_global(reinterpret_cast<SourceMeta*>"):
    if unsafe in source:
        raise SystemExit(f"ERROR: unsafe SourceMeta access remains: {unsafe}")
if changed:
    path.write_text(source)
EOF

echo "== [5b] setup.py: add CUDA 13 cccl include dir (compile fix)"
grep -q "/usr/local/cuda/include/cccl" setup.py || \
  sed -i "s#\(    include_dirs = \['csrc/'\]\)#\1\n    include_dirs.append('/usr/local/cuda/include/cccl')#" setup.py
grep -q "/usr/local/cuda/include/cccl" setup.py || { echo "ERROR: cccl include not added to setup.py"; exit 1; }

echo "== [6/6] rebuild + reinstall for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (no GPU needed)"
rm -rf build dist
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" python3 setup.py bdist_wheel
pip install dist/*.whl --force-reinstall --no-deps

echo "== verify installed cubin arches"
SO=$(find /usr/local/lib/python3*/dist-packages -maxdepth 1 -name "deep_ep_cpp*.so" | head -1)
[ -n "$SO" ] || { echo "ERROR: installed deep_ep_cpp shared object not found"; exit 1; }
CUBIN_LIST=$(cuobjdump --list-elf "$SO")
VERIFIED_SMS=()
for CUDA_ARCH in "${CUDA_ARCHES[@]}"; do
  SM="sm_${CUDA_ARCH/./}"
  grep -Eq "(^|[^[:alnum:]_])${SM}([^[:alnum:]_]|$)" <<< "$CUBIN_LIST" || {
    echo "ERROR: cubin arch $SM not found in $SO"
    printf '%s\n' "$CUBIN_LIST" | head
    exit 1
  }
  VERIFIED_SMS+=("$SM")
done
echo "== OK: DeepEP rebuilt (topk16 + hidden3584 + SourceMeta align + cccl) for ${VERIFIED_SMS[*]}"
