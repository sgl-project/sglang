#!/bin/bash
# build_one.sh: the build steps of the ONE Kimi-K3.1 VR200 serving image (docker/k31-vr200-one.Dockerfile).
# The SAME script runs in the Dockerfile
# (k31-vr200-one.Dockerfile: RUN bash /opt/k31-image/one/build_one.sh) and in the cluster bake (an enroot rootfs:
# enroot start --root --rw ... bash /opt/k31-image/one/build_one.sh), so the recipe of record and the gated image are the same steps.
#
# Inputs, already in place when it starts (the Dockerfile COPYs them, the bake unpacks them):
#   /sgl-workspace/sglang          the SGLang tree (file .k31-source-commit = its commit sha)
#   /sgl-workspace/sgl-nvidia-mega the plugin tree (file COMMIT = its commit sha)
#   /opt/k31-image/one/            this script + addons/ (public FlashInfer #8186 prims_ts patch) + addons/SHA256SUMS
# Base: the lmsysorg/sglang nightly of 2026-11-07 for CUDA 13.4 (SGLang 043cfc4, arm64), pinned by digest in the Dockerfile.
#
# Steps (BUILD_STEP <name> req|info rc=<rc>); every step runs and is logged in /opt/k31-image/one/build/<name>.log; the script
# exits with the number of FAILED req steps (0 = image complete). info steps never change the exit code.
#   versions     info  library versions of the base (torch, FlashInfer, DeepGEMM, sglang-kernel, ...)
#   net          info  reachability of the public sources the build fetches
#   addons       req   sha256 of every addon file
#   tree_git     req   one-commit git repo in the tree for setuptools-scm (the build context has no .git)
#   rust         req   Rust toolchain (rustup, minimal profile) when the base has no cargo
#   sglang       req   pip install --no-deps --no-build-isolation -e python (no dependency moves: this tree's pyproject pins torch 2.14.1)
#   treecore     req   Rust TreeCore runtime artifact built by the loader (auto mode) and bundled; never mode must load it
#   flashmla     req   sgl-kernel flashmla_ops from the tree's aot sources (docker/k31.Dockerfile step, public #48414 source)
#   sglk_py      req   sgl_kernel/flash_mla.py + flash_mla_mega.py from the tree's aot sources when the wheel lacks them
#                      spatial.py is NOT added: it has no public source.
#   dynblock     req   FlashInfer DynBlock 1024-expert header + jit-cache rebuild (docker/k31-image/patch_flashinfer.py,
#                      build_flashinfer_dynblock.py)
#   r1           req   FlashInfer fused_moe_trtllm_sm107 rebuild for SM107 (vr200/build_fi_sm107.py + install_k31_parts.sh)
#   prims_ts     req   FlashInfer prims_ts = main 26bb6dc9 tree (overlay_flashinfer_prims_ts.py, its 0.7.2 version gate bypassed:
#                      the base ships 0.7.3, served with this same overlay) + flashinfer-ai/flashinfer#8186
#   deepgemm     info  sgl-deep-gemm stays the base's (0.2.6 = the pyproject pin); build_deepgemm.py targets 0.2.3.post2 only
#   fa4          req   flash-attn-4 == 4.0.0b36 (every published point served it)
#   plugin       req   sgl-nvidia-mega editable install from /sgl-workspace/sgl-nvidia-mega + include check vs the tree
#   dynamo       req   Dynamo 1.6.0 + runtime patches 0001/0002 + K3.1 edits (docker/k31-image/build_dynamo.py),
#                      with CUDARC_CUDA_VERSION=13030 (cudarc 0.19.8 rejects the base's CUDA 13.4);
#                      fails if it moved torch / FlashInfer / DeepGEMM / sglang-kernel / triton / cutlass-dsl
#   dyn_sglang   req   Dynamo 1.6.0's SGLang chat processor (dynamo/frontend/sglang_processor.py, used by every
#                      --dyn-chat-processor sglang frontend) imports get_tokenizer from sglang.srt.utils.hf_transformers_utils, which
#                      this SGLang tree moved to sglang.srt.hf_transformers.tokenizer (same function): one exact-anchor edit adds the
#                      fallback import (old location first); the unedited file is kept in data/; the module must import
#   dyn_mmhash   req   Dynamo 1.6.0's supports_external_mm_hashes (dynamo/sglang/_compat.py, run by every Dynamo SGLang decode worker's
#                      handler init) gets an ImportError guard -> False (this SGLang has no MultimodalProcessorOutput)
#   smallm       req   the three SM107 GEMM tables the ported run configurations name, in the MR !2 tree at
#                      benchmark/kimi_k31_vr200/sm107_tables/ (git blobs pinned below; json loads)
#   spatial_scan info  any import of sgl_kernel.spatial in the tree (expected none)
#   verify       info  docker/k31-image/verify_image.py (expected notes: DeepGEMM marker absent, prims_ts marker != tree after #8186)
#   freeze       info  pip freeze + label
set -uo pipefail
umask 022
K=/opt/k31-image/one; T=/sgl-workspace/sglang; P=/sgl-workspace/sgl-nvidia-mega; A=$K/addons; L=$K/build
mkdir -p "$L" "$K/data"; : > "$K/steps.tsv"
export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH=/root/.cargo/bin:/opt/sglang/bin:/usr/local/cuda/bin:$PATH
JOBS=${K31_BUILD_JOBS:-64}; export MAX_JOBS=$JOBS
SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
RUST_TOOLCHAIN=1.92.0
NREQ=0; FAILED=""
CRIT='^(torch|flashinfer[-_]python|flashinfer[-_]cubin|sgl[-_]deep[-_]gemm|sglang[-_]kernel|triton|nvidia[-_]cutlass[-_]dsl)=='
freeze () { python3 -m pip list --format=freeze 2>/dev/null | LC_ALL=C sort; }
step () {  # step NAME req|info FUNCTION
  local name=$1 kind=$2 fn=$3 rc
  echo "BUILD_STEP_BEGIN $name $(date '+%F %T')"
  ( set -o pipefail; "$fn" ) > "$L/$name.log" 2>&1; rc=$?
  tail -n 12 "$L/$name.log" | cut -c1-240 | sed "s/^/  [$name] /"
  echo "BUILD_STEP $name $kind rc=$rc $(date '+%F %T')"
  printf '%s\t%s\t%s\n' "$name" "$kind" "$rc" >> "$K/steps.tsv"
  if [ "$rc" != 0 ]; then FAILED="$FAILED $name"; [ "$kind" = req ] && NREQ=$((NREQ + 1)); fi
  return 0
}

s_versions () {
  python3 - <<'PY'
import importlib.metadata as m
for d in ("torch", "flashinfer-python", "flashinfer-cubin", "sgl-deep-gemm", "sglang-kernel", "triton", "nvidia-cutlass-dsl",
          "flash-attn-4", "sgl-flash-mla", "ai-dynamo", "ai-dynamo-runtime", "mooncake-transfer-engine-cuda13", "nixl", "transformers",
          "sglang", "sgl-nvidia-mega", "sglang-router", "sgl-deep-ep"):
    try:
        print("VER", d, m.version(d))
    except m.PackageNotFoundError:
        print("VER", d, "absent")
PY
  echo "VER cuda $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*')"; command -v cargo && cargo --version || echo "VER cargo absent"
  freeze > "$K/pip-freeze.base.txt"; echo "base freeze $(wc -l < "$K/pip-freeze.base.txt") dists"
}
s_net () {
  local u
  for u in https://github.com https://codeload.github.com https://pypi.org/simple/flash-attn-4/ https://static.rust-lang.org/rustup/release-stable.toml \
           https://index.crates.io/config.json https://sh.rustup.rs http://ports.ubuntu.com/ubuntu-ports/; do
    echo "NET $(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$u" 2>&1 | tail -c 60) $u"
  done
}
s_addons () { (cd "$A" && sha256sum -c SHA256SUMS); }
s_tree_git () {
  cd "$T" || return 1
  git config --global --add safe.directory "$T"
  [ -s .k31-source-commit ] || { echo "no .k31-source-commit"; return 1; }
  if [ ! -d .git ]; then
    git init -q && git add -A && git -c user.name=k31 -c user.email=k31@localhost commit -qm "sglang tree $(cat .k31-source-commit)" || return 1
  fi
  git log --oneline -1; echo "tree $(cat .k31-source-commit) files $(git ls-files | wc -l)"
  [ -d rust ] && [ -f python/sglang/kernels/aot/CMakeLists.txt ] && [ -f python/sglang/kernels/aot/csrc/flashmla_extension.cc ]
}
s_rust () {
  if command -v cargo >/dev/null; then cargo --version; rustc --version; return 0; fi
  local triple; triple=$(uname -m)-unknown-linux-gnu
  curl --proto '=https' --tlsv1.2 -sSf -o /tmp/rustup-init "https://static.rust-lang.org/rustup/dist/$triple/rustup-init" || return 1
  chmod +x /tmp/rustup-init && /tmp/rustup-init -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN" --no-modify-path || return 1
  rm -f /tmp/rustup-init; cargo --version && rustc --version
}
s_sglang () {
  cd "$T" || return 1
  local before; before=$(freeze | grep -E "$CRIT" | md5sum)
  python3 -c "import setuptools_rust, setuptools_scm" 2>/dev/null || python3 -m pip install -q setuptools-rust setuptools-scm || return 1
  python3 -m pip install --no-deps --no-build-isolation -e python || return 1
  python3 -c "import sglang, importlib.metadata as m; print('sglang', sglang.__file__, m.version('sglang'))" || return 1
  [ "$(freeze | grep -E "$CRIT" | md5sum)" = "$before" ] || { echo "critical libraries moved"; return 1; }
}
s_treecore () {
  SGLANG_RUST_BUILD_MODE=auto python3 - <<'PY' || return 1
import glob, hashlib, shutil
import sglang.srt.mem_cache.rust_tree_core.extension as e
f = e.bindings.__file__
print("TreeCore binding (auto)", f, hashlib.sha256(open(f, "rb").read()).hexdigest())
assert hasattr(e, "RustUnifiedTreeCoreBinding"), "no RustUnifiedTreeCoreBinding"
dsts = glob.glob("/sgl-workspace/sglang/python/sglang/srt/mem_cache/rust_tree_core/mem_cache.cpython-*.so")
if len(dsts) == 1:
    dst = dsts[0]
    if hashlib.sha256(open(dst, "rb").read()).digest() != hashlib.sha256(open(f, "rb").read()).digest():
        shutil.copyfile(f, dst); print("installed as the bundled module", dst)
    print("in-package", dst, hashlib.sha256(open(dst, "rb").read()).hexdigest())
else:
    print("bundled module path count", len(dsts), "(loader artifact kept at", f, ")")
PY
  SGLANG_RUST_BUILD_MODE=never python3 -c "import sglang.srt.mem_cache.rust_tree_core.extension as e; print('never mode loads', e.bindings.__file__, e.RustUnifiedTreeCoreBinding.__name__)"
}
s_flashmla () {
  cd "$T/python/sglang/kernels/aot" || return 1
  CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake -S . -B /tmp/flashmla-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSKBUILD_SABI_COMPONENT=Development.SABIModule -DSKBUILD_SABI_VERSION=3.11 -DPython_EXECUTABLE="$(command -v python3)" \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -DSGL_KERNEL_ENABLE_SM100A=ON -DENABLE_BELOW_SM90=OFF -DENABLE_CCACHE=OFF || return 1
  cmake --build /tmp/flashmla-build --target flashmla_ops -j "$JOBS" > "$L/flashmla.ninja.log" 2>&1 || { grep -m5 -B2 -A8 error "$L/flashmla.ninja.log"; return 1; }
  cp /tmp/flashmla-build/flashmla_ops.abi3.so "$SITE/sgl_kernel/" && sha256sum "$SITE/sgl_kernel/flashmla_ops.abi3.so" && rm -rf /tmp/flashmla-build
}
s_sglk_py () {
  local src=$T/python/sglang/kernels/aot/python/sgl_kernel f
  for f in flash_mla.py flash_mla_mega.py; do
    [ -f "$src/$f" ] || { echo "tree lacks $src/$f"; return 1; }
    if [ -f "$SITE/sgl_kernel/$f" ]; then cmp -s "$src/$f" "$SITE/sgl_kernel/$f" && echo "$f: wheel copy == tree" || echo "$f: wheel copy differs from tree (kept wheel's)"
    else cp "$src/$f" "$SITE/sgl_kernel/$f" && echo "$f: added from the tree ($(sha256sum < "$src/$f" | cut -c1-16))"; fi
  done
  echo "INFO tree aot sgl_kernel vs installed sgl_kernel (python files):"; diff -rq --exclude=__pycache__ --exclude='*.so' "$src" "$SITE/sgl_kernel" | sed 's/^/  /' | head -40
  python3 -c "import ast, sys; [ast.parse(open(sys.argv[1] + '/' + f).read()) for f in ('flash_mla.py', 'flash_mla_mega.py')]; print('parse ok')" "$SITE/sgl_kernel"
}
s_dynblock () {
  python3 "$T/docker/k31-image/patch_flashinfer.py" && python3 "$T/docker/k31-image/build_flashinfer_dynblock.py"
}
s_r1 () {
  K31_BUILD_TMP=/tmp python3 "$T/docker/k31-image/vr200/build_fi_sm107.py" "$K/fi_sm107" || return 1
  bash "$T/docker/k31-image/vr200/install_k31_parts.sh" "$K/fi_sm107" none
}
s_prims_ts () {
  python3 -c "import flashinfer, importlib.metadata as m; print('flashinfer', m.version('flashinfer-python'))"
  python3 - "$T/docker/k31-image/overlay_flashinfer_prims_ts.py" <<'PY' || return 1
# Run the tree's overlay script unchanged except its 0.7.2 version gate (the base ships 0.7.3, served with this tree).
import importlib.metadata as m, runpy, sys
_orig = m.version
m.version = lambda name: "0.7.2" if name == "flashinfer-python" else _orig(name)
runpy.run_path(sys.argv[1], run_name="__main__")
PY
  (cd "$SITE" && git -c safe.directory='*' apply -p1 "$A/flashinfer-8186-prims-ts.diff") || return 1
  (cd "$SITE" && git -c safe.directory='*' apply -R --check -p1 "$A/flashinfer-8186-prims-ts.diff") && echo "FLASHINFER_8186 PRESENT" || return 1
  python3 - <<'PY'
import hashlib, pathlib, sysconfig
p = pathlib.Path(sysconfig.get_paths()["purelib"]) / "flashinfer/attention/prims_ts"
d = hashlib.sha256()
for f in sorted(p.rglob("*.py")):
    d.update(str(f.relative_to(p)).encode() + b"\0" + f.read_bytes())
print("prims_ts tree sha256", d.hexdigest(), len(list(p.rglob("*.py"))), "files")
PY
}
s_deepgemm () { python3 -c "import importlib.metadata as m; print('sgl-deep-gemm', m.version('sgl-deep-gemm'), '(kept; MR pyproject pins 0.2.6)')"; }
s_fa4 () {
  local v; v=$(python3 -c "import importlib.metadata as m; print(m.version('flash-attn-4'))" 2>/dev/null || echo absent); echo "flash-attn-4 base $v"
  [ "$v" = 4.0.0b36 ] && return 0
  python3 -m pip install --no-deps "flash-attn-4==4.0.0b36" || return 1
  python3 -c "import importlib.metadata as m; v = m.version('flash-attn-4'); print('flash-attn-4 now', v); assert v == '4.0.0b36'"
}
s_plugin () {
  [ -s "$P/COMMIT" ] || { echo "no $P/COMMIT"; return 1; }
  python3 -m pip uninstall -y sgl-nvidia-mega sgl_nvidia_mega 2>&1 | tail -2
  rm -rf "$SITE"/__editable__*sgl_nvidia_mega* "$SITE"/__pycache__/__editable__*sgl_nvidia_mega*
  (cd "$P" && python3 -m pip install --no-deps --no-build-isolation -e .) || return 1
  python3 -c "import sgl_nvidia_mega, importlib.metadata as m; print('sgl_nvidia_mega', sgl_nvidia_mega.__file__, m.version('sgl-nvidia-mega'))" || return 1
  echo "plugin commit $(cat "$P/COMMIT")"
  python3 -I "$T/docker/k31-image/vr200/verify_vr200_image.py" includes "$P" "$T/python/sglang/kernels/jit/include"
}
s_dynamo () {
  local before; before=$(freeze | grep -E "$CRIT")
  freeze > "$K/pip-freeze.pre-dynamo.txt"
  # The cudarc crate (0.19.8, in Dynamo 1.6.0's Cargo.lock) reads the CUDA version from nvcc and panics on 13.4 ("Unsupported cuda
  # toolkit version"). Its build.rs takes CUDARC_CUDA_VERSION=<major>0<minor>0 instead; 13030 = the newest it knows.
  # cudarc is built with dynamic-loading, so the 13.3 driver-API bindings load the base's CUDA 13.4 libraries at run time.
  export CUDARC_CUDA_VERSION=${CUDARC_CUDA_VERSION:-13030}; echo "CUDARC_CUDA_VERSION=$CUDARC_CUDA_VERSION"
  python3 "$T/docker/k31-image/build_dynamo.py" || return 1
  freeze > "$K/pip-freeze.post-dynamo.txt"
  echo "pip changes by the Dynamo step:"; diff "$K/pip-freeze.pre-dynamo.txt" "$K/pip-freeze.post-dynamo.txt" | sed 's/^/  /' | head -60
  [ "$(freeze | grep -E "$CRIT")" = "$before" ] || { echo "critical libraries moved by the Dynamo step"; return 1; }
  python3 -c "import importlib.metadata as m; print('ai-dynamo', m.version('ai-dynamo'), 'runtime', m.version('ai-dynamo-runtime'))"
}
s_dyn_sglang () {
  # Without it the Dynamo SGLang frontend fails at import (ModuleNotFoundError sglang.srt.utils.hf_transformers_utils).
  python3 -I - "$SITE/dynamo/frontend/sglang_processor.py" "$K/data/sglang_processor.py.orig" <<'PY' || return 1
import shutil, sys
path, keep = sys.argv[1:3]
old = "from sglang.srt.utils.hf_transformers_utils import get_tokenizer\n"
new = (
    "try:\n"
    "    from sglang.srt.utils.hf_transformers_utils import get_tokenizer\n"
    "except ModuleNotFoundError as _k31_exc:  # K3.1 image: this SGLang moved get_tokenizer to sglang.srt.hf_transformers.tokenizer\n"
    "    if _k31_exc.name != \"sglang.srt.utils.hf_transformers_utils\":\n"
    "        raise\n"
    "    from sglang.srt.hf_transformers.tokenizer import get_tokenizer\n"
)
src = open(path).read()
if new in src:
    print("already applied:", path)
    sys.exit(0)
n = src.count(old)
if n != 1:
    sys.exit("anchor found %d times in %s (want 1)" % (n, path))
shutil.copy2(path, keep)
open(path, "w").write(src.replace(old, new, 1))
print("edited", path, "; unedited copy", keep)
PY
  python3 -c "import dynamo.frontend.sglang_processor as m; print('dynamo.frontend.sglang_processor imports; get_tokenizer from', m.get_tokenizer.__module__)"
}
s_dyn_mmhash () {
  # Without it Dynamo SGLang decode workers fail at init: dynamo/sglang/_compat.py
  # supports_external_mm_hashes imports sglang.srt.managers.schedule_batch.MultimodalProcessorOutput, which this SGLang (and upstream
  # main since #48087) no longer has. Edit: on that ImportError the function returns False (caller-provided multimodal hashes not
  # enabled; K3.1 serving is text-only). Unedited copy kept in data/.
  python3 -I - "$SITE/dynamo/sglang/_compat.py" "$K/data/_compat.py.orig" <<'PY' || return 1
import shutil, sys
path, keep = sys.argv[1:3]
old = ("    # Deferred: schedule_batch imports torch and other SGLang runtime modules.\n"
       "    from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput\n")
new = ("    # Deferred: schedule_batch imports torch and other SGLang runtime modules.\n"
       "    try:\n"
       "        from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput\n"
       "    except ImportError as _k31_exc:  # K3.1 image: this SGLang no longer defines MultimodalProcessorOutput\n"
       "        if \"MultimodalProcessorOutput\" not in str(_k31_exc):\n"
       "            raise\n"
       "        return False\n")
src = open(path).read()
if new in src:
    print("already applied:", path)
    sys.exit(0)
n = src.count(old)
if n != 1:
    sys.exit("anchor found %d times in %s (want 1)" % (n, path))
shutil.copy2(path, keep)
open(path, "w").write(src.replace(old, new, 1))
print("edited", path, "; unedited copy", keep)
PY
  python3 -c "import dynamo.sglang._compat as m, dynamo.sglang.request_handlers.llm.decode_handler as d; print('dynamo.sglang._compat + decode_handler import')"
}
s_smallm () {
  local f h want rc=0
  # Pins: the git blob ids of the three tables in this tree (the loaders in models/kimi_k3.py + k3_ar_fusion.py read only
  # table / side / dense / fold / slice), including the MID prefill table.
  for fw in smallm_table_vr200_tp8_fold.json:19ffca55227dc26c0a27552422f867e1877c6096 sm107_table-stable.json:aa318d5737fcf899c0bd75719f427b6107a2723d sm107_table_mid_prefill.json:34ff8770ce8b226f2e0ea7baca77efa90f28a7b7; do
    f=$T/benchmark/kimi_k31_vr200/sm107_tables/${fw%%:*}; want=${fw#*:}
    h=$(git hash-object "$f") || { rc=1; continue; }; echo "git blob $h $f"
    [ "$h" = "$want" ] && python3 -c "import json, sys; json.load(open(sys.argv[1])); print('json ok')" "$f" || rc=1
  done
  return $rc
}
s_spatial_scan () {
  local n; n=$(grep -rIl --include='*.py' -E 'sgl_kernel\.spatial|from sgl_kernel import .*spatial|spatial_ops' "$T/python/sglang" | tee /dev/stderr | wc -l)
  echo "files importing sgl_kernel spatial: $n"; [ "$n" = 0 ]
}
s_verify () { python3 "$T/docker/k31-image/verify_image.py"; }
s_freeze () {
  freeze > "$K/pip-freeze.txt"; echo "final freeze $(wc -l < "$K/pip-freeze.txt") dists"
  echo "changes vs base:"; diff "$K/pip-freeze.base.txt" "$K/pip-freeze.txt" | sed 's/^/  /' | head -80
  find "$T" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null; rm -rf /root/.cache/pip /tmp/*.log
  return 0
}

echo "BUILD_ONE start $(date '+%F %T') host $(hostname) tree $(cat $T/.k31-source-commit 2>/dev/null) plugin $(cat $P/COMMIT 2>/dev/null) jobs $JOBS site $SITE"
step versions     info s_versions
step net          info s_net
step addons       req  s_addons
step tree_git     req  s_tree_git
step rust         req  s_rust
step sglang       req  s_sglang
step treecore     req  s_treecore
step flashmla     req  s_flashmla
step sglk_py      req  s_sglk_py
step dynblock     req  s_dynblock
step r1           req  s_r1
step prims_ts     req  s_prims_ts
step deepgemm     info s_deepgemm
step fa4          req  s_fa4
step plugin       req  s_plugin
step dynamo       req  s_dynamo
step dyn_sglang   req  s_dyn_sglang
step dyn_mmhash   req  s_dyn_mmhash
step smallm       req  s_smallm
step spatial_scan info s_spatial_scan
step verify       info s_verify
step freeze       info s_freeze
echo "k31-vr200-one tree $(cat $T/.k31-source-commit 2>/dev/null) plugin $(cat $P/COMMIT 2>/dev/null) failed_req=$NREQ failed:${FAILED:- none} built $(date -Is) ${SLURM_JOB_ID:+job $SLURM_JOB_ID}" >> /etc/k31-image.txt
echo "BUILD_ONE_END failed_req=$NREQ failed:${FAILED:- none}"
exit "$NREQ"
