#!/usr/bin/env bash
# A1-A5: bump the gfx942/gfx950 AITER pin to the Scout #175 candidate.
# Run from the root of your sglang checkout. Fails closed at every gate.
set -euo pipefail

OLD=285579b6211b0da30f35b91a890c29c823597cd1   # shipped pin (2026-10-17)
NEW=581579ab51ca26ddd4a08a6aa6aa983a78aaf276   # Scout #175 candidate (2026-11-07)
HOLD=07ff5f318b8a34334125a2287acbfcc56bd2898a  # gfx1250 — MUST NOT MOVE
F=docker/rocm.Dockerfile
BR=colin/bump-aiter-pin-581579ab5

[ -f "$F" ] || { echo "!! not in an sglang checkout (no $F)"; exit 1; }

# ---------- A1: the edit ----------
echo "== A1: pin bump =="
n_before=$(grep -c "$OLD" "$F" || true)
echo "   occurrences of the old pin: $n_before"
[ "$n_before" = "8" ] || { echo "!! expected 8, got $n_before — STOP and inspect"; exit 1; }

git checkout -b "$BR"
sed -i "s/$OLD/$NEW/g" "$F"

[ "$(grep -c "$OLD" "$F" || true)" = "0" ] || { echo "!! A1 FAILED: old pin remains"; exit 1; }
[ "$(grep -c "$NEW" "$F" || true)" = "8" ] || { echo "!! A1 FAILED: new pin not in 8 blocks"; exit 1; }
echo "   A1 OK — 8 blocks moved"

# ---------- A2: gfx1250 must be untouched ----------
echo "== A2: gfx1250 hold =="
[ "$(grep -c "$HOLD" "$F" || true)" = "2" ] || { echo "!! A2 FAILED: gfx1250 pin moved — ABORT"; git checkout -- "$F"; exit 1; }
echo "   A2 OK — gfx1250 still on 07ff5f318 (2 blocks)"

# ---------- A3: drop the dead torch_utils hand-patch ----------
echo "== A3: remove upstreamed torch_utils patch =="
python3 - "$F" <<'PY'
import re,sys
p=sys.argv[1]; s=open(p).read()
pat=re.compile(r"\n# torch 2\.11 Dynamo may pass a base torch\.Stream.*?\nPY\n", re.S)
if pat.search(s):
    open(p,"w").write(pat.sub("\n", s)); print("   A3 OK — patch block removed")
else:
    print("   A3 SKIP — block not found (already gone?)")
PY

# ---------- review gate ----------
echo
echo "================ DIFF — review before continuing ================"
git --no-pager diff --stat
echo
git --no-pager diff "$F" | head -60
echo "================================================================"
read -r -p "Looks right? commit, push and open the PR? [y/N] " ok
[ "$ok" = "y" ] || { echo "stopped, nothing committed"; exit 0; }

# ---------- A4: commit + PR body ----------
git add "$F"
git commit -m "[AMD] Bump AITER pin to 581579ab5 for gfx942/gfx950

Clears all 7 outstanding requests on #21302. gfx1250 held at 07ff5f318."

cat > /tmp/pr_body.md <<'BODY'
## Summary

Bumps the AITER pin for **gfx942 / gfx950** from `285579b62` (2026-10-17) to
`581579ab5` (2026-11-07), the AITER Scout #175 candidate.

**gfx1250 is unchanged** — both `gfx1250-*` blocks stay on `07ff5f318`.

## Why this commit

All **7** outstanding requests on #21302 are ancestors of `581579ab5`
(45 of the 52 requested PRs were already in the shipped pin):

| aiter PR | requested by | what |
|---|---|---|
| #5993 | amd-nprotaso | Qwen3.8 hc_mix kernel |
| #6622 | 1am9trash | v41 TP2 MoE config, shared-expert fusion |
| #5992 | ChangLiu0709 | causal conv1d prefill + GDN decode/verify |
| #6670 | amd-nprotaso | fused QSA sparse decode |
| #6464 | kkHuang-amd | mxfp4 FlyDSL a8w4 opt & replace |
| #6496 | ChangLiu0709 | GDN fused prefill |
| #6734 | omirosh | Hadamard fused into MXFP4 quant |

Resolved with `git merge-base --is-ancestor`, not by comparing merge dates.

## Validation

**Build** — the exact Dockerfile command for the gfx950 block
(`PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python setup.py build_ext --inplace`),
fresh container, MI355X:

| arm | result |
|---|---|
| `285579b62` (control) | rc=0, 2183 s, 133 modules |
| **`581579ab5`** | **rc=0, 2192 s, 143 modules** |

Nine seconds apart. All six FlyDSL AOT replay sweeps clean on the candidate:
MoE 10724, GEMM 23965, CONV 152, GDN_FUSED_PREFILL 48, FMHA_FP8 3112,
PA_DECODE_PLAN 96 — **0 failed** in every category.

**Blockers** — both closed:
- The `is_shuffled` MoE issue from Scout #174/#175 — fixed by #47870 and #48219,
  merged 2026-11-12, both ancestors of this tree.
- The Kimi-K3.1-Pro finding from #175 bisected to aiter#6464, but that model is
  not supported on the current test image (#48427 is still open). A sweep across
  the other models whose tuned configs #6464 rewrote found no effect:
  gpt-oss-120b overlaps across reps, DeepSeek-V4-Flash TP4 scores 0.970 → 0.965.

## Also in this PR

Removes the `csrc/cpp_itfs/torch_utils.py` hand-patch. It was upstreamed as
aiter#7101, which is an ancestor of this pin, and the upstream text is
byte-identical to the patch — it is now a no-op.

## What this does not establish

Ancestry proves containment, not that each PR's intended benefit is exercised.
Several are tuned-config additions whose effect appears only on one model at one
shape. Requesters should confirm their own cases.
BODY

echo "   A4: PR body written to /tmp/pr_body.md"

# ---------- A5: push + open ----------
git push -u origin "$BR"
gh pr create --title "[AMD] Bump AITER pin to 581579ab5 for gfx942/gfx950" \
             --body-file /tmp/pr_body.md

URL=$(gh pr view --json url -q .url)
gh pr edit "$URL" --add-label run-ci || echo "   !! add the 'run-ci' label by hand — it gates pr-test.yml"

echo
echo "== verify the posted body actually landed =="
gh pr view "$URL" --json body -q .body | grep -q "0 failed" \
  && echo "   body OK" || echo "   !! body truncated — re-post from /tmp/pr_body.md"
echo "   PR: $URL"
