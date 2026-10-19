#!/bin/bash
# Triage bundle for the A5/CANN9.2 all-zero-output incident.
# Collects version facts, then probes a live server. Read-only; safe to re-run.
# Usage: bash test/manual/ascend/a5_cann92_zero_output_triage.sh [port] [--selftest]
#   port      server port for the logprob probe (default 8001; skip probe if down)
#   --selftest  also run the repo NPU-graph/dspark parity checks (needs NPU + repo cwd)

PORT="${1:-8001}"
SELFTEST=0
[[ "${2:-}" == "--selftest" ]] && SELFTEST=1
OUT="a5_triage_$(date +%m%d_%H%M%S).log"

log() { echo -e "$@" | tee -a "$OUT"; }

log "==== A5/CANN9.2 zero-output triage $(date) ===="
log "== host: $(hostname)"

log "\n[1] CANN / driver"
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | tee -a "$OUT"
npu-smi info 2>/dev/null | head -20 | tee -a "$OUT"

log "\n[2] python stack"
python3 - <<'EOF' 2>&1 | tee -a "$OUT"
import torch, torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
try:
    import sglang
    print("sglang file:", sglang.__file__, "version:", getattr(sglang, "__version__", "?"))
except Exception as e:
    print("sglang import failed:", e)
try:
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
    print("is_npu_arch35:", is_npu_arch35())
except Exception as e:
    print("is_npu_arch35 check failed:", e)
EOF

log "\n[3] key packages"
pip list 2>/dev/null | grep -Ei "torch|triton|sgl|vllm|deep|ascend" | tee -a "$OUT"

log "\n[4] extra-ops provenance (build date must postdate the CANN9.2 image)"
ls -la --time-style=long-iso /usr/local/Ascend/extra-ops/ 2>/dev/null | tee -a "$OUT"
SO=$(ls /usr/local/Ascend/extra-ops/vllm_ascend_C*.so 2>/dev/null | head -1)
if [[ -n "$SO" ]]; then
    ldd "$SO" 2>/dev/null | grep -Ei "torch|ascend|npu|not found" | tee -a "$OUT"
fi
echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH" | tee -a "$OUT"
ls /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ 2>/dev/null | tee -a "$OUT"

log "\n[5] repo identity of the running checkout"
if git rev-parse --git-dir >/dev/null 2>&1; then
    git rev-parse HEAD | tee -a "$OUT"
    git log -1 --format="%ad %s" --date=iso | tee -a "$OUT"
    git status --porcelain | head -5 | tee -a "$OUT"
fi
grep -rn "SGLANG_ENABLE_SPEC_V2" python/sglang/srt/environ.py 2>/dev/null | tee -a "$OUT"
grep -rn "SGLANG_ENABLE_SPEC_V2" python/ 2>/dev/null | head -3 | tee -a "$OUT"

log "\n[6] logprob probe (NaN/garbage logits vs broken extraction)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy
curl -s -m 120 -X POST "http://127.0.0.1:${PORT}/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":4},"return_logprob":true,"logprob_start_len":0,"top_logprobs_num":3}' \
  > /tmp/probe.json 2>/dev/null
python3 - <<'EOF' 2>&1 | tee -a "$OUT"
import json, math
try:
    r = json.load(open("/tmp/probe.json"))
except Exception as e:
    print("probe failed (server down?):", e); raise SystemExit
print("output_ids:", r.get("output_ids"))
meta = r.get("meta_info", {})
print("spec_verify_ct:", meta.get("spec_verify_ct"),
      "proposed:", meta.get("spec_num_proposed_drafts"))
lp = r.get("meta_info", {}).get("output_token_logprobs") or r.get("output_token_logprobs")
if lp is None:
    print("no logprobs in response; keys:", list(r.keys()), list(meta.keys()))
else:
    flat = [v for row in lp if row for v in (row.values() if isinstance(row, dict) else [row])]
    bad = [v for v in flat if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
    print("logprob sample:", flat[:8])
    print("NaN/Inf logprobs:", len(bad), "-> forward broken" if bad else "-> extraction-side suspect")
EOF

if [[ "$SELFTEST" == 1 ]]; then
    log "\n[7] repo selftests (NPU graph capture + dspark epilogue parity)"
    export PYTHONPATH="$(pwd)/python:$PYTHONPATH"
    python3 test/manual/ascend/spike_dspark_npu_epilogue_capture.py 2>&1 | tail -5 | tee -a "$OUT"
    python3 -m pytest test/registered/npu/spec/test_npu_dspark_epilogue_parity.py -x -q 2>&1 | tail -5 | tee -a "$OUT"
fi

log "\n==== done: $OUT ===="
