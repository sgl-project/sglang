#!/usr/bin/env bash
# Single source of truth for whether SGLANG_ENABLE_ASYNC_ASSERT is on for a
# given CI stage. Both the gated PR test (_pr-test-stage.yml, keyed on the
# stage's self_name) and the manual Rerun Test workflow (rerun-test.yml, keyed
# on the stage resolved from each test file) call this, so the rule cannot
# drift between the two workflows.
#
# base-a runs the fwd-occupancy sanity kit. The async-assert probes launch
# extra per-step GPU work (NaN / Inf / OOB checks) that enlarges the GPU-feed
# bubble and depresses the occupancy measurement below its threshold, so async
# assert stays OFF in that stage. Every other stage keeps it ON for invariant
# coverage.
#
# Usage: async_assert_for_stage.sh <stage>   # echoes "true" | "false"
# <stage> accepts both forms seen across callers, e.g. "base-a" and
# "base-a-test-1-gpu-small".
set -euo pipefail
stage="${1:-}"
case "$stage" in
    base-a*) echo "false" ;;
    *) echo "true" ;;
esac
