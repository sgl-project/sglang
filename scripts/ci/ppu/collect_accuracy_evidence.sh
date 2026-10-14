#!/bin/bash
# Carry one accuracy entry's report off the NAS onto the run page and into the
# artifact directory.
#
# Run by every entry of test-ppu-accuracy.yml in the orchestration shell,
# held here for the same reason the board-side body is: those jobs cannot call a
# local composite action, because their runner resolves `uses: ./` on its own
# filesystem rather than in the container the checkout populated (measured, run
# 34085102675).
#
# The action carries no files out of the worker pod, only its logs, so the report
# travels over the NAS both sides mount: the pod writes under /mnt/wl_nas and this
# shell reads the same bytes under /wl_nas.
#
# What reaches the run page is the score, as a notice, whether or not it passed:
# a run that judged nothing because no baseline exists yet has still measured the
# only thing anyone wanted, and a reader should not have to download an artifact
# to see it.
#
# Reads:
#   ENTRY                       entry name, and what keeps parallel jobs'
#                               annotations apart on one run page
#   ACCURACY_RESULTS_ON_RUNNER  the run's own report directory on the NAS
#
# Usage: bash scripts/ci/ppu/collect_accuracy_evidence.sh
set -euo pipefail

: "${ENTRY:?the entry name is required}"
: "${ACCURACY_RESULTS_ON_RUNNER:?the NAS report directory is required}"

# The NAS copy is read, not moved. The pod writes the report as root and this
# shell is a different, non-root uid, so it can read those bytes but cannot
# unlink them: an rm here fails with EPERM on every file and turns this step red
# even when the suite passed, which is how the first Answer run on this board
# reported itself. What is left behind doubles as the on-cluster record once the
# artifact expires.
destination="${GITHUB_WORKSPACE}/ppu-accuracy-artifacts"
mkdir -p "${destination}"

if [ -d "${ACCURACY_RESULTS_ON_RUNNER}" ]; then
  # Named files rather than the whole tree, which is what the Answer and perf
  # collectors copy. Those directories hold kilobytes; this one also holds
  # EvalScope's work directory -- every prompt sent, every completion received,
  # every per-sample review, and the dataset and model caches redirected in
  # beside them -- which is gigabytes for a full split and contains generated
  # text. The verdict, the raw report and the tool log are what a reviewer needs;
  # the predictions stay on the NAS for whoever is debugging a specific answer.
  for name in result.json summary.md junit.xml evalscope.log trend.jsonl trend-error.json; do
    if [ -f "${ACCURACY_RESULTS_ON_RUNNER}/${name}" ]; then
      cp -a "${ACCURACY_RESULTS_ON_RUNNER}/${name}" "${destination}/"
    fi
  done
  # 环境证据仅复制白名单文件；诊断归档失败不改变原评测门禁。
  for name in environment.json environment-pods.tsv environment-pods.status; do
    if [ -f "${ACCURACY_RESULTS_ON_RUNNER}/${name}" ]; then
      cp -a "${ACCURACY_RESULTS_ON_RUNNER}/${name}" "${destination}/" 2>/dev/null ||
        echo "::warning::环境证据未能归档：${name}"
    fi
  done
  if [ -d "${ACCURACY_RESULTS_ON_RUNNER}/evalscope/reports" ]; then
    mkdir -p "${destination}/evalscope"
    cp -a "${ACCURACY_RESULTS_ON_RUNNER}/evalscope/reports" \
      "${destination}/evalscope/"
  fi
  echo "::notice::${ENTRY}: EvalScope predictions and reviews remain at ${ACCURACY_RESULTS_ON_RUNNER}/evalscope"
else
  echo "::warning::the pod produced no accuracy report at ${ACCURACY_RESULTS_ON_RUNNER}"
fi

if [ -f "${destination}/summary.md" ]; then
  # Annotations are what a reader of this run page actually gets here. This
  # runner drives the job through a container hook that does not share the job
  # container's filesystem with the runner process, so bytes written to
  # GITHUB_STEP_SUMMARY inside the container are dropped without an error: the
  # check runs of the first two Answer runs on this board both reported a summary
  # of length zero while their annotations arrived intact. Annotations travel
  # over this step's stdout, which does reach the runner, and they surface on the
  # run page and in the checks UI without downloading the artifact.
  #
  # The three channels are read back from summary.md rather than from result.json
  # because this container is not guaranteed a JSON parser -- no python3, no jq.
  # render_summary prefixes a measured line with "- MEASURED ", an unmeasured one
  # with "- FAIL " and a flagged one with "- WARN ", no other line in that
  # document starts with any of them, and test_ppu_accuracy_eval_unit locks the
  # shape.
  while IFS= read -r measured; do
    echo "::notice::${ENTRY}: ${measured#- MEASURED }"
  done < <(grep '^- MEASURED ' "${destination}/summary.md" || true)
  while IFS= read -r failure; do
    echo "::error::${ENTRY}: ${failure#- FAIL }"
  done < <(grep '^- FAIL ' "${destination}/summary.md" || true)
  while IFS= read -r warning; do
    echo "::warning::${ENTRY}: ${warning#- WARN }"
  done < <(grep '^- WARN ' "${destination}/summary.md" || true)
  counts=$(grep '^- Evaluations: ' "${destination}/summary.md" | head -1 || true)
  if [ -n "${counts}" ]; then
    echo "::notice::${ENTRY}: ${counts#- }"
  fi
  cat "${destination}/summary.md" >> "${GITHUB_STEP_SUMMARY}"
fi
