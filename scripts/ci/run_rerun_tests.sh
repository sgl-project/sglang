#!/usr/bin/env bash
# Usage: run_rerun_tests.sh <mode> <partition>
#
# Runs each line of $TEST_COMMAND the way rerun-test.yml's <mode> job runs a
# test file, then lists the failed ones in the log and the step summary.
# rerun-test.yml checks this script out at github.sha, like the workflow
# itself, so the PR checkout whose tests it runs cannot change it.
set -eo pipefail

mode=$1
partition=$2

case "$mode" in
  cuda)
    if [[ "$RUNNER_LABELS" == "1-gpu-5090" ]]; then
      source /etc/profile.d/sglang-ci.sh
    fi
    ;;
  multimodal_gen)
    # Run through the multimodal-gen CI's pytest runner so a rerun gets the
    # same retries of failed items as CI (performance checks are noisy).
    # Stop starting retries after 50 minutes, as the CI jobs do.
    export SGLANG_DIFFUSION_RETRY_DEADLINE=$(( $(date +%s) + 3000 ))
    ;;
  cpu)
    cd test/
    ;;
  amd)
    read -ra kvs <<< "$EXEC_ENV"
    exec_env=()
    for kv in "${kvs[@]}"; do
      exec_env+=(-e "$kv")
    done
    ;;
  *)
    echo "::error::Unknown mode: $mode"
    exit 1
    ;;
esac

# Runs command $2, the $1-th of the partition (0-based).
run_test() {
  local idx=$1 cmd=$2
  case "$mode" in
    cuda)
      # base-a perf tests (fwd_occupancy) need async assert off; match
      # _pr-test-stage.yml's base-a carve-out.
      local test_file="${cmd%% *}"
      test_file="${test_file%%::*}"
      local async_assert=true
      if grep -qE 'register_cuda_ci\(.*stage[[:space:]]*=[[:space:]]*"base-a"' "test/$test_file" 2>/dev/null; then
        async_assert=false
      fi
      (cd test/ && SGLANG_ENABLE_ASYNC_ASSERT=$async_assert python3 $cmd -f)
      ;;
    multimodal_gen)
      python3 -c 'import sys; from sglang.multimodal_gen.test.runner.pytest_runner import run_pytest; sys.exit(run_pytest(sys.argv[2:], junit_xml_path=sys.argv[1])[0])' \
        "$RUNNER_TEMP/rerun-test-$idx.xml" $cmd
      ;;
    cpu)
      python3 $cmd -f
      ;;
    amd)
      bash scripts/ci/amd/amd_ci_exec.sh "${exec_env[@]}" -w /sglang-checkout/test python3 $cmd -f
      ;;
  esac
}

cmds=()
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  cmds+=("$cmd")
done <<< "$TEST_COMMAND"
total=${#cmds[@]}
failed=()
suite_start=$SECONDS
for idx in "${!cmds[@]}"; do
  i=$((idx + 1))
  cmd="${cmds[$idx]}"
  if [[ "$mode" == "multimodal_gen" ]]; then
    label="pytest $cmd (multimodal-gen runner)"
  else
    label="python3 $cmd"
  fi
  echo ""
  echo "."
  echo "Begin ($i/$total): $label"
  echo "."
  file_start=$SECONDS
  if ! run_test "$idx" "$cmd"; then
    failed+=("$cmd")
  fi
  elapsed=$(( SECONDS - file_start ))
  echo "."
  echo "End ($i/$total): elapsed=${elapsed}s"
  echo "."
  echo ""
done
total_elapsed=$(( SECONDS - suite_start ))
if [ ${#failed[@]} -gt 0 ]; then
  echo "${#failed[@]}/$total test(s) failed in ${total_elapsed}s:"
  printf '  %s\n' "${failed[@]}"
  {
    echo "### Failed tests (partition $partition)"
    printf -- '- `%s`\n' "${failed[@]}"
  } >> "$GITHUB_STEP_SUMMARY"
  exit 1
fi
echo "All $total test(s) passed in ${total_elapsed}s"
