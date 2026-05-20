#!/usr/bin/env bash
# Pre-commit blocklist for session-artifact filename patterns.
#
# Wire as a Git pre-commit hook on the dev branch:
#
#   ln -s ../../development/pre-commit-blocklist.sh .git/hooks/pre-commit
#
# or invoke directly:
#
#   ./development/pre-commit-blocklist.sh
#
# Blocks any staged file whose basename matches HANDOFF*.md, SESSION_REPORT*.md,
# *.HANDOFF.md, or whose path starts with a top-level pensieve install (.pensieve
# is allowed under .gitignore but not in the git index). Per AC-12 of the
# refined Double Sparsity plan; matches the "no session artifacts in the
# upstream PR" hygiene contract.

set -euo pipefail

# Collect the list of staged files (Added/Modified/Renamed-To).
mapfile -t staged < <(git diff --cached --name-only --diff-filter=AMR 2>/dev/null || true)

if [[ ${#staged[@]} -eq 0 ]]; then
  exit 0
fi

violations=()
for path in "${staged[@]}"; do
  base=$(basename "$path")
  case "$base" in
    HANDOFF*.md|*.HANDOFF.md|SESSION_REPORT*.md|HANDOFF_NATIVE.md|HANDOFF_NATIVE_FINAL.md|SESSION_REPORT_*.md)
      violations+=("$path  (session-artifact filename)")
      ;;
  esac
  case "$path" in
    .pensieve/*|.pensieve)
      violations+=("$path  (pensieve install — must stay out of git)")
      ;;
    .humanize/*|.humanize)
      violations+=("$path  (humanize loop state — must stay out of git)")
      ;;
  esac
done

if [[ ${#violations[@]} -gt 0 ]]; then
  printf '\nERROR: pre-commit blocklist rejected the following staged paths:\n\n' >&2
  for v in "${violations[@]}"; do
    printf '  %s\n' "$v" >&2
  done
  printf '\nRemove these from the index before committing:\n' >&2
  printf '    git rm --cached -r <path>\n\n' >&2
  exit 1
fi

exit 0
