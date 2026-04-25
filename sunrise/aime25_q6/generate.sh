#!/usr/bin/env bash
# Regenerate sunrise/aime25_q6/test.jsonl from the canonical nemo_skills aime25 dataset.
#
# Run from inside an rcli container that has the nemo_skills venv set up (see
# journal 2026-04-21-024 §Step 6a), or adjust --source-jsonl to point at any
# local copy of nemo_skills/dataset/aime25/test.jsonl.
#
# Must be run after `ns prepare_data aime25` so the source jsonl exists.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

SOURCE="${SOURCE:-/workspace/nemo_skills-venv/lib/python3.12/site-packages/nemo_skills/dataset/aime25/test.jsonl}"

cd "$REPO_ROOT"
python3 sunrise/filter_nemo_skills_questions.py \
    --question-ids aime25-6 \
    --source-jsonl "$SOURCE" \
    --output-dir sunrise
echo "Wrote $HERE/test.jsonl"
