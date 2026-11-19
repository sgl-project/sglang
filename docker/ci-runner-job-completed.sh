#!/bin/bash
# Runner job-completed hook: kill what the job left on the GPUs, even when it was
# cancelled before its own cleanup step ran. Never fails the job.
python3 /opt/sglang-ci/killall.py || true
exit 0
