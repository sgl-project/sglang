#!/bin/bash
# Usage: TAG=x CONC=24 [DURATION=3600 PORT=30000] bash run_sa_point.sh (SA aiperf fork, ATOM flags)
source "$(dirname "$0")/atom_client_env.sh"
export AIPERF_BIN=/scratch/aiperf-sa-venv/bin/aiperf
export AIPERF_EXTRA="$ATOM_CLIENT_FLAGS"
: "${DURATION:=3600}"; : "${PORT:=30000}"
exec bash "$(dirname "$0")/bench_any.sh"
