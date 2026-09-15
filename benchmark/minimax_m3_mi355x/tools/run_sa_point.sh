#!/bin/bash
# Usage: TAG=x CONC=24 [DURATION=3600 PORT=30000] bash run_sa_point.sh   (SemiAnalysis aiperf fork, ATOM's client flags)
source "$(dirname "$0")/env.sh"
source "$M3_HERE/atom_client_env.sh"
export AIPERF_EXTRA="$ATOM_CLIENT_FLAGS"
: "${DURATION:=3600}" "${PORT:=30000}"
exec bash "$M3_HERE/bench_any.sh"
