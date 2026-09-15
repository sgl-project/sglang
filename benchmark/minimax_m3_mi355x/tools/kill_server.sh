#!/bin/bash
# Usage: bash kill_server.sh PORT  -- kills the launch_server for PORT, its process group, and orphaned sglang:: workers
PORT=${1:?}
for p in $(pgrep -f "launch_serve[r].*--port $PORT"); do pg=$(ps -o pgid= -p $p | tr -d ' '); kill -TERM $p 2>/dev/null; sleep 5; kill -9 -- -$pg 2>/dev/null; kill -9 $p 2>/dev/null; done
sleep 2
for p in $(pgrep -f "sglang::"); do pp=$(awk '{print $4}' /proc/$p/stat 2>/dev/null); [ "$pp" = "1" ] && kill -9 $p 2>/dev/null && echo "killed orphan $p"; done
sleep 3; rocm-smi --showmemuse 2>/dev/null | grep -E "GPU\[[0-7]\].*VRAM%" | tr '\n' ' '; echo
