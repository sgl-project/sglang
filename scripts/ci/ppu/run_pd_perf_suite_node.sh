#!/bin/bash
# Run one node's share of a prefill/decode-disaggregated performance entry.
#
# A PD group is two pods of one job: each runs this script, is told by the action
# which rank it is, and serves the role the reviewed config assigns to that rank.
# Rank 0 also runs the router the benchmark posts to and owns the report; the
# other node serves its half of the KV path and holds its board until rank 0 is
# done.  Neither of those decisions is made here -- they are made in
# `pd_perf_suite_kit` off the config -- so this script is only the setup a board
# needs before the suite can run, plus the per-node log every pod but worker-0
# would otherwise lose.
#
# A separate script from run_perf_suite_board.sh rather than a flag on it, for the
# reason that one is separate from the Answer board script: the two lines read
# different environment variables on purpose, so a change made for one cannot move
# the other's config.  What it does share is the tail -- run_perf_suite_node.sh --
# because keeping a node's log and its exact exit status is the same problem here.
#
# One thing the multi-node colocated entry does and this one deliberately does
# not: it derives no rendezvous address.  The two servers of a PD group do not
# join one process group -- each is a rank 0 of its own eight devices -- so there
# is nothing to meet at.  What the nodes exchange instead is an HTTP endpoint,
# published through the results directory by the suite itself.
#
# Reads:
#   SGLANG_PPU_PD_PERF_TEST_CONFIG   absolute path of the reviewed PD config, and
#                                    the one source for the visible devices and
#                                    the checkpoint
#   SGLANG_PPU_PD_PERF_RESULTS_DIR   absolute path every node of the group shares
#   PERF_SUITE                       registered suite name for run_suite.py
#   PERF_TIMEOUT_PER_FILE            seconds run_suite.py gives the registered file
#
# Usage: bash /workspace/source/scripts/ci/ppu/run_pd_perf_suite_node.sh
set -e

mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'PIPEOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
PIPEOF

: "${SGLANG_PPU_PD_PERF_TEST_CONFIG:?the reviewed PD config path is required}"
: "${SGLANG_PPU_PD_PERF_RESULTS_DIR:?the shared results directory is required}"
: "${PERF_SUITE:?the registered suite name is required}"
: "${PERF_TIMEOUT_PER_FILE:?the per-file timeout is required}"

# Everything below, into the shared directory, from this line rather than from
# the suite. The action streams worker-0 and no other pod, and `rank-N.log` is
# opened by `run_perf_suite_node.sh`, which is the last thing this script does --
# so a node that dies in the install, the JIT cache or the page-cache warm dies
# with its output nowhere. The fourth dispatch of this line is what that costs:
# the decode pod exited non-zero three minutes in, the run reported
# "failed=1" without naming a pod or a reason, the pods were gone by the time
# anyone looked, and the shared directory held not one byte of that node. Which
# left the whole pre-suite phase -- the only phase it could have died in -- with
# no evidence at all, on a line where every dispatch costs two boards.
#
# Appended rather than truncated so a retried node does not erase its first
# attempt, and guarded on both things the redirect needs -- a writable path under
# the shared directory, and the /dev/fd the process substitution opens -- because
# a failed redirection on `exec` ends the script under `set -e`, and a diagnostic
# that fails the run it exists to explain would be worse than no diagnostic.
PD_NODE_LOG="$SGLANG_PPU_PD_PERF_RESULTS_DIR/ranks/rank-${NODE_RANK:-0}-node.log"
if mkdir -p "$(dirname "$PD_NODE_LOG")" 2>/dev/null &&
  touch "$PD_NODE_LOG" 2>/dev/null &&
  [ -d /dev/fd ]; then
  exec > >(tee -a "$PD_NODE_LOG") 2>&1
else
  echo "could not open $PD_NODE_LOG; this node's log stays in its pod" >&2
fi

# Constant for every entry of this line, as on the single-board one:
# HF_HUB_OFFLINE keeps the tokenizer load off a cluster that has no network, and
# a PD run loads a tokenizer three times over -- once in each server and once in
# bench_serving; PPU_SUPPORTS_FP8=0 reflects the board, not a preference.
export SGLANG_IS_IN_CI=true
export CPLUS_INCLUDE_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/include
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub
export PPU_SUPPORTS_FP8=0

# v0.5.18 asserts sglang-kernel >= 0.4.6.post1 / flashinfer >= 0.6.17 whenever it
# sees CUDA, and PPU presents itself as CUDA. The image carries the vendor's PPU
# builds (0.4.3+v0.1.0.ppu2.1.1 / 0.6.12), which sit on a different release train
# and have no upgrade path here, so the comparison only blocks the launch --
# measured on the accuracy line, run 34900233869.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# _handle_ppu_backends enables SGLANG_SAIL_PLA_CUDA on PPU, which sends the GDN
# linear-attention decode into `pla.decode`. `pla` is absent from the v2.1.1
# image and nothing installs it, so the Qwen3.5 and Qwen3-Next entries die in
# CUDA graph capture with `No module named 'pla'` (measured on the accuracy
# line, run 34908645516). Off takes the community Triton kernel the same
# function falls through to. Drop this once the image ships `pla`.
export SGLANG_SAIL_PLA_CUDA=0

# `lo`, exactly as on the single-board entries and for the same reason: gloo
# carries the CPU side of every process group SGLang creates, and each server here
# is one node's worth of ranks in a single network namespace.  The peer of this
# pod is reached over HTTP and over RDMA, never over gloo, so the interface
# derivation the multi-node colocated entry performs has nothing to derive here.
export GLOO_SOCKET_IFNAME=lo

cd /workspace/source
if [[ -z "${PPU_E2E_PLAN:-}" ]]; then
  git config --global --add safe.directory /workspace/source
fi
SGLANG_PPU_SOURCE_REVISION=$(git rev-parse HEAD)
export SGLANG_PPU_SOURCE_REVISION

# The RoCE GID index, which the Mooncake transfer engine needs told even though
# no collective here crosses the node boundary -- each server is tp 8 inside one
# board, so pccl never leaves it, and what does leave it is the KV path.
# Mooncake picks the GID itself and cannot pick one here: its automatic discovery
# requires an IPv4-mapped-IPv6 GID, and these bonds carry no IPv4 at all, so it
# rejects every entry of a six-entry table and opens nothing --
# "No suitable GID found on mlx5_bond_1/", then "Failed to open device
# mlx5_bond_1 on port  with GID -1" on all four bonds, then a transfer engine
# that fails to initialize and a server that dies twenty seconds in.  That is
# upstream Mooncake #1593, and the first dispatch of this line hit it.  Told the
# index explicitly, Mooncake uses it: it reads MC_GID_INDEX first and falls back
# to NCCL_IB_GID_INDEX only in versions that carry that fallback, so the Mooncake
# name is the one set here -- and only that one, since NCCL_IB_GID_INDEX would
# also reach a pccl that currently works without it.  The script is the
# collective line's, because it reads sysfs rather than a suite's config: among
# the GIDs that are not link-local it takes the highest index, the RoCE v2 entry
# of the routable fd03::/8 address, and refuses to print a number if the devices
# of a host disagree.
MC_GID_INDEX=$(bash scripts/ci/ppu/answer_gid_index.sh)
export MC_GID_INDEX

# And the HTTP hops that must not go through a proxy.  Every one this line makes
# is inside the cluster: a server warming itself up over its own routable
# address, rank 0 waiting for its peer's `/health`, the benchmark posting to the
# router, and the router reaching both servers.  On the second dispatch of this
# line the first of those never arrived -- the prefill server's own
# `/model_info` came back as an nginx 404 for the full two minutes of the warmup
# loop while the uvicorn bound to that exact address logged no request at all,
# and the server was killed as a failed initialization.  Nothing here names a
# proxy: not this script, not the workflow, not the distributed action, and not
# the source commands this case was ported from.  So the bypass is what this line
# can set, and the variables it inherited are printed below, which is what tells
# a repeat of that 404 apart from a transparent redirect.
#
# `requests`, which the warmup and every readiness poll use, and the router's
# Rust client both read `no_proxy` and both accept a CIDR block; the benchmark's
# aiohttp reads neither, so it was never at risk.  The block is this node's own
# /24 -- the peer's out-of-band address is on it, so a peer discovered at runtime
# needs nothing added -- plus loopback, for the router the benchmark posts to.
#
# Which address that is takes four methods, and the order is what three dispatches
# taught.  The default route's source address was the only method until one run
# landed on two boards that have no default route: both pods exited thirty seconds
# in, having derived nothing, and the K8s events showed both containers started
# clean, so the whole failure was this derivation.  The boards that carry a
# default route are not a subset anyone controls -- the scheduler picks whichever
# two are free -- so no method that needs routing to the internet can be the last
# word.  Resolving the node's own name was the answer for two boards and not for a
# third: the next run put rank 0 on swu03, whose record exists in the DNS this
# fabric answers with, and the lookup a second into the container's life returned
# nothing anyway, which is what a resolver that is not up yet looks like.
#
# So the method that does not depend on either is asked first of the two: which
# address does this node use to reach the NAS it is already writing to.  The
# server is read out of the mount table rather than named here, the route to it is
# specific rather than default, and the question is answered by a file this script
# has already written -- the log holding this very line lives on that NAS, so a
# node that got this far has a working path to it.  On the bypass host of this
# fabric the three methods agree to the letter: the route to the NAS, the route to
# the internet, and DNS for its own name all return 11.161.48.214, which is the
# same agreement DNS and the route showed on swu07 and swu08.  `hostname -i` stays
# last, since it was already here and costs nothing, though it was empty on every
# board that needed it.
#
# SGLANG_HOST_IP carries the answer into the suite, which otherwise repeats the
# derivation itself: `get_local_ip_auto` probes by connecting a UDP socket
# towards 8.8.8.8, which needs the same default route, and its own fallback is
# the hostname lookup that was empty here -- so on those two boards the suite
# would raise `Can not get local ip` a minute after this script stopped
# protecting it.  Read first by that function, it also makes one address serve
# the bypass below, the endpoint each node publishes, and the address each
# server binds, rather than three derivations that agree only by luck.
address_towards() {
  ip route get "$1" 2>/dev/null |
    awk '{for (field = 1; field < NF; field++) if ($field == "src") print $(field + 1)}' |
    head -1
}
NODE_ADDRESS=$(address_towards 1.1.1.1)
NODE_ADDRESS_SOURCE="the source address of the default route"
if [ -z "$NODE_ADDRESS" ]; then
  # The NFS server behind the results directory, which is the deepest mount point
  # this directory sits under -- a node mounting more than one NAS has one that
  # holds the group's rank files and that is the one worth reaching.
  NAS_ADDRESS=$(awk -v dir="$SGLANG_PPU_PD_PERF_RESULTS_DIR/" '
    $3 ~ /^nfs/ {
      point = $2
      if (substr(point, length(point)) != "/") point = point "/"
      if (index(dir, point) == 1 && length(point) > length(deepest)) {
        deepest = point
        if (match($4, /addr=[0-9.]+/)) server = substr($4, RSTART + 5, RLENGTH - 5)
      }
    }
    END {print server}
  ' /proc/mounts)
  if [ -n "$NAS_ADDRESS" ]; then
    NODE_ADDRESS=$(address_towards "$NAS_ADDRESS")
    NODE_ADDRESS_SOURCE="the route to the NAS at $NAS_ADDRESS"
  fi
fi
if [ -z "$NODE_ADDRESS" ] && [ -n "${NODE_NAME:-}" ]; then
  # Asked more than once, and not because DNS is flaky: the one board where this
  # returned nothing had a record, and the query went out seconds after the
  # container started.  A resolver reached a moment later answers.
  for attempt in 1 2 3 4 5; do
    NODE_ADDRESS=$(getent ahostsv4 "$NODE_NAME" 2>/dev/null | awk '{print $1}' | head -1)
    if [ -z "$NODE_ADDRESS" ]; then
      NODE_ADDRESS=$(getent hosts "$NODE_NAME" 2>/dev/null | awk '$1 ~ /^[0-9]+[.]/ {print $1}' | head -1)
    fi
    if [ -n "$NODE_ADDRESS" ]; then
      break
    fi
    sleep 3
  done
  NODE_ADDRESS_SOURCE="DNS for $NODE_NAME"
fi
if [ -z "$NODE_ADDRESS" ]; then
  NODE_ADDRESS=$(hostname -i 2>/dev/null | awk '{print $1}')
  NODE_ADDRESS_SOURCE="hostname -i"
fi
if [ -z "$NODE_ADDRESS" ]; then
  echo "could not derive this node's own address, which the proxy bypass, the" \
    "endpoint this node publishes and the address its server binds all need;" \
    "tried the default route, the route to the NAS at" \
    "${NAS_ADDRESS:-an address the mount table did not give}," \
    "DNS for ${NODE_NAME:-an unnamed node}, and hostname -i" >&2
  exit 1
fi
export SGLANG_HOST_IP="$NODE_ADDRESS"
no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,$NODE_ADDRESS,$(echo "$NODE_ADDRESS" | cut -d. -f1-3).0/24"
export no_proxy
NO_PROXY="$no_proxy"
export NO_PROXY

# The device list and the checkpoint come out of the config the test will read: a
# second copy in the workflow could drift from what the suite validates against.
# nproc_per_node is a resource request and not device isolation, so narrowing the
# visible set here is what makes this node's tp 8 server a whole-board server
# rather than one sharing a board the test never checked it had to itself.
CUDA_VISIBLE_DEVICES=$(python3 -c "import json, os; print(','.join(str(int(device)) for device in json.load(open(os.environ['SGLANG_PPU_PD_PERF_TEST_CONFIG']))['hardware']['visible_devices']))")
export CUDA_VISIBLE_DEVICES
# PCCL detects an MNNVL fabric (shared UUID, cliqueSize 8) but the IMEX service
# is not provisioned on PPU, so fabric rendezvous aborts instead of falling
# back. sglang v0.5.18 additionally auto-enables custom all-reduce v2 multinode
# on any detected MNNVL fabric with tp<=8
# (server_args._handle_custom_all_reduce_v2_multinode), an opt-in that only
# holds on GB200/GB300. Disable both so the collective falls back to the legacy
# RoCE path. Each PD role may run single-node here, but the guard is harmless
# and keeps every PPU launch path uniform.
export PCCL_MNNVL_ENABLE=0
export SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE=0
PD_MODEL_PATH=$(python3 -c "import json, os; print(json.load(open(os.environ['SGLANG_PPU_PD_PERF_TEST_CONFIG']))['model']['path'])")
# Which role this node serves is the config's answer to the rank it was handed,
# and it is echoed rather than acted on: the suite resolves it again, and a log
# that names it is what makes a two-pod run readable. Read out of the JSON rather
# than through `pd_perf_eval_kit`, because this runs before the install.
PD_ROLE=$(python3 -c "import json, os; c = json.load(open(os.environ['SGLANG_PPU_PD_PERF_TEST_CONFIG'])); print('prefill' if int(os.environ.get('NODE_RANK', '0')) < c['disaggregation']['prefill_nodes'] else 'decode')")
# And the port that role's server will listen on, read the same way and for the
# same reason: only the probes below use it, and only to reach an address before
# anything of ours is behind it.
PD_ROLE_PORT=$(python3 -c "import json, os; c = json.load(open(os.environ['SGLANG_PPU_PD_PERF_TEST_CONFIG'])); print(c['disaggregation'][('prefill' if int(os.environ.get('NODE_RANK', '0')) < c['disaggregation']['prefill_nodes'] else 'decode') + '_port'])")
echo "source revision: $SGLANG_PPU_SOURCE_REVISION"
echo "test config:     $SGLANG_PPU_PD_PERF_TEST_CONFIG"
echo "suite:           $PERF_SUITE"
echo "node rank:       ${NODE_RANK:-0} of ${NNODES:-1} on ${NODE_NAME:-an unnamed node}"
echo "role:            $PD_ROLE"
echo "role port:       $PD_ROLE_PORT"
echo "results dir:     $SGLANG_PPU_PD_PERF_RESULTS_DIR"
echo "gloo interface:  $GLOO_SOCKET_IFNAME"
echo "gid index:       $MC_GID_INDEX"
echo "node address:    $NODE_ADDRESS (from $NODE_ADDRESS_SOURCE)"
echo "proxy bypass:    $no_proxy"
echo "inherited proxy: http_proxy=${http_proxy:-unset} https_proxy=${https_proxy:-unset} HTTP_PROXY=${HTTP_PROXY:-unset} HTTPS_PROXY=${HTTPS_PROXY:-unset}"
echo "visible devices: $CUDA_VISIBLE_DEVICES"
echo "checkpoint:      $PD_MODEL_PATH"

# Three probes, before the install and before a checkpoint is touched, at
# addresses nothing of ours is listening on yet.  They are here because the
# second and third dispatches of this line both died the same way: a prefill
# server bound to its own routable address on port 30000, logged no request for
# the two minutes its warmup spent asking itself for `/model_info`, and read an
# nginx 404 back every time.  A bound socket that never sees the packets sent to
# it means something on the node rewrites them, and Kubernetes hands out node
# ports from exactly 30000, which is why the ports moved below it.  The probes
# are what turned that reading into a measurement, and the fourth dispatch
# printed all three: this node's address on 30000 answered `HTTP 404`, which is
# the finding, since nothing of ours was behind it; 30000 on loopback answered
# `HTTP 404` as well; and the port this run uses refused, which is what a port
# with nothing behind it is supposed to do.
#
# The loopback answer is the part worth keeping in mind.  The rewrite matches on
# the port and not on the address, so 127.0.0.1 is inside it too, and the
# colocated line's habit of talking to 127.0.0.1 protects it from nothing: that
# line's 31000 is inside the same node-port range, and it has stayed up only
# because no service in this cluster has been handed 31000 yet.
#
# All three are one connect each, and none of them fails the run: a probe that
# comes back other than expected is read in the log next to the failure it
# explains.
for probe_target in "$NODE_ADDRESS 30000" "127.0.0.1 30000" "$NODE_ADDRESS $PD_ROLE_PORT"; do
  probe_address=${probe_target% *}
  probe_port=${probe_target#* }
  probe_answer=$(python3 -c "
import sys, urllib.error, urllib.request
try:
    with urllib.request.urlopen(f'http://{sys.argv[1]}:{sys.argv[2]}/model_info', timeout=5) as response:
        print(f'HTTP {response.status}')
except urllib.error.HTTPError as error:
    print(f'HTTP {error.code}')
except urllib.error.URLError as error:
    reason = error.reason
    print(type(reason).__name__ if isinstance(reason, BaseException) else reason)
except Exception as error:
    print(type(error).__name__)
" "$probe_address" "$probe_port")
  echo "port probe:      $probe_address:$probe_port answers $probe_answer with nothing of ours listening"
done

bash scripts/ci/ppu/ppu_install_dependency.sh

# The JIT cache is keyed on source mtime, and a fresh checkout stamps every file
# with the checkout time, so without this every run recompiles.
JIT_MTIME=$(git log -1 --format=%ct 2>/dev/null || echo 0)
if [ "$JIT_MTIME" != 0 ] && [ -d "python/sglang/kernels/jit" ]; then
  find python/sglang/kernels/jit -type f -exec touch -d @"$JIT_MTIME" {} +
  echo 'Pinned JIT source mtimes'
fi

# The JIT build directory moves off the shared NAS cache before anything can
# compile; the script it comes from records why.
# shellcheck source=scripts/ci/ppu/use_local_jit_cache.sh
. scripts/ci/ppu/use_local_jit_cache.sh

# Warmed on both nodes, unlike the pp-2 colocated entry that skips it: each role
# here loads the whole checkpoint into its own eight devices, so each node reads
# the whole tree and neither warm is wasted. It is the startup budget this
# protects, not the numbers.
bash scripts/ci/warm_page_cache.sh "$PD_MODEL_PATH"

# The tail is the colocated multi-node line's, because a PD node has the same two
# things to preserve: its own log, which the action streams for worker-0 only, and
# its exact exit status through the tee. It reads NODE_RANK for the file names and
# nothing else, so it needs no PD-specific variant.
if [[ -n "${PPU_E2E_PLAN:-}" ]]; then
  exec python3 scripts/ci/ppu/pr_e2e_worker.py suite
fi
cd test
bash ../scripts/ci/ppu/run_perf_suite_node.sh \
  "$PERF_SUITE" "$PERF_TIMEOUT_PER_FILE" "$SGLANG_PPU_PD_PERF_RESULTS_DIR/ranks"
