"""Per-tokenizer-worker split of the established client connections on the server port.

usage: conn_split.py PORT WORKERS [SAMPLES=3] [INTERVAL_S=3]
Prints "MAX TOTAL a/b/c/d" (largest worker first) for the sample with the largest per-worker count.
Exits 2 when the number of `sglang::tokenizer_worker` processes differs from WORKERS.
"""

import glob
import os
import re
import sys
import time


def established_inodes(port_hex):
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(table).readlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            fields = line.split()
            # local address ends with the port; state 01 is ESTABLISHED (the listener is 0A)
            if fields[1].endswith(":" + port_hex) and fields[3] == "01":
                inodes.add(fields[9])
    return inodes


def worker_pids():
    pids = []
    for proc in glob.glob("/proc/[0-9]*"):
        try:
            cmdline = open(proc + "/cmdline", "rb").read()
        except OSError:
            continue
        if cmdline.startswith(b"sglang::tokenizer_worker"):
            pids.append(int(proc[len("/proc/") :]))
    return pids


def socket_count(pid, inodes):
    count = 0
    for fd in glob.glob(f"/proc/{pid}/fd/*"):
        try:
            link = os.readlink(fd)
        except OSError:
            continue
        match = re.match(r"socket:\[(\d+)\]", link)
        if match and match.group(1) in inodes:
            count += 1
    return count


def main():
    port, workers = int(sys.argv[1]), int(sys.argv[2])
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    interval = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    port_hex = f"{port:04X}"
    best = None
    for i in range(samples):
        pids = worker_pids()
        if len(pids) != workers:
            print(f"ERROR: {len(pids)} sglang::tokenizer_worker processes, expected {workers}", file=sys.stderr)
            sys.exit(2)
        inodes = established_inodes(port_hex)
        counts = sorted((socket_count(pid=pid, inodes=inodes) for pid in pids), reverse=True)
        if best is None or counts[0] > best[0]:
            best = (counts[0], sum(counts), counts)
        if i + 1 < samples:
            time.sleep(interval)
    print(best[0], best[1], "/".join(str(c) for c in best[2]))


if __name__ == "__main__":
    main()
