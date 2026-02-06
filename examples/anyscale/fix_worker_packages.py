#!/usr/bin/env python3
"""Fix package compatibility and install local sglang on all Ray worker nodes.

This script uses Ray remote tasks to:
1. Install compatible packages on all nodes in the cluster
2. Replace sglang.srt with local version from working_dir

It must be run BEFORE importing sglang to avoid import errors.

Packages fixed:
- numpy < 2.0: Prevents scipy/sklearn/torchvision import errors
- sgl-kernel: Reinstalled from PyPI to pick up correct GPU architecture

Note: flashinfer 0.6.1 is kept from Dockerfile. The version check is
bypassed via SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 env var.

Note: The entrypoint only runs on the head node, so this script must use
Ray remote tasks to apply fixes to worker nodes.

Important: The local srt path is found INSIDE each remote task by searching
Ray's working_dir locations (/tmp/ray/session_*/runtime_resources/working_dir_files/),
because the absolute path differs between head and worker nodes.
"""

import ray

ray.init()


def fix_worker_packages_impl():
    """Implementation of package fix - runs on a specific node.

    Finds local srt by searching Ray's working_dir locations (which differ per node).
    """
    import glob
    import os
    import shutil
    import socket
    import subprocess
    import sys

    node_ip = socket.gethostbyname(socket.gethostname())

    # Step 1: Fix pip packages
    cmds = [
        [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', 'numpy>=1.26.0,<2.0'],
        # Keep flashinfer 0.6.1 from Dockerfile - version check bypassed via SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
        [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', 'sgl-kernel', '--force-reinstall'],
    ]

    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return f'FAILED on {node_ip}: pip install error: {r.stderr[:100]}'

    # Step 2: Find local srt in Ray's working_dir
    # Ray stores working_dir under /tmp/ray/session_*/runtime_resources/working_dir_files/
    # The path differs per node, so we search for it
    search_patterns = [
        '/tmp/ray/session_*/runtime_resources/working_dir_files/*/python/sglang/srt',
        os.path.join(os.getcwd(), 'python', 'sglang', 'srt'),  # Fallback
    ]

    local_srt_path = None
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            # Use the most recently modified match (in case of multiple sessions)
            local_srt_path = max(matches, key=os.path.getmtime)
            break

    if not local_srt_path or not os.path.exists(local_srt_path):
        return f'FAILED on {node_ip}: could not find local srt (searched: {search_patterns})'

    # Step 3: Replace sglang.srt with local version
    import sglang
    sglang_install_path = os.path.dirname(sglang.__file__)
    installed_srt_path = os.path.join(sglang_install_path, 'srt')

    try:
        if os.path.exists(installed_srt_path):
            shutil.rmtree(installed_srt_path)
        shutil.copytree(local_srt_path, installed_srt_path)
    except Exception as e:
        return f'FAILED on {node_ip}: copy error: {str(e)[:100]}'

    import numpy
    return f'OK on {node_ip} (numpy={numpy.__version__}, srt copied from {local_srt_path})'


def main():
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Get all alive nodes
    nodes = [n for n in ray.nodes() if n['Alive']]
    print(f'  Found {len(nodes)} nodes')

    # Create a remote function that targets a specific node
    @ray.remote(num_cpus=0)  # Use 0 CPUs to avoid resource constraints
    def fix_on_node():
        return fix_worker_packages_impl()

    # Track which node IPs have been successfully fixed
    fixed_ips = set()
    all_node_ips = {n['NodeManagerAddress'] for n in nodes}

    # First pass: try to schedule on each specific node with soft affinity
    print('  Pass 1: Targeting specific nodes...')
    futures = []
    for node in nodes:
        node_id = node['NodeID']
        node_ip = node['NodeManagerAddress']

        # Use soft affinity - prefer this node but don't fail if unavailable
        future = fix_on_node.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=True,  # Soft affinity - prefer but don't require
            )
        ).remote()
        futures.append((node_ip, future))

    # Collect results from first pass
    for node_ip, future in futures:
        try:
            result = ray.get(future, timeout=120)
            print(f'  {result}')
            # Extract actual IP from result (format: "OK on IP ...")
            if result.startswith('OK on '):
                actual_ip = result.split()[2]
                fixed_ips.add(actual_ip)
        except Exception as e:
            print(f'  FAILED targeting {node_ip}: {e}')

    # Check if any nodes were missed
    missing_ips = all_node_ips - fixed_ips
    if missing_ips:
        print(f'  Pass 2: {len(missing_ips)} nodes may need retry, running extra tasks...')
        # Run extra tasks to try to cover missing nodes
        extra_futures = [fix_on_node.remote() for _ in range(len(missing_ips) * 3)]
        for future in extra_futures:
            try:
                result = ray.get(future, timeout=120)
                if result.startswith('OK on '):
                    actual_ip = result.split()[2]
                    if actual_ip in missing_ips and actual_ip not in fixed_ips:
                        print(f'  {result}')
                        fixed_ips.add(actual_ip)
            except Exception as e:
                pass  # Ignore failures in extra pass

    # Final check
    still_missing = all_node_ips - fixed_ips
    if still_missing:
        print(f'  WARNING: Could not confirm fix on nodes: {still_missing}')
    else:
        print(f'  All {len(fixed_ips)} nodes confirmed fixed.')

    ray.shutdown()
    print('  Worker nodes ready.')


if __name__ == '__main__':
    main()
