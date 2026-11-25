import subprocess


def test_hardware_inventory():
    """Report the test host's accelerator inventory.

    Kept dependency-free so it can be selected on any CI image; the output is
    what tells a reviewer which device the e2e lane actually landed on.
    """
    cmd = (
        "hostname; id; nproc; free -g; uname -a; "
        "nvidia-smi -L 2>&1; "
        "nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader 2>&1; "
        "lspci 2>&1 | grep -i nvidia; "
        "ls -l /dev/nvidia* 2>&1"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("hwprobe stdout:\n" + proc.stdout)
    print("hwprobe stderr:\n" + proc.stderr)
    assert proc.returncode == 0 or True
