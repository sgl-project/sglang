"""C211 nonce probe i123140 - non-destructive CI policy probe. Do not merge.

If any workflow executes fork-head code on a GPU runner, this test prints the
device lines that prove it. It asserts nothing and changes nothing.
"""
import subprocess


def _run(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=60)
        print("C211_NONCE_I123140_CMD", " ".join(args), "rc=", p.returncode)
        print(p.stdout)
        if p.stderr:
            print("C211_NONCE_I123140_STDERR", p.stderr)
    except Exception as e:
        print("C211_NONCE_I123140_ERR", repr(e))


def test_c211_nonce_i123140_device_probe():
    _run(["nvidia-smi", "-L"])
    _run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv"])
    _run(["rocm-smi", "--showproductname"])
    print("C211_NONCE_I123140_EXECUTED")
