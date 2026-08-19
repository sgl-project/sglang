#!/usr/bin/env python3
"""TEST-ONLY (branch bingxche/rocm715-fullflow): report how torch enumerates GPUs.

The ROCm 7.15 images fail every test with AssertionError("Invalid device id"),
raised by torch.cuda.get_device_properties only when
``device < 0 or device >= device_count()`` where ``device`` comes from
current_device(). That means is_available() was true while current_device()
returned an index device_count() rejects, so the three have to be sampled
independently -- each in its own try/except, or the first failure hides the
rest. Delete with the rest of the test-only changes.
"""

import os
import sys


def show(label, fn):
    try:
        print(f"  {label} = {fn()!r}")
    except BaseException as exc:  # noqa: BLE001 - the failure is the datapoint
        print(f"  {label} !! {type(exc).__name__}: {exc}")


def main():
    print(f"  python = {sys.version.split()[0]}")
    try:
        import torch
    except BaseException as exc:  # noqa: BLE001
        print(f"  import torch !! {type(exc).__name__}: {exc}")
        return

    print(f"  torch = {torch.__version__}")
    show("torch.version.hip", lambda: torch.version.hip)
    show("cuda.is_available()", torch.cuda.is_available)
    show("cuda.device_count()", torch.cuda.device_count)
    show("cuda.current_device()", torch.cuda.current_device)
    show("cuda.is_initialized()", torch.cuda.is_initialized)

    try:
        count = torch.cuda.device_count()
    except BaseException:  # noqa: BLE001
        count = 0
    for i in range(count):
        show(f"cuda.get_device_properties({i})", lambda i=i: torch.cuda.get_device_properties(i))
    # The exact call the suites die on, with no argument, so it goes through
    # current_device() the same way get_device_sm() does.
    show("cuda.get_device_capability()", torch.cuda.get_device_capability)

    for var in sorted(k for k in os.environ if "VISIBLE_DEVICES" in k.upper()):
        print(f"  env {var} = {os.environ[var]!r}")

    # device_count() on ROCm is amdsmi-based and only falls back to the HIP
    # runtime count when amdsmi returns a negative number, so an amdsmi count of
    # exactly 0 gives device_count() 0 while is_available(), which asks the HIP
    # runtime directly, stays true. Sample each step of that path to see which
    # one produces the 0.
    print("  -- device_count internals --")
    show("_C._cuda_getDeviceCount()", torch._C._cuda_getDeviceCount)
    show("cuda._HAS_PYNVML", lambda: torch.cuda._HAS_PYNVML)
    show("cuda._parse_visible_devices()", lambda: torch.cuda._parse_visible_devices()[:8])
    show("cuda._raw_device_count_amdsmi()", torch.cuda._raw_device_count_amdsmi)
    show("cuda._device_count_amdsmi()", torch.cuda._device_count_amdsmi)

    print("  -- amdsmi directly --")

    def handles():
        import amdsmi

        amdsmi.amdsmi_init()
        return len(amdsmi.amdsmi_get_processor_handles())

    show("amdsmi module", lambda: __import__("amdsmi").__file__)
    show("amdsmi processor handles", handles)


if __name__ == "__main__":
    main()
