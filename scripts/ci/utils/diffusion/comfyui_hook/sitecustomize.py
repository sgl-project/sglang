# SPDX-License-Identifier: Apache-2.0
"""Give ComfyUI the same host-memory horizon SGLang is measured under.

ComfyUI's model management reads ``psutil.virtual_memory()`` to decide what to
keep resident, so on a machine with far more RAM than the deployment being
compared it simply uses more and looks faster. Measuring that compares the two
engines' *circumstances*, not their capability -- on one 32 GiB-horizon
experiment ComfyUI's anonymous footprint fell from 116 GiB to 30 GiB and it
changed loading strategy entirely.

``COMFY_FORCE_HOST_GIB`` caps what it believes the host has, mirroring the
budget the SGLang side is held to. Available memory is reported as that cap
minus this process's own anonymous resident set, which is the quantity ComfyUI
actually reasons about. Unset means no patching, so a normal ComfyUI run is
untouched.

Python imports this automatically when its directory is on ``PYTHONPATH``; the
comparison runner puts it there for every ComfyUI launch.
"""

import os


def _install() -> None:
    cap_gib = os.environ.get("COMFY_FORCE_HOST_GIB")
    if not cap_gib:
        return
    try:
        cap_bytes = int(float(cap_gib) * (1024**3))
    except ValueError:
        print(f"[comfy-hook] ignoring non-numeric COMFY_FORCE_HOST_GIB={cap_gib!r}")
        return
    if cap_bytes <= 0:
        return

    try:
        import psutil
    except ImportError:
        print("[comfy-hook] psutil not importable; host horizon NOT capped")
        return

    real_virtual_memory = psutil.virtual_memory
    process = psutil.Process()

    def _own_anon_bytes() -> int:
        """This process's anonymous RSS — what it has already spent."""
        try:
            info = process.memory_full_info()
            # USS is the closest portable stand-in for "pages only we hold".
            return int(getattr(info, "uss", None) or info.rss)
        except Exception:
            return 0

    def capped_virtual_memory():
        real = real_virtual_memory()
        used = min(_own_anon_bytes(), cap_bytes)
        available = max(cap_bytes - used, 0)
        return real._replace(
            total=cap_bytes,
            available=available,
            used=used,
            free=available,
            percent=round(100.0 * used / cap_bytes, 1),
        )

    def capped_swap_memory():
        # Swap is what lets an over-budget run survive instead of failing, so
        # leaving it visible would reopen the headroom the cap just closed.
        real = psutil.swap_memory()
        return real._replace(total=0, used=0, free=0, percent=0.0)

    psutil.virtual_memory = capped_virtual_memory
    psutil.swap_memory = capped_swap_memory
    print(
        f"[comfy-hook] host horizon capped to {cap_bytes / 1024**3:.1f} GiB "
        "(swap hidden) for comparison parity"
    )


_install()
