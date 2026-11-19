"""Build the zip consumed by Nightly's container_patches input."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def build(output):
    source = Path(__file__).resolve().parent
    with ZipFile(output, "w") as archive:
        for name in (
            "install.sh",
            "install.py",
            "mamba-pin-diagnostics.patch",
            "npu_pin_diagnostics.py",
        ):
            # Stable metadata makes it possible to verify the shipped archive
            # byte-for-byte against its reviewable source.
            info = ZipInfo(name, date_time=(2026, 11, 19, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (source / name).read_bytes())
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
