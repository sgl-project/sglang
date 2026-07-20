#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

"""Verify that SGLang wheels contain the canonical native gRPC contract."""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

WHEEL_PROTO_PATH = "sglang/srt/grpc/sglang.proto"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_proto = repo_root / "python" / WHEEL_PROTO_PATH
    source_contents = source_proto.read_bytes()

    for wheel_arg in sys.argv[1:]:
        wheel = Path(wheel_arg)
        if not wheel.is_file():
            raise SystemExit(f"wheel does not exist: {wheel}")
        try:
            with ZipFile(wheel) as archive:
                wheel_contents = archive.read(WHEEL_PROTO_PATH)
        except KeyError as err:
            raise SystemExit(f"{wheel} does not contain {WHEEL_PROTO_PATH}") from err
        except BadZipFile as err:
            raise SystemExit(f"invalid wheel archive: {wheel}") from err

        if wheel_contents != source_contents:
            raise SystemExit(
                f"{WHEEL_PROTO_PATH} in {wheel} does not match {source_proto}"
            )
        print(f"Verified {WHEEL_PROTO_PATH} in {wheel}")


if __name__ == "__main__":
    main()
