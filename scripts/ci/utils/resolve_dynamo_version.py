#!/usr/bin/env python3

import re
from time import sleep
from urllib.error import URLError
from urllib.request import urlopen

INDEX_URL = "https://pypi.nvidia.com"
VERSION = r"\d+\.\d+\.\d+\.dev\d{8}"
ARCHES = {"x86_64", "aarch64"}


def fetch_index(package):
    for attempt in range(3):
        try:
            with urlopen(f"{INDEX_URL}/{package}/", timeout=30) as response:
                return response.read().decode()
        except URLError:
            if attempt == 2:
                raise
            sleep(2)


def resolve_version(dynamo_index, runtime_index):
    dynamo_versions = set(
        re.findall(rf"ai_dynamo-({VERSION})-py3-none-any\.whl", dynamo_index)
    )
    runtime_arches = {}
    for version, arch in re.findall(
        rf"ai_dynamo_runtime-({VERSION})-cp310-abi3-manylinux_2_28_(x86_64|aarch64)\.whl",
        runtime_index,
    ):
        runtime_arches.setdefault(version, set()).add(arch)

    complete = dynamo_versions & {
        version for version, arches in runtime_arches.items() if arches >= ARCHES
    }
    return max(
        complete,
        key=lambda version: tuple(map(int, re.findall(r"\d+", version))),
        default="",
    )


if __name__ == "__main__":
    print(resolve_version(fetch_index("ai-dynamo"), fetch_index("ai-dynamo-runtime")))
