"""Unit tests for merging adjacent host pages of a staged HiCache write.

The staged page_first write copies device staging to the host. One copy per page
is slow for small pages, so ``kvcacheio/page_runs.h`` groups pages whose host
slots are adjacent. The header has no device code: the tests build it with the
host C++ compiler and replay the copies with ``memcpy``.
"""

import random
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import sglang
from sglang.srt.mem_cache.pool_host.common import (
    _CUDA_HOST_REGISTERED_RANGES_ATTR,
    host_buffer_is_one_pinned_range,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CSRC = Path(sglang.__file__).parent / "kernels" / "jit" / "csrc"
_CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

# Reads "num_pages page_size page_bytes merge" and one host token index per
# staged token. Prints the runs, then the host buffer after replaying one copy
# per run from a staging buffer whose byte i holds i % 251.
_DRIVER = r"""
#include "kvcacheio/page_runs.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  long long num_pages, page_size, page_bytes, merge, host_pages;
  if (std::scanf("%lld %lld %lld %lld %lld", &num_pages, &page_size, &page_bytes, &merge, &host_pages) != 5) return 2;
  std::vector<int64_t> dst(num_pages * page_size);
  for (auto& index : dst) {
    long long value;
    if (std::scanf("%lld", &value) != 1) return 2;
    index = value;
  }
  std::vector<unsigned char> staging(num_pages * page_bytes), host(host_pages * page_bytes, 255);
  for (size_t i = 0; i < staging.size(); ++i) staging[i] = static_cast<unsigned char>(i % 251);
  const long long token_bytes = page_bytes / page_size;
  sglang::for_each_host_page_run(
      dst.data(), num_pages, page_size, merge != 0, [&](int64_t first_page, int64_t run_pages) {
        std::printf("run %lld %lld\n", static_cast<long long>(first_page), static_cast<long long>(run_pages));
        std::memcpy(
            host.data() + dst[first_page * page_size] * token_bytes,
            staging.data() + first_page * page_bytes,
            run_pages * page_bytes);
      });
  for (unsigned char byte : host) std::printf("%d ", byte);
  std::printf("\n");
  return 0;
}
"""


@pytest.fixture(scope="module")
def page_runs_binary(tmp_path_factory):
    if _CXX is None:
        pytest.skip("needs a host C++ compiler")
    build = tmp_path_factory.mktemp("page_runs")
    source = build / "driver.cc"
    source.write_text(_DRIVER)
    binary = build / "driver"
    subprocess.run(
        [_CXX, "-std=c++17", "-O1", "-I", str(_CSRC), str(source), "-o", str(binary)],
        check=True,
    )
    return binary


def _replay(binary, host_pages, page_size, page_bytes, merge, host_page_count):
    tokens = [page * page_size + t for page in host_pages for t in range(page_size)]
    text = (
        f"{len(host_pages)} {page_size} {page_bytes} {int(merge)} {host_page_count}\n"
        + " ".join(map(str, tokens))
    )
    out = subprocess.run(
        [str(binary)], input=text, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    runs = [tuple(map(int, line.split()[1:])) for line in out if line.startswith("run")]
    return runs, out[-1]


def _expected_runs(host_pages, merge):
    runs = []
    for page, host_page in enumerate(host_pages):
        if merge and runs and host_page == host_pages[page - 1] + 1:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((page, 1))
    return runs


@pytest.mark.parametrize("page_size", [1, 16])
def test_runs_follow_adjacent_host_pages(page_runs_binary, page_size):
    cases = {
        "empty": [],
        "one page": [5],
        "one run": [3, 4, 5, 6],
        "descending": [6, 5, 4, 3],
        "two runs": [0, 1, 2, 7, 8],
        "mixed": [9, 0, 1, 5, 4, 2, 3, 11],
        "repeated step back": [1, 2, 2, 3],
    }
    for name, host_pages in cases.items():
        runs, _ = _replay(
            page_runs_binary, host_pages, page_size, 4 * page_size, True, 12
        )
        assert runs == _expected_runs(host_pages, True), name
        assert sum(count for _, count in runs) == len(host_pages), name


@pytest.mark.parametrize("page_size", [1, 3, 16])
def test_merged_copies_write_the_same_bytes(page_runs_binary, page_size):
    rng = random.Random(1234 + page_size)
    host_page_count = 96
    for _ in range(40):
        # Slices of the free list in allocation order, as the host pool hands
        # them out: long ascending runs with a few breaks.
        free = list(range(host_page_count))
        host_pages = []
        while free and len(host_pages) < 64:
            start = rng.randrange(len(free))
            length = rng.randint(1, 12)
            host_pages += free[start : start + length]
            del free[start : start + length]
        page_bytes = 8 * page_size
        merged_runs, merged = _replay(
            page_runs_binary, host_pages, page_size, page_bytes, True, host_page_count
        )
        single_runs, single = _replay(
            page_runs_binary, host_pages, page_size, page_bytes, False, host_page_count
        )
        assert merged == single
        assert single_runs == [(page, 1) for page in range(len(host_pages))]
        assert merged_runs == _expected_runs(host_pages, True)
        assert len(merged_runs) <= len(single_runs)


def test_one_pinned_range():
    buffer = torch.zeros(2, 8, 4)
    assert host_buffer_is_one_pinned_range(buffer)
    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, [])
    assert host_buffer_is_one_pinned_range(buffer)
    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, [(0, 64)])
    assert host_buffer_is_one_pinned_range(buffer)
    # A view answers for the tensor that was registered.
    assert host_buffer_is_one_pinned_range(buffer[0])
    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, [(0, 32), (32, 32)])
    assert not host_buffer_is_one_pinned_range(buffer)
    assert not host_buffer_is_one_pinned_range(buffer[1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
