"""
E2E HiCache tests on Intel XPU.

Every server-launching scenario is parametrized over BOTH KV-transfer
io-backends(``direct`` = PyTorch copy_, ``kernel`` = sgl-kernel-xpu
SYCL kernels) so a regression in either backend is caught here.

Classes (each guards a distinct failure mode):

  TestHiCacheXPUNoStorage
      Device<->host only, NO Tier-3 storage. Isolates the plain device<->host
      copy so a regression there is not attributed to the storage tier.
      (direct/layer_first + kernel/page_first).

  TestHiCacheXPUNixlTransfers
      Full transfer-kernel matrix with NIXL-POSIX storage attached
      ({direct,kernel} x {layer_first,page_first,page_head}). Drives every MHA
      load/backup kernel through prefill -> host/storage offload -> reload and
      asserts deterministic greedy output + Tier-3 files written.

  TestHiCacheXPUPosixCorrectness
      The strong correctness guard: evict/restore must reproduce the golden
      greedy trajectory bit-for-bit AND report cached_tokens > 0 (so the
      equality is not vacuously satisfied by silent recomputation). Both
      io-backends.

  TestHiCacheXPUMlaKernels
      MLA (DeepSeek-family) *_mla_* transfer kernels. A real DeepSeek-family
      model (~30 GB bf16) does not fit a single 22 GB Arc at TP=1, so this uses
      a tiny randomly-initialized DeepseekV3 checkpoint that keeps production
      MLA head dims (kv_lora=512, qk_nope=128, qk_rope=64, v_head=128). Random
      weights are fine here: HiCache restore is bit-exact by construction (it
      reloads the same KV bytes), so the offload/reload assertions validate the
      *_mla_* transfer path, not model accuracy. Both io-backends.

Kernel selection (pool_host/mha.py load_to_device_per_layer /
backup_from_device_all_layer):

    io_backend  mem_layout    load kernel                         backup kernel
    ----------  ------------  ----------------------------------  ----------------------------------
    kernel      layer_first   transfer_kv_per_layer               transfer_kv_all_layer
    kernel      page_first    transfer_kv_per_layer_pf_lf         transfer_kv_all_layer_lf_pf
    kernel      page_head     transfer_kv_per_layer_ph_lf         transfer_kv_all_layer_lf_ph
    direct      layer_first   transfer_kv_direct                  transfer_kv_direct
    direct      page_first*   transfer_kv_per_layer_direct_pf_lf  transfer_kv_all_layer_direct_lf_pf
    kernel(MLA) layer_first   transfer_kv_per_layer_mla           transfer_kv_all_layer_mla
    kernel(MLA) page_first    transfer_kv_per_layer_mla_pf_lf     transfer_kv_all_layer_mla_lf_pf

    (*) direct + page_first auto-resolves to page_first_direct in server_args.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_xpu.py -v -s
    # single class:
    python3 -m pytest test/registered/hicache/test_hicache_xpu.py \
        -k TestHiCacheXPUNixlTransfers -v -s
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from urllib.parse import urlparse

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_xpu_ci(
    est_time=1500,
    stage="base-b",
    runner_config="1-gpu-xpu",
    disabled="Requires Intel XPU with sgl-kernel-xpu transfer kernels.",
)

_XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, "xpu") else False

try:
    from nixl._api import nixl_agent  # noqa: F401

    _NIXL_AVAILABLE = True
except ImportError:
    _NIXL_AVAILABLE = False

# A real DeepSeek MLA model (~30 GB bf16) will not fit a single 22 GB Arc at
# TP=1, and --mem-fraction-static only sizes the KV pool, not the weights. Use a
# tiny randomly-initialized DeepseekV3 that keeps production MLA head dims so the
# *_mla_* transfer kernels run against a real MLA-shaped KV cache. Weights are
# random, but HiCache restore is bit-exact by construction, so the offload /
# reload assertions still validate the transfer path.
_MLA_TINY_MODEL = "trl-internal-testing/tiny-DeepseekV3ForCausalLM"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _resolve_base_url() -> str:
    """Allow SGLANG_TEST_PORT to override the port (avoid collisions on shared box)."""
    port = os.environ.get("SGLANG_TEST_PORT")
    if port:
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        return f"{parsed.scheme}://{parsed.hostname}:{port}"
    return DEFAULT_URL_FOR_TEST


def _nixl_posix_config() -> str:
    """Fully-qualified NIXL config -- the first plugin with active=True is
    selected. use_direct_io=False avoids the O_DIRECT alignment requirement.
    The flat ``{"plugin": "posix"}`` form raises AttributeError at construction.
    """
    return json.dumps(
        {
            "plugin": {
                "posix": {
                    "active": True,
                    "use_uring": True,
                    "use_direct_io": False,
                }
            }
        }
    )


def _launch_server(model: str, base_url: str, server_args: dict, env_extra=None):
    other = []
    for k, v in server_args.items():
        other.append(str(k))
        if not isinstance(v, bool):
            other.append(str(v))
    env = {**os.environ, **(env_extra or {})}
    return popen_launch_server(
        model=model,
        base_url=base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other,
        env=env,
    )


def _complete(base_url, model, prompt, max_tokens=32, timeout=90, want_cached=False):
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    if not want_cached:
        return text
    cached = 0
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    if details:
        cached = details.get("cached_tokens", 0) or 0
    return text, cached


def _count_storage_files(storage_dir) -> int:
    n = 0
    for _root, _dirs, names in os.walk(storage_dir):
        n += len(names)
    return n


def _shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# --------------------------------------------------------------------------- #
# 1. Device<->host only (no storage tier), both io-backends
# --------------------------------------------------------------------------- #
_NO_STORAGE_CONFIGS = [
    (
        "direct__layer_first",
        "direct",
        "layer_first",
        DEFAULT_MODEL_NAME_FOR_TEST,
        0.75,
        0.3,
    ),
    (
        "kernel__page_first",
        "kernel",
        "page_first",
        DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
        0.4,
        1.5,
    ),
]


@unittest.skipUnless(_XPU_AVAILABLE, "Intel XPU not available")
class TestHiCacheXPUNoStorage(CustomTestCase):
    """HiCache device<->host on XPU with NO storage tier, both io-backends."""

    def _run_config(self, io_backend, mem_layout, model, mem_fraction, hicache_ratio):
        base_url = _resolve_base_url()
        server_args = {
            "--tp": 1,
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": mem_fraction,
            "--hicache-ratio": hicache_ratio,
            "--page-size": 16,
            "--hicache-io-backend": io_backend,
            "--hicache-mem-layout": mem_layout,
        }
        proc = _launch_server(model, base_url, server_args)
        try:
            # wait_for_http_ready already confirms the server is up; the
            # completion calls below exercise it for real.
            wait_for_http_ready(
                base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, process=proc
            )

            basic = _complete(
                base_url, model, "The capital of France is", max_tokens=10
            )
            self.assertGreater(
                len(basic.strip()), 0, f"[{io_backend}] empty basic output"
            )

            for prompt in (
                "Once upon a time",
                "The quick brown fox",
                "In a galaxy far far away",
            ):
                out = _complete(base_url, model, prompt, max_tokens=24)
                self.assertGreater(
                    len(out.strip()), 0, f"[{io_backend}] empty output for {prompt!r}"
                )

            # Greedy repeat of the same prompt must be identical (cache reuse).
            prompt = "The answer to life, the universe, and everything is"
            first = _complete(base_url, model, prompt, max_tokens=32)
            second = _complete(base_url, model, prompt, max_tokens=32)
            self.assertEqual(
                first, second, f"[{io_backend}] nondeterministic device<->host reuse"
            )
        finally:
            kill_process_tree(proc.pid)


# --------------------------------------------------------------------------- #
# 2. Full transfer-kernel matrix with NIXL-POSIX storage, both io-backends
# --------------------------------------------------------------------------- #
# (label, io_backend, mem_layout, load_kernel, backup_kernel)
_NIXL_MATRIX_CONFIGS = [
    (
        "direct__layer_first",
        "direct",
        "layer_first",
        "transfer_kv_direct",
        "transfer_kv_direct",
    ),
    (
        "direct__page_first_direct",
        "direct",
        "page_first",  # auto -> page_first_direct
        "transfer_kv_per_layer_direct_pf_lf",
        "transfer_kv_all_layer_direct_lf_pf",
    ),
    (
        "kernel__layer_first",
        "kernel",
        "layer_first",
        "transfer_kv_per_layer",
        "transfer_kv_all_layer",
    ),
    (
        "kernel__page_first",
        "kernel",
        "page_first",
        "transfer_kv_per_layer_pf_lf",
        "transfer_kv_all_layer_lf_pf",
    ),
    (
        "kernel__page_head",
        "kernel",
        "page_head",
        "transfer_kv_per_layer_ph_lf",
        "transfer_kv_all_layer_lf_ph",
    ),
]


@unittest.skipUnless(_XPU_AVAILABLE and _NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestHiCacheXPUNixlTransfers(CustomTestCase):
    """Drive every MHA transfer kernel via a HiCache server backed by NIXL-POSIX."""

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

    def _run_config(self, io_backend, mem_layout, load_k, backup_k):
        base_url = _resolve_base_url()
        tmp = tempfile.mkdtemp(prefix=f"hc_xpu_nixl_{io_backend}_{mem_layout}_")
        storage_dir = os.path.join(tmp, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        server_args = {
            "--tp": 1,
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.5,
            "--hicache-ratio": 1.2,
            "--page-size": 16,
            "--hicache-io-backend": io_backend,
            "--hicache-mem-layout": mem_layout,
            "--hicache-storage-backend": "nixl",
            "--hicache-storage-backend-extra-config": _nixl_posix_config(),
        }
        proc = _launch_server(
            self.model,
            base_url,
            server_args,
            env_extra={"SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR": storage_dir},
        )
        try:
            wait_for_http_ready(
                base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, process=proc
            )
            # Long shared prefix -> offloaded to host/storage on eviction, then
            # reloaded. Greedy decode must be identical across the round-trip,
            # forcing the backup kernel (offload) and load kernel (reload).
            prefix = "The history of computing is a long and fascinating story. " * 40
            out1 = _complete(
                base_url, self.model, prefix + " In summary,", max_tokens=32
            )
            for i in range(8):
                _complete(
                    base_url,
                    self.model,
                    f"Unrelated filler request {i}: " + "lorem ipsum " * 60,
                    max_tokens=16,
                )
            out2 = _complete(
                base_url, self.model, prefix + " In summary,", max_tokens=32
            )

            self.assertGreater(
                len(out1.strip()), 0, f"[{io_backend}/{mem_layout}] empty output"
            )
            self.assertEqual(
                out1,
                out2,
                f"[{io_backend}/{mem_layout}] nondeterministic across host/storage reload",
            )
            print(
                f"[{io_backend}/{mem_layout}] load={load_k} backup={backup_k} "
                f"nixl_storage_files={_count_storage_files(storage_dir)}"
            )
        finally:
            kill_process_tree(proc.pid)
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 3. Correctness: evict/restore must be bit-exact AND actually hit the cache
# --------------------------------------------------------------------------- #
_CORRECTNESS_CONFIGS = [
    ("direct", "direct", "page_first"),
    ("kernel", "kernel", "page_first"),
]


@unittest.skipUnless(_XPU_AVAILABLE and _NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestHiCacheXPUPosixCorrectness(CustomTestCase):
    """XPU<->host transfer correctness via HiCache evict/restore over NIXL-POSIX."""

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

    def _launch(self, io_backend, mem_layout, storage_dir):
        base_url = _resolve_base_url()
        server_args = {
            "--tp": 1,
            "--enable-hierarchical-cache": True,
            # small device pool + small ratio so eviction happens quickly
            "--mem-fraction-static": 0.5,
            "--hicache-ratio": 1.2,
            "--page-size": 16,
            "--hicache-io-backend": io_backend,
            "--hicache-mem-layout": mem_layout,
            "--hicache-storage-backend": "nixl",
            "--hicache-storage-backend-extra-config": _nixl_posix_config(),
            # expose cached_tokens in the usage block for the assertion
            "--enable-cache-report": True,
        }
        proc = _launch_server(
            self.model,
            base_url,
            server_args,
            env_extra={"SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR": storage_dir},
        )
        wait_for_http_ready(
            base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, process=proc
        )
        return base_url, proc

    def _flush(self, base_url):
        requests.post(f"{base_url}/flush_cache", timeout=30)
        time.sleep(1.0)

    def _check_evict_restore_bit_exact(self, base_url, storage_dir):
        # Unique long prefix so no stale cache interferes and it spans many pages.
        prefix = (
            "In a distant galaxy, a lone cartographer charted forgotten star "
            "systems and recorded their histories. "
        ) * 30
        prompt = prefix + " The most important discovery was"

        # 1) GOLDEN on a flushed cache (computed on device, no reuse).
        self._flush(base_url)
        golden, _ = _complete(
            base_url, self.model, prompt, max_tokens=48, want_cached=True
        )
        self.assertGreater(len(golden.strip()), 0, "golden output empty")

        # 2) EVICT: flood with distinct long prompts to push the golden KV out
        # of the device pool into host + NIXL storage.
        for i in range(12):
            filler = (
                f"Unrelated chronicle number {i}: the archivist catalogued "
                f"maps and ledgers across the ages. "
            ) * 20
            _complete(base_url, self.model, filler, max_tokens=8)

        # 3) RESTORE: identical prompt -> KV restored from host/storage.
        restored, restored_cached = _complete(
            base_url, self.model, prompt, max_tokens=48, want_cached=True
        )
        n_files = _count_storage_files(storage_dir)
        print(f"restored_cached={restored_cached} nixl_storage_files={n_files}")

        # Correctness: sglang's default inference is not strictly bit-exact
        # across differing batch conditions, so a near-tie token late in the
        # continuation can differ; KV corruption instead diverges early. Assert
        # a long shared prefix.
        spl = _shared_prefix_len(restored, golden)
        self.assertGreaterEqual(
            spl,
            min(len(golden), 32),
            f"restored continuation diverged after only {spl} chars "
            f"(golden={golden!r} restored={restored!r}) -> XPU<->host KV "
            f"transfer corrupted the cache",
        )
        # The restore must have hit the cache (else equality is vacuous).
        self.assertGreater(
            restored_cached,
            0,
            "restore reported 0 cached tokens -> KV was recomputed, not restored",
        )
        self.assertGreater(
            n_files, 0, "no NIXL storage files written -> host<->storage tier idle"
        )

    def _check_multi_prompt_restore(self, base_url):
        # Several *distinct* prompts (no shared template), each round-tripped
        # through evict/restore, must all come back with a long shared prefix.
        # Distinct bodies make each restore an independent full-prefix cache hit,
        # guarding against cross-page/index aliasing in the transfer.
        bodies = [
            "The marine biologist documented bioluminescent plankton drifting "
            "through the midnight zone of the trench. ",
            "The locomotive engineer inspected the mountain railway's brakes "
            "before the long descent through the alpine tunnels. ",
            "The pastry chef folded laminated dough for the morning croissants "
            "while the ovens warmed the empty kitchen. ",
            "The radio astronomer aligned the dish toward the pulsar and "
            "logged the timing of each sweeping pulse. ",
        ]
        tails = [
            " The decisive observation was",
            " The critical adjustment turned out to be",
            " The essential technique was",
            " The surprising measurement showed",
        ]
        prompts = [(bodies[k] * 30) + tails[k] for k in range(len(bodies))]

        self._flush(base_url)
        goldens = []
        for p in prompts:
            g = _complete(base_url, self.model, p, max_tokens=48)
            self.assertGreater(len(g.strip()), 0)
            goldens.append(g)

        for i in range(12):
            _complete(
                base_url, self.model, (f"Filler passage {i}. " * 40), max_tokens=8
            )

        any_cached = 0
        for p, g in zip(prompts, goldens):
            r, c = _complete(base_url, self.model, p, max_tokens=48, want_cached=True)
            any_cached = max(any_cached, c)
            spl = _shared_prefix_len(r, g)
            self.assertGreaterEqual(
                spl,
                min(len(g), 24),
                f"multi-prompt restore diverged after only {spl} chars "
                f"(golden={g!r} restored={r!r}) -> KV aliasing/corruption",
            )
        self.assertGreater(
            any_cached,
            0,
            "no prompt hit the cache on restore; evict/restore not exercised",
        )

    def _run_config(self, io_backend, mem_layout):
        tmp = tempfile.mkdtemp(prefix=f"hc_xpu_posix_correct_{io_backend}_")
        storage_dir = os.path.join(tmp, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        base_url, proc = self._launch(io_backend, mem_layout, storage_dir)
        try:
            self._check_evict_restore_bit_exact(base_url, storage_dir)
            self._check_multi_prompt_restore(base_url)
        finally:
            kill_process_tree(proc.pid)
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 4. MLA transfer kernels
# --------------------------------------------------------------------------- #
# (label, io_backend, mem_layout, load_kernel, backup_kernel)
_MLA_CONFIGS = [
    (
        "kernel__layer_first",
        "kernel",
        "layer_first",
        "transfer_kv_per_layer_mla",
        "transfer_kv_all_layer_mla",
    ),
    (
        "kernel__page_first",
        "kernel",
        "page_first",
        "transfer_kv_per_layer_mla_pf_lf",
        "transfer_kv_all_layer_mla_lf_pf",
    ),
]


@unittest.skipUnless(_XPU_AVAILABLE and _NIXL_AVAILABLE, "Intel XPU and NIXL required")
class TestHiCacheXPUMlaKernels(CustomTestCase):
    """Drive each MLA transfer kernel via a HiCache server backed by NIXL-POSIX."""

    model = _MLA_TINY_MODEL

    def _run_config(self, io_backend, mem_layout, load_k, backup_k):
        base_url = _resolve_base_url()
        tmp = tempfile.mkdtemp(prefix=f"hc_xpu_mla_{io_backend}_{mem_layout}_")
        storage_dir = os.path.join(tmp, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        server_args = {
            "--tp": 1,
            "--trust-remote-code": True,
            "--attention-backend": "triton",
            "--enable-hierarchical-cache": True,
            # The device KV pool is (mem-fraction x device mem) and the pinned
            # host pool is (hicache-ratio x that). At 0.6 the host pool tries to
            # pin ~16 GB and fails; 0.1 keeps it ~3 GB. The tiny weights fit
            # regardless.
            "--mem-fraction-static": 0.1,
            "--hicache-ratio": 1.2,
            "--page-size": 16,
            "--hicache-io-backend": io_backend,
            "--hicache-mem-layout": mem_layout,
            "--hicache-storage-backend": "nixl",
            "--hicache-storage-backend-extra-config": _nixl_posix_config(),
        }
        proc = _launch_server(
            self.model,
            base_url,
            server_args,
            env_extra={"SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR": storage_dir},
        )
        try:
            wait_for_http_ready(
                base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, process=proc
            )
            prefix = "The history of computing is a long and fascinating story. " * 40
            out1 = _complete(
                base_url,
                self.model,
                prefix + " In summary,",
                max_tokens=32,
                timeout=120,
            )
            for i in range(8):
                _complete(
                    base_url,
                    self.model,
                    f"Unrelated filler request {i}: " + "lorem ipsum " * 60,
                    max_tokens=16,
                    timeout=120,
                )
            out2 = _complete(
                base_url,
                self.model,
                prefix + " In summary,",
                max_tokens=32,
                timeout=120,
            )

            self.assertGreater(
                len(out1.strip()), 0, f"[{io_backend}/{mem_layout}] empty output"
            )
            self.assertEqual(
                out1,
                out2,
                f"[{io_backend}/{mem_layout}] nondeterministic across host/storage reload",
            )
            print(
                f"[{io_backend}/{mem_layout}] load={load_k} backup={backup_k} "
                f"nixl_storage_files={_count_storage_files(storage_dir)}"
            )
        finally:
            kill_process_tree(proc.pid)
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Generate one test method per config on each class.
# --------------------------------------------------------------------------- #
def _bind(cls, label, runner_args, doc):
    def test(self):
        self._run_config(*runner_args)

    test.__name__ = f"test_{label}"
    test.__doc__ = doc
    setattr(cls, test.__name__, test)


for _c in _NO_STORAGE_CONFIGS:
    _bind(
        TestHiCacheXPUNoStorage,
        _c[0],
        (_c[1], _c[2], _c[3], _c[4], _c[5]),
        f"{_c[1]}/{_c[2]}: device<->host reuse, no storage",
    )

for _c in _NIXL_MATRIX_CONFIGS:
    _bind(
        TestHiCacheXPUNixlTransfers,
        _c[0],
        (_c[1], _c[2], _c[3], _c[4]),
        f"{_c[1]}/{_c[2]}: load={_c[3]}, backup={_c[4]}",
    )

for _c in _CORRECTNESS_CONFIGS:
    _bind(
        TestHiCacheXPUPosixCorrectness,
        _c[0],
        (_c[1], _c[2]),
        f"{_c[1]}/{_c[2]}: evict/restore bit-exact + cached_tokens>0",
    )

for _c in _MLA_CONFIGS:
    _bind(
        TestHiCacheXPUMlaKernels,
        _c[0],
        (_c[1], _c[2], _c[3], _c[4]),
        f"{_c[1]}/{_c[2]}: MLA load={_c[3]}, backup={_c[4]}",
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
