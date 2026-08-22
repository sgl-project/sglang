# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Core LoRA serving features: accuracy against HF+PEFT, multi-adapter batching,
the drainer, runtime load/unload, pinning, pool eviction, the radix cache, and
embedding models.

Adapters are synthetic with known-distinct weights, so most cases assert
nearest-reference routing rather than exact output.

The cases are in two groups: each feature in isolation, then feature
*combinations* with graph capture OFF (the "Feature combinations" section at the
bottom). Device-graph capture and tensor parallelism -- including the graph-ON
combinations -- live in test_lora_comb_xpu.py.
"""

import atexit
import multiprocessing as mp
import os
import shutil
import tempfile
import unittest
from typing import List, Optional

# XPU cannot be re-initialized in a forked child and SRTRunner spawns one, so the
# start method is forced here (pytest never runs the __main__ block below).
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

import torch

from sglang.test.lora_utils import BACKENDS
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    CustomTestCase,
    calculate_rouge_l,
    empty_gpu_cache,
)

DENSE_BACKENDS = [*BACKENDS, "torch_native"]

MEM_FRACTION_STATIC = 0.80
MAX_NEW_TOKENS = 16
LOGPROB_THRESHOLD = 2e-1

TEST_PROMPTS = [
    "AI is a field of computer science focused on",
    "The capital of France is",
]

# > 0 enables the scheduler's LoRADrainer, which frees slots for starving
# adapters once max_loras_per_batch fills. Must not change outputs.
LORA_DRAIN_WAIT_THRESHOLD = 0.1


# ==============================================================================
# Synthetic LoRA adapter fixtures
#
# Shared by the tests that need adapters with known-distinct weights rather than
# a published checkpoint: routing assertions have to tell adapters apart, which
# real adapters trained on the same task do not reliably allow.
# ==============================================================================

SYNTH_BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# q_proj/v_proj fuse into the column-parallel qkv_proj, which is what makes the
# TP cases a real test of sharding (LoRA A replicated, LoRA B sliced).
SYNTH_TARGET_MODULES = ["q_proj", "v_proj"]

SYNTH_LORA_RANK = 8
SYNTH_MAX_LORA_RANK = 16  # engines started without adapters must declare a max

# Calibrated: 0.02 shifts logprobs (enough to compare against HF) but leaves the
# argmax unchanged, so multi-adapter routing tests need 0.04 for distinct tokens.
SINGLE_LORA_B_STD = 0.02
MULTI_LORA_B_STD = 0.04

_temp_dirs = []


def lora_temp_dir(prefix: str) -> str:
    """Temp dir removed at interpreter exit. Fixtures are cached across test
    classes, so cleanup cannot live in tearDownClass."""
    path = tempfile.mkdtemp(prefix=prefix)
    _temp_dirs.append(path)
    return path


@atexit.register
def _cleanup_lora_temp_dirs():
    for path in _temp_dirs:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


def create_attention_lora_adapter(
    base_model_name: str,
    output_dir: str,
    seed: int = 0,
    lora_b_std: float = SINGLE_LORA_B_STD,
):
    """
    Create a randomly-initialized (untrained) LoRA adapter on the attention
    projections. PEFT zeroes lora_B, which would make the adapter a no-op, so
    lora_B gets non-zero weights and the adapter has a verifiable effect.

    Both lora_A and lora_B are seeded from one generator: PEFT draws lora_A from
    the *global* RNG, so seeding only lora_B leaves B@A varying run-to-run, which
    can make an adapter too weak to move any greedy token (reading as identical
    to base). Seeding both removes that flakiness at the source.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=SYNTH_LORA_RANK,
            lora_alpha=2 * SYNTH_LORA_RANK,
            target_modules=SYNTH_TARGET_MODULES,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name:
                param.normal_(
                    mean=0.0, std=1.0 / param.shape[1] ** 0.5, generator=generator
                )
            elif "lora_B" in name:
                param.normal_(mean=0.0, std=lora_b_std, generator=generator)

    peft_model.save_pretrained(output_dir)

    # A silently empty adapter would make every routing assertion vacuous.
    from safetensors import safe_open

    keys = safe_open(
        os.path.join(output_dir, "adapter_model.safetensors"), framework="pt"
    ).keys()
    assert any(
        t in k for k in keys for t in SYNTH_TARGET_MODULES
    ), f"Expected {SYNTH_TARGET_MODULES} LoRA weights in adapter, got: {sorted(keys)}"

    del peft_model, model
    empty_gpu_cache()


# Adapters are referred to by identity, not list index, so one cached adapter (or
# isolated-reference run) serves classes with different slot layouts.
ADAPTER_A = "A"
ADAPTER_B = "B"
ADAPTER_C = "C"
BASE_SLOT = None  # "no adapter" -- the base model
ADAPTER_SEEDS = {ADAPTER_A: 1, ADAPTER_B: 2, ADAPTER_C: 3}

_adapter_dirs = {}


def synth_adapter_path(
    identity: str,
    base_model: str = SYNTH_BASE_MODEL,
    lora_b_std: float = MULTI_LORA_B_STD,
) -> str:
    """Shared adapter for ``identity``, built on first use and cached."""
    key = (identity, base_model, lora_b_std)
    if key not in _adapter_dirs:
        path = lora_temp_dir(f"sglang_test_lora_{identity}_{lora_b_std}_")
        create_attention_lora_adapter(
            base_model, path, seed=ADAPTER_SEEDS[identity], lora_b_std=lora_b_std
        )
        _adapter_dirs[key] = path
    return _adapter_dirs[key]


def synth_slot_paths(slots, base_model: str = SYNTH_BASE_MODEL, **kwargs):
    """Adapter path per slot, ``None`` for the base-model slot."""
    return [
        None if s is BASE_SLOT else synth_adapter_path(s, base_model, **kwargs)
        for s in slots
    ]


def assert_nearest_reference_routing(
    test_case,
    *,
    batched: List[str],
    reference: dict,
    slots: List[Optional[str]],
    context: str = "",
):
    """
    Assert each batched output is closer (ROUGE-L) to its own adapter's isolated
    output than to any other adapter's.

    Exact reproduction is not required: batching changes kernel tiling, so tokens
    can differ slightly from a bs=1 run even when routing is correct. Nearest-
    reference is the strongest assertion that stays robust to that -- a swapped or
    dropped adapter moves the argmax to another slot, which this catches.

    The base slot is excluded from the reference set: base output is a prefix-like
    attractor that every adapter's output stays near, so including it would make
    the argmax uninformative.
    """
    test_case.assertEqual(
        len(batched),
        len(slots),
        f"{context} expected {len(slots)} outputs, got {len(batched)}",
    )

    adapter_slots = [i for i, s in enumerate(slots) if s is not BASE_SLOT]
    for i in adapter_slots:
        sims = {
            j: calculate_rouge_l([batched[i]], [reference[slots[j]]])[0]
            for j in adapter_slots
        }
        best = max(adapter_slots, key=lambda j: sims[j])
        print(
            f"{context} slot {i} (adapter={slots[i]}) self={sims[i]:.4f} "
            f"argmax=slot{best}"
        )
        test_case.assertEqual(
            best,
            i,
            f"{context} slot {i} (adapter={slots[i]}): batched output is closer to "
            f"another adapter's isolated output than to its own "
            f"(sims={ {j: round(sims[j], 3) for j in adapter_slots} }); "
            f"wrong adapter routed?",
        )

    # Distinct adapters on an identical prompt must not collapse.
    test_case.assertFalse(
        all(s == batched[0] for s in batched),
        f"{context} all outputs identical despite distinct adapters {slots}; "
        f"routing had no effect.",
    )


def lora_runner(
    *,
    backend: str = "triton",
    disable_radix: bool = True,
    model_type: str = "generation",
    **extra,
) -> SRTRunner:
    """SRTRunner with the kwargs shared by every case here. Graph capture is off
    throughout, for the combination cases as well as the isolated ones, so that a
    failure is never ambiguous between the feature and graph capture; the graph-ON
    interactions are covered in test_lora_comb_xpu.py."""
    return SRTRunner(
        SYNTH_BASE_MODEL,
        torch_dtype=torch.float16,
        model_type=model_type,
        lora_backend=backend,
        lora_target_modules=SYNTH_TARGET_MODULES,
        disable_cuda_graph=True,
        disable_radix_cache=disable_radix,
        mem_fraction_static=MEM_FRACTION_STATIC,
        port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        **extra,
    )


def generate(runner: SRTRunner, lora_paths, batched: bool = False):
    """Serve one request per entry of ``lora_paths`` on a shared prompt, returning
    the stripped outputs. ``batched`` forces all of them into one physical batch
    (sleep_on_idle on the runner keeps the scheduler from dribbling them out)."""
    forward = runner.batch_forward if batched else runner.forward
    outputs = forward(
        [TEST_PROMPTS[0]] * len(lora_paths),
        max_new_tokens=MAX_NEW_TOKENS,
        lora_paths=list(lora_paths),
    )
    return [s.strip() for s in outputs.output_strs]


_references = {}


def isolated_reference(slots):
    """
    Per-adapter baseline for the routing check, keyed by adapter identity so one
    run serves classes with different slot layouts. forward() runs each request as
    its own batch, with enough slots that nothing splits or drains.

    The backend is pinned to triton deliberately: the reference is the fixed
    fingerprint every backend is matched against, so a mis-routing backend cannot
    hide by shifting its own reference alongside its output.
    """
    key = tuple(slots)
    if key in _references:
        return _references[key]

    paths = synth_slot_paths(slots)
    adapters = [p for p in paths if p is not None]
    with lora_runner(
        lora_paths=adapters, max_loras_per_batch=len(adapters) + 1
    ) as runner:
        outputs = generate(runner, paths)
    empty_gpu_cache()

    _references[key] = {s: outputs[i] for i, s in enumerate(slots)}
    return _references[key]


# ==============================================================================
# Accuracy vs HuggingFace + PEFT
# ==============================================================================


class TestLoRAAccuracy(CustomTestCase):
    """
    A basic attention LoRA must match HF+PEFT on every dense backend. The HF
    reference is backend-independent, so it is computed once. This is the only
    *absolute* numerical check in the file; everything else asserts relative
    properties (routing, round-trip, determinism).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lora_path = synth_adapter_path(ADAPTER_A, lora_b_std=SINGLE_LORA_B_STD)
        with HFRunner(
            SYNTH_BASE_MODEL, torch_dtype=torch.float16, model_type="generation"
        ) as hf_runner:
            cls.hf_outputs = hf_runner.forward(
                TEST_PROMPTS,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[cls.lora_path] * len(TEST_PROMPTS),
            )
        empty_gpu_cache()

    def _check_backend(self, backend: str):
        with lora_runner(
            backend=backend, lora_paths=[self.lora_path], max_loras_per_batch=1
        ) as runner:
            srt_outputs = runner.forward(
                TEST_PROMPTS,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[self.lora_path] * len(TEST_PROMPTS),
            )
        empty_gpu_cache()

        for phase, srt, hf in (
            (
                "prefill",
                srt_outputs.top_input_logprobs,
                self.hf_outputs.top_input_logprobs,
            ),
            (
                "decode",
                srt_outputs.top_output_logprobs,
                self.hf_outputs.top_output_logprobs,
            ),
        ):
            for i in range(len(TEST_PROMPTS)):
                max_diff = torch.max(
                    torch.abs(torch.tensor(srt[i]) - torch.tensor(hf[i]))
                ).item()
                print(
                    f"[{backend}] prompt {i} {phase} logprob max_diff "
                    f"(SGLang vs HF): {max_diff:.6e}"
                )
                self.assertLess(
                    max_diff,
                    LOGPROB_THRESHOLD,
                    f"[{backend}] prompt {i}: {phase} logprob diff {max_diff:.6e} "
                    f"exceeds threshold {LOGPROB_THRESHOLD:.0e}",
                )

    def test_lora_matches_hf(self):
        for backend in DENSE_BACKENDS:
            with self.subTest(backend=backend):
                self._check_backend(backend)


# ==============================================================================
# Multi-LoRA routing: one physical batch, every slot on its own adapter
# ==============================================================================


class _RoutingTestCase(CustomTestCase):
    """
    Base for the multi-adapter routing tests. Subclasses declare the slot layout
    and any extra server args; the batched run and its reference share the rest,
    and the assertion lives here so the two halves cannot drift apart.

    max_loras_per_batch below the slot count is how the drainer case forces a
    running adapter out for a starving one.
    """

    SLOTS = [ADAPTER_A, ADAPTER_B, BASE_SLOT]
    MAX_LORAS_PER_BATCH = 3
    BACKENDS = BACKENDS
    RUNNER_ARGS = {}
    DISABLE_RADIX = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.paths = synth_slot_paths(cls.SLOTS)
        cls.reference = isolated_reference(cls.SLOTS)

    def _check_backend(self, backend: str):
        adapters = [p for p in self.paths if p is not None]
        with lora_runner(
            backend=backend,
            disable_radix=self.DISABLE_RADIX,
            lora_paths=adapters,
            max_loras_per_batch=self.MAX_LORAS_PER_BATCH,
            sleep_on_idle=True,
            **self.RUNNER_ARGS,
        ) as runner:
            batched = generate(runner, self.paths, batched=True)
        empty_gpu_cache()

        assert_nearest_reference_routing(
            self,
            batched=batched,
            reference=self.reference,
            slots=self.SLOTS,
            context=f"[{backend}]",
        )

    def _check_routing(self):
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._check_backend(backend)


class TestMultiLoRABatching(_RoutingTestCase):
    """Multi-LoRA batching (S-LoRA/Punica): two distinct adapters plus base in one
    physical batch, each request routed to its own adapter. Three slots for three
    requests, so nothing splits or drains."""

    BACKENDS = DENSE_BACKENDS

    def test_routing(self):
        self._check_routing()


class TestLoRADrainWaitThreshold(_RoutingTestCase):
    """Drainer (lora_drain_wait_threshold > 0): three distinct adapters through
    two slots forces the scheduler to drain a running adapter for a starving one.
    Draining is scheduling fairness, not correctness -- it changes *when* a request
    runs, never which adapter it uses -- so routing must be unaffected."""

    SLOTS = [ADAPTER_A, ADAPTER_B, ADAPTER_C]  # no base: 3 through 2 slots
    MAX_LORAS_PER_BATCH = 2
    RUNNER_ARGS = {"lora_drain_wait_threshold": LORA_DRAIN_WAIT_THRESHOLD}

    def test_routing(self):
        self._check_routing()


class TestLoRAOverlapLoading(_RoutingTestCase):
    """Overlap loading end-to-end: weights are staged on a side device stream
    concurrently with compute. It changes only how/when weights are staged, never
    the numerics, so the routing check applies unchanged.

    max_loaded_loras is required with overlap loading, within
    [max_loras_per_batch, 2x]."""

    RUNNER_ARGS = {"enable_lora_overlap_loading": True, "max_loaded_loras": 6}

    def test_routing(self):
        self._check_routing()


class TestLoRARadixCacheRouting(_RoutingTestCase):
    """Radix prefix cache ENABLED (every other case here disables it). Radix keys
    prefixes by adapter via extra_key, so the same prompt under two adapters lands
    in disjoint namespaces. All slots share one prompt -- the worst case for
    cross-contamination; if radix ever shared KV across adapters, a slot would look
    closest to another adapter's output."""

    DISABLE_RADIX = False

    def test_routing(self):
        self._check_routing()


# ==============================================================================
# Radix cache hit
# ==============================================================================


class TestLoRARadixCacheHit(CustomTestCase):
    """
    Serving the same (adapter, prompt) twice in one engine hits the first
    request's inserted prefix; with greedy decoding a correct hit reuses the exact
    KV, so equality is exact and a mismatch means corrupted or wrong-adapter KV.
    Serving a different adapter on the same prompt probes the other direction: the
    cache must not collapse distinct adapters onto one entry.
    """

    def test_radix_cache_hit_reproduces_and_keys_on_adapter(self):
        paths = synth_slot_paths([ADAPTER_A, ADAPTER_B])
        with lora_runner(
            disable_radix=False, lora_paths=paths, max_loras_per_batch=2
        ) as runner:
            (a_first,) = generate(runner, paths[:1])
            (a_second,) = generate(runner, paths[:1])  # cache hit
            (b,) = generate(runner, paths[1:])
        empty_gpu_cache()

        print(f"[radix-hit] A first={a_first!r} A second={a_second!r} B={b!r}")
        self.assertEqual(
            a_second,
            a_first,
            "radix cache hit did not reproduce adapter A's output (the prefix "
            "cache may have corrupted or dropped the adapter KV)",
        )
        self.assertNotEqual(
            b,
            a_first,
            "adapter B produced adapter A's cached output on the same prompt "
            "(the radix cache is not keying on the LoRA adapter)",
        )


# ==============================================================================
# Runtime adapter mutation
# ==============================================================================


class TestDynamicLoRAUpdate(CustomTestCase):
    """
    Dynamic load/unload (/load_lora_adapter, /unload_lora_adapter) on an engine
    that starts with NO adapters: load two at runtime, serve them by name, unload
    one and reload it. load/unload return a LoRAUpdateOutput and never raise; only
    a forward referencing an unknown name raises, and the probe uses a NEVER
    loaded name since an unloaded one would be transparently reloaded.
    """

    def _check_load_unload_reload(self, backend: str):
        with lora_runner(
            backend=backend,
            # No initial adapters -> enable_lora + max_lora_rank +
            # lora_target_modules are all required.
            lora_paths=None,
            enable_lora=True,
            max_lora_rank=SYNTH_MAX_LORA_RANK,
            max_loras_per_batch=2,
            max_loaded_loras=2,
        ) as runner:
            load_a = runner.load_lora_adapter(
                lora_name="adapter_a", lora_path=synth_adapter_path(ADAPTER_A)
            )
            load_b = runner.load_lora_adapter(
                lora_name="adapter_b", lora_path=synth_adapter_path(ADAPTER_B)
            )
            self.assertTrue(load_a.success, f"load_a: {load_a.error_message}")
            self.assertTrue(load_b.success, f"load_b: {load_b.error_message}")
            self.assertEqual(
                set(load_b.loaded_adapters or []),
                {"adapter_a", "adapter_b"},
                f"unexpected loaded set: {load_b.loaded_adapters}",
            )

            base, out_a, out_b = generate(runner, [None, "adapter_a", "adapter_b"])
            print(f"[{backend}] base={base!r} a={out_a!r} b={out_b!r}")
            self.assertNotEqual(out_a, base, "adapter_a matched base")
            self.assertNotEqual(out_b, base, "adapter_b matched base")
            self.assertNotEqual(out_a, out_b, "both adapters identical")

            with self.assertRaises(
                ValueError, msg="a never-loaded name must raise on forward"
            ):
                generate(runner, ["adapter_never_loaded"])

            unload = runner.unload_lora_adapter(lora_name="adapter_a")
            self.assertTrue(unload.success, f"unload: {unload.error_message}")
            self.assertNotIn(
                "adapter_a",
                set(unload.loaded_adapters or []),
                f"adapter_a still loaded: {unload.loaded_adapters}",
            )
            (b_after_unload,) = generate(runner, ["adapter_b"])
            self.assertEqual(
                b_after_unload, out_b, "adapter_b changed after unloading adapter_a"
            )

            # Reload explicitly, so the round-trip tested is the load path rather
            # than the forward-time auto-reload fallback.
            reload_a = runner.load_lora_adapter(
                lora_name="adapter_a", lora_path=synth_adapter_path(ADAPTER_A)
            )
            self.assertTrue(reload_a.success, f"reload: {reload_a.error_message}")
            (a_reloaded,) = generate(runner, ["adapter_a"])
            self.assertEqual(
                a_reloaded, out_a, "adapter_a changed after unload + reload"
            )
        empty_gpu_cache()

    def test_dynamic_load_unload_reload(self):
        self._check_load_unload_reload("triton")

    def test_dynamic_load_unload_reload_torch_native(self):
        self._check_load_unload_reload("torch_native")


class TestPinnedLoRAAdapters(CustomTestCase):
    """
    Pinned adapters (load_lora_adapter(pinned=True)) are excluded from eviction
    victim selection, so churning the single free slot must never evict one. Also
    probes the pin-all-slots guard: with 2 slots at most 1 may be pinned, and the
    rejection is success=False, not an exception.
    """

    def test_pinned_adapter_loads_and_survives_eviction(self):
        with lora_runner(
            lora_paths=None,
            enable_lora=True,
            max_lora_rank=SYNTH_MAX_LORA_RANK,
            max_loras_per_batch=2,
            # Roomy so eviction is driven by the per-batch pool, not this cap.
            max_loaded_loras=4,
        ) as runner:
            pinned = runner.load_lora_adapter(
                lora_name="pinned_a",
                lora_path=synth_adapter_path(ADAPTER_A),
                pinned=True,
            )
            unpinned = runner.load_lora_adapter(
                lora_name="unpinned_b", lora_path=synth_adapter_path(ADAPTER_B)
            )
            self.assertTrue(pinned.success, f"pinned load: {pinned.error_message}")
            self.assertTrue(
                unpinned.success, f"unpinned load: {unpinned.error_message}"
            )

            over_pin = runner.load_lora_adapter(
                lora_name="pinned_c",
                lora_path=synth_adapter_path(ADAPTER_C),
                pinned=True,
            )
            self.assertFalse(
                over_pin.success,
                "pinning a second adapter should be rejected (would pin all slots)",
            )
            self.assertIn(
                "pin all slots",
                over_pin.error_message or "",
                f"unexpected over-pinning error: {over_pin.error_message!r}",
            )

            (baseline,) = generate(runner, ["pinned_a"])
            for _ in range(3):
                # pinned_c was rejected by the guard, so churn with unpinned_b.
                generate(runner, ["unpinned_b"])
            (after_churn,) = generate(runner, ["pinned_a"])
        empty_gpu_cache()

        print(f"[pinning] baseline={baseline!r} after churn={after_churn!r}")
        self.assertEqual(
            after_churn,
            baseline,
            "pinned adapter output changed after serving other adapters (it may "
            "have been evicted despite being pinned)",
        )


class TestLoRAEvictionEndToEnd(CustomTestCase):
    """
    End-to-end pool eviction with one batch slot: serving a second adapter forces
    the first out, and serving it again reloads it (mem_pool.py
    select_victim/evict/reload). Asserts correctness under eviction rather than
    which adapter was evicted -- output is policy-invariant, so victim identity is
    test_lora_eviction_policy.py's job.
    """

    def _check_eviction(self, *, policy: str, backend: str = "triton"):
        label = f"{policy},{backend}"
        paths = synth_slot_paths([ADAPTER_A, ADAPTER_B])
        with lora_runner(
            backend=backend,
            lora_paths=paths,
            max_loras_per_batch=1,
            lora_eviction_policy=policy,
        ) as runner:
            (a_first,) = generate(runner, paths[:1])
            (b_first,) = generate(runner, paths[1:])  # evicts A
            (a_again,) = generate(runner, paths[:1])  # must reload A
            (b_again,) = generate(runner, paths[1:])  # must reload B
        empty_gpu_cache()

        print(f"[evict:{label}] A={a_first!r} B={b_first!r}")
        self.assertNotEqual(
            a_first, b_first, f"[{label}] adapters A and B produced identical output"
        )
        self.assertEqual(
            a_again, a_first, f"[{label}] adapter A changed after eviction + reload"
        )
        self.assertEqual(
            b_again, b_first, f"[{label}] adapter B changed after eviction + reload"
        )

    def test_eviction_lru(self):
        self._check_eviction(policy="lru")

    def test_eviction_fifo(self):
        self._check_eviction(policy="fifo")

    def test_eviction_lru_torch_native(self):
        self._check_eviction(policy="lru", backend="torch_native")

    def test_eviction_fifo_torch_native(self):
        self._check_eviction(policy="fifo", backend="torch_native")


# ==============================================================================
# Embedding LoRA
# ==============================================================================


class TestEmbeddingLoRA(CustomTestCase):
    """
    LoRA on an embedding model (last-token pooling, L2-normalized): LoRA on
    q_proj/v_proj perturbs attention, which propagates to the pooled embedding.
    Checked HF-free via cosine similarity (unit vectors, so cosine == dot):
    unit-norm, the adapter moves the embedding off base, the same adapter
    reproduces, and two adapters stay distinct -- the embedding analog of the
    routing check used for generation. A is encoded twice to separate "the adapter
    moved the embedding" from "the embedding is nondeterministic".

    SRTRunner.forward drops lora_path on the embedding path, so the adapter is
    applied via engine.encode(prompt=..., lora_path=...) directly.
    """

    EFFECT_MAX_COS = 0.999  # base-vs-adapter must be below this
    REPEAT_MIN_COS = 0.999  # the same adapter twice must be above this

    @staticmethod
    def _cos(a, b) -> float:
        ta = torch.tensor(a, dtype=torch.float32)
        tb = torch.tensor(b, dtype=torch.float32)
        return float(torch.dot(ta, tb) / (ta.norm() * tb.norm() + 1e-12))

    def _check_embedding_backend(self, backend: str):
        paths = synth_slot_paths([ADAPTER_A, ADAPTER_B])
        with lora_runner(
            backend=backend,
            model_type="embedding",
            lora_paths=paths,
            max_loras_per_batch=3,
        ) as runner:

            def encode(lora_path):
                resp = runner.engine.encode(prompt=TEST_PROMPTS[0], lora_path=lora_path)
                return (resp[0] if isinstance(resp, list) else resp)["embedding"]

            vectors = {
                "base": encode(None),
                "a": encode(paths[0]),
                "a2": encode(paths[0]),
                "b": encode(paths[1]),
            }
        empty_gpu_cache()

        for name in ("base", "a", "b"):
            norm = torch.tensor(vectors[name], dtype=torch.float32).norm().item()
            print(f"[{backend}] {name} ||v||={norm:.4f}")
            self.assertAlmostEqual(
                norm, 1.0, delta=1e-2, msg=f"{name} embedding is not unit-norm"
            )

        cos_base_a = self._cos(vectors["base"], vectors["a"])
        cos_base_b = self._cos(vectors["base"], vectors["b"])
        cos_a_a = self._cos(vectors["a"], vectors["a2"])
        cos_a_b = self._cos(vectors["a"], vectors["b"])
        print(
            f"[{backend}] cos(base,A)={cos_base_a:.5f} cos(base,B)={cos_base_b:.5f} "
            f"cos(A,A)={cos_a_a:.5f} cos(A,B)={cos_a_b:.5f}"
        )

        self.assertLess(
            cos_base_a,
            self.EFFECT_MAX_COS,
            f"adapter A did not change the embedding (cos(base,A)={cos_base_a:.5f})",
        )
        self.assertLess(
            cos_base_b,
            self.EFFECT_MAX_COS,
            f"adapter B did not change the embedding (cos(base,B)={cos_base_b:.5f})",
        )
        self.assertGreater(
            cos_a_a,
            self.REPEAT_MIN_COS,
            f"adapter A embedding not reproducible (cos(A,A)={cos_a_a:.5f})",
        )
        self.assertLess(
            cos_a_b,
            cos_a_a,
            f"adapter A and B embeddings not distinct (cos(A,B)={cos_a_b:.5f} "
            f"not < cos(A,A)={cos_a_a:.5f})",
        )

    def test_embedding_lora_applies_and_routes(self):
        self._check_embedding_backend("triton")

    def test_embedding_lora_applies_and_routes_torch_native(self):
        self._check_embedding_backend("torch_native")


# ==============================================================================
# Feature combinations (graph OFF)
#
# Everything above isolates ONE serving feature. test_lora_comb_xpu.py covers
# combinations, but every case there holds disable_cuda_graph=False, so a failure
# has two candidate causes: the feature interaction, or graph capture. These cases
# close that gap by combining the same features with graph capture OFF, matching
# the rest of this file -- so a failure here localizes the bug to the interaction
# itself, and a case that fails there but passes here implicates graph capture.
#
# Only combinations whose parts are individually covered above appear here; each
# pairs features that contend for the SAME state, which is where interactions
# actually break:
#
#   B1. drainer x radix cache      -- both reorder/reuse work across adapters
#   B2. overlap loading x drainer  -- graph-OFF twin of C5, to separate the
#                                     interaction from graph capture
#   B3. overlap loading x eviction -- a slot is staged asynchronously while the
#                                     victim selector is reclaiming slots
#   B4. pinning x drainer          -- the drainer picks adapters to displace and
#                                     must respect the pin
#   B5. dynamic load x radix cache -- an adapter loaded at runtime must not
#                                     inherit a previous adapter's cached prefix
# ==============================================================================


class TestDrainerWithRadixCache(_RoutingTestCase):
    """B1. Drainer x radix cache. Both features reuse work across adapters: the
    drainer reorders *which* adapter runs when, and radix reuses KV keyed by
    adapter. Combined, a prefix inserted before a drain must not be handed to the
    adapter that displaced it. Three adapters through two slots forces draining
    while every slot shares one prompt (maximal prefix contention)."""

    SLOTS = [ADAPTER_A, ADAPTER_B, ADAPTER_C]
    MAX_LORAS_PER_BATCH = 2
    RUNNER_ARGS = {"lora_drain_wait_threshold": LORA_DRAIN_WAIT_THRESHOLD}
    DISABLE_RADIX = False

    def test_routing(self):
        self._check_routing()


class TestOverlapLoadingWithDrainer(_RoutingTestCase):
    """B2. Overlap loading x drainer. Overlap loading stages weights on a side
    stream while the drainer concurrently displaces adapters, so a staged copy can
    land in a slot the drainer has already reclaimed. This is the graph-OFF twin
    of C5 in test_lora_comb_xpu.py -- if that fails and this passes, the fault is
    graph capture rather than the interaction."""

    SLOTS = [ADAPTER_A, ADAPTER_B, ADAPTER_C]
    MAX_LORAS_PER_BATCH = 2
    RUNNER_ARGS = {
        "enable_lora_overlap_loading": True,
        "max_loaded_loras": 4,
        "lora_drain_wait_threshold": LORA_DRAIN_WAIT_THRESHOLD,
    }

    def test_routing(self):
        self._check_routing()


class TestOverlapLoadingWithEviction(CustomTestCase):
    """B3. Overlap loading x pool eviction. With a single batch slot, every switch
    evicts -- and with overlap loading the replacement is staged asynchronously, so
    a reload races the victim's teardown. Asserts the same reload-fidelity property
    as TestLoRAEvictionEndToEnd (each adapter reproduces its own first output), which
    is what a torn or half-staged copy would break."""

    def _check_eviction_with_overlap(self, backend: str):
        paths = synth_slot_paths([ADAPTER_A, ADAPTER_B])
        with lora_runner(
            backend=backend,
            lora_paths=paths,
            max_loras_per_batch=1,
            enable_lora_overlap_loading=True,
            max_loaded_loras=2,
        ) as runner:
            (a_first,) = generate(runner, paths[:1])
            (b_first,) = generate(runner, paths[1:])  # evicts A while staging B
            (a_again,) = generate(runner, paths[:1])  # restage A over B
            (b_again,) = generate(runner, paths[1:])  # restage B over A
        empty_gpu_cache()

        print(f"[overlap+evict:{backend}] A={a_first!r} B={b_first!r}")
        self.assertNotEqual(
            a_first,
            b_first,
            f"[{backend}] adapters A and B produced identical output",
        )
        self.assertEqual(
            a_again,
            a_first,
            f"[{backend}] adapter A changed after eviction + overlapped reload",
        )
        self.assertEqual(
            b_again,
            b_first,
            f"[{backend}] adapter B changed after eviction + overlapped reload",
        )

    def test_overlap_loading_with_eviction(self):
        self._check_eviction_with_overlap("triton")

    def test_overlap_loading_with_eviction_csgmv(self):
        self._check_eviction_with_overlap("csgmv")


class TestPinnedAdapterWithDrainer(CustomTestCase):
    """B4. Pinning x drainer. Pinning excludes an adapter from eviction victim
    selection; the drainer independently chooses adapters to displace when slots
    are contended. Together the drainer must not displace the pinned adapter.
    TestPinnedLoRAAdapters churns a single unpinned adapter with the drainer off --
    here the drainer is active and two adapters contend for the one free slot."""

    def test_pinned_adapter_survives_drainer(self):
        with lora_runner(
            lora_paths=None,
            enable_lora=True,
            max_lora_rank=SYNTH_MAX_LORA_RANK,
            max_loras_per_batch=2,
            max_loaded_loras=4,
            lora_drain_wait_threshold=LORA_DRAIN_WAIT_THRESHOLD,
        ) as runner:
            pinned = runner.load_lora_adapter(
                lora_name="pinned_a",
                lora_path=synth_adapter_path(ADAPTER_A),
                pinned=True,
            )
            self.assertTrue(pinned.success, f"pinned load: {pinned.error_message}")
            for name, identity in (("churn_b", ADAPTER_B), ("churn_c", ADAPTER_C)):
                loaded = runner.load_lora_adapter(
                    lora_name=name, lora_path=synth_adapter_path(identity)
                )
                self.assertTrue(loaded.success, f"{name}: {loaded.error_message}")

            (baseline,) = generate(runner, ["pinned_a"])
            # Two unpinned adapters contending for the single free slot is what
            # engages the drainer; the pinned slot must stay untouched throughout.
            for _ in range(3):
                generate(runner, ["churn_b", "churn_c"], batched=True)
            (after_churn,) = generate(runner, ["pinned_a"])
        empty_gpu_cache()

        print(f"[pin+drain] baseline={baseline!r} after churn={after_churn!r}")
        self.assertEqual(
            after_churn,
            baseline,
            "pinned adapter output changed after the drainer displaced adapters "
            "(it may have been drained despite being pinned)",
        )


class TestDynamicLoadWithRadixCache(CustomTestCase):
    """B5. Dynamic load/unload x radix cache. An adapter loaded at runtime reuses a
    pool slot a previous adapter occupied, while radix holds prefixes keyed by
    adapter. A freshly loaded adapter must not inherit the cached prefix of the one
    it replaced -- the failure mode is silent (plausible text from stale KV), so it
    is asserted as "B does not reproduce A's cached output on the same prompt".
    TestLoRARadixCacheHit covers the same property with adapters supplied at
    startup; the runtime-load path reaches it through different bookkeeping."""

    def test_runtime_loaded_adapter_does_not_inherit_cached_prefix(self):
        with lora_runner(
            disable_radix=False,
            lora_paths=None,
            enable_lora=True,
            max_lora_rank=SYNTH_MAX_LORA_RANK,
            max_loras_per_batch=1,  # one slot: B must reuse A's
            max_loaded_loras=2,
        ) as runner:
            load_a = runner.load_lora_adapter(
                lora_name="adapter_a", lora_path=synth_adapter_path(ADAPTER_A)
            )
            self.assertTrue(load_a.success, f"load_a: {load_a.error_message}")
            (a_first,) = generate(runner, ["adapter_a"])
            (a_cached,) = generate(runner, ["adapter_a"])  # radix hit

            # Load B *after* A's prefix is in the cache, so B is the first request
            # to occupy the slot with a populated cache behind it.
            load_b = runner.load_lora_adapter(
                lora_name="adapter_b", lora_path=synth_adapter_path(ADAPTER_B)
            )
            self.assertTrue(load_b.success, f"load_b: {load_b.error_message}")
            (b_first,) = generate(runner, ["adapter_b"])

            unload = runner.unload_lora_adapter(lora_name="adapter_b")
            self.assertTrue(unload.success, f"unload: {unload.error_message}")
            (a_after,) = generate(runner, ["adapter_a"])
        empty_gpu_cache()

        print(f"[dynload+radix] A={a_first!r} A(cached)={a_cached!r} B={b_first!r}")
        self.assertEqual(
            a_cached,
            a_first,
            "radix cache hit did not reproduce adapter A's output",
        )
        self.assertNotEqual(
            b_first,
            a_first,
            "a runtime-loaded adapter reproduced the previous adapter's cached "
            "output on the same prompt (stale prefix reused across adapters)",
        )
        self.assertEqual(
            a_after,
            a_first,
            "adapter A changed after a runtime-loaded adapter reused its slot",
        )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
