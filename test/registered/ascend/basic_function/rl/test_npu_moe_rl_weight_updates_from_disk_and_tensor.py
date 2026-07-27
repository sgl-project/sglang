"""
E2E Test Suite: NPU MoE RL Weight Updates
==========================================
End-to-end tests for From Disk and From Tensor weight update methods in NPU + MoE scenarios.

Model: Qwen/Qwen3-30B-A3B (BF16, Unquantized MoE)
Hardware: NPU

## Test Matrix

| Test Case                             | Method        | TP  |
|--------------------------------------|--------------|-----|
| TestNPUMoEWeightUpdateFromDiskTP1       | From Disk    | 1   |
| TestNPUMoEWeightUpdateFromDiskTP2       | From Disk    | 2   |
| TestNPUMoEWeightUpdateFromTensorTP1     | From Tensor  | 1   |
| TestNPUMoEWeightUpdateFromTensorTP2     | From Tensor  | 2   |


"""

import os
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH as QWEN3_30B_A3B_INSTRUCT,
)
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_WEIGHTS_PATH as QWEN3_30B_A3B,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=700,
    suite="full-2-npu-a3",
    nightly=True,
)


# ──────────────────────────────────────────────────────────────────────
# Common NPU server launch args
# ──────────────────────────────────────────────────────────────────────
def _npu_server_args(tp_size=1, **extra):
    """Build NPU server launch arguments."""
    args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.95",
        # "--disable-cuda-graph",
        "--max-running-requests",
        "8",
        "--tp-size",
        str(tp_size),
    ]
    args.extend(
        f"--{k.replace('_','-')}={v}" if v is not True else f"--{k.replace('_','-')}"
        for k, v in extra.items()
    )
    return args


# ──────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────
def _get_decode_logprob_signature(base_url, *, max_new_tokens=64, temperature=0.0):
    """
    Get the full logprob signature for a deterministic decode.

    Returns:
        dict: {
            "text": generated text,
            "token_ids": [int, ...],     # output token ID sequence
            "logprobs": [float, ...],    # output token log probability sequence
        }
    """
    payload = {
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
    }
    resp = requests.post(
        f"{base_url}/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    ret = resp.json()
    output_token_logprobs = ret["meta_info"].get("output_token_logprobs")
    assert (
        output_token_logprobs is not None
    ), "missing output_token_logprobs in response"
    assert len(output_token_logprobs) > 0, "empty output_token_logprobs"
    return {
        "text": ret["text"],
        "token_ids": [int(x[1]) for x in output_token_logprobs],
        "logprobs": [float(x[0]) for x in output_token_logprobs],
    }


def _assert_logprob_signature_equal(a, b, *, atol=1e-4, msg=""):
    """Assert that logprob signatures from two decodes are identical."""
    assert a["text"] == b["text"], f"{msg}text mismatch: {a['text']!r} != {b['text']!r}"
    assert (
        a["token_ids"] == b["token_ids"]
    ), f"{msg}token_ids mismatch: {a['token_ids']} != {b['token_ids']}"
    assert len(a["logprobs"]) == len(
        b["logprobs"]
    ), f"{msg}logprobs length mismatch: {len(a['logprobs'])} != {len(b['logprobs'])}"
    for idx, (la, lb) in enumerate(zip(a["logprobs"], b["logprobs"])):
        assert abs(la - lb) <= atol, (
            f"{msg}logprob diff at idx={idx}: {la} vs {lb} "
            f"(delta={abs(la - lb):.6f}, tol={atol})"
        )


def _assert_logprob_signature_different(a, b, *, msg=""):
    """Assert that two decodes produce different output (weights actually changed)."""
    assert a["token_ids"] != b["token_ids"] or any(
        abs(la - lb) > 1e-4 for la, lb in zip(a["logprobs"], b["logprobs"])
    ), f"{msg}Expected signatures to differ, but they are identical"


# ──────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────
class _BaseNPUMoEWeightUpdateTest(CustomTestCase):
    """Base class for NPU MoE weight update tests.

    Subclasses must set:
        - model: primary model path
        - alt_model: alternative model path for roundtrip validation (From Disk only)
        - tp_size: tensor parallelism size
    """

    model = None
    alt_model = None
    tp_size = 1
    base_url = DEFAULT_URL_FOR_TEST
    server_process = None

    @classmethod
    def setUpClass(cls):
        if cls.model is None:
            raise NotImplementedError("Subclass must set 'model'")

    def setUp(self):
        self._launch_server()

    def _launch_server(self, extra_args=None):
        args = list(_npu_server_args(tp_size=self.tp_size))
        if extra_args:
            args.extend(extra_args)
        self.server_process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env={**os.environ, "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1"},
        )

    def tearDown(self):
        if self.server_process is not None:
            kill_process_tree(self.server_process.pid)
            self.server_process = None
        # torch.multiprocessing forkserver workers may escape process tree;
        # kill them explicitly to avoid NPU resource leaks
        try:
            import subprocess as sp

            sp.run(["pkill", "-f", "multiprocessing.forkserver"], capture_output=True)
        except Exception:
            pass

    # ── HTTP helpers ────────────────────────────────────────────────
    def _post(self, endpoint, payload, timeout=300):
        resp = requests.post(
            f"{self.base_url}{endpoint}", json=payload, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    # ── Verification ──────────────────────────────────────────────────
    def _run_decode(self):
        return _get_decode_logprob_signature(self.base_url)

    def _verify_idempotent_update(self, update_fn):
        """
        Verify idempotency of same-model updates:
          baseline → update(same_model) → updated
          assert baseline == updated

        This is the most basic validation — weights unchanged, output should not change.
        If NPU format is lost after update, kernel will receive wrong shapes → output changes → assert fails.
        """
        baseline = self._run_decode()
        result = update_fn()
        assert result.get("success"), f"Update failed: {result}"
        updated = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            updated,
            msg="Same-model update should not change output (idempotent). "
            "NPU format may have been lost after update.",
        )
        return baseline

    def _verify_roundtrip(self, baseline, model_a, model_b, update_fn):
        """
        Roundtrip validation:
          baseline (model_a) → update(model_b) → changed → update(model_a) → restored
          assert baseline == restored
        """
        # Switch to model_b
        r1 = update_fn(model_b)
        assert r1.get("success"), f"Update to {model_b} failed: {r1}"
        changed = self._run_decode()
        _assert_logprob_signature_different(
            baseline,
            changed,
            msg=f"Switching from {model_a} to {model_b} should change output",
        )

        # Switch back to model_a
        r2 = update_fn(model_a)
        assert r2.get("success"), f"Update back to {model_a} failed: {r2}"
        restored = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            restored,
            msg=f"Switching back to {model_a} should restore exact output",
        )


# ══════════════════════════════════════════════════════════════════════
# Test 1: From Disk — TP=1
# ══════════════════════════════════════════════════════════════════════
class TestNPUMoEWeightUpdateFromDiskTP1(_BaseNPUMoEWeightUpdateTest):
    """
    Verify From Disk correctness under NPU + MoE (BF16).

    Checkpoints:
      1. Same-model update keeps logprobs unchanged (idempotency)
      2. Different model update changes logprobs (weights actually take effect)
      3. Switching back restores logprobs exactly (roundtrip consistency)
      4. Both flush_cache=True and False modes

    [Test Category] RL Weight Update
    [Test Target] POST /update_weights_from_disk + TP=1
    """

    model = QWEN3_30B_A3B
    alt_model = QWEN3_30B_A3B_INSTRUCT
    tp_size = 1

    # All From Disk TP=1 tests share a single server to avoid repeated startup/shutdown
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        args = list(_npu_server_args(tp_size=cls.tp_size))
        cls.server_process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env={**os.environ, "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.server_process is not None:
            kill_process_tree(cls.server_process.pid)
            cls.server_process = None
        try:
            import subprocess as sp

            sp.run(["pkill", "-f", "multiprocessing.forkserver"], capture_output=True)
        except Exception:
            pass

    def setUp(self):
        pass  # Server launched in setUpClass, not managed per-test

    def tearDown(self):
        pass  # Server shut down in tearDownClass

    def _update(self, model_path, flush_cache=True):
        return self._post(
            "/update_weights_from_disk",
            {
                "model_path": model_path,
                "flush_cache": flush_cache,
            },
        )

    def test_01_idempotent_update_flush_cache(self):
        """Same-model From Disk update (flush_cache=True) — idempotency.

        Verify: after updating with the same model path, decode logprobs should be identical.
        """
        self._verify_idempotent_update(
            lambda: self._update(self.model, flush_cache=True)
        )

    def test_02_idempotent_update_no_flush_cache(self):
        """Same-model From Disk update (flush_cache=False) — idempotency.

        Verify: flush_cache=False does not affect weight update correctness.
        """
        self._verify_idempotent_update(
            lambda: self._update(self.model, flush_cache=False)
        )

    def test_03_roundtrip_instruct_and_back(self):
        """Roundtrip validation — base ↔ instruct.

        Flow:
          1. baseline = decode(base)
          2. update(instruct) → decode → should differ from baseline
          3. update(base) → decode → should be identical to baseline
        """
        baseline = self._run_decode()
        self._verify_roundtrip(
            baseline,
            self.model,
            self.alt_model,
            lambda m: self._update(m, flush_cache=True),
        )

    def test_04_decode_still_works_after_update(self):
        """Consistency of consecutive decodes after update.

        Verify: two consecutive decodes after update should produce identical results
        (ruling out randomness/cache pollution).
        """
        self._update(self.model, flush_cache=True)
        sig1 = self._run_decode()
        sig2 = self._run_decode()
        _assert_logprob_signature_equal(
            sig1,
            sig2,
            msg="Two consecutive decodes after update should be identical",
        )

    def test_05_abort_all_requests(self):
        """Update with abort_all_requests.

        Verify: abort_all_requests=True does not block correct weight update,
        and post-update decodes are consistent.
        """
        result = self._post(
            "/update_weights_from_disk",
            {
                "model_path": self.model,
                "flush_cache": True,
                "abort_all_requests": True,
            },
        )
        assert result.get("success"), f"Update with abort_all_requests failed: {result}"
        sig1 = self._run_decode()
        sig2 = self._run_decode()
        _assert_logprob_signature_equal(
            sig1,
            sig2,
            msg="Two consecutive decodes after abort_all_requests update should be identical",
        )


# ══════════════════════════════════════════════════════════════════════
# Test 2: From Disk — TP=2
# ══════════════════════════════════════════════════════════════════════
class TestNPUMoEWeightUpdateFromDiskTP2(_BaseNPUMoEWeightUpdateTest):
    """
    Verify From Disk correctness under TP=2.

    When TP=2, w13_weight and w2_weight are sharded along the dim dimension.

    [Test Category] RL Weight Update
    [Test Target] POST /update_weights_from_disk + TP2
    """

    model = QWEN3_30B_A3B
    tp_size = 2

    def _update(self, model_path):
        return self._post(
            "/update_weights_from_disk",
            {
                "model_path": model_path,
                "flush_cache": True,
            },
        )

    def test_01_idempotent_update_tp2(self):
        """TP=2 same-model From Disk update — idempotency.

        Verify: TP sharding does not affect weight update correctness.
        """
        baseline = self._run_decode()
        result = self._update(self.model)
        assert result.get("success"), f"Update failed: {result}"
        updated = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            updated,
            msg="TP2: same-model update should be idempotent",
        )

    def test_02_multi_round_update(self):
        """TP=2 multi-round updates — no memory leak, no cumulative drift.

        After 2 consecutive same-model updates, decode result should match baseline.
        """
        baseline = self._run_decode()
        for i in range(2):
            result = self._update(self.model)
            assert result.get("success"), f"Round {i} update failed: {result}"
        updated = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            updated,
            msg="TP2: output should be unchanged after 2 rounds of same-model update",
        )


# ══════════════════════════════════════════════════════════════════════
# Test 3: From Tensor — TP=1
# ══════════════════════════════════════════════════════════════════════
class TestNPUMoEWeightUpdateFromTensorTP1(_BaseNPUMoEWeightUpdateTest):
    """Verify load_format="direct" updating MoE expert weights under NPU + MoE + TP=1.

    [Test Category] RL Weight Update
    [Test Target] POST /update_weights_from_tensor
    """

    model = QWEN3_30B_A3B
    tp_size = 1

    def _update_via_direct(self):
        import base64

        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        named_tensors = self._load_layer0_moe_weights()

        monkey_patch_torch_reductions()
        rank_tensors = [
            (name, tensor.to("npu").clone()) for name, tensor in named_tensors
        ]
        serializer = MultiprocessingSerializer()
        serialized_bytes = serializer.serialize(rank_tensors)
        del rank_tensors

        serialized_str = base64.b64encode(serialized_bytes).decode("utf-8")
        serialized_tensors = [serialized_str] * self.tp_size

        return self._post(
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_tensors,
                "load_format": "direct",
                "flush_cache": True,
            },
            timeout=600,
        )

    def _load_layer0_moe_weights(self):
        """Read MoE weights from checkpoint, fuse + transpose to match model param shapes.

        checkpoint: gate/up [inter, hidden], down [hidden, inter]
        model param: w13 [E, hidden, 2*inter], w2 [E, inter, hidden]
        """
        import json

        from safetensors import safe_open
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        num_experts = config.num_experts
        inter_size = config.moe_intermediate_size
        hidden_size = config.hidden_size

        index_path = f"{self.model}/model.safetensors.index.json"
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        layer_prefix = "model.layers.0.mlp.experts."
        gate_weight = torch.zeros(
            num_experts, inter_size, hidden_size, dtype=torch.bfloat16
        )
        up_weight = torch.zeros(
            num_experts, inter_size, hidden_size, dtype=torch.bfloat16
        )
        down_weight = torch.zeros(
            num_experts, hidden_size, inter_size, dtype=torch.bfloat16
        )

        shard_to_experts = {}
        for eid in range(num_experts):
            for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
                key = f"{layer_prefix}{eid}.{suffix}"
                shard = weight_map[key]
                shard_to_experts.setdefault(shard, set()).add(eid)

        for shard_file, expert_ids in shard_to_experts.items():
            shard_path = f"{self.model}/{shard_file}"
            with safe_open(shard_path, framework="pt") as f:
                for eid in expert_ids:
                    gate_weight[eid] = f.get_tensor(
                        f"{layer_prefix}{eid}.gate_proj.weight"
                    )
                    up_weight[eid] = f.get_tensor(f"{layer_prefix}{eid}.up_proj.weight")
                    down_weight[eid] = f.get_tensor(
                        f"{layer_prefix}{eid}.down_proj.weight"
                    )

        w13 = torch.cat([gate_weight, up_weight], dim=1)
        w2 = down_weight
        return [
            ("model.layers.0.mlp.experts.w13_weight", w13),
            ("model.layers.0.mlp.experts.w2_weight", w2),
        ]

    def test_01_direct_tp1_moe(self):
        """From Tensor (load_format="direct") TP=1 MoE layer.

        Same-model weights → no-op, decode should match baseline.
        """
        baseline = self._run_decode()
        result = self._update_via_direct()
        assert result.get("success"), f"Update failed: {result}"
        updated = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            updated,
            msg="From Tensor (direct, TP=1): weight update broke decode",
        )


# ══════════════════════════════════════════════════════════════════════
# Test 4: From Tensor — TP=2
# ══════════════════════════════════════════════════════════════════════
class TestNPUMoEWeightUpdateFromTensorTP2(_BaseNPUMoEWeightUpdateTest):
    """Verify From Tensor under TP=2 across three load_format paths and two layer types.

    TP=2: ~28.7 GiB per chip, leaving enough headroom for _weight_loader_impl transpose workspace.

    Test data:
      Reads layer 0 weights from safetensors checkpoint
      (real values, ~3 GB), not full 60 GB model.

    [Test Category] RL Weight Update
    [Test Target] POST /update_weights_from_tensor
    """

    model = QWEN3_30B_A3B
    tp_size = 2

    def _update_via_load_weights(self):
        """load_format=None → goes through model.load_weights() path.

        Sends per-expert weights in checkpoint format (e.g. model.layers.0.mlp.experts.0.gate_proj.weight).

        """
        import base64

        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        named_tensors = self._load_layer0_moe_weights_per_expert()

        # Move tensors to NPU before serializing to avoid ForkingPickler
        # using resource_sharer FD sharing for CPU tensors (fails across HTTP processes).
        # NPU tensors go through NPU-specific reduction path, serialized data is self-contained.
        monkey_patch_torch_reductions()
        rank_tensors = [
            (name, tensor.to("npu").clone()) for name, tensor in named_tensors
        ]
        serializer = MultiprocessingSerializer()
        serialized_bytes = serializer.serialize(rank_tensors)

        # Release NPU tensor memory after serialization
        del rank_tensors

        serialized_str = base64.b64encode(serialized_bytes).decode("utf-8")
        serialized_tensors = [serialized_str] * self.tp_size

        return self._post(
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_tensors,
                "load_format": None,
                "flush_cache": True,
            },
            timeout=600,
        )

    def _update_via_flattened_bucket(self):
        """load_format="flattened_bucket" → FlattenedTensorBucket path.

        Packs named tensors into a single flattened tensor + metadata.
        """
        import base64

        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        named_tensors = self._load_layer0_moe_weights_per_expert()
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)

        # Move flattened_tensor to NPU before serializing to avoid ForkingPickler
        # using resource_sharer FD sharing for CPU tensors (fails across HTTP processes).
        # Note: no clone() because get_flattened_tensor() produces a single large tensor,
        # clone would double memory, test process NPU pool cannot fit two copies.
        monkey_patch_torch_reductions()
        flattened = bucket.get_flattened_tensor().to("npu")
        payload = {
            "flattened_tensor": flattened,
            "metadata": bucket.metadata,
        }

        serializer = MultiprocessingSerializer()
        serialized_bytes = serializer.serialize(payload)

        # Release NPU tensor memory after serialization
        del flattened, payload

        serialized_str = base64.b64encode(serialized_bytes).decode("utf-8")

        return self._post(
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": [serialized_str] * self.tp_size,
                "load_format": "flattened_bucket",
                "flush_cache": True,
            },
            timeout=600,
        )

    def _update_via_direct_attn_tp2(self):
        """Shard layer 0 attention weights by TP=2, fuse QKV, serialize per-rank independently.

        checkpoint weights:
          q_proj: [4096, 2048] → chunk dim=0 → [2048, 2048] per rank
          k_proj: [512, 2048]  → chunk dim=0 → [256, 2048] per rank
          v_proj: [512, 2048]  → chunk dim=0 → [256, 2048] per rank
          fuse per rank: cat([q,k,v], dim=0) → [2560, 2048]
          o_proj: [2048, 4096] → chunk dim=1 → [2048, 2048] per rank

        Generates different serialized_named_tensors per rank to match
        the direct path's TP sharding requirements.
        """
        import base64
        import json

        from safetensors import safe_open

        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        layer_prefix = "model.layers.0.self_attn."

        # Read layer 0 attention weights from safetensors
        index_path = f"{self.model}/model.safetensors.index.json"
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        q_key = f"{layer_prefix}q_proj.weight"
        k_key = f"{layer_prefix}k_proj.weight"
        v_key = f"{layer_prefix}v_proj.weight"
        o_key = f"{layer_prefix}o_proj.weight"

        needed_keys = [q_key, k_key, v_key, o_key]
        shard_to_keys = {}
        for key in needed_keys:
            shard = weight_map[key]
            shard_to_keys.setdefault(shard, []).append(key)

        raw = {}
        for shard_file, keys in shard_to_keys.items():
            shard_path = f"{self.model}/{shard_file}"
            with safe_open(shard_path, framework="pt") as f:
                for key in keys:
                    raw[key] = f.get_tensor(key)

        q_full = raw[q_key]
        k_full = raw[k_key]
        v_full = raw[v_key]
        o_full = raw[o_key]

        monkey_patch_torch_reductions()
        serializer = MultiprocessingSerializer()
        serialized_tensors = []

        for tp_rank in range(self.tp_size):
            device = f"npu:{tp_rank}"
            # Shard by TP
            q_shard = q_full.chunk(self.tp_size, dim=0)[tp_rank]
            k_shard = k_full.chunk(self.tp_size, dim=0)[tp_rank]
            v_shard = v_full.chunk(self.tp_size, dim=0)[tp_rank]
            # fuse QKV per rank → qkv_proj.weight sent to server
            qkv_shard = torch.cat([q_shard, k_shard, v_shard], dim=0)
            # o_proj chunk dim=1 → RowParallelLinear shard
            o_shard = o_full.chunk(self.tp_size, dim=1)[tp_rank]

            rank_tensors = [
                (f"{layer_prefix}qkv_proj.weight", qkv_shard.to(device)),
                (f"{layer_prefix}o_proj.weight", o_shard.to(device)),
            ]
            serialized_bytes = serializer.serialize(rank_tensors)
            serialized_str = base64.b64encode(serialized_bytes).decode("utf-8")
            serialized_tensors.append(serialized_str)
            del rank_tensors, q_shard, k_shard, v_shard, qkv_shard, o_shard

        return self._post(
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_tensors,
                "load_format": "direct",
                "flush_cache": True,
            },
            timeout=300,
        )

    def _update_via_direct_moe_tp2(self):
        """load_format="direct" TP=2 MoE layer.

        Reads layer 0 MoE weights from checkpoint, narrows gate/up along inter dim
        by 1/tp_size separately, then fuses into w13; w2 narrows along inter dim.
        Aligns with product code _load_w13 sharding logic.

        w13 (ColumnParallel): gate/up narrow → [E, inter/2, hidden] → cat → [E, inter, hidden]
        w2  (RowParallel):    narrow dim=2 → [E, hidden, inter/2]
        """
        import base64
        import json

        from safetensors import safe_open
        from transformers import AutoConfig

        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        num_experts = config.num_experts
        inter_size = config.moe_intermediate_size
        hidden_size = config.hidden_size

        index_path = f"{self.model}/model.safetensors.index.json"
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        layer_prefix = "model.layers.0.mlp.experts."

        # Group reads by shard (same logic as _load_layer0_moe_weights)
        shard_to_experts = {}
        for eid in range(num_experts):
            for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
                key = f"{layer_prefix}{eid}.{suffix}"
                shard = weight_map[key]
                shard_to_experts.setdefault(shard, set()).add(eid)

        gate_weight = torch.zeros(
            num_experts, inter_size, hidden_size, dtype=torch.bfloat16
        )
        up_weight = torch.zeros(
            num_experts, inter_size, hidden_size, dtype=torch.bfloat16
        )
        down_weight = torch.zeros(
            num_experts, hidden_size, inter_size, dtype=torch.bfloat16
        )

        for shard_file, expert_ids in shard_to_experts.items():
            shard_path = f"{self.model}/{shard_file}"
            with safe_open(shard_path, framework="pt") as f:
                for eid in expert_ids:
                    gate_weight[eid] = f.get_tensor(
                        f"{layer_prefix}{eid}.gate_proj.weight"
                    )
                    up_weight[eid] = f.get_tensor(f"{layer_prefix}{eid}.up_proj.weight")
                    down_weight[eid] = f.get_tensor(
                        f"{layer_prefix}{eid}.down_proj.weight"
                    )

        # TP-shard gate/up first THEN fuse: product code _load_w13 narrows gate and up
        # individually along dim=1 (inter dim), then concatenates into w13's two halves.
        # Fusing first then chunking would give rank 0 all gate, rank 1 all up — logically wrong.
        inter_per_tp = inter_size // self.tp_size
        w2_full = down_weight  # [E, hidden, inter]

        monkey_patch_torch_reductions()
        serializer = MultiprocessingSerializer()
        serialized_tensors = []

        for tp_rank in range(self.tp_size):
            device = f"npu:{tp_rank}"
            # ColumnParallel: gate/up each take 1/tp_size along dim=1 (inter)
            gate_shard = gate_weight[
                :, tp_rank * inter_per_tp : (tp_rank + 1) * inter_per_tp, :
            ]  # [E, inter/2, hidden]
            up_shard = up_weight[
                :, tp_rank * inter_per_tp : (tp_rank + 1) * inter_per_tp, :
            ]  # [E, inter/2, hidden]
            w13_shard = torch.cat([gate_shard, up_shard], dim=1)  # [E, inter, hidden]
            # RowParallel: w2 dim=2 (inter) take 1/tp_size
            w2_shard = w2_full[
                :, :, tp_rank * inter_per_tp : (tp_rank + 1) * inter_per_tp
            ]  # [E, hidden, inter/2]

            rank_tensors = [
                ("model.layers.0.mlp.experts.w13_weight", w13_shard.to(device)),
                ("model.layers.0.mlp.experts.w2_weight", w2_shard.to(device)),
            ]
            serialized_bytes = serializer.serialize(rank_tensors)
            serialized_str = base64.b64encode(serialized_bytes).decode("utf-8")
            serialized_tensors.append(serialized_str)
            del rank_tensors, gate_shard, up_shard, w13_shard, w2_shard

        return self._post(
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_tensors,
                "load_format": "direct",
                "flush_cache": True,
            },
            timeout=600,
        )

    def _load_layer0_moe_weights_per_expert(self):
        """Read layer 0 MoE expert weights from safetensors checkpoint, per-expert format.

        Returns checkpoint-format named tensors for load_format=None / flattened_bucket
        paths. Product code's load_weights() completes fuse and TP sharding via
        expert_params_mapping.

        Example names:
          model.layers.0.mlp.experts.0.gate_proj.weight  — [inter, hidden]
          model.layers.0.mlp.experts.0.up_proj.weight    — [inter, hidden]
          model.layers.0.mlp.experts.0.down_proj.weight  — [hidden, inter]
          ... (3 params per expert)
        """
        import json

        from safetensors import safe_open
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        num_experts = config.num_experts

        index_path = f"{self.model}/model.safetensors.index.json"
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        layer_prefix = "model.layers.0.mlp.experts."

        # Group reads by shard
        shard_to_experts = {}
        for eid in range(num_experts):
            for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
                key = f"{layer_prefix}{eid}.{suffix}"
                shard = weight_map[key]
                shard_to_experts.setdefault(shard, set()).add(eid)

        result = []
        for shard_file, expert_ids in shard_to_experts.items():
            shard_path = f"{self.model}/{shard_file}"
            with safe_open(shard_path, framework="pt") as f:
                for eid in expert_ids:
                    result.append(
                        (
                            f"{layer_prefix}{eid}.gate_proj.weight",
                            f.get_tensor(f"{layer_prefix}{eid}.gate_proj.weight"),
                        )
                    )
                    result.append(
                        (
                            f"{layer_prefix}{eid}.up_proj.weight",
                            f.get_tensor(f"{layer_prefix}{eid}.up_proj.weight"),
                        )
                    )
                    result.append(
                        (
                            f"{layer_prefix}{eid}.down_proj.weight",
                            f.get_tensor(f"{layer_prefix}{eid}.down_proj.weight"),
                        )
                    )

        return result

    def test_01_load_weights_path(self):
        """From Tensor (load_format=None) — real no-op content verification.

        Reads layer 0 MoE weights from safetensors checkpoint (real values),
        sends back to server via From Tensor. Real weights → true no-op.

        Verification:
          1. API success (copy stage passed)
          2. Decode output matches baseline exactly (logprobs token-by-token)
        """
        baseline = self._run_decode()
        result = self._update_via_load_weights()
        assert result.get("success"), f"Update failed at copy stage: {result}"
        updated = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            updated,
            msg="From Tensor (load_weights): weight update broke decode. ",
        )

    def test_02_flattened_bucket(self):
        """From Tensor (load_format="flattened_bucket") — real no-op content verification.

        Same as load_weights test, but packed via FlattenedTensorBucket → reconstruct → model.load_weights().
        """
        baseline = self._run_decode()
        result = self._update_via_flattened_bucket()
        assert result.get("success"), f"Update (flattened_bucket) failed: {result}"
        updated = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            updated,
            msg=f"From Tensor (flattened_bucket): decode diverged after weight update. ",
        )

    def test_03_direct_tp2_non_moe(self):
        """load_format="direct" TP=2 non-MoE attention layer.

        Reads layer 0 attention from checkpoint, shards by rank, fuses QKV → direct path.
        Same-model weights → no-op, decode should match baseline.
        """
        baseline = self._run_decode()
        result = self._update_via_direct_attn_tp2()
        assert result.get("success"), f"Update failed: {result}"
        updated = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            updated,
            msg="From Tensor (direct, TP=2, non-MoE): weight update broke decode",
        )

    def test_04_direct_tp2_moe(self):
        """load_format="direct" TP=2 MoE expert layer.

        Reads MoE weights from checkpoint, fuses + TP-shards → direct path.
        Same-model weights → no-op, decode should match baseline.
        """
        baseline = self._run_decode()
        result = self._update_via_direct_moe_tp2()
        assert result.get("success"), f"Update failed: {result}"
        updated = self._run_decode()
        _assert_logprob_signature_equal(
            baseline,
            updated,
            msg="From Tensor (direct, TP=2, MoE): weight update broke decode",
        )


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main()
