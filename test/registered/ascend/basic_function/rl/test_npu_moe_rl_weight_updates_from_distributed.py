"""
E2E Test: NPU MoE From Distributed Weight Update (Multi-Process)
=================================================================
Simulates a real RL disaggregated scenario: trainer broadcasts weights → inference receives and applies them.

Design:
  - mp.spawn creates 2 processes, dispatched to trainer / inference
  - Rank 0 (trainer): reads layer 0 MoE per-expert weights from safetensors,
    HCCL broadcast (supports flattened_bucket / standard per-tensor paths)
  - Rank 1 (inference): SGLang Server, uses expert_params_mapping
    to auto-merge per-expert weights into FusedMoE format

Model: Qwen3-30B-A3B (BF16)
Hardware: NPUs

Key design:
  - Trainer reads per-expert checkpoint-format weights from safetensors and broadcasts them,
    mimicking AutoModelForCausalLM.from_pretrained() named_parameters() behavior in VERL/Ray RLHF.
  - Server load_weights uses expert_params_mapping to match per-expert weight names,
    auto-invoking FusedMoE.weight_loader(param, weight, name, shard_id, expert_id).
  - FlattenedTensorBucket packs N tensors into a single flat buffer,
    avoiding fragmentation and OOM from per-tensor broadcasts.
"""

import gc
import json
import logging
import os
import socket
import time
import unittest

import requests
import torch
import torch.multiprocessing as mp
from safetensors import safe_open

from sglang.srt.utils import init_custom_process_group, kill_process_tree
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
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
    est_time=500,
    suite="full-4-npu-a3",
    nightly=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)

# ──────────────────────────────────────────────────────────────────────
# Model & Communication Config
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH = QWEN3_30B_A3B
INSTRUCT_MODEL_PATH = QWEN3_30B_A3B_INSTRUCT

# Only update the first N_TEST_EXPERTS experts to control per-broadcast data volume
N_TEST_EXPERTS = 8

GROUP_NAME = "test_weight_update_e2e"
MASTER_ADDR = "127.0.0.1"
BACKEND = "hccl"


def _build_expert_weight_names(num_experts, layer=0):
    """Generate per-expert HF checkpoint format weight name list."""
    names = []
    for i in range(num_experts):
        for sfx in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
            names.append(f"model.layers.{layer}.mlp.experts.{i}.{sfx}")
    return names


# ──────────────────────────────────────────────────────────────────────
# Verification Utilities
# ──────────────────────────────────────────────────────────────────────
def _get_decode_logprob_signature(base_url, *, max_new_tokens=16):
    """Send /generate request and extract decode logprob signature for content verification."""
    payload = {
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    ret = resp.json()
    output_token_logprobs = ret["meta_info"]["output_token_logprobs"]
    assert output_token_logprobs is not None
    assert len(output_token_logprobs) > 0
    logger.info(
        "[verify] generate elapsed=%.1fs text=%s tokens=%d logprobs=%s",
        elapsed,
        ret["text"][:40],
        len(output_token_logprobs),
        [float(x[0]) for x in output_token_logprobs[:4]],
    )
    return {
        "text": ret["text"],
        "token_ids": [int(x[1]) for x in output_token_logprobs],
        "logprobs": [float(x[0]) for x in output_token_logprobs],
    }


def _assert_logprob_signature_equal(a, b, *, atol=1e-4, msg=""):
    """Verify two decode logprob signatures match (text, token_ids, logprobs)."""
    logger.info("[verify] baseline text: %s", a["text"])
    logger.info("[verify] updated  text: %s", b["text"])
    assert a["text"] == b["text"], f"{msg}text mismatch: {a['text']!r} != {b['text']!r}"
    assert a["token_ids"] == b["token_ids"], f"{msg}token_ids mismatch"
    assert len(a["logprobs"]) == len(b["logprobs"]), f"{msg}logprobs len mismatch"
    for idx, (la, lb) in enumerate(zip(a["logprobs"], b["logprobs"])):
        assert abs(la - lb) <= atol, (
            f"{msg}logprob diff at idx={idx}: {la} vs {lb} "
            f"(delta={abs(la - lb):.6f}, tol={atol})"
        )


def _assert_logprob_signature_different(a, b, *, msg=""):
    """Verify that two decodes have different token_ids or logprobs (weights actually changed)."""
    logger.info("[verify] baseline  text: %s", a["text"])
    logger.info("[verify] changed  text: %s", b["text"])
    assert a["token_ids"] != b["token_ids"] or any(
        abs(la - lb) > 1e-4 for la, lb in zip(a["logprobs"], b["logprobs"])
    ), f"{msg}Expected signatures to differ, but they are identical"


# ──────────────────────────────────────────────────────────────────────
# Multi-Process Entry
# ──────────────────────────────────────────────────────────────────────
def _init_process(
    rank,
    master_port,
    world_size,
    barrier,
    result_queue,
    tp_size,
    alt_model_path=None,
    load_format="flattened_bucket",
):
    """mp.spawn entry: rank=0 → trainer, rank=1 → inference.

    alt_model_path: if set, perform roundtrip verification
    load_format: "flattened_bucket" or None (standard path)
    """
    logger.info(
        "[init] rank=%d dispatching: master_port=%d world_size=%d tp_size=%d "
        "roundtrip=%s load_format=%s",
        rank,
        master_port,
        world_size,
        tp_size,
        alt_model_path is not None,
        load_format,
    )
    if rank == 0:
        trainer_device = 1 if tp_size == 1 else tp_size
        logger.info("[init] rank=0 → trainer (npu:%d)", trainer_device)
        _trainer_process(
            rank,
            world_size,
            master_port,
            barrier,
            result_queue,
            trainer_device,
            alt_model_path,
            load_format,
        )
    elif rank == 1:
        logger.info("[init] rank=1 → inference")
        _inference_process(
            rank,
            world_size,
            master_port,
            barrier,
            result_queue,
            tp_size,
            alt_model_path,
            load_format,
        )
    else:
        logger.error("[init] unexpected rank=%d, aborting", rank)
        result_queue.put("FAIL")
        result_queue.put(f"Unexpected rank: {rank}")


# ──────────────────────────────────────────────────────────────────────
# Rank 0: Trainer Process
# ──────────────────────────────────────────────────────────────────────
def _trainer_process(
    rank,
    world_size,
    master_port,
    barrier,
    result_queue,
    device=1,
    alt_model_path=None,
    load_format="flattened_bucket",
):
    """Trainer: reads per-expert weights from safetensors → HCCL broadcast.

    load_format:
      flattened_bucket  → FlattenedTensorBucket packs into 1 flat buffer
      None              → per-tensor broadcast (standard path)

    Aligned with real RL scenarios (VERL/Ray RLHF):
      trainer reads per-expert weights from checkpoint and broadcasts,
      server load_weights auto-merges via expert_params_mapping.
    """
    try:
        torch.npu.set_device(device)
        logger.info(
            "[trainer] started on npu:%d, world_size=%d, roundtrip=%s load_format=%s",
            device,
            world_size,
            alt_model_path is not None,
            load_format,
        )

        init_method = f"tcp://{MASTER_ADDR}:{master_port}"
        logger.info(
            "[trainer] init custom process group: init_method=%s rank=%d world_size=%d",
            init_method,
            rank,
            world_size,
        )
        group = init_custom_process_group(
            backend=BACKEND,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            group_name=GROUP_NAME,
        )
        torch.npu.synchronize()
        barrier.wait()
        logger.info("[trainer] group ready, barrier passed")

        def _read_safetensors(model_path):
            expert_names = _build_expert_weight_names(N_TEST_EXPERTS)
            index_path = f"{model_path}/model.safetensors.index.json"
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shard_to_keys = {}
            for key in expert_names:
                shard = weight_map[key]
                shard_to_keys.setdefault(shard, []).append(key)
            all_tensors = {}
            for shard_file, keys in shard_to_keys.items():
                with safe_open(f"{model_path}/{shard_file}", framework="pt") as f:
                    for key in keys:
                        t = f.get_tensor(key)
                        all_tensors[key] = t
                        logger.info(
                            "[trainer]   %s shape=%s dtype=%s",
                            key,
                            list(t.shape),
                            t.dtype,
                        )
            return [(name, all_tensors[name]) for name in expert_names], all_tensors

        def _broadcast_standard(named_tensors, round_label):
            """Standard path: per-tensor broadcast."""
            logger.info(
                "[trainer] [%s] broadcasting %d tensors (standard path)",
                round_label,
                len(named_tensors),
            )
            t0 = time.perf_counter()
            total_bytes = 0
            for _, tensor in named_tensors:
                t = tensor.to(f"npu:{device}")
                torch.distributed.broadcast(t, src=0, group=group)
                total_bytes += t.numel() * t.element_size()
            torch.npu.synchronize()
            elapsed = time.perf_counter() - t0
            logger.info(
                "[trainer] [%s] broadcast %d tensors, %.1f MB in %.1fs",
                round_label,
                len(named_tensors),
                total_bytes / 1e6,
                elapsed,
            )

        def _broadcast_flattened(named_tensors, round_label):
            """FlattenedTensorBucket path: pack into 1 flat buffer → broadcast."""
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor().to(f"npu:{device}")
            total_bytes = flattened_tensor.numel() * flattened_tensor.element_size()
            logger.info(
                "[trainer] [%s] broadcasting %d tensors via flattened_bucket (%.1f MB)",
                round_label,
                len(named_tensors),
                total_bytes / 1e6,
            )
            t0 = time.perf_counter()
            torch.distributed.broadcast(flattened_tensor, src=0, group=group)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - t0
            logger.info(
                "[trainer] [%s] broadcast: %d bytes in %.1fs",
                round_label,
                flattened_tensor.numel(),
                elapsed,
            )
            del flattened_tensor, bucket

        _broadcast = (
            _broadcast_flattened
            if load_format == "flattened_bucket"
            else _broadcast_standard
        )

        def _do_broadcast(model_path, round_label):
            named_tensors, all_tensors = _read_safetensors(model_path)
            logger.info(
                "[trainer] [%s] read %d tensors from %s, %.1f MB",
                round_label,
                len(named_tensors),
                model_path,
                sum(t.numel() * t.element_size() for _, t in named_tensors) / 1e6,
            )
            _broadcast(named_tensors, round_label)
            del all_tensors, named_tensors
            gc.collect()

        if alt_model_path is not None:
            _do_broadcast(alt_model_path, "instruct")
            torch.npu.empty_cache()
            _do_broadcast(MODEL_PATH, "base")
        else:
            _do_broadcast(MODEL_PATH, "noop")

        # Cleanup
        torch.distributed.destroy_process_group(group)
        torch.npu.empty_cache()
        logger.info("[trainer] exit (cleanup done)")
    except Exception as e:
        logger.error("[trainer] failed: %s", e, exc_info=True)
        result_queue.put("FAIL")
        result_queue.put(f"Trainer failed: {type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────
# Rank 1: Inference Process
# ──────────────────────────────────────────────────────────────────────
def _inference_process(
    _rank,
    world_size,
    master_port,
    barrier,
    result_queue,
    tp_size=1,
    alt_model_path=None,
    load_format="flattened_bucket",
):
    """Inference: start SGLang Server → baseline decode → receive weights → decode verification.

    load_format: "flattened_bucket" or None (standard broadcast path)
    """
    torch.npu.set_device(0)

    mem_fraction = "0.95" if tp_size == 1 else "0.7"

    server_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        mem_fraction,
        "--max-running-requests",
        "8",
        "--tp-size",
        str(tp_size),
    ]
    if tp_size == 1:
        server_args.append("--disable-cuda-graph")

    base_url = DEFAULT_URL_FOR_TEST

    logger.info(
        "[inference] launching server: tp_size=%d args=%s", tp_size, server_args
    )
    t_launch = time.perf_counter()
    process = popen_launch_server(
        MODEL_PATH,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=server_args,
        env={**os.environ, "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1"},
    )
    logger.info(
        "[inference] server ready: %s (launch took %.1fs)",
        base_url,
        time.perf_counter() - t_launch,
    )

    def _get_names_dtypes_shapes():
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        hidden_size = config.hidden_size
        moe_intermediate_size = (
            getattr(config, "moe_intermediate_size", None) or config.intermediate_size
        )
        num_experts = config.num_experts

        names = _build_expert_weight_names(N_TEST_EXPERTS)
        dtypes = ["bfloat16"] * len(names)
        shapes = []
        for name in names:
            if "down_proj" in name:
                shapes.append([hidden_size, moe_intermediate_size])
            else:
                shapes.append([moe_intermediate_size, hidden_size])
        logger.info(
            "[inference] config: hidden=%d moe_intermediate=%d total_experts=%d test_experts=%d",
            hidden_size,
            moe_intermediate_size,
            num_experts,
            N_TEST_EXPERTS,
        )
        return names, dtypes, shapes

    def _do_update(label):
        """Call /update_weights_from_distributed."""
        t0 = time.perf_counter()
        payload = {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "group_name": GROUP_NAME,
            "flush_cache": True,
        }
        if load_format is not None:
            payload["load_format"] = load_format
        resp = requests.post(
            f"{base_url}/update_weights_from_distributed",
            json=payload,
            timeout=600,
        )
        elapsed = time.perf_counter() - t0
        assert resp.json()[
            "success"
        ], f"update_weights_from_distributed [{label}] failed: {resp.json()}"
        logger.info(
            "[inference] update_weights_from_distributed [%s]: OK (%.1fs)",
            label,
            elapsed,
        )

    try:
        # 1. Baseline decode
        baseline = _get_decode_logprob_signature(base_url)
        logger.info(
            "[inference] baseline decode: %s... tokens=%d",
            baseline["text"][:50],
            len(baseline["token_ids"]),
        )

        # 2. Build weight names/shapes
        names, dtypes, shapes = _get_names_dtypes_shapes()

        # 3. Init weight update communication group
        logger.info(
            "[inference] init_weights_update_group: master_port=%d", master_port
        )
        resp = requests.post(
            f"{base_url}/init_weights_update_group",
            json={
                "master_address": MASTER_ADDR,
                "master_port": master_port,
                "rank_offset": world_size
                - tp_size,  # trainer takes 1 rank → server starts from rank 1
                "world_size": world_size,
                "group_name": GROUP_NAME,
                "backend": BACKEND,
            },
            timeout=60,
        )
        assert resp.json()[
            "success"
        ], f"init_weights_update_group failed: {resp.json()}"
        logger.info("[inference] init_weights_update_group: OK")

        barrier.wait()
        logger.info("[inference] barrier passed, waiting for broadcast")

        if alt_model_path is not None:
            # ── Roundtrip Verification ────────────────────────────
            # Round 1: receive instruct weights → decode should change
            logger.info("[inference] === roundtrip round 1: receive instruct ===")

            try:
                _do_update("instruct")
                instruct_decode = _get_decode_logprob_signature(base_url)
                logger.info(
                    "[inference] instruct decode: %s... tokens=%d",
                    instruct_decode["text"][:50],
                    len(instruct_decode["token_ids"]),
                )
                _assert_logprob_signature_different(
                    baseline,
                    instruct_decode,
                    msg="Roundtrip round 1: instruct weights should change decode output. ",
                )
                logger.info(
                    "[inference] round 1 PASS: decode changed (instruct weights took effect)"
                )
            except Exception as e:
                logger.error("[inference] round 1 failed: %s", e)
                result_queue.put("FAIL")
                result_queue.put(f"Round 1 (instruct) failed: {type(e).__name__}: {e}")
                return

            # Round 2: receive base weights → decode should restore
            logger.info("[inference] === roundtrip round 2: receive base ===")

            try:
                _do_update("base")
                restored = _get_decode_logprob_signature(base_url)
                logger.info(
                    "[inference] restored decode: %s... tokens=%d",
                    restored["text"][:50],
                    len(restored["token_ids"]),
                )
                _assert_logprob_signature_equal(
                    baseline,
                    restored,
                    msg="Roundtrip round 2: base weights should restore decode. ",
                )
                logger.info("[inference] round 2 PASS: decode restored to baseline")
                result_queue.put("PASS")
            except Exception as e:
                logger.error("[inference] round 2 failed: %s", e)
                result_queue.put("FAIL")
                result_queue.put(
                    f"Round 2 (base restore) failed: {type(e).__name__}: {e}"
                )
                return
        else:
            # ── No-op Verification ──────────────────────────────────

            try:
                _do_update("noop")
                updated = _get_decode_logprob_signature(base_url)
                logger.info(
                    "[inference] post-update decode: %s... tokens=%d",
                    updated["text"][:50],
                    len(updated["token_ids"]),
                )
                _assert_logprob_signature_equal(
                    baseline,
                    updated,
                    msg="From Distributed: weights verification failed. ",
                )
                logger.info("[inference] PASS: logprob signature matches baseline")
                result_queue.put("PASS")
            except requests.exceptions.ConnectionError as e:
                logger.error("[inference] server crashed after weight update: %s", e)
                result_queue.put("FAIL")
                result_queue.put(f"Server crashed after weight update: {e}")
            except AssertionError as e:
                logger.error("[inference] content verification failed: %s", e)
                result_queue.put("FAIL")
                result_queue.put(f"{type(e).__name__}: {e}")
            except Exception as e:
                logger.error("[inference] unexpected error: %s", e)
                result_queue.put("FAIL")
                result_queue.put(f"{type(e).__name__}: {e}")

    except Exception as e:
        logger.error("[inference] setup failed: %s", e, exc_info=True)
        result_queue.put("FAIL")
        result_queue.put(f"Inference setup failed: {type(e).__name__}: {e}")

    finally:
        # Clean up communication group
        logger.info("[inference] cleaning up...")
        try:
            resp = requests.post(
                f"{base_url}/destroy_weights_update_group",
                json={"group_name": GROUP_NAME},
                timeout=30,
            )
            logger.info(
                "[inference] destroy_weights_update_group: %s",
                "OK" if resp.status_code == 200 else f"status={resp.status_code}",
            )
        except Exception as e:
            logger.warning("[inference] destroy_weights_update_group failed: %s", e)
        kill_process_tree(process.pid)
        logger.info("[inference] process tree killed")


# ──────────────────────────────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────────────────────────────
class TestNPUMoEUpdateWeightsDistributedE2E(CustomTestCase):
    """NPU + MoE + From Distributed multi-process E2E test.

    Coverage:
      - TP=1/2 x no-op/roundtrip x flattened_bucket/standard path
      - 6 test methods, 2-3 chips

    Simulates real RL disaggregated architecture (VERL/Ray RLHF):
      trainer process (rank 0) → safetensors reads per-expert weights → HCCL broadcast
      inference process (rank 1) → SGLang Server → load_weights → decode logprob verification

    [Test Category] RL Weight Update + Distributed
    [Test Target] POST /update_weights_from_distributed, POST /generate (logprobs)
    """

    def _run_e2e(
        self, tp_size, world_size, alt_model_path=None, load_format="flattened_bucket"
    ):
        """Common test logic.

        Args:
            tp_size: server TP size
            world_size: total HCCL communicator size (trainer + tp_ranks)
            alt_model_path: if set, perform instruct→base roundtrip verification
            load_format: "flattened_bucket" or None (standard broadcast path)
        """
        assert torch.npu.device_count() >= (
            tp_size + 1
        ), f"At least {tp_size + 1} NPU devices required"

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            free_port = s.getsockname()[1]
        logger.info(
            "=== test start: tp_size=%d world_size=%d master_port=%d roundtrip=%s load_format=%s ===",
            tp_size,
            world_size,
            free_port,
            alt_model_path is not None,
            load_format,
        )

        result_queue = mp.Queue()
        barrier = mp.Barrier(
            2, timeout=300
        )  # sync trainer + inference mp.spawn processes

        logger.info(
            "[runner] spawning 2 processes (tp_size=%d world_size=%d master_port=%d)",
            tp_size,
            world_size,
            free_port,
        )
        context = mp.spawn(
            _init_process,
            args=(
                free_port,
                world_size,
                barrier,
                result_queue,
                tp_size,
                alt_model_path,
                load_format,
            ),
            nprocs=2,
            join=False,
        )
        logger.info("[runner] processes spawned, waiting for result...")

        # Use result_queue as the main signal: inference puts "PASS"/"FAIL" immediately after verification.
        # Do NOT rely on context.join()+is_alive(); mp.spawn process exit has cleanup delay, is_alive() has races.
        try:
            result = result_queue.get(timeout=600)
            logger.info("[runner] result received: %s", result)
        except Exception:
            self.fail(
                "No result reported within 600s — likely HCCL deadlock or server crash"
            )

        # Clean up spawn processes (may have brief cleanup delay, join then terminate as fallback)
        context.join(timeout=10)
        for p in context.processes:
            if p.is_alive():
                p.terminate()

        if result == "FAIL":
            try:
                detail = result_queue.get(timeout=5)
            except Exception:
                detail = "no details"
            self.fail(f"Content verification failed: {detail}")
        elif result == "PASS":
            logger.info("=== test PASS ===")
            return
        else:
            self.fail(f"Unexpected result: {result}")

    def test_noop_flattened_tp1(self):
        """TP=1 no-op flattened_bucket: same model broadcast, decode should match baseline."""
        self._run_e2e(tp_size=1, world_size=2, load_format="flattened_bucket")

    def test_noop_flattened_tp2(self):
        """TP=2 no-op flattened_bucket: additionally verify column-parallel w13/w2 TP sharding correctness."""
        self._run_e2e(tp_size=2, world_size=3, load_format="flattened_bucket")

    def test_roundtrip_flattened_tp1(self):
        """TP=1 roundtrip flattened_bucket: instruct weights should change decode, base weights should restore baseline."""
        self._run_e2e(
            tp_size=1,
            world_size=2,
            alt_model_path=INSTRUCT_MODEL_PATH,
            load_format="flattened_bucket",
        )

    def test_roundtrip_flattened_tp2(self):
        """TP=2 roundtrip flattened_bucket: instruct/base weight switch correctness under TP sharding."""
        self._run_e2e(
            tp_size=2,
            world_size=3,
            alt_model_path=INSTRUCT_MODEL_PATH,
            load_format="flattened_bucket",
        )

    def test_noop_standard_tp1(self):
        """TP=1 no-op standard: cover load_format=None per-tensor broadcast."""
        self._run_e2e(tp_size=1, world_size=2, load_format=None)

    def test_roundtrip_standard_tp1(self):
        """TP=1 roundtrip standard: instruct→base switch validity under per-tensor broadcast."""
        self._run_e2e(
            tp_size=1,
            world_size=2,
            alt_model_path=INSTRUCT_MODEL_PATH,
            load_format=None,
        )


if __name__ == "__main__":
    unittest.main()
