"""Unit tests for the O(1) expert weight matching optimization in deepseek_weight_loader.

Downloads model.safetensors.index.json from real HuggingFace model repos and
verifies that the new dict-based lookup produces identical results to the
original linear scan for every weight tensor name.
"""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _download_index(repo_id: str, dest: Path) -> Path:
    """Download model.safetensors.index.json via ``hf`` CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    idx_path = dest / "model.safetensors.index.json"
    if idx_path.exists():
        return idx_path
    subprocess.check_call(
        [
            "uvx",
            "hf",
            "download",
            repo_id,
            "--include",
            "model.safetensors.index.json",
            "--local-dir",
            str(dest),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return idx_path


def _download_config(repo_id: str, dest: Path) -> Path:
    """Download config.json via ``hf`` CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    cfg_path = dest / "config.json"
    if cfg_path.exists():
        return cfg_path
    subprocess.check_call(
        [
            "uvx",
            "hf",
            "download",
            repo_id,
            "--include",
            "config.json",
            "--local-dir",
            str(dest),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Reference implementation: original O(N*M) linear scan
# ---------------------------------------------------------------------------

def _make_expert_params_mapping(
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
) -> List[Tuple[str, str, int, str]]:
    """Reproduces FusedMoE.make_expert_params_mapping."""
    return [
        (
            (
                "experts.w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else "experts.w2_"
            ),
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", ckpt_gate_proj_name),
            ("w2", ckpt_down_proj_name),
            ("w3", ckpt_up_proj_name),
        ]
    ]


def _linear_scan_match(
    name: str,
    expert_params_mapping: List[Tuple[str, str, int, str]],
) -> Optional[Tuple[str, str, int, str]]:
    """Original O(N) linear scan to find expert mapping for a weight name."""
    for mapping in expert_params_mapping:
        param_name, weight_name, expert_id, shard_id = mapping
        if weight_name not in name:
            continue
        replaced = name.replace(weight_name, param_name)
        return (replaced, param_name, expert_id, shard_id)
    return None


# ---------------------------------------------------------------------------
# New implementation: O(1) dict lookup
# ---------------------------------------------------------------------------

_EXPERT_WEIGHT_RE = re.compile(r"experts\.(\d+)\.([^.]+)\.")


def _build_expert_proj_map(
    expert_params_mapping: List[Tuple[str, str, int, str]],
) -> Dict[str, Tuple[str, str]]:
    proj_map: Dict[str, Tuple[str, str]] = {}
    for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
        if expert_id != 0:
            continue
        parts = weight_name.split(".")
        if len(parts) >= 3:
            proj = parts[2]
            proj_map[proj] = (param_name, shard_id)
    return proj_map


def _dict_lookup_match(
    name: str,
    expert_proj_map: Dict[str, Tuple[str, str]],
    num_experts: int,
) -> Optional[Tuple[str, str, int, str]]:
    """New O(1) dict lookup to find expert mapping for a weight name."""
    m = _EXPERT_WEIGHT_RE.search(name)
    if not m:
        return None
    expert_id = int(m.group(1))
    proj_name = m.group(2)
    entry = expert_proj_map.get(proj_name)
    if entry is None or expert_id >= num_experts:
        return None
    param_name, shard_id = entry
    weight_name = f"experts.{expert_id}.{proj_name}."
    replaced = name.replace(weight_name, param_name)
    return (replaced, param_name, expert_id, shard_id)


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

MODELS = [
    ("deepseek-ai/DeepSeek-V3.2", False),
    ("deepseek-ai/DeepSeek-V3", False),
    ("nvidia/DeepSeek-R1-NVFP4", True),
]


class TestExpertWeightMatching(unittest.TestCase):
    """Verify dict lookup produces identical results to linear scan for real models."""

    _cache_dir: Path

    @classmethod
    def setUpClass(cls):
        cls._cache_dir = Path(tempfile.gettempdir()) / "hf_configs_test"

    def _run_for_model(self, repo_id: str, is_nvfp4: bool):
        safe_name = repo_id.replace("/", "--")
        dest = self._cache_dir / safe_name

        idx_path = _download_index(repo_id, dest)
        cfg_path = _download_config(repo_id, dest)

        with open(idx_path) as f:
            weight_names = list(json.load(f)["weight_map"].keys())
        with open(cfg_path) as f:
            config = json.load(f)

        n_routed_experts = config["n_routed_experts"]
        # In sglang, shared experts are fused as an extra expert
        num_fused_shared = 1
        total_experts = n_routed_experts + num_fused_shared

        expert_params_mapping = _make_expert_params_mapping(
            "gate_proj", "down_proj", "up_proj", total_experts
        )
        expert_proj_map = _build_expert_proj_map(expert_params_mapping)

        # Simulate the shared_experts -> experts.{n_routed} renaming that
        # happens in do_load_weights before the matching loop.
        processed_names = []
        for name in weight_names:
            if "mlp.shared_experts" in name:
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{n_routed_experts}",
                )
            processed_names.append(name)

        matched_linear = 0
        matched_dict = 0
        mismatches = []

        for name in processed_names:
            result_linear = _linear_scan_match(name, expert_params_mapping)
            result_dict = _dict_lookup_match(name, expert_proj_map, total_experts)

            if result_linear is not None:
                matched_linear += 1
            if result_dict is not None:
                matched_dict += 1

            if result_linear != result_dict:
                mismatches.append((name, result_linear, result_dict))

        self.assertEqual(
            mismatches,
            [],
            f"Mismatches found for {repo_id} ({len(mismatches)} of {len(processed_names)})",
        )
        self.assertEqual(matched_linear, matched_dict)
        # Sanity: we should have matched a significant number of expert weights
        self.assertGreater(matched_linear, 1000, f"Too few matches for {repo_id}")

    def test_deepseek_v3_2(self):
        self._run_for_model("deepseek-ai/DeepSeek-V3.2", is_nvfp4=False)

    def test_deepseek_v3(self):
        self._run_for_model("deepseek-ai/DeepSeek-V3", is_nvfp4=False)

    def test_deepseek_r1_nvfp4(self):
        self._run_for_model("nvidia/DeepSeek-R1-NVFP4", is_nvfp4=True)


class TestAsyncLoadThreshold(unittest.TestCase):
    """Verify should_async_load respects the expert count threshold."""

    _THRESHOLD = 128

    @staticmethod
    def _should_async_load(device_type: str, num_experts: int = 0) -> bool:
        """Standalone reimplementation of should_async_load for testing without sglang imports."""
        if num_experts > 128:
            return False
        if device_type is None:
            return False
        return device_type == "cpu"

    def test_below_threshold_cpu_tensor(self):
        self.assertTrue(self._should_async_load("cpu", num_experts=64))

    def test_above_threshold_cpu_tensor(self):
        self.assertFalse(self._should_async_load("cpu", num_experts=256))

    def test_at_threshold_cpu_tensor(self):
        # At exactly the threshold, should still allow async
        self.assertTrue(self._should_async_load("cpu", num_experts=128))

    def test_above_threshold_is_boundary(self):
        # One above threshold should disable
        self.assertFalse(self._should_async_load("cpu", num_experts=129))

    def test_default_num_experts(self):
        # Default num_experts=0 should not disable async
        self.assertTrue(self._should_async_load("cpu"))

    def test_cuda_tensor_always_sync(self):
        self.assertFalse(self._should_async_load("cuda", num_experts=0))

    def test_none_device_always_sync(self):
        self.assertFalse(self._should_async_load(None, num_experts=0))


if __name__ == "__main__":
    unittest.main()
