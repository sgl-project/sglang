"""E2E test: source patcher + dumper + comparator on SGLang server.

Patches Qwen3MoeDecoderLayer.forward (and related methods) to insert
dumper.dump() calls at 7 points, launches servers with Qwen3-30B-A3B
(MOE model), runs inference, verifies patched dump fields exist, then
runs comparator to verify numerical consistency.

Test cases:
- test_patch_dump_and_compare: TP=2 baseline vs TP=4 target
- test_dp_attention: TP=2 baseline vs TP=2+DP=2+dp-attention target

The dumper.apply_source_patches() auto-injects ``from ... import dumper``
so the YAML only needs ``dumper.dump(...)`` calls.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import requests

pytestmark = pytest.mark.filterwarnings(
    "ignore:Unknown config option. asyncio_mode:pytest.PytestConfigWarning",
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="nightly-4-gpu", nightly=True)

MODEL = "Qwen/Qwen3-30B-A3B"
BASELINE_TP = 2
TARGET_TP = 4
EXP_NAME = "e2e_source_patcher"
DUMPER_FILTER = "layer_id in [0, 1, 2]"

_FIELDS_TO_VERIFY: list[str] = [
    # decoder layer level (aligned with miles)
    "layer_input",
    "attn_output",
    "pre_mlp_residual",
    "mlp_output",
    # attention internals
    "attn_pre_o_proj",
    # moe internals
    "moe_router_logits",
    "moe_expert_output",
]

PATCH_CONFIG_YAML: str = """\
patches:
  # --- decoder layer level (aligned with miles test) ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                  hidden_states,
                  residual,
                  forward_batch,
                  captured_last_layer_outputs=captured_last_layer_outputs,
                  **kwargs,
              )
          )
        append: "dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h[tp:partial]')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h[tp:partial]')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h[tp]')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial]')"
"""

PATCH_CONFIG_DP_ATTENTION_YAML: str = """\
patches:
  # --- decoder layer level (aligned with miles test) ---
  # In dp-attention mode: attn tensors are NOT TP-sharded (attn_tp_size=1),
  # and mlp_output is already all-reduced inside forward_normal().
  # layer_input is dumped after prepare_attn which DP-distributes tokens,
  # so it needs dp:=attn_dp to filter to the non-empty DP rank.
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                  hidden_states,
                  residual,
                  forward_batch,
                  captured_last_layer_outputs=captured_last_layer_outputs,
                  **kwargs,
              )
          )
        append: "dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated dp:=attn_dp')"
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h # tp:replicated')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h # tp:replicated')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[tp:partial]')"
"""


class TestSourcePatcherE2ESGLang:
    """E2E: patch Qwen3Moe forward -> dump -> compare."""

    def test_patch_dump_and_compare(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=4 target."""
        _run_e2e_scenario(
            tmp_path=tmp_path,
            target_tp=TARGET_TP,
        )

    def test_dp_attention(self, tmp_path: Path) -> None:
        """TP=2 baseline vs TP=2+DP=2+dp-attention target.

        In dp-attention mode (attn_tp_size=1, attn_dp_size=2), attention
        tensors are NOT TP-sharded and mlp_output is already all-reduced.
        A separate patch config with corrected dims is used for the target.
        """
        _run_e2e_scenario(
            tmp_path=tmp_path,
            target_tp=BASELINE_TP,
            extra_target_server_args=["--dp", "2", "--enable-dp-attention"],
            target_patch_config_yaml=PATCH_CONFIG_DP_ATTENTION_YAML,
        )


# --------------------------------- helpers ---------------------------------


def _run_e2e_scenario(
    *,
    tmp_path: Path,
    target_tp: int,
    extra_target_server_args: Optional[list[str]] = None,
    target_patch_config_yaml: Optional[str] = None,
) -> None:
    """Full e2e: write patch config -> baseline run -> target run -> compare."""
    base_url: str = DEFAULT_URL_FOR_TEST

    baseline_config_path: Path = tmp_path / "patch_config.yaml"
    baseline_config_path.write_text(PATCH_CONFIG_YAML)

    target_config_path: Path = tmp_path / "patch_config_target.yaml"
    target_config_path.write_text(target_patch_config_yaml or PATCH_CONFIG_YAML)

    baseline_dir: Path = tmp_path / "baseline"
    _run_server_and_generate(
        dump_dir=baseline_dir,
        config_path=baseline_config_path,
        tp=BASELINE_TP,
        base_url=base_url,
    )
    _verify_patched_fields(dump_dir=baseline_dir, field_names=_FIELDS_TO_VERIFY)

    target_dir: Path = tmp_path / "target"
    _run_server_and_generate(
        dump_dir=target_dir,
        config_path=target_config_path,
        tp=target_tp,
        base_url=base_url,
        extra_server_args=extra_target_server_args,
    )
    _verify_patched_fields(dump_dir=target_dir, field_names=_FIELDS_TO_VERIFY)

    baseline_exp: Path = baseline_dir / EXP_NAME
    target_exp: Path = target_dir / EXP_NAME

    cmd: list[str] = [
        "python",
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_exp),
        "--target-path",
        str(target_exp),
        "--output-format",
        "json",
        "--allow-skipped-pattern",
        "input_ids|positions",
    ]

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    debug_file: Path = _save_comparator_output(
        stdout=result.stdout, stderr=result.stderr
    )
    print(f"Comparator debug output: {debug_file}")

    assert result.returncode == 0, (
        f"Comparator failed (rc={result.returncode}). " f"Debug output: {debug_file}"
    )


def _run_server_and_generate(
    *,
    dump_dir: Path,
    config_path: Path,
    tp: int,
    base_url: str,
    extra_server_args: Optional[list[str]] = None,
) -> None:
    """Launch SGLang server with source patcher + dumper, send a generate request."""
    env: dict[str, str] = {
        **os.environ,
        "DUMPER_SOURCE_PATCHER_CONFIG": str(config_path),
        "DUMPER_DIR": str(dump_dir),
        "DUMPER_EXP_NAME": EXP_NAME,
        "DUMPER_SERVER_PORT": "reuse",
    }

    server_args: list[str] = [
        "--tp",
        str(tp),
        "--max-total-tokens",
        "128",
        "--mem-fraction-static",
        "0.5",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]
    if extra_server_args:
        server_args.extend(extra_server_args)

    proc = popen_launch_server(
        MODEL,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=server_args,
        env=env,
    )
    try:
        requests.post(
            f"{base_url}/dumper/configure",
            json={
                "enable": True,
                "filter": DUMPER_FILTER,
                "cleanup_previous": True,
            },
        ).raise_for_status()

        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
        )
        assert resp.status_code == 200, f"Generate failed: {resp.text}"
    finally:
        kill_process_tree(proc.pid)


def _verify_patched_fields(*, dump_dir: Path, field_names: list[str]) -> None:
    """Verify that patched dump fields exist as .pt files."""
    for field in field_names:
        matches: list[Path] = list(dump_dir.rglob(f"*name={field}*.pt"))
        assert len(matches) > 0, (
            f"Expected patched field '{field}' not found under {dump_dir}. "
            f"Available files: {sorted(f.name for f in dump_dir.rglob('*.pt'))[:20]}"
        )


def _save_comparator_output(*, stdout: str, stderr: str) -> Path:
    """Save comparator stdout+stderr to a temp file that persists for debugging."""
    fd, path_str = tempfile.mkstemp(prefix="comparator_e2e_", suffix=".log", dir="/tmp")
    with os.fdopen(fd, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n=== STDERR ===\n")
        f.write(stderr)
    return Path(path_str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
