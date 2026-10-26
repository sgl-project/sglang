"""Simulator-only CPU DSPark argument validation; imports no device runtime."""

import functools
import importlib
import os
import re
from collections.abc import Mapping


def _enabled():
    return (
        os.environ.get("SGLANG_SIMULATOR_BOOTSTRAP") == "1"
        and os.environ.get("SGLANG_USE_CPU_ENGINE") == "1"
    )


def _require(condition, message):
    if not condition:
        raise ValueError("CPU DSPark simulator " + message)


def _field(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _validate(native, server_args, cfg):
    # Validate everything before appending even one native declaration.
    _require(cfg.speculative_algorithm == "DSPARK", "requires DSPARK")
    _require(
        type(cfg.speculative_dspark_block_size) is int
        and cfg.speculative_dspark_block_size == 5,
        "requires explicit draft Q5",
    )
    verify_width = cfg.speculative_num_draft_tokens
    _require(
        verify_width is None or type(verify_width) is int and verify_width == 6,
        "requires target Q6",
    )
    for name in ("speculative_num_steps", "speculative_eagle_topk"):
        value = getattr(cfg, name)
        _require(
            value is None or type(value) is int and value == 1, f"requires {name}=1"
        )
    for name in (
        "speculative_draft_model_path",
        "speculative_draft_model_revision",
    ):
        # Native dispatch can already have declared revision='main'. That is
        # not an explicit input pin. Never probe/download a checkpoint here.
        value = getattr(server_args, name, None)
        _require(
            type(value) is str and bool(value.strip()), f"requires explicit {name}"
        )
        _require(
            getattr(cfg, name, None) == value,
            f"does not support resolving another {name}",
        )
    _require(
        re.fullmatch("[0-9a-f]{40}", cfg.speculative_draft_model_revision) is not None,
        "requires a full lowercase draft revision commit",
    )
    max_running = getattr(server_args, "max_running_requests", None)
    _require(
        type(max_running) is int
        and max_running > 0
        and type(cfg.max_running_requests) is int
        and cfg.max_running_requests == max_running,
        "requires positive explicit max_running_requests",
    )
    from sglang_simulator.compat import (
        native_joint_reference_requested,
        native_reference_requested,
    )

    joint = native_joint_reference_requested()
    sizes = dict.fromkeys(
        (
            "tp_size",
            "ep_size",
            "dp_size",
            "pp_size",
            "attn_cp_size",
            "dcp_size",
            "nnodes",
        ),
        1,
    )
    if joint:
        sizes.update(tp_size=2, attn_dp_size=2)
    for name, expected in sizes.items():
        value = getattr(cfg, name, None)
        _require(
            type(value) is int and value == expected,
            f"requires physical {name}={expected}",
        )
    if joint:
        # TP2 / attention-DP2 / CP1 yields singleton attention; native DP1
        # still owns exactly two workers. Do not divide the request limit here.
        _require(getattr(cfg, "enable_dp_lm_head", None) is True, "requires DP LM head")
        _require(
            getattr(cfg, "enable_dp_attention_local_control_broadcast", None) is True,
            "requires native attention-DP local control broadcast",
        )
        _require(
            getattr(cfg, "moe_a2a_backend", None) == "none", "requires built-in TP MoE"
        )
        _require(
            getattr(cfg, "speculative_moe_a2a_backend", None) in (None, "none"),
            "requires the same draft TP MoE backend",
        )
        _require(
            getattr(cfg, "enable_two_batch_overlap", False) is False,
            "does not support two-batch overlap",
        )

    native_reference = native_reference_requested()
    _require(
        cfg.disable_overlap_schedule is (not native_reference),
        "requires source overlap for native regions"
        if native_reference
        else "requires overlap disabled",
    )
    _require(cfg.disable_cuda_graph is True, "requires physical CUDA graphs disabled")
    for phase in ("decode", "prefill"):
        backend = getattr(cfg, "cuda_graph_backend_" + phase, None)
        _require(
            backend is None or backend == "disabled",
            f"requires physical {phase} CUDA graphs disabled",
        )
        graph = getattr(cfg, "cuda_graph_config", None)
        if graph is not None:
            _require(
                _field(_field(graph, phase), "backend") == "disabled",
                f"requires effective physical {phase} CUDA graphs disabled",
            )
    _require(
        getattr(cfg, "disaggregation_mode", "null") == "null",
        "does not support disaggregation",
    )
    for name in (
        "enable_torch_compile",
        "enable_multi_layer_eagle",
        "speculative_adaptive",
        "speculative_adaptive_config",
        "speculative_dspark_align_verify_tokens_to_graph_tier",
        "speculative_dspark_sps_table_path",
        "speculative_dspark_confidence_sts_path",
        "speculative_token_map",
        "speculative_use_block_verification",
        "speculative_use_rejection_sampling",
        "enable_strict_thinking",
        "constrained_json_whitespace_pattern",
        "constrained_json_disable_any_whitespace",
    ):
        _require(not getattr(cfg, name, None), f"does not support {name}")
    if not joint:
        for name in ("enable_dp_attention", "enable_dp_lm_head"):
            _require(not getattr(cfg, name, None), f"does not support {name}")
    for name in (
        "speculative_draft_window_size",
        "speculative_draft_sink_size",
        "constrained_json_max_whitespace_cnt",
    ):
        _require(getattr(cfg, name, None) is None, f"does not support {name}")
    _require(
        getattr(cfg, "speculative_attention_mode", "prefill") == "prefill",
        "requires speculative_attention_mode='prefill'",
    )
    for name in (
        "speculative_accept_threshold_single",
        "speculative_accept_threshold_acc",
    ):
        value = getattr(cfg, name, 1.0)
        _require(type(value) in (float, int) and value == 1, f"does not support {name}")
    # The native pipeline declares xgrammar as an inert default before this
    # hook. An explicit backend or constrained request remains unsupported.
    _require(
        getattr(server_args, "grammar_backend", None) is None,
        "does not support an explicit grammar_backend",
    )
    _require(
        native.envs.SGLANG_RAGGED_VERIFY_MODE.get() == "static",
        "requires SGLANG_RAGGED_VERIFY_MODE=static",
    )
    _require(
        not native.envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get(),
        "does not support DP speculative prefill coordination",
    )
    _require(
        not native.envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER.get(),
        "does not support the confidence prefix scheduler",
    )
    _require(
        native.envs.SGLANG_SIMULATE_ACC_LEN.get() <= 0,
        "does not support native simulated acceptance",
    )
    _require(
        not os.environ.get("SGLANG_SIMULATOR_COORDINATOR") or native_reference,
        "requires native reference regions for the coordinator",
    )


def install_cpu_dspark_args():
    """Install once under both bootstrap flags, preserving every non-CPU call.

    Only the simulator's explicit static Q5/Q6 CPU path bypasses the native
    accelerator-only handler. Target topology/cost identity stays in the
    predictor configuration. Physical execution is single-rank by default;
    explicit joint source intent permits only the two-singleton TP2 domain.
    """
    if not _enabled():
        return False
    native = importlib.import_module("sglang.srt.arg_groups.speculative_hook")
    original = native._handle_dspark
    if getattr(original, "_sglang_simulator_cpu_dspark_args", False):
        return True

    @functools.wraps(original)
    def handle(server_args):
        if not _enabled():
            return original(server_args)
        cfg = native.resolving_view(server_args)
        if cfg.device != "cpu":
            return original(server_args)
        _validate(native, server_args, cfg)
        native.declare_resolution(
            server_args,
            "sglang_simulator.cpu_dspark_args",
            speculative_num_steps=1,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=6,
        )

    handle._sglang_simulator_cpu_dspark_args = True
    native._handle_dspark = handle
    return True
