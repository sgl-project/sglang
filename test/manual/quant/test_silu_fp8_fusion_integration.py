"""Checkpoint-backed tests of the ModelOpt static-FP8 model entry points.

Set SGLANG_TEST_STATIC_FP8_MODEL to a local Qwen3-8B-FP8 checkpoint and select
free devices through CUDA_VISIBLE_DEVICES. From the repository root:
    SGLANG_TEST_STATIC_FP8_TP=1 SGLANG_TEST_REQUIRE_STATIC_FP8=1 python -m pytest -q -s \
        test/manual/quant/test_silu_fp8_fusion_integration.py

Repeat TP2/4 with sufficient GPUs; select "not breakable_inductor_field" there.
Missing checkpoint/hardware skips unless require mode is set. TP count is not a
runtime allowlist. Tests compare off/on tokens and selected/top-5 logprobs, and
associate fused kernels with Graph launches per rank. Worker factories below are
importable top-level functions, not separate pytest tests. Synthetic shared MLP
checks need the initialized worker and are not MoE checkpoint quality evidence.
"""

import json
import os
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def run_module_checks(real_mlp, target: Path):
    from sglang.kernels.ops.quantization.silu_and_mul_static_fp8 import (
        silu_and_mul_static_fp8,
    )
    from sglang.srt.distributed.parallel_state import graph_capture
    from sglang.srt.layers.quantization.fp8_utils import static_quant_fp8
    from sglang.srt.layers.quantization.modelopt_fp8_input import ModelOptFp8Input
    from sglang.srt.layers.quantization.silu_fp8_fusion import make_silu_fp8_fusion
    from sglang.srt.models.qwen2 import Qwen2MLP
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
    from sglang.srt.runtime_context import get_parallel

    rank = torch.distributed.get_rank()
    tp = get_parallel().tp_size
    result = dict(
        status="running",
        tp=tp,
        synthetic_weights=True,
        cases=[],
        producer_byte_checks=0,
        legacy_contract_checks=0,
        negative_checks=0,
        graph_checks=0,
    )

    def save():
        if rank == 0:
            (target / "module-tests.json").write_text(json.dumps(result, indent=2))

    def expect_error(fn):
        try:
            fn()
        except (ValueError, TypeError):
            result["negative_checks"] += 1
        else:
            raise AssertionError("invalid payload was accepted")

    original_dtype = torch.get_default_dtype()
    try:
        with (
            torch.inference_mode(),
            torch.random.fork_rng(devices=[torch.cuda.current_device()]),
            torch.device("cuda"),
        ):
            torch.manual_seed(9271)
            for dtype in (torch.bfloat16, torch.float16):
                torch.set_default_dtype(dtype)
                for cls, width in [
                    (Qwen2MoeMLP, 256),
                    (Qwen2MoeMLP, 512),
                    (Qwen2MoeMLP, 1024),
                    (Qwen2MLP, 12288),
                ]:
                    kwargs = dict(
                        hidden_size=4096,
                        intermediate_size=width * tp,
                        hidden_act="silu",
                        quant_config=real_mlp.down_proj.quant_method.quant_config,
                        prefix="validation.mlp",
                        allow_silu_fp8_quant=True,
                    )
                    if cls is Qwen2MoeMLP:
                        kwargs["reduce_results"] = False
                    mlp = cls(**kwargs)
                    for linear in (mlp.gate_up_proj, mlp.down_proj):
                        linear.weight.copy_(
                            torch.randn(linear.weight.shape, dtype=dtype).to(
                                torch.float8_e4m3fn
                            )
                        )
                        linear.weight_scale.fill_(0.01)
                        linear.input_scale.fill_(0.02)
                        linear.quant_method.process_weights_after_loading(linear)
                    down, fusion = mlp.down_proj, mlp._silu_fp8_fusion
                    assert fusion is not None
                    assert make_silu_fp8_fusion(down, allowed=False) is None
                    result["negative_checks"] += 1
                    for m in (1, 7, 16, 129, 512, 2048):
                        x = torch.randn(m, 4096, dtype=dtype)
                        gate, _ = mlp.gate_up_proj(x)
                        activated = mlp.act_fn(gate)
                        qref, _ = static_quant_fp8(
                            activated, down.input_scale, repeat_scale=True
                        )
                        value = fusion(gate)
                        assert isinstance(value, ModelOptFp8Input)
                        assert (
                            value.scale is down.input_scale
                            and value.orig_dtype == dtype
                        )
                        assert torch.equal(
                            value.qx.view(torch.uint8), qref.view(torch.uint8)
                        )
                        assert torch.equal(
                            value.row_scales, down.input_scale.expand(m, 1)
                        )
                        result["producer_byte_checks"] += 1
                        expected_down = down.quant_method.apply(down, activated)
                        # Patches are assertions against accidental second quantization,
                        # not alternate implementations of the consumer.
                        with patch(
                            "sglang.srt.layers.quantization.fp8_utils.static_quant_fp8",
                            side_effect=AssertionError("double quantization"),
                        ):
                            for payload in (
                                value,
                                (value.qx, value.scale),
                                (value.qx, value.scale, dtype),
                            ):
                                actual_down = down.quant_method.apply(down, payload)
                                assert torch.equal(actual_down, expected_down)
                                result["legacy_contract_checks"] += 1
                        mlp._silu_fp8_fusion = None
                        expected = mlp(x)
                        mlp._silu_fp8_fusion = fusion
                        actual = mlp(x)
                        assert torch.equal(actual, expected), (
                            cls.__name__,
                            dtype,
                            m,
                            width,
                        )
                        assert torch.isfinite(actual).all()
                        if m in (1, 129, 2048):
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                mlp(x)
                            torch.cuda.current_stream().wait_stream(stream)
                            graph = torch.cuda.CUDAGraph()
                            # SGLang must register graph collective buffers and
                            # enter its graph communicator modes for TP>1.
                            with graph_capture(stream=stream):
                                with torch.cuda.graph(graph, stream=stream):
                                    captured = mlp(x)
                            for _ in range(2):
                                x.copy_(torch.randn_like(x))
                                down.input_scale.mul_(1.0625)
                                mlp._silu_fp8_fusion = None
                                expected = mlp(x)
                                mlp._silu_fp8_fusion = fusion
                                graph.replay()
                                torch.cuda.synchronize()
                                assert torch.equal(captured, expected)
                                result["graph_checks"] += 1
                            del graph, captured
                        result["cases"].append(
                            dict(cls=cls.__name__, dtype=str(dtype), m=m, local_k=width)
                        )
                        save()
                    expect_error(lambda: down.quant_method.apply(down, value.qx))
                    expect_error(
                        lambda: down.quant_method.apply(
                            down, (value.qx, value.scale.clone())
                        )
                    )
                    expect_error(
                        lambda: down.quant_method.apply(
                            down, (value.qx, value.scale, torch.float32)
                        )
                    )
                    expect_error(lambda: down.quant_method.apply(down, (value.qx,)))
                    expect_error(
                        lambda: down.quant_method.apply(
                            down,
                            ModelOptFp8Input(
                                value.qx,
                                value.scale,
                                dtype,
                                torch.empty(1, device="cuda"),
                            ),
                        )
                    )
                    assert fusion(gate[:, ::2]) is None
                    result["negative_checks"] += 1
                    unaligned = torch.empty(gate.numel() + 1, dtype=dtype)[1:].view_as(
                        gate
                    )
                    assert unaligned.is_contiguous()
                    assert fusion(unaligned) is None
                    result["negative_checks"] += 1
                    # Actual same-class caller without explicit permission stays off.
                    kwargs["allow_silu_fp8_quant"] = False
                    unauthorized = cls(**kwargs)
                    assert unauthorized._silu_fp8_fusion is None
                    result["negative_checks"] += 1
                    del unauthorized, mlp, fusion
                q, rows = silu_and_mul_static_fp8(
                    torch.empty(0, 32, dtype=dtype), torch.ones(1, dtype=torch.float32)
                )
                assert q.shape == (0, 16) and rows.shape == (0, 1)
        result["status"] = "passed"
        print(
            "PRODUCTION_MODULE_TESTS_PASSED", tp, rank, len(result["cases"]), flush=True
        )
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        torch.set_default_dtype(original_dtype)
        save()


def run_boundary_checks(target):
    from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
    from sglang.srt.layers.quantization.silu_fp8_fusion import (
        make_silu_fp8_fusion,
        silu_static_fp8_supported,
    )
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM
    from sglang.srt.runtime_context import get_exec, get_lora, get_parallel, get_spec

    result = dict(status="running", synthetic=True, checks=[])
    dtype = torch.get_default_dtype()
    try:
        with torch.inference_mode(), torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            kw = dict(
                vocab_size=128,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=2,
                num_attention_heads=8,
                num_key_value_heads=4,
                head_dim=64,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_num_key_heads=4,
                linear_num_value_heads=8,
                full_attention_interval=2,
                moe_intermediate_size=512,
                shared_expert_intermediate_size=256,
                num_experts=8,
                num_experts_per_tok=2,
                rope_parameters=dict(rope_type="default", rope_theta=10000000.0),
            )
            quant = ModelOptFp8Config(
                is_checkpoint_fp8_serialized=True, packed_modules_mapping={}
            )
            config = Qwen3_5MoeTextConfig(**kw)
            model = Qwen3_5MoeForCausalLM(config, quant, prefix="boundary")
            assert len(model.layers) == 2
            for layer in model.layers:
                shared = layer.mlp.shared_expert
                assert shared is not None and shared._silu_fp8_fusion is not None
                assert not layer.mlp.enable_shared_expert_fusion
                for linear in (shared.gate_up_proj, shared.down_proj):
                    linear.weight.copy_(
                        torch.randn_like(linear.weight, dtype=torch.bfloat16).to(
                            torch.float8_e4m3fn
                        )
                    )
                    linear.weight_scale.fill_(0.01)
                    linear.input_scale.fill_(0.02)
                    linear.quant_method.process_weights_after_loading(linear)
                x = torch.randn(129, 512, dtype=torch.bfloat16)
                fusion = shared._silu_fp8_fusion
                shared._silu_fp8_fusion = None
                expected = shared(x)
                shared._silu_fp8_fusion = fusion
                actual = shared(x)
                assert torch.equal(expected, actual)
                result["checks"].append(
                    type(layer).__name__ + ": actual standalone shared forward exact"
                )
            down = model.layers[0].mlp.shared_expert.down_proj
            # Policy-only test: no TP8 process group or collective is executed.
            # A larger TP value alone must not reject otherwise valid local data.
            with get_parallel().override(tp_size=8), patch.object(down, "tp_size", 8):
                assert make_silu_fp8_fusion(down, allowed=True) is not None
                with patch.object(down, "input_size_per_partition", 17):
                    assert make_silu_fp8_fusion(down, allowed=True) is None
            result["checks"].append(
                "TP count is not a policy allowlist; local alignment remains required"
            )
            for bag, change in (
                # ParallelContext.override accepts topology fields only;
                # these two flags are leaves of its published config bag.
                (get_parallel()._config, dict(enable_layernorm_sp=True)),
                (get_parallel()._config, dict(enable_dp_attention=True)),
                (get_lora(), dict(enable_lora=True)),
                (get_spec(), dict(speculative_algorithm="EAGLE")),
                (get_exec().deterministic, dict(enable_deterministic_inference=True)),
                (get_exec().graph, dict(enable_torch_compile=True)),
            ):
                with bag.override(**change):
                    assert make_silu_fp8_fusion(down, allowed=True) is None
                result["checks"].append("config fallback: " + next(iter(change)))
            for obj, field, value in (
                (down.quant_method, "use_marlin", True),
                (down.quant_method, "_static_fp8_sm89", False),
                (down.quant_method, "enable_flashinfer_bmm", True),
                (down, "input_is_parallel", False),
                (down, "use_decode_attn_tp", True),
            ):
                with patch.object(obj, field, value):
                    assert make_silu_fp8_fusion(down, allowed=True) is None
                result["checks"].append("metadata fallback: " + field)
            with envs.SGLANG_ENABLE_SILU_STATIC_FP8_FUSION.override(False):
                assert make_silu_fp8_fusion(down, allowed=True) is None
            result["checks"].append("global off")
            # Capability is checked at initialization, once; no compiler-version
            # whitelist or compiler mutation is involved in this fallback test.
            silu_static_fp8_supported.cache_clear()
            with patch(
                "sglang.srt.layers.quantization.fp8_utils.cutlass_fp8_supported",
                return_value=False,
            ):
                assert not silu_static_fp8_supported()
                assert make_silu_fp8_fusion(down, allowed=True) is None
            silu_static_fp8_supported.cache_clear()
            assert silu_static_fp8_supported()
            result["checks"].append("unsupported consumer capability fallback")
            del model

            class UnapprovedSubclass(Qwen3_5MoeForCausalLM):
                pass

            for name, cls, cfg, extras in (
                ("generic base with MoE config", Qwen3_5ForCausalLM, config, {}),
                ("unapproved subclass", UnapprovedSubclass, config, {}),
                (
                    "nextn",
                    Qwen3_5ForCausalLM,
                    config,
                    dict(is_nextn=True, allow_silu_fp8_quant=True),
                ),
                ("Qwen3.5 dense", Qwen3_5ForCausalLM, Qwen3_5TextConfig(**kw), {}),
            ):
                model = cls(cfg, quant, prefix="boundary.negative", **extras)
                for layer in model.layers:
                    mlp = getattr(layer.mlp, "shared_expert", layer.mlp)
                    assert mlp._silu_fp8_fusion is None
                result["checks"].append(name + ": no implicit authorization")
                del model
            already_fused = Qwen2MoeSparseMoeBlock(
                0,
                Qwen3_5MoeTextConfig(**dict(kw, shared_expert_intermediate_size=512)),
                quant,
                prefix="boundary.fused",
                support_shared_expert_fusion=True,
                enable_cuda_shared_expert_fusion=True,
                allow_silu_fp8_quant=True,
            )
            assert (
                already_fused.enable_shared_expert_fusion
                and already_fused.shared_expert is None
            )
            result["checks"].append("already routed-fused shared expert remains absent")
        result["status"] = "passed"
        print("PRODUCTION_BOUNDARIES_PASSED", len(result["checks"]), flush=True)
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        torch.set_default_dtype(dtype)
        (target / "boundary-tests.json").write_text(json.dumps(result, indent=2))


def make_boundary_hook(config):
    # Factories execute during initialized worker setup, outside forward.
    # Constructor-only decisions must never be tested inside a forward hook.
    run_boundary_checks(Path(config["target"]))
    return None


def make_mlp_hook(config):
    done = False
    target = Path(config["target"])

    def hook(module, args, output):
        nonlocal done
        if done or torch.cuda.is_current_stream_capturing():
            return
        done = True
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            # Module TP tests run on all ranks; only rank zero writes evidence.
            if config["module_tests"]:
                run_module_checks(module, target)
            return
        record = dict(
            fusion_present=module._silu_fp8_fusion is not None,
            module_type=type(module).__name__,
            consumer_type=type(module.down_proj.quant_method).__name__,
        )
        (target / "entrypoint.json").write_text(json.dumps(record, indent=2))
        assert record["fusion_present"] == (config["flag"] == "1"), record
        if config["module_tests"]:
            run_module_checks(module, target)

    return hook


def make_consumer_hook(config):
    done = False

    def hook(module, args, output):
        nonlocal done
        if done or torch.cuda.is_current_stream_capturing():
            return
        done = True
        from sglang.srt.layers.quantization.modelopt_fp8_input import ModelOptFp8Input

        value = args[0]
        prequantized = isinstance(value, ModelOptFp8Input)
        assert prequantized == (config["flag"] == "1")
        if prequantized:
            assert value.scale is module.input_scale
            assert value.orig_dtype == output[0].dtype == module.orig_dtype
        record = dict(
            prequantized=prequantized,
            input_type=type(value).__name__,
            output_dtype=str(output[0].dtype),
        )
        if torch.distributed.get_rank() == 0:
            (Path(config["target"]) / "consumer.json").write_text(
                json.dumps(record, indent=2)
            )

    return hook


def assert_fusion_graph_replay(events, enabled):
    fused = [
        e
        for e in events
        if e.get("cat") == "kernel"
        and "silu_and_mul_static_fp8_kernel" in e.get("name", "")
    ]
    launches = [
        e
        for e in events
        if e.get("cat") == "cuda_runtime" and "cudaGraphLaunch" in e.get("name", "")
    ]
    correlations = {
        e["args"]["correlation"]
        for e in launches
        if e.get("args", {}).get("correlation") is not None
    }
    assert bool(fused) == enabled
    assert launches, "No actual CUDA Graph replay observed"
    for event in fused:
        args = event.get("args", {})
        assert args.get("graph id", 0) > 0, "Fused kernel is not in a CUDA Graph"
        assert args.get("correlation") in correlations, (
            "Fused kernel is not associated with a CUDA Graph launch"
        )
    return dict(fused=len(fused), graph_launches=len(launches), graph_fused=len(fused))


@pytest.mark.parametrize("graph_id,correlation", [(0, 7), (5, 8), (5, None)])
def test_reject_unassociated_fusion(graph_id, correlation):
    events = [
        dict(cat="cuda_runtime", name="cudaGraphLaunch", args=dict(correlation=7)),
        dict(
            cat="kernel",
            name="silu_and_mul_static_fp8_kernel",
            args={"graph id": graph_id, "correlation": correlation},
        ),
    ]
    with pytest.raises(AssertionError, match="Fused kernel"):
        assert_fusion_graph_replay(events, True)


def test_accept_associated_fusion():
    events = [
        dict(cat="cuda_runtime", name="cudaGraphLaunch", args=dict(correlation=7)),
        dict(
            cat="kernel",
            name="silu_and_mul_static_fp8_kernel",
            args={"graph id": 5, "correlation": 7},
        ),
    ]
    assert assert_fusion_graph_replay(events, True)["graph_fused"] == 1
    assert assert_fusion_graph_replay(events[:1], False)["graph_fused"] == 0


@pytest.fixture(scope="module")
def model_path():
    from sglang.srt.layers.quantization.silu_fp8_fusion import (
        silu_static_fp8_supported,
    )

    model = os.environ.get("SGLANG_TEST_STATIC_FP8_MODEL")
    if not model or not Path(model).is_dir():
        if os.environ.get("SGLANG_TEST_REQUIRE_STATIC_FP8") == "1":
            pytest.fail("A local SGLANG_TEST_STATIC_FP8_MODEL checkpoint is required")
        pytest.skip(
            "Set SGLANG_TEST_STATIC_FP8_MODEL to a local Qwen3-8B-FP8 checkpoint"
        )
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (8, 9)
        or not silu_static_fp8_supported()
    ):
        if os.environ.get("SGLANG_TEST_REQUIRE_STATIC_FP8") == "1":
            pytest.fail("This run requires the qualified SM89 environment")
        pytest.skip("Requires the qualified SM89 environment")
    return model


def require_visible_gpus(tp):
    if torch.cuda.device_count() < tp:
        if os.environ.get("SGLANG_TEST_REQUIRE_STATIC_FP8") == "1":
            pytest.fail(f"Requires {tp} visible GPUs")
        pytest.skip(f"Requires {tp} visible GPUs")


@pytest.mark.parametrize("required", ["0", "1"])
def test_insufficient_gpu_contract(monkeypatch, required):
    monkeypatch.setenv("SGLANG_TEST_REQUIRE_STATIC_FP8", required)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    outcome = pytest.fail.Exception if required == "1" else pytest.skip.Exception
    with pytest.raises(outcome, match="Requires 2 visible GPUs"):
        require_visible_gpus(2)


@pytest.mark.parametrize("mode", ["eager", "default_graph", "breakable_inductor_field"])
def test_delivered_entrypoint(model_path, tmp_path, monkeypatch, mode):
    import gzip

    from transformers import AutoTokenizer

    import sglang

    tp = int(os.environ.get("SGLANG_TEST_STATIC_FP8_TP", "1"))
    assert tp > 0, "SGLANG_TEST_STATIC_FP8_TP must be positive"
    require_visible_gpus(tp)
    helpers = str(Path(__file__).parent.resolve())
    monkeypatch.setenv(
        "PYTHONPATH", helpers + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    inputs = []
    for length in (128, 2048):
        for index in range(4):
            seed = tokenizer.encode(
                f"Example {index}: Explain why addition and multiplication are useful. ",
                add_special_tokens=False,
            )
            inputs.append((seed * (length // len(seed) + 1))[:length])
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    results = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("SGLANG_ENABLE_SILU_STATIC_FP8_FUSION", flag)
        target = tmp_path / ("off" if flag == "0" else "on")
        target.mkdir()
        hooks = []
        if mode == "eager":
            hooks = [
                dict(
                    name="modules",
                    target_modules=["model.layers.0.mlp"],
                    hook_factory="test_silu_fp8_fusion_integration:make_mlp_hook",
                    config=dict(
                        target=str(target), flag=flag, module_tests=flag == "1"
                    ),
                ),
                dict(
                    name="consumer",
                    target_modules=["model.layers.0.mlp.down_proj"],
                    hook_factory="test_silu_fp8_fusion_integration:make_consumer_hook",
                    config=dict(target=str(target), flag=flag),
                ),
            ]
            if flag == "1" and tp == 1:
                hooks.insert(
                    0,
                    dict(
                        name="boundaries",
                        target_modules=["model.layers.0.mlp"],
                        hook_factory="test_silu_fp8_fusion_integration:make_boundary_hook",
                        config=dict(target=str(target)),
                    ),
                )
        engine = None
        outputs = []
        try:
            engine = sglang.Engine(
                model_path=model_path,
                tp_size=tp,
                dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                random_seed=9271,
                context_length=4096,
                max_running_requests=8,
                max_total_tokens=8192,
                mem_fraction_static=0.6,
                chunked_prefill_size=2048,
                disable_radix_cache=True,
                disable_cuda_graph=mode == "eager",
                cuda_graph_max_bs_decode=4,
                log_level="warning",
                forward_hooks=hooks,
                **(
                    {"cuda_graph_tc_compiler": "inductor"}
                    if mode == "breakable_inductor_field"
                    else {}
                ),
            )
            for ids in inputs:
                repeats = []
                for _ in range(2):
                    response = engine.generate(
                        input_ids=ids,
                        sampling_params=dict(
                            temperature=0, max_new_tokens=32, ignore_eos=True
                        ),
                        return_logprob=True,
                        top_logprobs_num=5,
                    )
                    meta = response["meta_info"]
                    tokens = [entry[1] for entry in meta["output_token_logprobs"]]
                    assert len(tokens) == meta["completion_tokens"] == 32
                    assert meta["prompt_tokens"] == len(ids)
                    repeats.append(
                        dict(
                            tokens=tokens,
                            logprobs=meta["output_token_logprobs"],
                            top_logprobs=meta["output_top_logprobs"],
                        )
                    )
                outputs.append(repeats)
                (target / "outputs.json").write_text(json.dumps(outputs, indent=2))
                assert repeats[0] == repeats[1], (
                    "Same-flag controlled generation changed"
                )
            if mode == "eager":
                assert json.loads((target / "entrypoint.json").read_text())[
                    "fusion_present"
                ] == (flag == "1")
                assert json.loads((target / "consumer.json").read_text())[
                    "prequantized"
                ] == (flag == "1")
                if flag == "1":
                    assert (
                        json.loads((target / "module-tests.json").read_text())["status"]
                        == "passed"
                    )
                    if tp == 1:
                        assert (
                            json.loads((target / "boundary-tests.json").read_text())[
                                "status"
                            ]
                            == "passed"
                        )
            else:
                # No numerical hooks in this model: independently observe actual
                # replay and the fused producer rather than just a Graph flag.
                engine.start_profile(
                    output_dir=str(target / "profile"),
                    activities=["CPU", "GPU"],
                    record_shapes=True,
                    profile_prefix="silu-static-fp8",
                )
                engine.generate(
                    input_ids=inputs[-1],
                    sampling_params=dict(
                        temperature=0, max_new_tokens=8, ignore_eos=True
                    ),
                )
                engine.stop_profile()
                traces = list((target / "profile").rglob("*.trace.json.gz"))
                assert len(traces) == tp
                observations = []
                for path in traces:
                    with gzip.open(path, "rt") as handle:
                        events = json.load(handle)["traceEvents"]
                    observations.append(
                        dict(
                            file=path.name,
                            **assert_fusion_graph_replay(events, flag == "1"),
                        )
                    )
                (target / "graph-observations.json").write_text(
                    json.dumps(observations, indent=2)
                )
        finally:
            if engine is not None:
                engine.shutdown()
        results[flag] = outputs
    assert results["0"] == results["1"], "Flag-off/on tokens or logprobs differ"
