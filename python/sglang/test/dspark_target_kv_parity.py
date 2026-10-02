"""Fixed-input bridge to the pinned SpecForge backbone, with no target decoder."""

from __future__ import annotations

import copy
import json
import socket
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import Qwen3Config

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.dspark_target_kv import DSparkTargetKVDraftModel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.speculative.dspark_components.dspark_target_kv_artifact import (
    audit_target_kv_checkpoint,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    SharedHeadTransform,
    read_target_kv_draft_contract,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_encoder import (
    TargetKVContextEncoder,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.training_capture.identity import local_safetensors_digest
from sglang.srt.training_capture.protocol import digest_bytes
from sglang.test.dspark_target_kv_utils import make_target_kv_contract
from sglang.test.training_capture_utils import make_snapshot


class SpecForgeKVReference(nn.Module):
    """Test adapter: the existing training backbone consumes encoded KV features."""

    def __init__(self, config, contract, *, attention_backend="eager"):
        super().__init__()
        from specforge.modeling.draft.dspark import DSparkDraftModel

        config = copy.deepcopy(config)
        config._attn_implementation = attention_backend
        config.dflash_config = {
            "projector_type": "dspark",
            "target_layer_ids": contract.kv.selected_layer_ids,
            "markov_rank": config.markov_rank,
            "markov_head_type": config.markov_head_type,
            "mask_token_id": contract.sequence.mask_token_id,
            "enable_confidence_head": False,
            "shift_label": True,
        }
        self.backbone = DSparkDraftModel(config)
        # SpecForge's production hidden-input projection is outside this bridge.
        self.backbone.fc = nn.Identity()
        self.backbone.hidden_norm = nn.Identity()
        self.backbone.kv_encoder = TargetKVContextEncoder(contract)
        self.transform = SharedHeadTransform.decode(contract.teacher.output_transform)

    def forward(self, *, tensors, anchor, embed, head, previous_tokens):
        contract = self.backbone.kv_encoder.contract
        width = contract.sequence.prediction_count
        device = embed.weight.device
        prefix = {
            name: value[:anchor].to(device).detach()
            for name, value in tensors.items()
            if name.startswith("target_")
        }
        positions = tensors["position_ids"][:anchor].to(device)
        encoded = self.backbone.kv_encoder(prefix, positions)
        block = torch.full((1, width), contract.sequence.mask_token_id, device=device)
        block[0, 0] = tensors["token_ids"][anchor]
        layers = []
        handles = [
            layer.register_forward_hook(lambda _, __, output: layers.append(output))
            for layer in self.backbone.layers
        ]
        try:
            hidden = self.backbone(
                position_ids=torch.cat(
                    (positions, torch.arange(anchor, anchor + width, device=device))
                )[None],
                target_hidden=encoded[None],
                noise_embedding=embed(block.long()),
                use_cache=False,
                is_causal=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        base = self.transform.apply(head(hidden))
        corrected = self.backbone.apply_logits_head(
            base, prev_token_ids=previous_tokens, hidden_states=hidden
        )
        return {
            "encoded": encoded,
            "layers": layers,
            "hidden": hidden,
            "base": base,
            "corrected": corrected,
        }


def make_parity_checkpoint(directory, head_type, dtype=torch.bfloat16):
    directory = Path(directory)
    directory.mkdir()
    contract = make_target_kv_contract()
    contract = msgspec.structs.replace(
        contract,
        encoder=msgspec.structs.replace(contract.encoder, hidden_size=128),
        teacher=msgspec.structs.replace(
            contract.teacher,
            output_transform=json.dumps(
                {"logit_scale": 0.7, "final_logit_softcapping": 2.0}
            ),
        ),
    )
    config = Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=256,
        max_position_embeddings=128,
        attention_dropout=0.0,
        num_target_layers=4,
        block_size=3,
        markov_rank=8,
        markov_head_type=head_type,
        enable_confidence_head=False,
        mask_token_id=255,
        architectures=["DSparkTargetKVDraftModel"],
        input_mode="target_kv",
        target_kv_contract=msgspec.to_builtins(contract),
        dtype=str(dtype).removeprefix("torch."),
    )
    torch.manual_seed(1729)
    with set_default_torch_dtype(dtype):
        reference = SpecForgeKVReference(config, contract)
        embed = nn.Embedding(config.vocab_size, config.hidden_size)
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    embed.weight.requires_grad_(False)
    head.weight.requires_grad_(False)
    config.save_pretrained(directory)
    save_file(reference.backbone.state_dict(), str(directory / "model.safetensors"))
    manifest, objects = make_snapshot(response_length=12)
    tensors = {
        name: torch.cat(
            [
                objects[obj.key]
                for obj in sorted(
                    (obj for obj in manifest.objects if obj.name == name),
                    key=lambda obj: obj.token_range[0] if obj.kind == "kv" else 0,
                )
            ]
        )
        for name in {obj.name for obj in manifest.objects}
    }
    return reference, embed, head, tensors


class ServingKVParityRunner:
    """Real serving layers, fused context writer and Triton paged attention."""

    def __init__(
        self, directory, *, embed, head, dtype=torch.bfloat16, context_capacity=256
    ):
        self.device = "cuda"
        self.row_stride = context_capacity * 3
        self.args = ServerArgs(
            model_path=str(directory),
            attention_backend="triton",
            cuda_graph_backend_decode="disabled",
            cuda_graph_backend_prefill="disabled",
        )
        set_global_server_args_for_scheduler(self.args)
        self.config = ModelConfig(
            str(directory),
            is_draft_model=True,
            speculative_algorithm="DSPARK",
            dtype=str(dtype).removeprefix("torch."),
        )
        self.contract = read_target_kv_draft_contract(self.config.hf_config)
        self.mapping = ReqToTokenPool(
            4, context_capacity, self.device, enable_memory_saver=False
        )
        self.pool = MHATokenToKVPool(
            4 * self.row_stride,
            1,
            dtype,
            self.config.hf_config.num_key_value_heads,
            self.config.head_dim,
            self.config.hf_config.num_hidden_layers,
            self.device,
            enable_memory_saver=False,
        )
        with set_default_torch_dtype(dtype):
            self.model = DSparkTargetKVDraftModel(self.config.hf_config).to(self.device)
        self.model.load_weights(
            load_file(str(Path(directory) / "model.safetensors")).items()
        )
        head.org_vocab_size = head.out_features
        self.model.attach_shared_modules(embed_tokens=embed, lm_head=head)
        runner = SimpleNamespace(
            req_to_token_pool=self.mapping,
            token_to_kv_pool=self.pool,
            token_to_kv_pool_allocator=None,
            sliding_window_size=None,
            model_config=self.config,
            server_args=self.args,
            device=self.device,
            gpu_id=0,
            is_draft_worker=True,
            page_size=1,
        )
        self.backend = TritonAttnBackend(runner)

    @torch.no_grad()
    def forward(self, *, tensors, anchors, previous_tokens):
        width = self.contract.sequence.prediction_count
        blocks, locations, positions = [], [], []
        for row, anchor in enumerate(anchors):
            # Deliberately use non-contiguous slots and distinct request rows.
            slots = (
                torch.arange(anchor + width, device=self.device) * 3
                + row * self.row_stride
                + 1
            )
            self.mapping.req_to_token[row + 1, : len(slots)] = slots.to(torch.int32)
            self.model.write_target_kv(
                target_kv={
                    name: value[:anchor].to(self.device)
                    for name, value in tensors.items()
                    if name.startswith("target_")
                },
                pool=self.pool,
                positions=tensors["position_ids"][:anchor].to(self.device),
                cache_loc=slots[:anchor],
            )
            block = torch.full(
                (width,),
                self.contract.sequence.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            block[0] = tensors["token_ids"][anchor]
            blocks.append(block)
            locations.append(slots[anchor:])
            positions.append(torch.arange(anchor, anchor + width, device=self.device))
        batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=len(anchors),
            input_ids=torch.cat(blocks),
            req_pool_indices=torch.arange(1, len(anchors) + 1, device=self.device),
            seq_lens=torch.tensor(anchors, dtype=torch.int32, device=self.device),
            out_cache_loc=torch.cat(locations),
            seq_lens_sum=sum(anchors) + len(anchors) * width,
            seq_lens_cpu=torch.tensor(anchors) + width,
            positions=torch.cat(positions),
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=make_draft_block_spec_info(
                draft_token_num=width, device=self.device
            ),
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self.backend.init_forward_metadata(batch)
        layers = []
        handles = [
            layer.register_forward_hook(
                lambda _, __, output: layers.append((output[0] + output[1]).clone())
            )
            for layer in self.model.layers
        ]
        try:
            with forward_context(ForwardContext(attn_backend=self.backend)):
                hidden = self.model(
                    batch.input_ids, batch.positions, batch
                ).hidden_states
        finally:
            for handle in handles:
                handle.remove()
        hidden = hidden.view(len(anchors), width, -1)
        base, _ = self.model.compute_base_logits(hidden)
        corrected = self.model.markov_head.apply_block_logits(
            base,
            token_ids=previous_tokens,
            hidden_states=hidden,
        )
        return {
            "layers": [value.view(len(anchors), width, -1) for value in layers],
            "hidden": hidden,
            "base": base,
            "corrected": corrected,
        }


@contextmanager
def single_gpu_parity_context():
    with patch.dict("os.environ", {"SGLANG_RAGGED_VERIFY_MODE": "static"}):
        torch.cuda.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                backend="nccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


class FixedInputParityError(AssertionError):
    def __init__(self, report):
        self.report = report
        failures = [
            name for name, stage in report["stages"].items() if not stage["passed"]
        ]
        super().__init__(f"fixed-input parity failed: {', '.join(failures)}")


def compare_parity_outputs(actual, expected, *, rtol, atol):
    """Retain every failing stage; nonfinite values never certify parity."""
    stages = {}
    pairs = {
        **{
            f"layer.{index}": (
                value,
                torch.cat([row["layers"][index] for row in expected]),
            )
            for index, value in enumerate(actual["layers"])
        },
        **{
            key: (actual[key], torch.cat([row[key] for row in expected]))
            for key in ("hidden", "base", "corrected")
        },
    }
    for name, (value, wanted) in pairs.items():
        if value.shape != wanted.shape or value.dtype != wanted.dtype:
            stages[name] = {"passed": False, "error": "shape or dtype mismatch"}
            continue
        error = (value.detach().float() - wanted.detach().float()).abs()
        finite = torch.isfinite(error)
        mismatched = ~(finite & (error <= atol + rtol * wanted.detach().float().abs()))
        stages[name] = {
            "passed": not mismatched.any().item(),
            "elements": error.numel(),
            "mismatched": mismatched.sum().item(),
            "nonfinite": (~finite).sum().item(),
            "max_abs": error.max().item() if finite.all() else None,
            "rms": error.square().mean().sqrt().item() if finite.all() else None,
        }
        print(json.dumps({"parity_stage": name, **stages[name]}), flush=True)
    report = {
        "rtol": rtol,
        "atol": atol,
        "stages": stages,
        "max_abs": {name: stage.get("max_abs") for name, stage in stages.items()},
    }
    if not all(stage["passed"] for stage in stages.values()):
        raise FixedInputParityError(report)
    return report


def check_fixed_input_parity(
    reference, serving, embed, head, tensors, anchors, *, rtol=0.03, atol=0.03
):
    width = serving.contract.sequence.prediction_count
    previous = torch.zeros(len(anchors), width, dtype=torch.long, device="cuda")
    for row, anchor in enumerate(anchors):
        tokens = tensors["token_ids"][anchor : anchor + width]
        previous[row, : len(tokens)] = tokens.to("cuda").long()
    expected = [
        reference(
            tensors=tensors,
            anchor=anchor,
            embed=embed,
            head=head,
            previous_tokens=previous[row : row + 1],
        )
        for row, anchor in enumerate(anchors)
    ]
    actual = serving.forward(tensors=tensors, anchors=anchors, previous_tokens=previous)
    report = compare_parity_outputs(actual, expected, rtol=rtol, atol=atol)

    # Test-only CE+TV reference over saved teacher rows; partial final windows
    # select valid response rows before gathering IDs or normalizing logits.
    sums, count = [], 0
    for anchor, output in zip(anchors, expected, strict=True):
        positions = torch.arange(
            anchor + 1, min(anchor + width + 1, len(tensors["token_ids"]))
        )
        positions = positions[tensors["loss_mask"][positions].bool()]
        rows = torch.searchsorted(tensors["logits_positions"].long(), positions)
        torch.testing.assert_close(tensors["logits_positions"][rows].long(), positions)
        logits = output["corrected"][0, (positions - anchor - 1).cuda()].float()
        labels = tensors["token_ids"][positions].long().cuda()
        ids = tensors["teacher_topk_ids"][rows].long().cuda()
        probability = (
            (
                tensors["teacher_topk_logits"][rows].cuda()
                - tensors["teacher_logsumexp"][rows, None].cuda()
            )
            .exp()
            .detach()
        )
        log_z = logits.logsumexp(-1)
        ce = log_z - logits.gather(-1, labels[:, None]).squeeze(-1)
        tv = 0.5 * (
            probability - (logits.gather(-1, ids) - log_z[:, None]).exp()
        ).abs().sum(-1)
        sums.append((ce + serving.contract.training.lambda_tv * tv).sum())
        count += len(positions)
    assert count > 0
    loss = torch.stack(sums).sum() / count
    reference.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {}
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        norm = parameter.grad.float().abs().sum().item()
        assert norm > 0, name
        group = name.split(".")[1]
        gradients[group] = gradients.get(group, 0.0) + norm
    assert embed.weight.grad is None and head.weight.grad is None
    return {
        **report,
        "anchors": anchors,
        "valid_labels": count,
        "loss": loss.item(),
        "gradient_l1": gradients,
    }


def _validate_captured_checkpoint(directory, target_path, *, attention_backend):
    """Exercise a runtime fixture using only its saved KV/teacher and shared weights."""
    directory, target_path = Path(directory), Path(target_path)
    config = Qwen3Config.from_pretrained(directory, local_files_only=True)
    contract = read_target_kv_draft_contract(config)
    assert local_safetensors_digest(target_path) == contract.teacher.weights_revision
    fixture_path = directory / "validation" / "inputs.safetensors"
    assert (
        digest_bytes(fixture_path.read_bytes())
        == contract.validation.golden_fixture_sha256
    )
    tensors = load_file(str(fixture_path))
    with set_default_torch_dtype(torch.bfloat16):
        reference = SpecForgeKVReference(
            config, contract, attention_backend=attention_backend
        ).cuda()
        embed = nn.Embedding(config.vocab_size, config.hidden_size).cuda()
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).cuda()
    reference.backbone.load_state_dict(
        load_file(str(directory / "model.safetensors")), strict=True
    )
    shared = {}
    for path in target_path.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in ("model.embed_tokens.weight", "lm_head.weight"):
                if name in source.keys():  # noqa: SIM118 - safe_open is not iterable
                    shared[name] = source.get_tensor(name)
    target_config = Qwen3Config.from_pretrained(target_path, local_files_only=True)
    if target_config.tie_word_embeddings:
        shared["lm_head.weight"] = shared["model.embed_tokens.weight"]
    with torch.no_grad():
        embed.weight.copy_(shared["model.embed_tokens.weight"])
        head.weight.copy_(shared["lm_head.weight"])
    embed.weight.requires_grad_(False)
    head.weight.requires_grad_(False)
    serving = ServingKVParityRunner(
        directory,
        embed=embed,
        head=head,
        context_capacity=len(tensors["token_ids"]) + config.block_size,
    )
    first_response = int(tensors["logits_positions"][0])
    report = check_fixed_input_parity(
        reference,
        serving,
        embed,
        head,
        tensors,
        [first_response - 1, first_response, len(tensors["token_ids"]) - 2],
        rtol=contract.validation.parity_rtol,
        atol=contract.validation.parity_atol,
    )
    report["fixture_sha256"] = contract.validation.golden_fixture_sha256
    report["weights_sha256"] = digest_bytes(
        (directory / "model.safetensors").read_bytes()
    )
    report["reference_adapter"] = "specforge_backbone_with_shared_kv_encoder"
    report["reference_attention"] = attention_backend
    return report


def validate_captured_checkpoint(directory, target_path, *, attention_backend="eager"):
    """Write a success/failure report; an older pass cannot survive a failed rerun."""
    directory = Path(directory)
    report_path = directory / "validation" / "parity.json"
    report_path.unlink(missing_ok=True)
    from specforge.modeling.draft import dflash, dflash_kernels, dspark

    report = {
        "status": "running",
        "reference_adapter": "specforge_backbone_with_shared_kv_encoder",
        "reference_attention": attention_backend,
        "serving_attention": "triton",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "transformers_version": version("transformers"),
        "specforge_sources": {
            module.__name__: digest_bytes(Path(module.__file__).read_bytes())
            for module in (dflash, dflash_kernels, dspark)
        },
        "artifact_sha256": {
            name: digest_bytes((directory / name).read_bytes())
            for name in (
                "config.json",
                "model.safetensors",
                "validation/inputs.safetensors",
            )
        },
    }

    def write_report():
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(report_path)

    write_report()
    profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
    try:
        with (
            profile,
            patch(
                "transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward",
                side_effect=AssertionError(
                    "target decoder execution is forbidden in cached-data parity"
                ),
            ),
        ):
            report.update(
                _validate_captured_checkpoint(
                    directory, target_path, attention_backend=attention_backend
                )
            )
        report["status"] = "passed"
    except Exception as error:
        if isinstance(error, FixedInputParityError):
            report.update(error.report)
        report.update(status="failed", error=str(error))
        raise
    finally:
        # SDPA chooses among several kernels whose BF16 results can differ.
        report["observed_sdpa_operators"] = sorted(
            event.key
            for event in profile.key_averages()
            if "scaled_dot_product" in event.key
        )
        write_report()
    try:
        audit_target_kv_checkpoint(directory)
    except Exception as error:
        report.update(status="failed", error=f"artifact audit failed: {error}")
        write_report()
        raise
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument(
        "--reference-attention",
        choices=("eager", "sdpa", "flex_attention"),
        default="eager",
    )
    arguments = parser.parse_args()
    with single_gpu_parity_context():
        print(
            json.dumps(
                validate_captured_checkpoint(
                    arguments.checkpoint,
                    arguments.target_path,
                    attention_backend=arguments.reference_attention,
                )
            ),
            flush=True,
        )
