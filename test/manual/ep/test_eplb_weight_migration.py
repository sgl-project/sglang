"""Bounded EPLB migration check for CPU/Gloo or Ascend/HCCL.

Example: torchrun --standalone --nproc-per-node=2 \
    test/manual/ep/test_eplb_weight_migration.py --device npu --weight-format nd

Add --w8a8-postprocess to execute the real NPU W8A8 post-load processor and
DeepSeek EPLB parameter registration on synthetic tensors. No model is loaded.
Add --no-fused-shared to cover full expert tensors, including unchanged slots
next to the offset-zero slice that exposed NZ-to-NZ storage-wide copies.
"""

import argparse
import datetime
import json
import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


def _metadata(mapping, rank):
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    num_logical = max(mapping) + 1
    physical = torch.tensor([mapping, mapping], device=_DEVICE)
    inverse = torch.full((2, num_logical, len(mapping)), -1, device=_DEVICE)
    for expert in range(num_logical):
        slots = [index for index, value in enumerate(mapping) if value == expert]
        inverse[:, expert, : len(slots)] = torch.tensor(slots, device=_DEVICE)
    return ExpertLocationMetadata._init_raw(
        ep_size=2,
        physical_to_logical_map=physical,
        logical_to_all_physical_map=inverse,
        moe_ep_rank=rank,
    )


def _expert_tensor(expert, layer, kind):
    # Values encode both the logical expert and positions inside each tensor.
    if kind < 2:
        values = torch.arange(512).reshape(16, 32)
        return ((values + expert * 7 + layer * 11 + kind) % 127).to(torch.int8)
    values = torch.arange(32).to(torch.float32) / 32
    values = values + expert + layer * 16 + kind
    return values.to(torch.bfloat16 if kind == 2 else torch.float32)


def _processed_weights(initial, rank, layer, device, weight_format, shared_ids):
    with ExitStack() as stack:
        if device == "cpu":
            # Avoid querying hardware architecture while importing the model.
            stack.enter_context(
                patch(
                    "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                    return_value=False,
                )
            )
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        get_weights = DeepseekV2MoE.get_moe_weights
        from sglang.srt.hardware_backend.npu.quantization import moe_methods

        experts = torch.nn.Module()
        experts.num_local_experts = 4 + len(shared_ids)
        ids = initial[rank * 4 : (rank + 1) * 4] + shared_ids
        names, kinds = [], []
        for weight_kind, prefix in enumerate(("w13", "w2")):
            for suffix, kind in (
                ("weight", weight_kind),
                ("weight_scale", 2),
                ("weight_offset", 3),
            ):
                tensor = torch.stack(
                    [_expert_tensor(expert, layer, kind) for expert in ids]
                )
                if kind < 2:
                    tensor = tensor.transpose(1, 2).contiguous()
                else:
                    tensor = tensor.float().unsqueeze(-1)
                name = prefix + "_" + suffix
                experts.register_parameter(
                    name, torch.nn.Parameter(tensor.to(device), requires_grad=False)
                )
                names.append(name)
                kinds.append(kind)
        experts.correction_bias = torch.nn.Parameter(
            torch.ones(5, device=device), requires_grad=False
        )
        if weight_format == "nd":
            stack.enter_context(
                patch.object(
                    moe_methods, "npu_format_cast", side_effect=lambda tensor: tensor
                )
            )
        else:
            import torch_npu

            stack.enter_context(
                patch.object(
                    moe_methods,
                    "npu_format_cast",
                    side_effect=lambda tensor: torch_npu.npu_format_cast(tensor, 29),
                )
            )
        method = (
            object.__new__(moe_methods.NPUW8A8Int8MoEMethod)
            if device == "cpu"
            else moe_methods.NPUW8A8Int8MoEMethod()
        )
        for prefix in ("w13", "w2"):
            method.process_weights_after_loading(experts, prefix)
        routed = get_weights(
            SimpleNamespace(experts=experts, num_fused_shared_experts=len(shared_ids))
        )
        full = [getattr(experts, name).data for name in names]
        assert len(routed) == len(full) == 6
        assert all(
            view.data_ptr() == tensor.data_ptr()
            for view, tensor in zip(routed, full, strict=True)
        )
        return full, routed, kinds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--weight-format", choices=("nd", "nz"), default="nd")
    parser.add_argument("--w8a8-postprocess", action="store_true")
    parser.add_argument("--no-fused-shared", action="store_true")
    parser.add_argument("--all-layers-per-update", action="store_true")
    args = parser.parse_args()
    shared_ids = [] if args.no_fused_shared else [31]
    if args.device == "cpu" and args.weight_format != "nd":
        parser.error("NZ requires --device npu")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("launch with torchrun --nproc-per-node=2")
    rank = int(os.environ["RANK"])
    if args.device == "npu":
        import torch_npu

        # format_cast otherwise silently falls back to ND in the default
        # standalone environment, unlike SGLang's NPU backend initialization.
        torch.npu.config.allow_internal_format = True
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    global _DEVICE
    _DEVICE = args.device

    from sglang.srt.eplb.expert_location import set_global_expert_location_metadata
    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    publish(
        ServerArgs(
            model_path="dummy",
            device=args.device,
            tp_size=2,
            ep_size=2,
            ep_dispatch_algorithm="static",
        ),
        role="test",
    )
    dist.init_process_group(
        backend="hccl" if args.device == "npu" else "gloo",
        timeout=datetime.timedelta(seconds=60),
    )
    updates = 0
    try:
        for initial, layouts in (
            (
                list(range(8)),
                [[7, 0, 4, 3, 2, 5, 6, 1], [4, 5, 6, 7, 0, 1, 2, 3], list(range(8))],
            ),
            (
                [0, 0, 1, 1, 2, 2, 3, 3],
                [
                    [2, 0, 2, 1, 3, 1, 0, 3],
                    [3, 3, 2, 2, 1, 1, 0, 0],
                    [0, 0, 1, 1, 2, 2, 3, 3],
                ],
            ),
        ):
            current = _metadata(initial, rank)
            set_global_expert_location_metadata(current, allow_overwrite=True)
            map_pointers = {
                name: value.data_ptr()
                for name, value in vars(current).items()
                if isinstance(value, torch.Tensor)
            }
            full_weights = {}
            routed = {}
            kinds = list(range(4))
            for layer in range(2):
                if args.w8a8_postprocess:
                    full_weights[layer], routed[layer], kinds = _processed_weights(
                        initial,
                        rank,
                        layer,
                        args.device,
                        args.weight_format,
                        shared_ids,
                    )
                    continue
                full_weights[layer] = []
                for kind in range(4):
                    # The final slot represents a fused shared expert. EPLB
                    # receives only the routed view and must preserve this slot.
                    tensor = torch.stack(
                        [
                            _expert_tensor(expert, layer, kind)
                            for expert in initial[rank * 4 : (rank + 1) * 4]
                            + shared_ids
                        ]
                    ).to(_DEVICE)
                    if args.weight_format == "nz" and kind < 2:
                        tensor = torch_npu.npu_format_cast(tensor, 29)
                    full_weights[layer].append(tensor)
                routed[layer] = [tensor[:4] for tensor in full_weights[layer]]
            weight_pointers = [
                [tensor.data_ptr() for tensor in tensors] for tensors in routed.values()
            ]
            if args.weight_format == "nz":
                for tensors in full_weights.values():
                    for kind, tensor in zip(kinds, tensors, strict=True):
                        if kind < 2:
                            assert torch_npu.get_npu_format(tensor) == 29
            updater = ExpertLocationUpdater()
            if args.device == "cpu":
                # The updater's first-use accelerator cache flush has no CPU
                # equivalent; all migration and metadata operations stay real.
                updater._first_execution = False
            expected_maps = [initial.copy(), initial.copy()]
            for layout in layouts:
                target = _metadata(layout, rank)
                layer_chunks = [[0, 1]] if args.all_layers_per_update else [[0], [1]]
                for layer_ids in layer_chunks:
                    missing = updater.update(
                        routed, target, layer_ids, nnodes=1, rank=rank
                    )
                    assert missing == {}, missing
                    for layer in layer_ids:
                        expected_maps[layer] = layout.copy()
                    for lid, tensors in full_weights.items():
                        for kind, tensor in zip(kinds, tensors, strict=True):
                            expected = torch.stack(
                                [
                                    _expert_tensor(expert, lid, kind)
                                    for expert in expected_maps[lid][
                                        rank * 4 : (rank + 1) * 4
                                    ]
                                    + shared_ids
                                ]
                            )
                            torch.testing.assert_close(
                                tensor.cpu(), expected, rtol=0, atol=0
                            )
                            if args.weight_format == "nz" and kind < 2:
                                assert torch_npu.get_npu_format(tensor) == 29
                    assert weight_pointers == [
                        [tensor.data_ptr() for tensor in tensors]
                        for tensors in routed.values()
                    ]
                    for name, pointer in map_pointers.items():
                        assert getattr(current, name).data_ptr() == pointer, name
                    torch.testing.assert_close(
                        current.physical_to_logical_map_cpu, torch.tensor(expected_maps)
                    )
                    if 1 in layer_ids:
                        for name in map_pointers:
                            torch.testing.assert_close(
                                getattr(current, name),
                                getattr(target, name),
                                rtol=0,
                                atol=0,
                            )
                    dist.barrier()
                    updates += len(layer_ids)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "device": args.device,
                        "weight_format": args.weight_format,
                        "world_size": 2,
                        "layer_updates": updates,
                        "model_loaded": False,
                        "processed_w8a8": args.w8a8_postprocess,
                        "fused_shared_experts": len(shared_ids),
                        "all_layers_per_update": args.all_layers_per_update,
                    }
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
