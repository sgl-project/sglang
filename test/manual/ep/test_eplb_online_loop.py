"""CPU/Gloo or NPU/HCCL loop with synthetic routing and real migration.

Run with torchrun --standalone --nproc-per-node=2 \
    test/manual/ep/test_eplb_online_loop.py

Add --rebalance-threshold 0.9 to also check that uniform routing eventually
skips migration with balancedness reporting enabled, as required upstream.

P/D roles run sequentially with separate configurations. This checks manager
behavior, not PD request transfer or device graph execution.
"""

import argparse
import datetime
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--rebalance-threshold", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.rebalance_threshold <= 1:
        parser.error("--rebalance-threshold must be in [0, 1]")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("launch with torchrun --nproc-per-node=2")
    rank = int(os.environ["RANK"])
    if args.device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))

    from sglang.srt.eplb.eplb_manager import EPLBManager
    from sglang.srt.eplb.expert_distribution import (
        ExpertDistributionRecorder,
        set_global_expert_distribution_recorder,
    )
    from sglang.srt.eplb.expert_location import (
        ExpertLocationMetadata,
        ModelConfigForExpertLocation,
        set_global_expert_location_metadata,
    )
    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
    from sglang.srt.runtime_context import SpawnRanks, get_forward, publish
    from sglang.srt.server_args import ServerArgs

    class ObservedUpdater(ExpertLocationUpdater):
        def __init__(self):
            super().__init__()
            if args.device == "cpu":
                self._first_execution = False  # CPU has no accelerator cache flush.
            self.completed_chunks = 0

        def update(self, *args, **kwargs):
            missing = super().update(*args, **kwargs)
            self.completed_chunks += 1
            return missing

    class ObservedManager(EPLBManager):
        skipped_decisions = 0

        def _check_rebalance_needed(self, utilization):
            needed = super()._check_rebalance_needed(utilization)
            if not needed:
                self.skipped_decisions += 1
            return needed

    dist.init_process_group(
        backend="hccl" if args.device == "npu" else "gloo",
        timeout=datetime.timedelta(seconds=60),
    )
    reports = []
    try:
        # Only model metadata and the hardware selection are substituted.
        # Recorder, all-reduce, planner, manager and P2P updater stay real.
        with (
            patch.object(
                ModelConfigForExpertLocation,
                "from_model_config",
                return_value=ModelConfigForExpertLocation(2, 8, 1),
            ),
            (
                patch(
                    "sglang.srt.eplb.expert_distribution.get_device", return_value="cpu"
                )
                if args.device == "cpu"
                else nullcontext()
            ),
        ):
            for role in ("prefill", "decode"):
                publish(
                    ServerArgs(
                        model_path="dummy",
                        device=args.device,
                        tp_size=2,
                        ep_size=2,
                        disaggregation_mode=role,
                        moe_a2a_backend="deepep",
                        deepep_mode=(
                            "auto"
                            if args.device == "npu"
                            else "normal"
                            if role == "prefill"
                            else "low_latency"
                        ),
                        enable_eplb=True,
                        ep_dispatch_algorithm="static",
                        eplb_algorithm="deepseek",
                        eplb_rebalance_num_iterations=4,
                        eplb_rebalance_layers_per_chunk=1,
                        eplb_min_rebalancing_utilization_threshold=args.rebalance_threshold,
                        expert_balancedness_report_mode=(
                            "server_log" if args.rebalance_threshold < 1 else "off"
                        ),
                        expert_distribution_recorder_mode="stat",
                        expert_distribution_recorder_buffer_size=4,
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                model_config = SimpleNamespace()
                metadata = ExpertLocationMetadata.init_trivial(
                    model_config, moe_ep_rank=rank
                )
                set_global_expert_location_metadata(metadata, allow_overwrite=True)
                model = SimpleNamespace(
                    routed_experts_weights_of_layer={
                        layer: [
                            torch.tensor(
                                [
                                    expert + layer * 16
                                    for expert in range(rank * 4, rank * 4 + 4)
                                ],
                                dtype=torch.float32,
                                device=args.device,
                            ).reshape(4, 1)
                        ]
                        for layer in range(2)
                    }
                )
                pointers = [
                    tensors[0].data_ptr()
                    for tensors in model.routed_experts_weights_of_layer.values()
                ]
                map_pointer = metadata.physical_to_logical_map.data_ptr()
                recorder = ExpertDistributionRecorder.init_new(metadata, rank)
                if args.device == "cpu":
                    recorder._accumulator._first_dump = False
                set_global_expert_distribution_recorder(recorder)
                updater = ObservedUpdater()
                weight_updater = SimpleNamespace(update_weights_from_disk=None)
                manager = ObservedManager(
                    model_config=model_config,
                    get_model=lambda: model,
                    get_expert_location_updater=lambda: updater,
                    get_expert_backup_client=lambda: None,
                    get_weight_updater=lambda: weight_updater,
                )
                step = 0

                def forward(traffic, step):
                    get_forward().set("is_extend_in_batch", role == "prefill")
                    with recorder.with_forward_pass(step, SimpleNamespace()):
                        for layer in range(2):
                            with recorder.with_current_layer(layer):
                                if role == "prefill" or args.device == "npu":
                                    logical_ids = (
                                        torch.repeat_interleave(
                                            torch.arange(8), traffic.long()
                                        )
                                        if rank == 0
                                        else torch.empty(0, dtype=torch.long)
                                    )
                                    physical_ids = (
                                        metadata.logical_to_rank_dispatch_physical_map[
                                            layer
                                        ][logical_ids.to(args.device)]
                                    )
                                    recorder.on_select_experts(
                                        physical_ids.reshape(-1, 1)
                                    )
                                    if role == "decode":
                                        # NPU counts valid sender routes; receiver
                                        # counters must not double-count them.
                                        recorder.on_deepep_dispatch_low_latency(
                                            torch.full((4,), 999, device=args.device)
                                        )
                                else:
                                    local_experts = (
                                        metadata.physical_to_logical_map_cpu[
                                            layer, rank * 4 : (rank + 1) * 4
                                        ]
                                    )
                                    recorder.on_select_experts(
                                        torch.tensor([[0, 1]], device=args.device)
                                    )
                                    recorder.on_deepep_dispatch_low_latency(
                                        traffic[local_experts].to(args.device)
                                    )
                    manager.on_forward_pass_end()

                phase_reports = []
                hot = (0, 1)
                for phase in range(2):
                    old_map = metadata.physical_to_logical_map_cpu.clone()
                    traffic = torch.ones(8, dtype=torch.int32)
                    traffic[list(hot)] = 40
                    before_peak = max(
                        traffic[old_map[0]].reshape(2, 4).sum(dim=1)
                    ).item()
                    target_chunks = updater.completed_chunks + 2
                    for _ in range(20):
                        forward(traffic, step)
                        step += 1
                        if updater.completed_chunks >= target_chunks:
                            break
                    else:
                        raise AssertionError("periodic rebalance did not complete")
                    new_map = metadata.physical_to_logical_map_cpu.clone()
                    assert not torch.equal(old_map, new_map)
                    after_peak = max(
                        traffic[new_map[0]].reshape(2, 4).sum(dim=1)
                    ).item()
                    assert after_peak < before_peak, (before_peak, after_peak)
                    for layer, tensors in model.routed_experts_weights_of_layer.items():
                        expected = (
                            new_map[layer, rank * 4 : (rank + 1) * 4].float()
                            + layer * 16
                        )
                        torch.testing.assert_close(tensors[0].flatten().cpu(), expected)
                        assert tensors[0].data_ptr() == pointers[layer]
                    assert metadata.physical_to_logical_map.data_ptr() == map_pointer
                    phase_reports.append(
                        {
                            "peak_before": before_peak,
                            "peak_after": after_peak,
                            "hot": list(hot),
                        }
                    )
                    # Shift the hot experts onto the same rank in the NEW map.
                    hot = tuple(new_map[0, :2].tolist())
                    dist.barrier()
                if args.rebalance_threshold < 1:
                    # Keep the same manager/history alive as the routing turns
                    # uniform. Once old skew ages out, decisions must skip the
                    # updater on BOTH ranks, using the upstream reporting-enabled path.
                    uniform = torch.ones(8, dtype=torch.int32)
                    for index in range(100):
                        if index == 84:
                            stable_map = metadata.physical_to_logical_map_cpu.clone()
                            chunks_before_stable_tail = updater.completed_chunks
                            skipped_before_stable_tail = manager.skipped_decisions
                        forward(uniform, step)
                        step += 1
                    assert updater.completed_chunks == chunks_before_stable_tail
                    assert manager.skipped_decisions > skipped_before_stable_tail
                    assert torch.equal(metadata.physical_to_logical_map_cpu, stable_map)
                reports.append(
                    {
                        "role": role,
                        "steps": step,
                        "completed_chunks": updater.completed_chunks,
                        "idle_sender_rank": 1 if role == "prefill" else None,
                        "phases": phase_reports,
                        "rebalance_threshold": args.rebalance_threshold,
                        "skipped_decisions": manager.skipped_decisions,
                    }
                )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "scope": f"synthetic routing / real {args.device} EPLB loop",
                        "reports": reports,
                    }
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
