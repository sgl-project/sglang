import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from sglang.srt.managers.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.managers.io_struct import UpdateExpertLocationReqInput, UpdateExpertLocationReqOutput
from sglang.srt.managers.schedule_batch import get_global_expert_location_metadata
from sglang.srt.model_executor.model_weight_updater import ModelWeightUpdater
from sglang.srt.model_loader.weight_utils import ModelParamNameInfo, ModelParamNameInfoMoe
from sglang.srt.poll_based_barrier import PollBasedBarrier
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class ExpertLocationUpdater:
    def __init__(self, model_runner: "ModelRunner"):
        self._model_runner = model_runner
        self._model_weight_updater = ModelWeightUpdater(
            init_pin_memory=model_runner.server_args.expert_location_updater_mode == "pin_memory",
            load_format=model_runner.server_args.load_format,
            model_config=model_runner.model_config,
            model=model_runner.model,
            device=model_runner.device,
        )
        self._prepare_end_barrier = PollBasedBarrier(noop=False)
        self._ongoing_req: Optional[UpdateExpertLocationReqInput] = None

    def start(self, req: UpdateExpertLocationReqInput):
        assert self._ongoing_req is None
        self._ongoing_req = req

        interesting_logical_experts_of_layer = _compute_interesting_logical_experts_of_layer(
            old_expert_location_metadata=get_global_expert_location_metadata(),
            new_expert_location_metadata=req.expert_location_metadata,
            ep_rank=self._model_runner.tp_rank,
        )

        self._model_weight_updater.start_prepare(
            weight_filter=lambda name: self._weight_filter(name, interesting_logical_experts_of_layer),
        )

    def event_loop_step(self) -> List[UpdateExpertLocationReqOutput]:
        outputs = []

        if self._model_weight_updater.poll_prepare_end():
            self._prepare_end_barrier.local_arrive()

        if self._prepare_end_barrier.poll_global_arrived():
            outputs.append(self._act())

        return outputs

    def _act(self):
        torch.distributed.barrier()

        get_global_expert_distribution_recorder().flush_buffer_depending_on_expert_location_metadata()

        get_global_expert_location_metadata().update(self._ongoing_req.expert_location_metadata)
        if self._model_runner.tp_rank == 0 and get_bool_env_var(
                "SGLANG_LOG_EXPERT_LOCATION_METADATA"
        ):
            logger.info(
                f"Updated expert_location_metadata: {get_global_expert_location_metadata().debug_str()}"
            )

        self._model_weight_updater.act()

        torch.distributed.barrier()

        assert self._ongoing_req is not None
        self._ongoing_req = None

        return UpdateExpertLocationReqOutput()

    def _weight_filter(self, name: str, interesting_logical_experts_of_layer: Dict[int, List[int]]):
        info: ModelParamNameInfo = self._model_runner.model.get_param_name_info(name)
        return (
                isinstance(info, ModelParamNameInfoMoe)
                and (info.expert_id in interesting_logical_experts_of_layer[info.layer_id])
        )


def _compute_interesting_logical_experts_of_layer(
        old_expert_location_metadata: ExpertLocationMetadata,
        new_expert_location_metadata: ExpertLocationMetadata,
        ep_rank: int,
) -> Dict[int, List[int]]:
    num_layers = old_expert_location_metadata.num_layers
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts

    def _get_partial_physical_to_logical_map(meta: ExpertLocationMetadata, layer_id: int):
        return meta.physical_to_logical_map[layer_id,
               num_local_physical_experts * ep_rank: num_local_physical_experts * (ep_rank + 1)]

    interesting_logical_experts_of_layer = {}
    for layer_id in range(num_layers):
        old_partial_map = _get_partial_physical_to_logical_map(old_expert_location_metadata, layer_id)
        new_partial_map = _get_partial_physical_to_logical_map(new_expert_location_metadata, layer_id)
        interesting_logical_experts_of_layer[layer_id] = new_partial_map[new_partial_map != old_partial_map].tolist()
    return interesting_logical_experts_of_layer
