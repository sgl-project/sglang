import logging
from typing import TYPE_CHECKING

import torch
from sglang.srt.managers.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.managers.io_struct import UpdateExpertLocationReqInput
from sglang.srt.managers.schedule_batch import get_global_expert_location_metadata
from sglang.srt.model_executor.model_weight_updater import ModelWeightUpdater
from sglang.srt.model_loader.weight_utils import ModelParamNameInfo, ModelParamNameInfoMoe
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class ExpertLocationUpdater:
    def __init__(self, model_runner: "ModelRunner"):
        self._model_runner = model_runner
        self._model_weight_updater = ModelWeightUpdater(
            init_pin_memory=TODO,
            load_format=TODO,
            model_config=model_runner.model_config,
            model=model_runner.model,
            device=model_runner.device,
        )

    def start_prepare(self):
        self._model_weight_updater.start_prepare(
            weight_filter=lambda name: self._weight_filter(name),
        )

    def act(self, recv_req: UpdateExpertLocationReqInput):
        logger.info("update_expert_location start")
        torch.distributed.barrier()

        get_global_expert_distribution_recorder().flush_buffer_depending_on_expert_location_metadata()

        get_global_expert_location_metadata().update(recv_req.expert_location_metadata)
        if self._model_runner.tp_rank == 0 and get_bool_env_var(
            "SGLANG_LOG_EXPERT_LOCATION_METADATA"
        ):
            logger.info(
                f"Updated expert_location_metadata: {get_global_expert_location_metadata().debug_str()}"
            )

        # We may be able to further reduce lock time by faster copying, pre-transfering, etc
        self._model_runner.update_weights_from_disk(
            model_path=self._model_runner.model_config.model_path,
            load_format=self._model_runner.server_args.load_format,
            param_categories=["moe"],
        )

        logger.info("update_expert_location end")
        torch.distributed.barrier()

    def _weight_filter(self, name: str):
        info: ModelParamNameInfo = self._model_runner.model.get_param_name_info()
        if not isinstance(info, ModelParamNameInfoMoe):
            return False

        return TODO
