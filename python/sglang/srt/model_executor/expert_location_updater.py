import logging
from typing import TYPE_CHECKING

import torch
from sglang.srt.managers.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.managers.io_struct import UpdateExpertLocationReqInput
from sglang.srt.managers.schedule_batch import get_global_expert_location_metadata
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class ExpertLocationUpdater:
    def __init__(self, model_runner: "ModelRunner"):
        self._model_runner = model_runner

    def prepare(self):
        TODO

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
