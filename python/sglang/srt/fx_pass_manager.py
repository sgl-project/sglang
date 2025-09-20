import logging

import torch
import torch.fx as fx
import torch.nn as nn

from sglang.srt.passes import SiLUMulFusionPass

logger = logging.getLogger(__name__)


class SGLangFXPassManager:
    def __init__(self):
        self.passes = [SiLUMulFusionPass()]

    def apply_passes(self, model: nn.Module) -> nn.Module:
        logger.info("Starting SGLang FX Pass optimization...")

        try:
            tracer = fx.Tracer()
            graph = tracer.trace(model)
            fx_model = fx.GraphModule(model, graph)

            total_modifications = 0
            for pass_instance in self.passes:
                if pass_instance.is_applicable(graph):
                    logger.info(f"Applying {pass_instance.name}...")
                    modified = pass_instance.apply(graph)
                    if modified:
                        total_modifications += 1
                        logger.info(f"{pass_instance.name} applied successfully")
                else:
                    logger.debug(f"Skipping {pass_instance.name} (not applicable)")

            fx_model.recompile()

            logger.info(
                f"FX Pass optimization completed. Applied {total_modifications} passes."
            )
            return fx_model

        except Exception as e:
            logger.warning(f"FX Pass optimization failed: {e}")
            return model


def apply_sglang_fx_optimization(model: nn.Module) -> nn.Module:
    pass_manager = SGLangFXPassManager()
    return pass_manager.apply_passes(model)
