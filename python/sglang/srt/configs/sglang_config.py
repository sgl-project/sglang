import copy
import logging
from dataclasses import replace
from functools import lru_cache

from pydantic import Field
from transformers import PretrainedConfig

from sglang.srt.configs.compilation_config import CompilationConfig, CompilationMode
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class SGLangConfig:
    """Dataclass which contains all sglang-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # Model configuration.
    model_config: ModelConfig = Field(default=None)

    # Device configuration.
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)

    # Compilation configuration
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)

    # The ID of the SGLang instance.
    instance_id: str = ""

    def with_hf_config(
        self,
        hf_config: PretrainedConfig,
        architectures: list[str] | None = None,
    ) -> "SGLangConfig":
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures

        model_config = copy.deepcopy(self.model_config)
        model_config.hf_config = hf_config

        return replace(self, model_config=model_config)


_current_sglang_config: SGLangConfig | None = None
_current_prefix: str | None = None


def set_current_sglang_config(sglang_config: SGLangConfig, prefix: str | None = None):
    """
    Temporarily set the current SGLang config.
    Used during model initialization.
    We save the current SGLang config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the SGLang config to determine how to dispatch.
    """
    global _current_sglang_config, _current_prefix
    old_sglang_config = _current_sglang_config
    old_prefix = _current_prefix

    try:
        _current_sglang_config = sglang_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        if (
            # TODO(yuan-luo) compilation mode check
            sglang_config.compilation_config.mode
            == CompilationMode.SGLANG_COMPILE
        ):
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                sglang_config.model_config.model,
            )
    finally:
        _current_sglang_config = old_sglang_config
        _current_prefix = old_prefix
        # Clear the compilation config cache when context changes
        get_cached_compilation_config.cache_clear()


@lru_cache(maxsize=1)
def get_cached_compilation_config():
    """Cache config to avoid repeated calls to get_current_sglang_config()"""
    return get_current_sglang_config().compilation_config


def get_current_sglang_config() -> SGLangConfig:
    if _current_sglang_config is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the sglang config. In that case, we set a default
        # config.
        logger.warning("Current SGLang config is not set.")
        return SGLangConfig()
    return _current_sglang_config
