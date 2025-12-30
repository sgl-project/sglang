import copy
import logging
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig

from sglang.srt.compilation.compilation_config import CompilationConfig, CompilationMode
from sglang.srt.configs.device_config import DeviceConfig

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
else:
    ModelConfig = Any

logger = logging.getLogger(__name__)


class SGLangConfig:
    """Dataclass which contains all sglang-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    def __init__(
        self,
        model_config: ModelConfig = None,
        device_config: DeviceConfig = DeviceConfig,
        compilation_config: CompilationConfig = CompilationConfig,
    ):
        self.model_config = model_config
        self.device_config = device_config
        self.compilation_config = compilation_config
        self.self_id = ""

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


@contextmanager
def set_current_sglang_config(
    sglang_config: SGLangConfig, check_compile=False, prefix: str | None = None
):
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
        # TODO(): custom op check
        if check_compile:
            pass

        if (
            check_compile
            # TODO(): compilation mode check
            and sglang_config.compilation_config.mode == CompilationMode.SGLANG_COMPILE
        ):
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " https://github.com/sgl-project/sglang/issues/new/choose"
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
