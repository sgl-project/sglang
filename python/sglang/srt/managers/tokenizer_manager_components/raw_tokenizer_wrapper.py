from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.srt.managers.mm_utils import TensorTransportMode
from sglang.srt.managers.multimodal_processor import (
    get_mm_processor,
    import_processors,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class RawTokenizerWrapper:
    tokenizer: Optional[Any] = None
    processor: Optional[Any] = None
    mm_processor: Optional[Any] = None
    async_dynamic_batch_tokenizer: Optional[AsyncDynamicbatchTokenizer] = None

    def init_tokenizer_and_processor(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
    ) -> None:
        # Initialize tokenizer and processor
        if model_config.is_multimodal:
            import_processors("sglang.srt.multimodal.processors")
            if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
                import_processors(mm_process_pkg, overwrite=True)
            _processor = _get_processor_wrapper(server_args)
            transport_mode = _determine_tensor_transport_mode(server_args)

            # We want to parallelize the image pre-processing so we create an executor for it
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            self.mm_processor = get_mm_processor(
                model_config.hf_config,
                server_args,
                _processor,
                transport_mode,
                model_config=model_config,
            )

            if server_args.skip_tokenizer_init:
                self.tokenizer = self.processor = None
            else:
                self.processor = _processor
                self.tokenizer = get_tokenizer_from_processor(self.processor)
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            self.mm_processor = self.processor = None

            if server_args.skip_tokenizer_init:
                self.tokenizer = None
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )

        # Initialize async dynamic batch tokenizer if enabled (common for both multimodal and non-multimodal)
        if (
            server_args.enable_dynamic_batch_tokenizer
            and not server_args.skip_tokenizer_init
        ):
            self.async_dynamic_batch_tokenizer = AsyncDynamicbatchTokenizer(
                self.tokenizer,
                max_batch_size=server_args.dynamic_batch_tokenizer_batch_size,
                batch_wait_timeout_s=server_args.dynamic_batch_tokenizer_batch_timeout,
            )
        else:
            self.async_dynamic_batch_tokenizer = None


def _get_processor_wrapper(server_args):
    try:
        processor = get_processor(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            use_fast=not server_args.disable_fast_image_processor,
            tokenizer_backend=server_args.tokenizer_backend,
        )
    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:
            logger.info(
                f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
            )
            processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=True,
                tokenizer_backend=server_args.tokenizer_backend,
            )
        else:
            raise e
    return processor


def _determine_tensor_transport_mode(server_args: ServerArgs) -> TensorTransportMode:
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"
