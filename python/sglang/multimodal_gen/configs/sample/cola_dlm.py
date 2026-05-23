from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class ColaDLMSamplingParams(SamplingParams):
    """Sampling parameters for Cola-DLM text diffusion model."""

    data_type: DataType = DataType.TEXT
    return_file_paths_only: bool = False
    num_inference_steps: int = 16
    guidance_scale: float = 7.0
    max_new_tokens: int = 256
    block_size: int = 16
    patch_size: int = 1
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    T: float = 1000.0
    pad_token_id: int = 100277
    eos_token_id: int = 100257
