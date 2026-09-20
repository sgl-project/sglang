import torch

from sglang.multimodal_gen.configs.models.dits.krea2 import Krea2DitConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.models.dits.krea2 import (
    Krea2Transformer2DModel,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_krea2_online_fp8_quantizes_transformer_linears():
    _ensure_single_process_parallel_runtime()

    quant_config = Fp8Config(
        ignored_layers=[
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.ff.up",
        ]
    )

    with torch.device("meta"):
        model = Krea2Transformer2DModel(
            config=Krea2DitConfig(),
            hf_config={},
            quant_config=quant_config,
        )

    block = model.transformer_blocks[0]

    assert isinstance(block.attn.to_q.quant_method, UnquantizedLinearMethod)
    assert isinstance(block.attn.to_k.quant_method, Fp8LinearMethod)
    assert isinstance(block.attn.to_v.quant_method, Fp8LinearMethod)
    assert isinstance(block.attn.to_out[0].quant_method, Fp8LinearMethod)

    assert isinstance(block.ff.gate.quant_method, Fp8LinearMethod)
    assert isinstance(block.ff.up.quant_method, UnquantizedLinearMethod)
    assert isinstance(block.ff.down.quant_method, Fp8LinearMethod)

    text_block = model.text_fusion.layerwise_blocks[0]
    assert isinstance(text_block.attn.to_q.quant_method, Fp8LinearMethod)
    assert isinstance(text_block.ff.gate.quant_method, Fp8LinearMethod)

    # Numerically sensitive boundary layers remain BF16-native.
    assert isinstance(model.img_in, torch.nn.Linear)
    assert isinstance(model.time_mod_proj, torch.nn.Linear)
    assert isinstance(model.final_layer.linear, torch.nn.Linear)
    assert isinstance(model.text_fusion.projector, torch.nn.Linear)
