"""External KV calibration metadata follows the model's construction layout."""

import json
import socket
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import ApertusConfig, Glm4Config, LlamaConfig, OPTConfig, Qwen2Config

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.models import hunyuan
from sglang.srt.models.apertus import ApertusModel
from sglang.srt.models.arcee import ArceeModel
from sglang.srt.models.glm4 import Glm4Model
from sglang.srt.models.hunyuan import HunYuanMoEV1ForCausalLM
from sglang.srt.models.llama import LlamaModel
from sglang.srt.models.mimo_v2 import MiMoV2Model
from sglang.srt.models.opt import OPTModel
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.solar import SolarModel
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

MODELS = {
    "llama": (LlamaModel, LlamaConfig),
    "qwen2": (Qwen2Model, Qwen2Config),
    "glm4": (Glm4Model, Glm4Config),
    "opt": (OPTModel, OPTConfig),
    "apertus": (ApertusModel, ApertusConfig),
    "arcee": (ArceeModel, LlamaConfig),
    "solar": (SolarModel, LlamaConfig),
    "hunyuan": (HunYuanMoEV1ForCausalLM, LlamaConfig),
    "mimo": (MiMoV2Model, LlamaConfig),
}


def build_model(kind):
    cls, cfg = MODELS[kind]
    extra = {}
    if kind == "apertus":
        extra.update(
            hidden_act="xielu",
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
    if kind == "arcee":
        extra.update(hidden_act="relu2")
    if kind == "solar":
        extra.update(bskcn_1=[0], bskcn_2=[], bskcn_3=[1], bskcn_4=[])
    if kind == "hunyuan":
        extra.update(attention_head_dim=16, num_experts=1)
    if kind == "mimo":
        extra.update(
            layernorm_epsilon=1e-6,
            hybrid_layer_pattern=[0, 1],
            swa_num_attention_heads=8,
            swa_num_key_value_heads=4,
            swa_head_dim=16,
            sliding_window_size=16,
            moe_layer_freq=[0, 0],
        )
    config = cfg(
        hidden_size=128,
        intermediate_size=256,
        ffn_dim=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=32,
        word_embed_proj_dim=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **extra,
    )
    with torch.device("cuda"):
        return cls(config)


def attention_layers(model):
    if isinstance(model, OPTModel):
        layers = model.decoder.layers
    elif isinstance(model, HunYuanMoEV1ForCausalLM):
        layers = model.model.layers
    else:
        layers = model.layers
    return [layer.self_attn.attn for layer in layers]


def calibration_data(model, tp_size, version):
    return dict(
        model_type=model.config.model_type,
        kv_cache=dict(
            dtype="float8_e4m3fn",
            scaling_factor={
                rank: {
                    layer: (rank * 7 + layer * 3 + version + 1) / 32
                    for layer in range(model.config.num_hidden_layers)
                }
                for rank in range(tp_size)
            },
        ),
    )


def loading_scope(changed):
    if not changed:
        return nullcontext()
    # Width4/rank3 is valid during loading even when CI constructed at TP1.
    # No model construction or kernel executes within this temporary scope.
    return get_parallel().override(
        tp_size=4,
        tp_rank=3,
        tp_group=None,
        attn_tp_size=4,
        attn_tp_rank=3,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=4,
        moe_tp_rank=3,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


def load_scales(model, directory, *, tp_rank, tp_size, version=0, changed=False):
    data = calibration_data(model, tp_size, version)
    path = Path(directory) / "calibration.json"
    path.write_text(json.dumps(data))
    if isinstance(model, HunYuanMoEV1ForCausalLM):
        # This legacy native endpoint parses the file, then rejects the missing
        # kv_scale attribute. Verify its real selection and preserve that guard.
        selected = []
        original = hunyuan.kv_cache_scales_loader

        def record(*args, **kwargs):
            rows = list(original(*args, **kwargs))
            selected.extend(rows)
            return rows

        with (
            patch.object(hunyuan, "kv_cache_scales_loader", record),
            loading_scope(changed),
        ):
            try:
                model.load_kv_cache_scales(str(path))
            except RuntimeError as error:
                assert "KV cache scaling factor attribute" in str(error)
            else:
                raise AssertionError("Expected the existing HunYuan kv_scale guard")
        assert selected == list(data["kv_cache"]["scaling_factor"][tp_rank].items())
        return [(layer.k_scale, layer.v_scale) for layer in attention_layers(model)]
    with loading_scope(changed):
        model.load_kv_cache_scales(str(path))
    result = [(layer.k_scale, layer.v_scale) for layer in attention_layers(model)]
    expected = [
        (data["kv_cache"]["scaling_factor"][tp_rank][i],) * 2
        for i in range(model.config.num_hidden_layers)
    ]
    assert result == expected, (type(model).__name__, result, expected)
    return result


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestKvCalibrationLoaderLayout(CustomTestCase):
    def setUp(self):
        if torch.distributed.is_initialized():
            self.skipTest("requires an isolated distributed test process")
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)
        torch.set_default_dtype(torch.bfloat16)
        server = ServerArgs(model_path="dummy", device="cuda", tp_size=1)
        publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        torch.cuda.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        self.addCleanup(destroy_distributed_environment)
        initialize_model_parallel()
        self.addCleanup(destroy_model_parallel)
        initialize_dp_attention(server)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = tmp.name

    def check_loads(self, changed):
        for kind in MODELS:
            model = build_model(kind)
            for version in (0, 13):
                with self.subTest(kind=kind, version=version):
                    load_scales(
                        model,
                        self.directory,
                        tp_rank=0,
                        tp_size=1,
                        version=version,
                        changed=changed,
                    )

    def test_real_model_calibration_after_scope_exit(self):
        self.check_loads(True)

    def test_existing_invalid_file_fallback_keeps_loaded_scales(self):
        for kind in MODELS:
            model = build_model(kind)
            expected = load_scales(model, self.directory, tp_rank=0, tp_size=1)
            for changed in (False, True):
                for error in ("missing", "json", "model_type", "dtype", "ranks"):
                    path = Path(self.directory) / "invalid.json"
                    data = calibration_data(model, 1, 0)
                    if error == "missing":
                        path = Path(self.directory) / "missing.json"
                    elif error == "json":
                        path.write_text("{")
                    else:
                        if error == "model_type":
                            data["model_type"] = "wrong_model_type"
                        if error == "dtype":
                            data["kv_cache"]["dtype"] = "float32"
                        if error == "ranks":
                            data["kv_cache"]["scaling_factor"] = {}
                        path.write_text(json.dumps(data))
                    with loading_scope(changed):
                        model.load_kv_cache_scales(str(path))
                    self.assertEqual(
                        [(a.k_scale, a.v_scale) for a in attention_layers(model)],
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
