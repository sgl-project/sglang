"""Import and exercise real upstream GPU modules, not extracted AST methods."""

import importlib.metadata
import json
import os

import torch

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "versions": {
                k: importlib.metadata.version(k)
                for k in ("transformers", "triton", "sglang")
            },
        }
    ),
    flush=True,
)
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

print(json.dumps({"indexer_file": dsa_indexer.__file__}), flush=True)
publish(ServerArgs(model_path="dummy", device="cuda"), role="test")
torch.set_default_dtype(torch.bfloat16)
kwargs = dict(
    hidden_size=512,
    index_n_heads=4,
    index_head_dim=128,
    rope_head_dim=64,
    index_topk=32,
    q_lora_rank=128,
    max_position_embeddings=32768,
    rope_theta=10000,
    layer_id=0,
    scale_fmt=None,
    is_neox_style=False,
)
with torch.device("cuda"):
    os.environ["SGLANG_ROCM_DSA_INDEXER_PROJECTION_FUSION"] = "0"
    control = dsa_indexer.Indexer(**kwargs)
    os.environ["SGLANG_ROCM_DSA_INDEXER_PROJECTION_FUSION"] = "1"
    candidate = dsa_indexer.Indexer(**kwargs)
assert not control.use_dsa_indexer_projection_fusion
assert candidate.use_dsa_indexer_projection_fusion
assert not candidate.use_dsa_indexer_fusion
with torch.inference_mode():
    torch.manual_seed(20260913)
    wk = torch.randn_like(control.wk.weight) * 0.01
    wg = torch.randn_like(control.weights_proj.weight) * 0.01
    control.wk.weight.copy_(wk)
    control.weights_proj.weight.copy_(wg)
    candidate.wk_weights_proj.weight.copy_(torch.cat((wk, wg)))
    for count in (1, 8, 16, 257, 8192):
        x = torch.randn((count, 512), device="cuda", dtype=torch.bfloat16)
        expected = torch.cat((control.wk(x)[0], control.weights_proj(x)[0]), dim=-1)
        actual = candidate.wk_weights_proj(x)[0]
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        print(json.dumps({"tokens": count, "projection_parity": "pass"}), flush=True)
print("REAL_MODULE_PREFLIGHT_PASS", flush=True)
