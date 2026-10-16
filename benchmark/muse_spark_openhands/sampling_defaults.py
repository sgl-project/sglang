"""Record checkpoint defaults and the matching EvalScope request parameters."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from transformers import GenerationConfig

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.sampling.sampling_params import SamplingParams

model, output = map(Path, sys.argv[1:3])
raw = json.loads((model / "generation_config.json").read_text())
config = GenerationConfig.from_pretrained(str(model), local_files_only=True)
defaults = ModelConfig.get_default_sampling_params(
    SimpleNamespace(sampling_defaults="model", hf_generation_config=config)
)
params = SamplingParams(**defaults)
params.normalize(None)
params.verify(1 << 30)
effective = {key: getattr(params, key) for key in ("temperature", "top_p", "top_k")}
request = {**defaults, "ignore_eos": True}
output.write_text(json.dumps({"generation_config": raw, "effective": effective, "request": request}, indent=2) + "\n")
print("Effective checkpoint sampling:", effective)
