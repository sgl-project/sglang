import multiprocessing as mp
from abc import ABC

import torch

from sglang.test.runners import SRTRunner

PROMPT = (
    "What is the range of the numeric output of a sigmoid node in a neural network?"
)
RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

CONVS = [
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
]


class BaseNoHFRewardModelTest(ABC):
    """Base test class for reward model testing that doesn't compare with HF.

    This is for models that only need to verify SGLang can run them successfully.
    """

    # Required attributes for subclasses
    model_path: str

    # Optional attributes with defaults
    torch_dtype: torch.dtype = torch.float16
    tp_size: int = 4
    trust_remote_code: bool = True
    disable_cuda_graph: bool = True
    mem_fraction_static: float = 0.8

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_assert_close_reward_scores(self):
        """Test that the model can generate reward scores."""
        srt_runner_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "disable_cuda_graph": self.disable_cuda_graph,
            "tp_size": self.tp_size,
            "mem_fraction_static": self.mem_fraction_static,
        }

        with SRTRunner(
            self.model_path,
            torch_dtype=self.torch_dtype,
            model_type="reward",
            **srt_runner_kwargs,
        ) as srt_runner:
            prompts = srt_runner.tokenizer.apply_chat_template(CONVS, tokenize=False)
            srt_outputs = srt_runner.forward(prompts)
        srt_scores = torch.tensor(srt_outputs.scores)
        print(f"accuracy: {srt_scores}")
        self.assertIsInstance(srt_scores, torch.Tensor)
