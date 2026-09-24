from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.arg_groups.model_override_base import model_config_of
from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs


def _denoiser_statistics(logits: torch.Tensor, temperatures: torch.Tensor):
    processed_logits = logits / temperatures[:, None, None]
    log_probabilities = torch.log_softmax(processed_logits, dim=-1, dtype=torch.float32)
    probabilities = log_probabilities.exp()
    token_entropies = -(log_probabilities * probabilities).sum(dim=-1)
    return probabilities, token_entropies, logits.argmax(dim=-1)


_compiled_denoiser_statistics = torch.compile(_denoiser_statistics, dynamic=True)


def _sample_denoiser(probabilities: torch.Tensor, generator: torch.Generator):
    # The exponential-race formulation samples the same categorical distribution
    # as multinomial. Probabilities come from softmax, so multinomial's repeated
    # validation and device-to-host synchronization are unnecessary here.
    noise = torch.empty_like(probabilities).exponential_(1.0, generator=generator)
    return (probabilities / noise).argmax(dim=-1)


class Gemma4Renoise(DllmAlgorithm):
    supported_architectures = ("DiffusionGemmaForBlockDiffusion",)
    requires_separate_context_encoding = True
    required_attention_backend = "triton"

    @classmethod
    def configure_server_args(cls, server_args: ServerArgs) -> None:
        cfg = resolving_view(server_args)
        if cfg.device != "cuda":
            raise ValueError(
                "DiffusionGemma currently supports CUDA/ROCm GPU execution only"
            )

        model_config = model_config_of(server_args)
        hf_config = model_config.hf_config
        if cfg.pp_size > 1:
            raise ValueError("DiffusionGemma does not support pipeline parallelism")
        if cfg.dcp_size > 1 or cfg.attn_cp_size > 1:
            raise ValueError("DiffusionGemma does not support context parallelism")
        if model_config.quantization is not None:
            raise ValueError("DiffusionGemma does not support quantized loading")
        if not (
            getattr(hf_config, "tie_word_embeddings", False)
            and getattr(hf_config.text_config, "tie_word_embeddings", False)
        ):
            raise ValueError("DiffusionGemma requires tied word embeddings")

        declare_resolution(
            server_args,
            cls.__name__,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config,
                Phase.PREFILL,
                backend=Backend.DISABLED,
            ),
        )

    @classmethod
    def validate_request(cls, req: Req) -> Optional[str]:
        sp = req.sampling_params
        unsupported = [
            name
            for name, value, default in (
                ("frequency_penalty", sp.frequency_penalty, 0.0),
                ("presence_penalty", sp.presence_penalty, 0.0),
                ("repetition_penalty", sp.repetition_penalty, 1.0),
                ("min_new_tokens", sp.min_new_tokens, 0),
                ("min_p", sp.min_p, 0.0),
                ("sampling_seed", sp.sampling_seed, None),
            )
            if value != default
        ]
        if sp.logit_bias:
            unsupported.append("logit_bias")
        unsupported.extend(
            name
            for name in ("json_schema", "regex", "ebnf", "structural_tag")
            if getattr(sp, name) is not None
        )
        unsupported.extend(
            name
            for name, value in (
                ("return_logprob", req.return_logprob),
                ("top_logprobs_num", req.logprob.top_logprobs_num),
                ("token_ids_logprob", req.logprob.token_ids_logprob),
                (
                    "return_flat_raw_top_logprobs",
                    req.return_flat_raw_top_logprobs,
                ),
                ("custom_logit_processor", req.custom_logit_processor),
                ("return_sampling_mask", req.return_sampling_mask),
            )
            if value
        )
        if not unsupported:
            return None
        return (
            "DiffusionGemma with Gemma4Renoise does not support request-level "
            + ", ".join(unsupported)
            + "; sampling is governed by the renoise schedule."
        )

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        algorithm_config = config.algorithm_config or {}
        if not isinstance(algorithm_config, dict):
            raise ValueError("Gemma4Renoise configuration must be a mapping")

        def section(name: str) -> dict:
            value = algorithm_config.get(name) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            return value

        self.max_denoising_steps = algorithm_config.get("max_denoising_steps", 48)
        sampler_config = section("sampler_config")
        self.entropy_bound = sampler_config.get("entropy_bound", 0.1)
        temperature_config = section("temperature_schedule")
        self.t_min = temperature_config.get("t_min", algorithm_config.get("t_min", 0.4))
        self.t_max = temperature_config.get("t_max", algorithm_config.get("t_max", 0.8))
        stopping_config = section("stopping_config")
        # Zero disables adaptive convergence: entropy cannot be negative.
        self.confidence_threshold = stopping_config.get(
            "confidence_threshold",
            algorithm_config.get("confidence_threshold", 0.005),
        )
        self.stability_threshold = stopping_config.get(
            "stability_threshold", algorithm_config.get("stability_threshold", 1)
        )
        self.seed = algorithm_config.get("seed")
        self.vocab_size = None
        self.embed_tokens = None

        if (
            not isinstance(self.max_denoising_steps, int)
            or self.max_denoising_steps < 1
        ):
            raise ValueError("max_denoising_steps must be at least 1")
        if self.entropy_bound <= 0:
            raise ValueError("entropy_bound must be positive")
        if self.t_min < 0 or self.t_max <= self.t_min:
            raise ValueError("temperature_schedule must satisfy 0 <= t_min < t_max")
        if (
            not isinstance(self.stability_threshold, int)
            or self.stability_threshold < 0
            or self.confidence_threshold < 0
        ):
            raise ValueError("invalid stopping_config")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> DllmRunOutput:
        if not forward_batch.forward_mode.is_dllm_extend():
            if forward_batch.input_ids.numel() == 0:
                return None, [], None, None, False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], None, None, out.can_run_graph

        self.vocab_size = model_runner.model_config.hf_config.text_config.vocab_size
        self.embed_tokens = model_runner.model.get_input_embeddings()
        return super().run(model_runner, forward_batch, algo_states)

    def _block_start_list(self, forward_batch: ForwardBatch) -> List[int]:
        return [0] * forward_batch.batch_size

    def max_steps(self, block_size: int) -> int:
        return self.max_denoising_steps

    def _temperature(self, step: int) -> float:
        return self.t_min + (self.t_max - self.t_min) * (
            step / self.max_denoising_steps
        )

    def _request_seed(self, forward_batch: ForwardBatch, index: int) -> int:
        if self.seed is not None:
            return self.seed
        rid = forward_batch.rids[index] if forward_batch.rids else str(index)
        return int.from_bytes(
            hashlib.blake2b(rid.encode(), digest_size=8).digest(), "little"
        )

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        device = forward_batch.input_ids.device
        states = []
        for index in range(forward_batch.batch_size):
            generator = torch.Generator(device=device)
            generator.manual_seed(self._request_seed(forward_batch, index))
            current = torch.randint(
                self.vocab_size,
                (self.block_size,),
                device=device,
                generator=generator,
            )
            states.append(
                {
                    "step": self.max_denoising_steps,
                    "current": current,
                    "argmax": current,
                    "history": [],
                    "self_conditioning": None,
                    "rng_state": generator.get_state(),
                    "finished": False,
                }
            )
        return states

    def prepare_inputs(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        states: List[Any],
    ) -> None:
        self._write_input_ids(forward_batch, states)

        signals = [state["self_conditioning"] for state in states]
        signal = next((value for value in signals if value is not None), None)
        if signal is not None:
            batched = torch.zeros(
                (len(states), self.block_size, signal.shape[-1]),
                dtype=signal.dtype,
                device=signal.device,
            )
            for index, value in enumerate(signals):
                if value is not None:
                    batched[index].copy_(value)
            signal = batched.view(len(states) * self.block_size, -1)

        forward_batch.input_embeds = model_runner.model.prepare_dllm_input_embeds(
            forward_batch.input_ids, signal
        )

    def _write_input_ids(self, forward_batch: ForwardBatch, states: List[Any]) -> None:
        current = torch.stack([state["current"] for state in states])
        forward_batch.input_ids.copy_(current.view(-1))

    def _soft_embeddings(self, probabilities: torch.Tensor) -> torch.Tensor:
        weight = self.embed_tokens.weight
        sharded = isinstance(self.embed_tokens, VocabParallelEmbedding)
        if sharded:
            shard = self.embed_tokens.shard_indices
            start, end = shard.org_vocab_start_index, shard.org_vocab_end_index
            # Slice before casting: only materialize this rank's vocabulary.
            probabilities = probabilities[..., start:end]
            weight = weight[: end - start]
        probabilities = probabilities.to(weight.dtype)
        soft_embeds = torch.matmul(probabilities, weight)
        if sharded and self.embed_tokens.tp_size > 1:
            soft_embeds = tensor_model_parallel_all_reduce(soft_embeds)
        scale = torch.as_tensor(
            self.embed_tokens.embed_scale, dtype=weight.dtype, device=weight.device
        )
        return soft_embeds * scale

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        logits = full_logits.view(
            forward_batch.batch_size, self.block_size, self.vocab_size
        )
        if all(state["finished"] for state in states):
            self._write_input_ids(forward_batch, states)
            return [True] * len(states)

        temperatures = logits.new_tensor(
            [self._temperature(state["step"]) for state in states]
        )
        statistics = (
            _compiled_denoiser_statistics if logits.is_cuda else _denoiser_statistics
        )
        probabilities, token_entropies, argmax_tokens = statistics(logits, temperatures)
        sorted_entropy, indices = torch.sort(token_entropies, dim=-1)
        cumulative_entropy = torch.cumsum(sorted_entropy, dim=-1)
        sorted_selected = cumulative_entropy - sorted_entropy <= self.entropy_bound
        selected = torch.zeros_like(sorted_selected).scatter(
            -1, indices, sorted_selected
        )
        confident = token_entropies.mean(dim=-1) < self.confidence_threshold
        finished = []
        active = []
        for index, state in enumerate(states):
            if state["finished"]:
                continue
            active.append(index)
            generator = torch.Generator(device=logits.device)
            generator.set_state(state["rng_state"])
            denoiser = _sample_denoiser(probabilities[index], generator)
            random_canvas = torch.randint(
                self.vocab_size,
                (self.block_size,),
                device=logits.device,
                generator=generator,
            )
            state["current"] = torch.where(selected[index], denoiser, random_canvas)
            argmax = argmax_tokens[index]
            history = state["history"]
            if self.stability_threshold == 0:
                stable = torch.ones((), dtype=torch.bool, device=logits.device)
            elif len(history) == self.stability_threshold:
                stable = (torch.stack(history) == argmax).all()
            else:
                stable = torch.zeros((), dtype=torch.bool, device=logits.device)
            history.append(argmax)
            if len(history) > self.stability_threshold:
                history.pop(0)
            state["step"] -= 1
            state["argmax"] = argmax
            state["rng_state"] = generator.get_state()
            finished.append((stable & confident[index]) | (state["step"] == 0))

        # One device-to-host synchronization for the whole batch. In particular,
        # stability checks must not call torch.equal once per request/history.
        done = [state["finished"] for state in states]
        for index, value in zip(active, torch.stack(finished).tolist()):
            states[index]["finished"] = value
            done[index] = value

        soft_embeds = None
        if not all(done):
            soft_embeds = self._soft_embeddings(
                probabilities.reshape(-1, self.vocab_size)
            ).view(len(states), self.block_size, -1)
        for index in active:
            state = states[index]
            if state["finished"]:
                state["current"] = state["argmax"]
                state["self_conditioning"] = None
            else:
                state["self_conditioning"] = soft_embeds[index]
        self._write_input_ids(forward_batch, states)
        return done


Algorithm = Gemma4Renoise
