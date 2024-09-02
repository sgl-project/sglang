from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ChoicesDecision:
    decision: str
    meta_info: Optional[Dict[str, Any]] = None


class ChoicesSamplingMethod(ABC):

    @property
    def requires_unconditional_logprobs(self) -> bool:
        return False

    @abstractmethod
    def __call__(
        self,
        *,
        choices: List[str],
        normalized_prompt_logprobs: List[float],
        input_token_logprobs: List[List[Any]],
        output_token_logprobs: List[List[Any]],
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,
    ) -> ChoicesDecision: ...


class TokenLengthNormalized(ChoicesSamplingMethod):

    def __call__(
        self,
        *,
        choices: List[str],
        normalized_prompt_logprobs: List[float],
        input_token_logprobs: List[List[Any]],
        output_token_logprobs: List[List[Any]],
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,
    ) -> ChoicesDecision:
        """Select the option with the highest token length normalized prompt logprob."""
        best_choice = choices[np.argmax(normalized_prompt_logprobs)]
        meta_info = {
            "normalized_prompt_logprobs": normalized_prompt_logprobs,
            "input_token_logprobs": input_token_logprobs,
            "output_token_logprobs": output_token_logprobs,
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)


token_length_normalized = TokenLengthNormalized()


class GreedyTokenSelection(ChoicesSamplingMethod):

    def __call__(
        self,
        *,
        choices: List[str],
        normalized_prompt_logprobs: List[float],
        input_token_logprobs: List[List[Any]],
        output_token_logprobs: List[List[Any]],
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,
    ) -> ChoicesDecision:
        """Select the option based on greedy logprob selection. For overlapping options
        where one option is a subset of a longer option, extend the shorter option using
        its average logprob for comparison against the longer option."""

        num_options = len(choices)
        max_tokens = max(len(option) for option in input_token_logprobs)
        logprob_matrix = self._build_logprob_matrix(
            input_token_logprobs, max_tokens, num_options
        )
        remaining = self._greedy_selection(logprob_matrix, num_options, max_tokens)

        best_choice = choices[remaining[0]]
        meta_info = {
            "normalized_prompt_logprobs": normalized_prompt_logprobs,
            "input_token_logprobs": input_token_logprobs,
            "output_token_logprobs": output_token_logprobs,
            "greedy_logprob_matrix": logprob_matrix.tolist(),
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)

    def _build_logprob_matrix(self, input_token_logprobs, max_tokens, num_options):
        logprob_matrix = np.zeros((num_options, max_tokens))
        for i, option in enumerate(input_token_logprobs):
            actual_logprobs = [token[0] for token in option]
            avg_logprob = np.mean(actual_logprobs)
            logprob_matrix[i, : len(option)] = actual_logprobs
            if len(option) < max_tokens:
                logprob_matrix[i, len(option) :] = avg_logprob
        return logprob_matrix

    def _greedy_selection(self, logprob_matrix, num_options, max_tokens):
        remaining = np.arange(num_options)
        for j in range(max_tokens):
            max_logprob = np.max(logprob_matrix[remaining, j])
            remaining = remaining[logprob_matrix[remaining, j] == max_logprob]
            if len(remaining) == 1:
                break
        return remaining


greedy_token_selection = GreedyTokenSelection()


class UnconditionalLikelihoodNormalized(ChoicesSamplingMethod):

    @property
    def requires_unconditional_logprobs(self) -> bool:
        return True

    def __call__(
        self,
        *,
        choices: List[str],
        normalized_prompt_logprobs: List[float],
        input_token_logprobs: List[List[Any]],
        output_token_logprobs: List[List[Any]],
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,
    ) -> ChoicesDecision:
        """Select the option with the highest average token logprob once normalized by
        the unconditional token logprobs.

        The first unconditional token logprob is assumed to be None. If so, it is
        replaced with 0 for the purposes of normalization."""

        if unconditional_token_logprobs is None:
            raise ValueError(
                "Unconditional token logprobs are required for this method."
            )

        normalized_unconditional_prompt_logprobs = self._normalize_logprobs(
            input_token_logprobs, unconditional_token_logprobs
        )

        best_choice = choices[np.argmax(normalized_unconditional_prompt_logprobs)]
        meta_info = {
            "normalized_prompt_logprobs": normalized_prompt_logprobs,
            "input_token_logprobs": input_token_logprobs,
            "output_token_logprobs": output_token_logprobs,
            "unconditional_token_logprobs": unconditional_token_logprobs,
            "normalized_unconditional_prompt_logprobs": normalized_unconditional_prompt_logprobs,
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)

    def _normalize_logprobs(self, input_token_logprobs, unconditional_token_logprobs):
        normalized_unconditional_prompt_logprobs = []
        for inputs, unconditionals in zip(
            input_token_logprobs, unconditional_token_logprobs
        ):
            inputs_logprobs = np.array([token[0] for token in inputs])
            unconditionals_logprobs = np.array([token[0] for token in unconditionals])
            unconditionals_logprobs[0] = unconditionals_logprobs[0] or 0
            normalized_unconditional_prompt_logprobs.append(
                float(np.mean(inputs_logprobs - unconditionals_logprobs))
            )
        return normalized_unconditional_prompt_logprobs


unconditional_likelihood_normalized = UnconditionalLikelihoodNormalized()
