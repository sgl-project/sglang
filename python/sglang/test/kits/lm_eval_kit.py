"""
This module provides a mixin class for running lm-eval harness evaluations
against SGLang servers
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml


@contextmanager
def scoped_env_vars(new_env: dict[str, str] | None):
    """Context manager to temporarily set environment variables."""
    if not new_env:
        yield
        return

    old_values = {}
    new_keys = []

    try:
        for key, value in new_env.items():
            if key in os.environ:
                old_values[key] = os.environ[key]
            else:
                new_keys.append(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old_values.items():
            os.environ[key] = value
        for key in new_keys:
            os.environ.pop(key, None)


class LMEvalMixin:
    """
    Mixin class for running lm-eval harness evaluations.
    """

    other_args: list[str] = []
    model_config_name: str = ""
    default_rtol: float = 0.08

    def test_lm_eval(self):
        """Run lm-eval evaluation and validate results."""
        # Flush cache before evaluation
        requests.get(self.base_url + "/flush_cache")

        eval_config = yaml.safe_load(
            Path(self.model_config_name).read_text(encoding="utf-8")
        )
        results = self.launch_lm_eval(eval_config)

        rtol = eval_config.get("rtol", self.default_rtol)

        success = True
        for task in eval_config["tasks"]:
            for metric in task["metrics"]:
                ground_truth = metric["value"]
                measured_value = results["results"][task["name"]][metric["name"]]
                print(
                    f"{task['name']} | {metric['name']}: "
                    f"ground_truth={ground_truth:.3f} | "
                    f"measured={measured_value:.3f} | rtol={rtol}"
                )
                success = success and np.isclose(
                    ground_truth, measured_value, rtol=rtol
                )

        self.assertTrue(success, f"lm-eval validation failed")

    def launch_lm_eval(self, eval_config: dict[str, Any]) -> dict:
        """
        Args:
            eval_config: Configuration dictionary with model and task settings
        """
        import lm_eval

        batch_size = eval_config.get("batch_size", "auto")
        backend = eval_config.get("backend", "local-completions")
        num_concurrent = eval_config.get("num_concurrent", 1)

        model_args = {
            "model": eval_config["model_name"],
            "base_url": self.base_url + "/v1/completions",
            "num_concurrent": num_concurrent,
        }

        env_vars = eval_config.get("env_vars", None)
        with scoped_env_vars(env_vars):
            results = lm_eval.simple_evaluate(
                model=backend,
                model_args=model_args,
                tasks=[task["name"] for task in eval_config["tasks"]],
                num_fewshot=eval_config.get("num_fewshot", 0),
                limit=eval_config.get("limit", None),
                apply_chat_template=eval_config.get("apply_chat_template", False),
                fewshot_as_multiturn=eval_config.get("fewshot_as_multiturn", False),
                gen_kwargs=eval_config.get("gen_kwargs"),
                batch_size=batch_size,
            )

        return results
