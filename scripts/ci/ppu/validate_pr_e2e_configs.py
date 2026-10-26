"""Validate catalog configs with the actual evaluation kits in the CPU environment."""

from select_pr_e2e import ROOT, load_catalog, read_json


def validate_configs(entries):
    from sglang.test.kits import (
        accuracy_eval_kit,
        answer_eval_kit,
        pd_perf_eval_kit,
        perf_eval_kit,
    )

    validators = {
        "answer": answer_eval_kit.validate_test_config,
        "accuracy": accuracy_eval_kit.validate_test_config,
        "perf": perf_eval_kit.validate_test_config,
        "pd": pd_perf_eval_kit.validate_pd_test_config,
    }
    for entry in entries:
        config = read_json(ROOT / entry["config"])
        try:
            validators[entry["category"]](config)
        except (
            answer_eval_kit.AnswerEvalError,
            accuracy_eval_kit.AccuracyEvalError,
            perf_eval_kit.PerfEvalError,
        ) as error:
            expected = {}
            if not entry["blocked"] or str(error) != expected.get(entry["config"]):
                raise
        else:
            if entry["blocked"]:
                raise ValueError(f"Reassess resolved catalog blocker: {entry['id']}")
    print(
        f"Schema-checked {len(entries)} catalog configs (blocked entries remain unselectable)"
    )


if __name__ == "__main__":
    validate_configs(load_catalog()["entries"])
