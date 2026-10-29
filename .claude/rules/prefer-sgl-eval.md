# Accuracy evals: use sgl-eval

Use [sgl-eval](https://github.com/sgl-project/sgl-eval) whenever you measure a
model's accuracy.

- **Against a server:** run the `sgl-eval` CLI from its own virtualenv, not the
  server's env, at the version pinned in the `test` extra of
  `python/pyproject.toml`.

  ```bash
  sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --num-examples 200
  ```

- **In a test:** use `sglang.test.sgl_eval_utils.run_sgl_eval` or the kits
  that wrap it. A new `GSM8KMixin` test sets `gsm8k_backend = "sgl_eval"`.

- **Sampling:** use the model's recommended settings (temperature, top_p,
  thinking, max tokens) unless the user says otherwise. sgl-eval overrides
  them with its own defaults, so pass them explicitly, or use
  `--load-preset-from-model-id` when the model has a preset. `--thinking` only
  sets the `thinking` chat-template key; if the model's template reads a
  different one (commonly `enable_thinking`), pass that key with
  `--chat-template-kwarg` instead. When comparing against a CI test, use that
  test's settings (printed in its log as `sgl-eval <name> generation: {...}`).

Do not use the deprecated `sglang.test.run_eval` or `few_shot_gsm8k`, `lm-eval`,
or a hand-written scorer. The exceptions are evals sgl-eval doesn't ship, and
base models, since sgl-eval only sends chat requests.

Only compare scores from the same harness and settings, and say which ones you
used when you report a number.
