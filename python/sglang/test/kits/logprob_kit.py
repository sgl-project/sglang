"""Reusable cross-mode log-probability test kit.

Generates tokens from a target server, then scores the same token sequence
via prefill on a baseline server (or the same server). Compares all logprob
artifacts between the two to verify correctness.

Verified artifacts:
    - output_token_logprobs: per-token value comparison
    - input_token_logprobs:  per-token value comparison
    - output_top_logprobs:   top-k token IDs and values
    - input_top_logprobs:    top-k token IDs and values
    - output_token_ids_logprobs: values for user-specified token IDs
    - input_token_ids_logprobs:  values for user-specified token IDs
    - logprob_start_len:     boundary correctness
    - return_text_in_logprobs: structural validation

"""

import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import requests

CROSS_MODE_PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Today is a sunny day and I like",
]
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_TOP_LOGPROBS_NUM = 5
DEFAULT_PROBE_TOKEN_IDS = [1, 2, 10, 100, 1000]
DEFAULT_DECIMAL_PLACES = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compare_token_logprobs(test_case, target_lps, baseline_lps, places, tag):
    """Compare per-token logprob values, skipping None entries."""
    test_case.assertEqual(
        len(target_lps),
        len(baseline_lps),
        msg=f"[{tag}] length mismatch: {len(target_lps)} vs {len(baseline_lps)}",
    )

    diffs = []
    for i in range(len(target_lps)):
        t_val, t_id = target_lps[i][0], target_lps[i][1]
        b_val, b_id = baseline_lps[i][0], baseline_lps[i][1]

        if t_val is None or b_val is None:
            continue

        test_case.assertEqual(
            t_id,
            b_id,
            msg=f"[{tag}] token_id mismatch at pos {i}: {t_id} vs {b_id}",
        )
        test_case.assertAlmostEqual(
            t_val,
            b_val,
            places=places,
            msg=f"[{tag}] pos {i}: target={t_val:.6f} vs baseline={b_val:.6f}",
        )
        diffs.append(abs(t_val - b_val))

    if diffs:
        print(f"[{tag}] max|diff|={max(diffs):.6f} mean|diff|={np.mean(diffs):.6f}")


def _compare_top_logprobs(test_case, target_tops, baseline_tops, places, tag):
    """Compare top-k logprobs: common token IDs and their values."""
    test_case.assertEqual(
        len(target_tops),
        len(baseline_tops),
        msg=f"[{tag}] length mismatch",
    )

    for pos in range(len(target_tops)):
        if target_tops[pos] is None or baseline_tops[pos] is None:
            continue
        dec_top = {t[1]: t[0] for t in target_tops[pos] if t[0] is not None}
        scr_top = {t[1]: t[0] for t in baseline_tops[pos] if t[0] is not None}
        common_ids = set(dec_top.keys()) & set(scr_top.keys())
        test_case.assertGreater(
            len(common_ids),
            0,
            msg=f"[{tag}] pos {pos}: no common top-k IDs",
        )
        for tid in common_ids:
            test_case.assertAlmostEqual(
                dec_top[tid],
                scr_top[tid],
                places=places,
                msg=f"[{tag}] pos {pos} tid={tid}: "
                f"target={dec_top[tid]:.6f} vs baseline={scr_top[tid]:.6f}",
            )


def _compare_ids_logprobs(test_case, target_ids, baseline_ids, places, tag):
    """Compare token_ids_logprob values for user-specified token IDs."""
    test_case.assertEqual(
        len(target_ids),
        len(baseline_ids),
        msg=f"[{tag}] length mismatch",
    )

    for pos in range(len(target_ids)):
        if target_ids[pos] is None or baseline_ids[pos] is None:
            continue
        dec_map = {t[1]: t[0] for t in target_ids[pos]}
        scr_map = {t[1]: t[0] for t in baseline_ids[pos]}
        test_case.assertEqual(
            set(dec_map.keys()),
            set(scr_map.keys()),
            msg=f"[{tag}] pos {pos}: token IDs differ",
        )
        for tid in dec_map:
            test_case.assertAlmostEqual(
                dec_map[tid],
                scr_map[tid],
                places=places,
                msg=f"[{tag}] pos {pos} tid={tid}: "
                f"target={dec_map[tid]:.6f} vs baseline={scr_map[tid]:.6f}",
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_with_logprobs(
    url,
    prompt_or_ids,
    max_new_tokens,
    top_logprobs_num,
    token_ids_logprob=None,
    logprob_start_len=-1,
    return_text_in_logprobs=False,
    sampling_params=None,
):
    """Send a generate request with logprob options to an SGLang server.

    Args:
        url: Server base URL.
        prompt_or_ids: Text string or list of token IDs.
        max_new_tokens: Number of tokens to generate.
        top_logprobs_num: Number of top logprobs to return.
        token_ids_logprob: Optional token IDs for the token_ids_logprob artifact.
        logprob_start_len: Starting position for input logprobs (-1 to skip).
        return_text_in_logprobs: Include token text in logprob tuples.
        sampling_params: Optional dict of sampling parameters.  Defaults to
            ``{"temperature": 0, "ignore_eos": True}``.  ``max_new_tokens``
            is always set from the explicit argument.
    """
    if isinstance(prompt_or_ids, str):
        payload = {"text": prompt_or_ids}
    else:
        payload = {"input_ids": prompt_or_ids}

    if sampling_params is None:
        sampling_params = {"temperature": 0, "ignore_eos": True}
    sampling_params = {**sampling_params, "max_new_tokens": max_new_tokens}

    payload.update(
        {
            "sampling_params": sampling_params,
            "return_logprob": True,
            "top_logprobs_num": top_logprobs_num,
            "logprob_start_len": logprob_start_len,
            "return_text_in_logprobs": return_text_in_logprobs,
        }
    )
    if token_ids_logprob is not None:
        payload["token_ids_logprob"] = token_ids_logprob

    response = requests.post(url + "/generate", json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Server returned {response.status_code}: {response.text}")
    return response.json()


def hf_logprobs_for_sequence(hf_model, token_ids, temperature=1.0):
    """Run a HuggingFace forward pass and return per-position log-prob vectors.

    Returns a ``[T, V]`` tensor where ``logprobs[i]`` is the log-softmax of
    ``logits[i] / temperature``.  ``logits[i]`` predicts ``token[i+1]``.

    Args:
        hf_model: A HuggingFace ``AutoModelForCausalLM`` instance.
        token_ids: List of token IDs forming the sequence.
        temperature: Temperature divisor applied before log-softmax.
    """
    import torch
    import torch.nn.functional as F

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=hf_model.device)
    with torch.inference_mode():
        logits = hf_model(input_ids=input_tensor, return_dict=True).logits[0]
    if temperature != 1.0:
        logits = logits / temperature
    return F.log_softmax(logits.float(), dim=-1)


def assert_position_logprobs_match(
    test_case,
    ref_lp_vec,
    sgl_token_logprob,
    sgl_top_logprobs=None,
    sgl_ids_logprobs=None,
    token_ids=None,
    top_k=5,
    rtol=0.20,
    atol=0.0,
    tag="",
):
    """Compare SGLang logprob output against a reference vector at one position.

    Checks token-level logprob, top-k logprobs, and specified-token-ID
    logprobs against the reference ``[V]`` log-prob vector (e.g. from HF).

    Positions whose SGLang logprob value is ``None`` (e.g. position 0 of
    input logprobs) are silently skipped.

    Args:
        test_case: ``unittest.TestCase`` instance for assertions.
        ref_lp_vec: ``[V]`` reference log-prob vector (torch.Tensor).
        sgl_token_logprob: ``(logprob, token_id, text)`` tuple from SGLang.
        sgl_top_logprobs: List of ``(logprob, token_id, text)`` tuples, or
            ``None`` to skip top-k comparison.
        sgl_ids_logprobs: List of ``(logprob, token_id, text)`` tuples for
            user-specified IDs, or ``None`` to skip.
        token_ids: Token IDs corresponding to *sgl_ids_logprobs*.
        top_k: Number of top logprobs to compare.
        rtol: Relative tolerance for ``torch.allclose``.
        atol: Absolute tolerance for ``torch.allclose``.
        tag: Descriptive tag for error messages.
    """
    import torch

    sgl_val, sgl_idx = sgl_token_logprob[0], sgl_token_logprob[1]
    if sgl_val is None:
        return

    ref_val = ref_lp_vec[sgl_idx].item()
    test_case.assertTrue(
        torch.allclose(
            torch.tensor([ref_val]),
            torch.tensor([float(sgl_val)]),
            rtol=rtol,
            atol=atol,
        ),
        msg=f"[{tag}] token logprob mismatch: ref={ref_val:.6f} vs SGL={sgl_val:.6f}",
    )

    if sgl_top_logprobs is not None:
        ref_topk_vals, _ = torch.topk(ref_lp_vec, k=top_k, dim=-1)
        sgl_vals = [float(t[0]) for t in sgl_top_logprobs if t and t[0] is not None]
        if sgl_vals:
            sgl_topk = torch.tensor(
                sgl_vals[:top_k],
                dtype=torch.float32,
                device=ref_lp_vec.device,
            )
            k = min(ref_topk_vals.numel(), sgl_topk.numel())
            test_case.assertTrue(
                torch.allclose(ref_topk_vals[:k], sgl_topk[:k], rtol=rtol, atol=atol),
                msg=f"[{tag}] top-k mismatch",
            )

    if sgl_ids_logprobs is not None and token_ids:
        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=ref_lp_vec.device)
        ref_ids_vals = ref_lp_vec[ids_tensor]
        sgl_ids_vals = torch.tensor(
            [float(v) for v, _, _ in sgl_ids_logprobs],
            dtype=torch.float32,
            device=ref_lp_vec.device,
        )
        test_case.assertTrue(
            torch.allclose(ref_ids_vals, sgl_ids_vals, rtol=rtol, atol=atol),
            msg=f"[{tag}] token-IDs mismatch",
        )


def run_logprob_cross_mode_check(
    test_case,
    prompts=None,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    top_logprobs_num=DEFAULT_TOP_LOGPROBS_NUM,
    token_ids_logprob=None,
    return_text_in_logprobs=False,
    decimal_places=DEFAULT_DECIMAL_PLACES,
    baseline_url=None,
):
    """Run cross-mode logprob comparison across all artifacts.

    1. Generate tokens from ``test_case.base_url`` (with logprob_start_len=0).
    2. Score the full sequence via prefill on the baseline server.
    3. Compare output_token_logprobs, input_token_logprobs, output_top_logprobs,
       input_top_logprobs, output_token_ids_logprobs, input_token_ids_logprobs.

    When *baseline_url* is ``None``, the target server itself is used for the
    scoring step.  Because scoring uses ``max_new_tokens=0`` (pure prefill),
    speculative decoding is not involved, making the target server a valid
    non-speculative baseline for its own decode logprobs.

    Args:
        test_case: ``unittest.TestCase`` instance (must have ``base_url``).
        prompts: List of prompt strings.
        max_new_tokens: Tokens to generate per prompt.
        top_logprobs_num: Top-k count for top_logprobs.
        token_ids_logprob: Token IDs for the token_ids_logprob artifact.
        return_text_in_logprobs: Whether to include token text.
        decimal_places: ``assertAlmostEqual`` precision (``places``).
        baseline_url: URL of the baseline server.  Defaults to ``test_case.base_url``.
    """
    target_url = test_case.base_url
    if baseline_url is None:
        baseline_url = target_url
    if prompts is None:
        prompts = list(CROSS_MODE_PROMPTS)
    if token_ids_logprob is None:
        token_ids_logprob = list(DEFAULT_PROBE_TOKEN_IDS)

    for round_idx, prompt in enumerate(prompts):
        tag_prefix = f"round {round_idx}"
        print(f"\n--- Cross-mode check {tag_prefix}: {prompt!r} ---")

        # Step 1: generate from target with logprob_start_len=0
        gen_res = generate_with_logprobs(
            target_url,
            prompt,
            max_new_tokens,
            top_logprobs_num,
            token_ids_logprob,
            logprob_start_len=0,
            return_text_in_logprobs=return_text_in_logprobs,
        )
        meta = gen_res["meta_info"]
        P = meta["prompt_tokens"]

        input_token_ids = [t[1] for t in meta["input_token_logprobs"]]
        output_token_ids = [t[1] for t in meta["output_token_logprobs"]]
        full_sequence = input_token_ids + output_token_ids

        # Step 2: score the full sequence via prefill on baseline
        score_res = generate_with_logprobs(
            baseline_url,
            full_sequence,
            max_new_tokens=0,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            logprob_start_len=0,
            return_text_in_logprobs=return_text_in_logprobs,
        )
        score_meta = score_res["meta_info"]

        # ----- output_token_logprobs -----
        _compare_token_logprobs(
            test_case,
            meta["output_token_logprobs"],
            score_meta["input_token_logprobs"][P:],
            decimal_places,
            f"{tag_prefix} output_token_logprobs",
        )

        # ----- input_token_logprobs -----
        _compare_token_logprobs(
            test_case,
            meta["input_token_logprobs"],
            score_meta["input_token_logprobs"][:P],
            decimal_places,
            f"{tag_prefix} input_token_logprobs",
        )

        # ----- output_top_logprobs -----
        if top_logprobs_num > 0:
            _compare_top_logprobs(
                test_case,
                meta["output_top_logprobs"],
                score_meta["input_top_logprobs"][P:],
                decimal_places,
                f"{tag_prefix} output_top_logprobs",
            )

            # ----- input_top_logprobs -----
            _compare_top_logprobs(
                test_case,
                meta["input_top_logprobs"],
                score_meta["input_top_logprobs"][:P],
                decimal_places,
                f"{tag_prefix} input_top_logprobs",
            )

        # ----- output_token_ids_logprobs -----
        if token_ids_logprob:
            _compare_ids_logprobs(
                test_case,
                meta["output_token_ids_logprobs"],
                score_meta["input_token_ids_logprobs"][P:],
                decimal_places,
                f"{tag_prefix} output_token_ids_logprobs",
            )

            # ----- input_token_ids_logprobs -----
            target_in_ids = meta.get("input_token_ids_logprobs")
            baseline_in_ids = score_meta.get("input_token_ids_logprobs")
            if target_in_ids is not None and baseline_in_ids is not None:
                _compare_ids_logprobs(
                    test_case,
                    target_in_ids,
                    baseline_in_ids[:P],
                    decimal_places,
                    f"{tag_prefix} input_token_ids_logprobs",
                )

        # ----- logprob_start_len boundary -----
        test_case.assertEqual(
            len(meta["input_token_logprobs"]),
            P,
            msg=f"{tag_prefix}: input_token_logprobs length != prompt_tokens",
        )

        # ----- return_text_in_logprobs structural check -----
        if return_text_in_logprobs:
            for lp in meta["output_token_logprobs"]:
                test_case.assertIsNotNone(
                    lp[2],
                    msg=f"{tag_prefix}: output token text should not be None",
                )

        print(f"--- {tag_prefix} passed ---")


def run_logprob_start_len_check(
    test_case,
    prompts=None,
    max_new_tokens=8,
    start_lens=None,
):
    """Verify logprob_start_len boundary correctness.

    Uses ``test_case.base_url``.  For each prompt and start_len, verifies that:
    - ``len(input_token_logprobs) == prompt_tokens - logprob_start_len``
    - ``len(input_top_logprobs)``  matches
    - ``len(output_token_logprobs) == max_new_tokens``
    """
    if prompts is None:
        prompts = list(CROSS_MODE_PROMPTS)
    target_url = test_case.base_url

    for prompt in prompts:
        probe_res = generate_with_logprobs(
            target_url,
            prompt,
            max_new_tokens=1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            logprob_start_len=-1,
        )
        P = probe_res["meta_info"]["prompt_tokens"]

        test_start_lens = (
            start_lens if start_lens is not None else [0, 1, P // 2, P - 1]
        )

        for sl in test_start_lens:
            if sl >= P:
                continue
            with test_case.subTest(prompt=prompt, logprob_start_len=sl):
                res = generate_with_logprobs(
                    target_url,
                    prompt,
                    max_new_tokens,
                    top_logprobs_num=5,
                    token_ids_logprob=None,
                    logprob_start_len=sl,
                )
                meta = res["meta_info"]

                expected_len = P - sl
                test_case.assertEqual(
                    len(meta["input_token_logprobs"]),
                    expected_len,
                    msg=f"start_len={sl}: input_token_logprobs len",
                )
                test_case.assertEqual(
                    len(meta["input_top_logprobs"]),
                    expected_len,
                    msg=f"start_len={sl}: input_top_logprobs len",
                )
                test_case.assertEqual(
                    len(meta["output_token_logprobs"]),
                    max_new_tokens,
                )
                test_case.assertEqual(
                    meta["prompt_tokens"],
                    sl + len(meta["input_token_logprobs"]),
                )

                print(
                    f"[logprob_start_len={sl} {prompt!r}] "
                    f"input={len(meta['input_token_logprobs'])}, "
                    f"output={len(meta['output_token_logprobs'])}"
                )


def run_logprob_check(test_case: unittest.TestCase, arg):
    """Verify logprob shape correctness for a single parameter combination.

    Args:
        test_case: ``unittest.TestCase`` instance (must have ``self.base_url``).
        arg: Tuple of (input_len, output_len, temperature, logprob_start_len,
             return_logprob, top_logprobs_num).
    """
    (
        input_len,
        output_len,
        temperature,
        logprob_start_len,
        return_logprob,
        top_logprobs_num,
    ) = arg
    input_ids = list(range(input_len))

    response = requests.post(
        test_case.base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
        },
    )
    res = response.json()

    test_case.assertEqual(res["meta_info"]["prompt_tokens"], input_len)
    test_case.assertEqual(res["meta_info"]["completion_tokens"], output_len)

    if return_logprob:
        test_case.assertEqual(
            len(res["meta_info"]["input_token_logprobs"]) + logprob_start_len,
            res["meta_info"]["prompt_tokens"],
        )
        test_case.assertEqual(
            len(res["meta_info"]["output_token_logprobs"]), output_len
        )

        if top_logprobs_num:
            test_case.assertEqual(
                len(res["meta_info"]["input_top_logprobs"]) + logprob_start_len,
                res["meta_info"]["prompt_tokens"],
            )
            test_case.assertEqual(
                len(res["meta_info"]["output_top_logprobs"]), output_len
            )

            for i in range(output_len):
                test_case.assertEqual(
                    len(res["meta_info"]["output_top_logprobs"][i]),
                    top_logprobs_num,
                )

                if temperature == 0:
                    rank = 0
                    while rank < len(res["meta_info"]["output_top_logprobs"][i]):
                        try:
                            test_case.assertListEqual(
                                res["meta_info"]["output_token_logprobs"][i],
                                res["meta_info"]["output_top_logprobs"][i][rank],
                            )
                            break
                        except AssertionError:
                            # Tie-breaking: allow next rank when values match.
                            if (
                                res["meta_info"]["output_top_logprobs"][i][rank][0]
                                == res["meta_info"]["output_top_logprobs"][i][rank + 1][
                                    0
                                ]
                            ):
                                rank += 1
                            else:
                                raise


def run_logprob_mixed_check(
    test_case,
    input_lens=None,
    output_lens=None,
    logprob_start_lens=None,
    max_workers=8,
):
    """Stress-test logprob shape correctness with many parameter combinations.

    Uses ``test_case.base_url``.  Sends concurrent requests with various
    (input_len, output_len, logprob_start_len, return_logprob,
    top_logprobs_num) combos and verifies that the returned array lengths
    are all correct.

    Args:
        test_case: ``unittest.TestCase`` instance (must have ``base_url``).
        input_lens: List of input lengths to test.
        output_lens: List of output lengths to test.
        logprob_start_lens: List of logprob_start_len values to test.
        max_workers: Concurrency for the thread pool.
    """
    if input_lens is None:
        input_lens = [200, 500, 1000, 2000]
    if output_lens is None:
        output_lens = [4, 8]
    if logprob_start_lens is None:
        logprob_start_lens = [0, 100, 300, 800, 1998]

    args = []
    temperature = 0
    for input_len in input_lens:
        for output_len in output_lens:
            for start_len in logprob_start_lens:
                for return_logprob in [True, False]:
                    for top_logprobs_num in [0, 5]:
                        if start_len >= input_len:
                            continue
                        args.append(
                            (
                                input_len,
                                output_len,
                                temperature,
                                start_len,
                                return_logprob,
                                top_logprobs_num,
                            )
                        )

    random.shuffle(args)
    print(f"[logprob_mixed] running {len(args)} parameter combinations")
    func = partial(run_logprob_check, test_case)
    with ThreadPoolExecutor(max_workers) as executor:
        list(executor.map(func, args))

    print(f"[logprob_mixed] all {len(args)} combinations passed")


# ---------------------------------------------------------------------------
# Public API – Mixin class
# ---------------------------------------------------------------------------


class LogprobCrossModeMixin:
    """Mixin providing cross-mode logprob test methods.

    Mix into a test class that has ``self.base_url`` pointing to the target
    server.  Override class attributes to customise behaviour.

    Example::

        class TestEagleLogprobs(EagleServerBase, LogprobCrossModeMixin):
            logprob_decimal_places = 2
    """

    logprob_decimal_places = DEFAULT_DECIMAL_PLACES
    logprob_max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    logprob_top_k = DEFAULT_TOP_LOGPROBS_NUM
    logprob_prompts = None
    logprob_probe_token_ids = None

    def test_cross_mode_logprobs(self):
        """Compare decode logprobs against prefill scoring for all artifacts."""
        print(f"Testing cross-mode logprobs for {self.base_url}")
        run_logprob_cross_mode_check(
            self,
            prompts=self.logprob_prompts,
            max_new_tokens=self.logprob_max_new_tokens,
            top_logprobs_num=self.logprob_top_k,
            token_ids_logprob=self.logprob_probe_token_ids,
            decimal_places=self.logprob_decimal_places,
        )

    def test_cross_mode_logprob_start_len(self):
        """Verify logprob_start_len boundary behaviour."""
        run_logprob_start_len_check(
            self,
            prompts=self.logprob_prompts,
        )

    def test_cross_mode_return_text_in_logprobs(self):
        """Verify return_text_in_logprobs structural correctness."""
        run_logprob_cross_mode_check(
            self,
            prompts=(self.logprob_prompts or CROSS_MODE_PROMPTS)[:1],
            max_new_tokens=8,
            return_text_in_logprobs=True,
            decimal_places=self.logprob_decimal_places,
        )

    def test_cross_mode_logprob_mixed(self):
        """Stress-test logprob shape correctness with many parameter combos."""
        run_logprob_mixed_check(self)
