"""One greedy reference on the source, then compare the original restored run.

No engine lifecycle, retry or performance assertion. Uses the frozen public
helper for actual HTTP, output-token and weight-version validation.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("capture", "compare"))
    p.add_argument("--public-helper", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--engine-url")
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--reference", type=Path)
    p.add_argument("--source-receipt", type=Path)
    p.add_argument("--restore-result", type=Path)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "frozen_public_restore", a.public_helper
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    a.out.mkdir(parents=True, exist_ok=False)
    if a.phase == "capture":
        helper.require(
            a.engine_url and a.prompt_file, "Reference needs actual source and prompt"
        )
        http = helper.HTTP(a.engine_url, a.out, a.timeout)
        _, identity = helper.inspect_engine(http)
        prompt = a.prompt_file.read_text()
        request = dict(
            text=prompt,
            stream=True,
            return_logprob=True,
            logprob_start_len=-1,
            sampling_params=dict(temperature=0, max_new_tokens=a.max_new_tokens),
        )
        stream = helper.SSE(http, request, "uninterrupted-reference")
        output = helper.token_output(stream.finish()["raw"])
        _, after = helper.inspect_engine(http)
        helper.require(
            after == identity
            and output["weight_version"] == identity["weight_version"],
            "Reference weights changed or output version missing",
        )
        helper.require(
            output["stop_reason"] not in (None, "abort", "aborted")
            and 8 < len(output["token_ids"]) <= a.max_new_tokens,
            "Reference must finish and leave an active-interruption opportunity",
        )
        result = dict(
            status="REFERENCE_OBSERVED",
            identity=identity,
            output=output,
            prompt=helper.file_ref(a.prompt_file),
            sampling_params=request["sampling_params"],
            original_HTTP=http.refs,
            captured_unix_s=time.time(),
        )
        helper.write_json(a.out / "reference.json", result)
    else:
        helper.require(
            all((a.reference, a.source_receipt, a.restore_result)),
            "Missing original results",
        )
        reference = json.loads(a.reference.read_text())
        source = json.loads(a.source_receipt.read_text())
        restored = json.loads(a.restore_result.read_text())
        helper.require(
            reference.get("status") == "REFERENCE_OBSERVED"
            and restored.get("status") == "PASS"
            and source.get("status") == "CHECKPOINT_READY_SOURCE_LEFT_PAUSED",
            "Original reference/restore checks did not complete",
        )
        helper.require(
            reference["sampling_params"]
            == dict(temperature=0, max_new_tokens=source["max_new_tokens"]),
            "Reference sampling/budget differs from the frozen export invocation",
        )
        helper.require(
            reference["identity"] == source["identity"]
            and reference["identity"] == restored["initial_inspect"]["identity"],
            "Reference/source/target identity differs",
        )
        original = reference["output"]["token_ids"]
        resumed = restored["output"]["token_ids"]
        prefix = source["prefix"]["token_ids"]
        helper.require(
            prefix == original[: len(prefix)] and resumed == original,
            "Greedy interrupted-prefix/continuation differs from uninterrupted reference",
        )
        helper.require(
            restored["output"]["logprobs"] == reference["output"]["logprobs"],
            "Greedy continuation logprobs differ from uninterrupted reference",
        )
        helper.require(
            restored["output"]["stop_reason"] == reference["output"]["stop_reason"],
            "Reference/restore completion reason differs",
        )
        result = dict(
            status="PASS_GREEDY_REFERENCE_TOKEN_IDENTITY",
            token_count=len(original),
            interrupted_prefix_tokens=len(prefix),
            prefix_equal=True,
            continuation_equal=True,
            logprob_count=len(reference["output"]["logprobs"]),
            logprobs_equal=True,
            source_reference=helper.file_ref(a.reference),
            source_checkpoint=helper.file_ref(a.source_receipt),
            target_restore=helper.file_ref(a.restore_result),
            scope="This fixed prompt/greedy run only; not general numerical equivalence or a benchmark",
        )
        helper.write_json(a.out / "correctness.json", result)
    print(json.dumps(dict(status=result["status"], output=str(a.out))))


if __name__ == "__main__":
    main()
