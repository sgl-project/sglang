"""`evalscope perf` with one extra API plugin, `openai_openhands_replay`, registered through
evalscope's plugin registry.

usage: evalscope_client.py --history-tokenizer DIR <evalscope perf arguments, with --api openai_openhands_replay>

The plugin is the pinned `openai` plugin with two changes. Neither touches timing or token accounting:

* The text that the multi-turn strategy replays as the next turn's assistant history is the streamed
  content + reasoning_content, minus the tokenizer's special-token spellings and declared Harmony protocol controls. The released checkpoint marks Harmony controls special=false, so the adapter must identify those
  controls explicitly; this is a documented plain-text replay normalization. The Beam response parser forces skip_special_tokens=False, and with
  ignore_eos the model continues past <|return|> into Harmony-tagged text that the parser passes through as
  content. The Beam chat template rejects such history with HTTP 400 ("Pass plain text in API fields, not Harmony
  wire tags"), and even when it does not, the tags would re-enter the prompt as control tokens.
* The server's final usage record (prompt, completion and, when reported, cached tokens) is kept with the raw
  stream chunks in evalscope's database, so validation can read what the server reported for every request.
"""

import argparse
import os
import sys

from evalscope.perf.arguments import add_argument
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.registry import register_api

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_tool import special_token_pattern  # noqa: E402

API_NAME = "openai_openhands_replay"
RECORD_KEY = "openhands_replay"


@register_api(API_NAME)
class OpenHandsReplayPlugin(OpenaiPlugin):
    special_tokens = None

    async def process_request(self, client_session, url, headers, body):
        output = await super().process_request(client_session, url, headers, body)
        if output.success:
            streamed = output.generated_text
            output.generated_text = self.special_tokens.sub("", streamed)
            # No `choices` key, so evalscope's own response parsing skips this entry.
            output.response_messages.append(
                {
                    RECORD_KEY: {
                        "prompt_tokens": output.prompt_tokens,
                        "completion_tokens": output.completion_tokens,
                        "cached_tokens": output.real_cached_tokens,
                        "special_token_chars_removed": len(streamed)
                        - len(output.generated_text),
                    }
                }
            )
        return output


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--history-tokenizer", required=True)
    known, rest = pre.parse_known_args()
    OpenHandsReplayPlugin.special_tokens = special_token_pattern(
        known.history_tokenizer
    )

    parser = argparse.ArgumentParser(prog="evalscope perf (openhands replay)")
    add_argument(parser)
    args = parser.parse_args(rest)
    if args.api != API_NAME:
        parser.error(f"--api must be {API_NAME}")
    run_perf_benchmark(args)


if __name__ == "__main__":
    main()
