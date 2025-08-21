# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Faster constrained decoding with jump forward decoding / compressed finite state machine.
Reference: https://lmsys.org/blog/2024-02-05-compressed-fsm/
"""

import dataclasses
import logging
from collections import defaultdict
from typing import Optional

import interegular
from interegular import InvalidSyntax
from outlines.caching import cache

from sglang.srt.utils import get_bool_env_var

try:
    # outlines >= 0.1.0
    from outlines_core.fsm.outlines_core_rs import FSMInfo
    from outlines_core.fsm.regex import make_byte_level_fsm, make_deterministic_fsm
except ImportError:
    # outlines <= 0.0.46
    from outlines.fsm.regex import FSMInfo, make_byte_level_fsm, make_deterministic_fsm

IP_REGEX = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"

# Env var was set in sglang.srt.server_args.ServerArgs.__post__init__
DISABLE_DISK_CACHE = get_bool_env_var("SGLANG_DISABLE_OUTLINES_DISK_CACHE", "true")

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class JumpEdge:
    symbol: str = None
    symbol_next_state: int = None
    byte: int = None
    byte_next_state: int = None


def disk_cache(expire: Optional[float] = None, typed=False, ignore=()):
    if not DISABLE_DISK_CACHE:
        return cache(expire, typed, ignore)
    else:
        return lambda fn: None


@disk_cache()
def init_state_to_jump_forward(regex_string):
    try:
        regex_pattern = interegular.parse_pattern(regex_string)
    except InvalidSyntax as e:
        logger.warning(f"skip invalid regex: {regex_string}, {e=}")
        return

    byte_fsm = make_byte_level_fsm(regex_pattern.to_fsm().reduce(), keep_utf8=True)
    regex_fsm, _ = make_deterministic_fsm(byte_fsm)

    fsm_info: FSMInfo = regex_fsm.fsm_info

    symbol_to_id = fsm_info.alphabet_symbol_mapping
    id_to_symbol = {}
    for symbol, id_ in symbol_to_id.items():
        id_to_symbol.setdefault(id_, []).append(symbol)

    transitions = fsm_info.transitions

    outgoings_ct = defaultdict(int)
    # NOTE(lsyin): Final states can lead to terminate, so they have one outgoing edge naturally
    for s in fsm_info.finals:
        outgoings_ct[s] = 1

    state_to_jump_forward = {}
    for (state, id_), next_state in transitions.items():
        if id_ == fsm_info.alphabet_anything_value:
            # Arbitrarily symbol cannot be recognized as jump forward
            continue

        symbols = id_to_symbol[id_]
        for c in symbols:
            if len(c) > 1:
                # Skip byte level transitions like c = "5E"
                continue

            outgoings_ct[state] += 1
            if outgoings_ct[state] > 1:
                if state in state_to_jump_forward:
                    del state_to_jump_forward[state]
                break

            state_to_jump_forward[state] = JumpEdge(
                symbol=c,
                symbol_next_state=next_state,
            )

    # Process the byte level jump forward
    outgoings_ct = defaultdict(int)
    for s in fsm_info.finals:
        outgoings_ct[s] = 1

    for (state, id_), next_state in transitions.items():
        if id_ == fsm_info.alphabet_anything_value:
            continue
        symbols = id_to_symbol[id_]
        for c in symbols:
            byte_ = None
            if len(c) == 1 and ord(c) < 0x80:
                # ASCII character
                byte_ = ord(c)
            elif len(c) > 1:
                # FIXME: This logic is due to the leading \x00
                # https://github.com/outlines-dev/outlines/pull/930
                byte_ = int(symbols[0][1:], 16)

            if byte_ is not None:
                outgoings_ct[state] += 1
                if outgoings_ct[state] > 1:
                    if state in state_to_jump_forward:
                        del state_to_jump_forward[state]
                    break
                e = state_to_jump_forward.get(state, JumpEdge())
                e.byte = byte_
                e.byte_next_state = next_state
                state_to_jump_forward[state] = e

    return state_to_jump_forward


class OutlinesJumpForwardMap:
    def __init__(self, regex_string):
        self.state_to_jump_forward = init_state_to_jump_forward(regex_string)

    def jump_forward_symbol(self, state):
        jump_forward_str = ""
        next_state = state
        while state in self.state_to_jump_forward:
            e = self.state_to_jump_forward[state]
            if e.symbol is None:
                break
            jump_forward_str += e.symbol
            next_state = e.symbol_next_state
            state = next_state

        return jump_forward_str, next_state

    def jump_forward_byte(self, state):
        if state not in self.state_to_jump_forward:
            return None

        jump_forward_bytes = []
        next_state = None
        while state in self.state_to_jump_forward:
            e = self.state_to_jump_forward[state]
            assert e.byte is not None and e.byte_next_state is not None
            jump_forward_bytes.append((e.byte, e.byte_next_state))
            next_state = e.byte_next_state
            state = next_state

        return jump_forward_bytes

    def is_jump_forward_symbol_state(self, state):
        return (
            state in self.state_to_jump_forward
            and self.state_to_jump_forward[state].symbol is not None
        )


def test_main(regex_string):
    jump_forward_map = OutlinesJumpForwardMap(regex_string)
    for state, e in jump_forward_map.state_to_jump_forward.items():
        if e.symbol is not None:
            jump_forward_str, next_state = jump_forward_map.jump_forward_symbol(state)
            print(f"{state} -> {next_state}", jump_forward_str)
        bytes_ = jump_forward_map.jump_forward_byte(state)
        print(f"{state} -> {bytes_[-1][1]}", [hex(b) for b, _ in bytes_])


if __name__ == "__main__":
    import outlines

    outlines.caching.clear_cache()
    test_main(r"The google's DNS sever address is " + IP_REGEX)
    test_main(r"霍格沃茨特快列车|霍比特人比尔博")
    # 霍格: \xe9\x9c\x8d \xe6\xa0\xbc ...
    # 霍比: \xe9\x9c\x8d \xe6\xaf\x94 ...

    test_main(r"[-+]?[0-9]+[ ]*")
