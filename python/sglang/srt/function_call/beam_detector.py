"""Beam response-template fallback detector."""

from sglang.srt.parser.beam import BEAM_RESPONSE_TEMPLATE
from sglang.srt.parser.response_template import ResponseTemplateToolDetector


class BeamDetector(ResponseTemplateToolDetector):
    response_template = BEAM_RESPONSE_TEMPLATE
    constrains_every_turn = True
    # Beam opens every assistant message with an analysis block, empty or not.
    grammar_requires_reasoning = True
