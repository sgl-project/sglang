"""Beam response-template fallback detector."""

from sglang.srt.parser.beam import BEAM_RESPONSE_TEMPLATE
from sglang.srt.parser.response_template import ResponseTemplateToolDetector


class BeamDetector(ResponseTemplateToolDetector):
    response_template = BEAM_RESPONSE_TEMPLATE
    constrains_every_turn = True
    grammar_requires_reasoning = True
