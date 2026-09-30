"""Beam response-template fallback detector."""

from sglang.srt.parser.beam import BEAM_RESPONSE_TEMPLATE
from sglang.srt.parser.response_template import ResponseTemplateToolDetector


class BeamDetector(ResponseTemplateToolDetector):
    response_template = BEAM_RESPONSE_TEMPLATE
