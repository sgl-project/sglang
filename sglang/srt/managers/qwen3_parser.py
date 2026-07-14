import logging
from typing import Tuple

class ProductionQwen3Detector:
    """
    Production-grade Qwen3 Reasoning Parser to fix infinite thinking loops.
    """
    START_TAG = "<" + "think>"
    END_TAG = "<" + "/think>"
    MAX_INPUT_SIZE = 1024 * 1024  # 1MB

    def __init__(self, *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.in_thinking = False
        self.thinking_finished = False
        self.buffer = ""

    def detect_and_parse(self, text: str) -> Tuple[str, str]:
        if not text:
            return "", ""

        combined = self.buffer + text
        self.buffer = ""

        if len(combined) > self.MAX_INPUT_SIZE:
            logging.critical("[Qwen3 Parser] Input exceeded 1MB. Force resetting.")
            self.reset()
            return "", combined

        reasoning_out = []
        content_out = []

        while combined:
            target_tag = self.END_TAG if self.in_thinking else self.START_TAG
            safe_length = self._calculate_safe_length(combined, target_tag)
            
            safe_text = combined[:safe_length]
            self.buffer = combined[safe_length:]
            combined = "" 

            tag_idx = safe_text.find(target_tag)

            if tag_idx != -1:
                before_tag = safe_text[:tag_idx]
                after_tag = safe_text[tag_idx + len(target_tag):]

                if not self.in_thinking:
                    if before_tag: content_out.append(before_tag)
                    self.in_thinking = True
                else:
                    if before_tag: reasoning_out.append(before_tag)
                    self.in_thinking = False
                    self.thinking_finished = True

                combined = after_tag + self.buffer
                self.buffer = ""
            else:
                if safe_text:
                    if self.in_thinking:
                        reasoning_out.append(safe_text)
                    else:
                        content_out.append(safe_text)
                
                if self.buffer:
                    break

        return "".join(reasoning_out), "".join(content_out)

    def _calculate_safe_length(self, text: str, tag: str) -> int:
        """Prefix buffering: if text ends with a partial tag prefix, truncate it to buffer."""
        if not text or not tag:
            return len(text)
        for i in range(len(tag) - 1, 0, -1):
            if text.endswith(tag[:i]):
                return len(text) - i
        return len(text)

class StreamingParserAdapter:
    """Adapter to manage parser lifecycle per request."""
    def __init__(self):
        self.parser = ProductionQwen3Detector()
        self.is_initialized = False

    def init_request(self):
        self.parser.reset()
        self.is_initialized = True

    def process_chunk(self, chunk: str) -> Tuple[str, str]:
        if not self.is_initialized:
            self.init_request()
        return self.parser.detect_and_parse(chunk)

    def get_final_state(self) -> dict:
        return {
            "in_thinking": self.parser.in_thinking,
            "thinking_finished": self.parser.thinking_finished,
            "buffer": self.parser.buffer
        }
