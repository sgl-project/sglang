import unittest

from sglang.srt.parser.harmony_parser import (
    CanonicalStrategy,
    Event,
    HarmonyParser,
    TextStrategy,
    Token,
    iter_tokens,
    prefix_hold,
)
from sglang.test.test_utils import CustomTestCase


class TestEvent(CustomTestCase):
    def test_init(self):
        """Test Event dataclass initialization."""
        event = Event("reasoning", "content")
        self.assertEqual(event.event_type, "reasoning")
        self.assertEqual(event.content, "content")


class TestToken(CustomTestCase):
    def test_init(self):
        """Test Token dataclass initialization."""
        token = Token("START", 0, 7)
        self.assertEqual(token.type, "START")
        self.assertEqual(token.start, 0)
        self.assertEqual(token.end, 7)


class TestPrefixHold(CustomTestCase):
    def test_empty_text(self):
        """Test prefix_hold with empty text."""
        emit, hold = prefix_hold("", ["<|start|>"])
        self.assertEqual(emit, "")
        self.assertEqual(hold, "")

    def test_no_matching_prefixes(self):
        """Test prefix_hold with no matching prefixes."""
        emit, hold = prefix_hold("hello world", ["<|start|>", "<|end|>"])
        self.assertEqual(emit, "hello world")
        self.assertEqual(hold, "")

    def test_partial_token_suffix(self):
        """Test prefix_hold with partial token at end."""
        emit, hold = prefix_hold("hello <|ret", ["<|return|>"])
        self.assertEqual(emit, "hello ")
        self.assertEqual(hold, "<|ret")

    def test_multiple_potential_matches(self):
        """Test prefix_hold with multiple potential matches."""
        emit, hold = prefix_hold("text <|", ["<|start|>", "<|end|>"])
        self.assertEqual(emit, "text ")
        self.assertEqual(hold, "<|")

    def test_exact_token_match(self):
        """Test prefix_hold with exact token match."""
        emit, hold = prefix_hold("text <|start|>", ["<|start|>"])
        self.assertEqual(emit, "text <|start|>")
        self.assertEqual(hold, "")


class TestIterTokens(CustomTestCase):
    def test_empty_text(self):
        """Test iter_tokens with empty text."""
        tokens = list(iter_tokens(""))
        self.assertEqual(tokens, [])

    def test_plain_text(self):
        """Test iter_tokens with plain text."""
        tokens = list(iter_tokens("hello world"))
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, "TEXT")
        self.assertEqual(tokens[0].start, 0)
        self.assertEqual(tokens[0].end, 11)

    def test_single_token(self):
        """Test iter_tokens with single structural token."""
        tokens = list(iter_tokens("<|start|>"))
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, "START")
        self.assertEqual(tokens[0].start, 0)
        self.assertEqual(tokens[0].end, 9)

    def test_mixed_content(self):
        """Test iter_tokens with mixed text and tokens."""
        tokens = list(iter_tokens("text<|start|>more text"))
        self.assertEqual(len(tokens), 3)

        self.assertEqual(tokens[0].type, "TEXT")
        self.assertEqual(tokens[0].start, 0)
        self.assertEqual(tokens[0].end, 4)

        self.assertEqual(tokens[1].type, "START")
        self.assertEqual(tokens[1].start, 4)
        self.assertEqual(tokens[1].end, 13)

        self.assertEqual(tokens[2].type, "TEXT")
        self.assertEqual(tokens[2].start, 13)
        self.assertEqual(tokens[2].end, 22)

    def test_unknown_token_partial_suffix(self):
        """Test iter_tokens with unknown token that could be partial."""
        tokens = list(iter_tokens("text <|ret"))
        self.assertEqual(len(tokens), 2)

        self.assertEqual(tokens[0].type, "TEXT")
        self.assertEqual(tokens[0].start, 0)
        self.assertEqual(tokens[0].end, 5)

        self.assertEqual(tokens[1].type, "TEXT")
        self.assertEqual(tokens[1].start, 5)
        self.assertEqual(tokens[1].end, 10)

    def test_unknown_token_middle(self):
        """Test iter_tokens with unknown token in middle."""
        tokens = list(iter_tokens("text <|weird|> more <|start|>"))
        self.assertEqual(len(tokens), 5)

        self.assertEqual(tokens[0].type, "TEXT")
        self.assertEqual(tokens[1].type, "TEXT")  # "<|"
        self.assertEqual(tokens[2].type, "TEXT")  # "weird|> more "
        self.assertEqual(tokens[3].type, "START")
        # No trailing text token since it ends with a known token

    def test_all_structural_tokens(self):
        """Test iter_tokens recognizes all structural tokens."""
        text = "<|start|><|channel|><|message|><|constrain|><|end|><|call|><|return|>"
        tokens = list(iter_tokens(text))

        expected_types = [
            "START",
            "CHANNEL",
            "MESSAGE",
            "CONSTRAIN",
            "END",
            "CALL",
            "RETURN",
        ]
        self.assertEqual(len(tokens), len(expected_types))

        for token, expected_type in zip(tokens, expected_types):
            self.assertEqual(token.type, expected_type)


class TestCanonicalStrategy(CustomTestCase):
    def setUp(self):
        self.strategy = CanonicalStrategy()

    def test_init(self):
        """Test CanonicalStrategy initialization."""
        self.assertIn("<|start|>", self.strategy.guard_tokens)
        self.assertIn("<|constrain|>", self.strategy.guard_tokens)

    def test_extract_channel_type(self):
        """Test _extract_channel_type method."""
        self.assertEqual(self.strategy._extract_channel_type("analysis"), "analysis")
        self.assertEqual(
            self.strategy._extract_channel_type("commentary to=functions.tool"),
            "commentary",
        )
        self.assertEqual(self.strategy._extract_channel_type("final to=user"), "final")
        self.assertEqual(self.strategy._extract_channel_type("ANALYSIS"), "analysis")
        self.assertIsNone(self.strategy._extract_channel_type("unknown"))

    def test_parse_single_analysis_block(self):
        """Test parsing single analysis block."""
        text = "<|channel|>analysis<|message|>Let me think about this<|end|>"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[0].content, "Let me think about this")
        self.assertEqual(remaining, "")

    def test_parse_single_commentary_block(self):
        """Test parsing single commentary block."""
        text = "<|channel|>commentary<|message|>User-visible message<|end|>"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, "User-visible message")
        self.assertEqual(remaining, "")

    def test_parse_single_final_block(self):
        """Test parsing single final block."""
        text = "<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, "The answer is 42")
        self.assertEqual(remaining, "")

    def test_parse_tool_call_commentary(self):
        """Test parsing tool call on commentary channel."""
        text = '<|channel|>commentary to=functions.get_weather<|message|>{"location": "SF"}<|call|>'
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "tool_call")
        self.assertEqual(events[0].content, '{"location": "SF"}')
        self.assertEqual(remaining, "")

    def test_parse_tool_call_analysis(self):
        """Test parsing built-in tool call on analysis channel."""
        text = '<|channel|>analysis to=browser.search<|message|>{"query": "SGLang"}<|call|>'
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "tool_call")
        self.assertEqual(events[0].content, '{"query": "SGLang"}')
        self.assertEqual(remaining, "")

    def test_parse_complex_sequence(self):
        """Test parsing complex sequence with multiple blocks."""
        text = (
            "<|channel|>analysis<|message|>Need to use function get_weather.<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.get_weather<|message|>"
            '{"location":"San Francisco"}<|call|>'
        )
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[0].content, "Need to use function get_weather.")
        self.assertEqual(events[1].event_type, "tool_call")
        self.assertEqual(events[1].content, '{"location":"San Francisco"}')
        self.assertEqual(remaining, "")

    def test_parse_with_interspersed_text(self):
        """Test parsing with plain text between blocks."""
        text = (
            "Some text "
            "<|channel|>analysis<|message|>reasoning<|end|>"
            " more text "
            "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
            " trailing text"
        )
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 4)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, "Some text ")
        self.assertEqual(events[1].event_type, "reasoning")
        self.assertEqual(events[1].content, "reasoning")
        self.assertEqual(events[2].event_type, "normal")
        self.assertEqual(events[2].content, " more text ")
        self.assertEqual(events[3].event_type, "normal")
        self.assertEqual(events[3].content, "answer trailing text")
        self.assertEqual(remaining, "")

    def test_parse_incomplete_block(self):
        """Test parsing incomplete block (streaming scenario)."""
        text = "<|channel|>analysis<|message|>partial content"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[0].content, "partial content")
        self.assertEqual(remaining, "<|channel|>analysis<|message|>")

    def test_parse_partial_token_suffix(self):
        """Test parsing with partial token at end."""
        text = "complete text <|ret"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, "complete text ")
        self.assertEqual(remaining, "<|ret")

    def test_parse_tool_response_message(self):
        """Test parsing tool response message (no channel)."""
        text = '<|start|>functions.get_weather to=assistant<|message|>{"sunny": true}<|end|>'
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, '{"sunny": true}')
        self.assertEqual(remaining, "")

    def test_parse_empty_content_blocks(self):
        """Test parsing blocks with empty content."""
        text = "<|channel|>analysis<|message|><|end|>"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[0].content, "")
        self.assertEqual(remaining, "")

    def test_parse_commentary_filler_between_blocks(self):
        """Test that 'commentary' filler between <|call|> and <|channel|> is filtered out."""
        # This pattern occurs when the model generates malformed output
        text = (
            '<|channel|>commentary to=functions.get_weather<|message|>{"location":"SF"}<|call|>'
            "commentary"  # This should be filtered out
            '<|channel|>commentary to=functions.get_temp<|message|>{"location":"NYC"}<|call|>'
        )
        events, remaining = self.strategy.parse(text)

        # Should have 2 tool calls, no "commentary" normal text
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "tool_call")
        self.assertEqual(events[0].content, '{"location":"SF"}')
        self.assertEqual(events[1].event_type, "tool_call")
        self.assertEqual(events[1].content, '{"location":"NYC"}')
        self.assertEqual(remaining, "")

        # Verify no "commentary" text was emitted as normal content
        normal_events = [e for e in events if e.event_type == "normal"]
        commentary_events = [
            e for e in normal_events if "commentary" in e.content.lower()
        ]
        self.assertEqual(
            len(commentary_events), 0, "Commentary filler should be filtered out"
        )


class TestTextStrategy(CustomTestCase):
    def setUp(self):
        self.strategy = TextStrategy()

    def test_init(self):
        """Test TextStrategy initialization."""
        self.assertIn("analysis_then_final", self.strategy.patterns)

    def test_parse_analysis_then_final(self):
        """Test parsing analysis then final format."""
        text = "analysis I need to think about this. assistantfinal The answer is 42."
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[0].content, "I need to think about this.")
        self.assertEqual(events[1].event_type, "normal")
        self.assertEqual(events[1].content, "The answer is 42.")
        self.assertEqual(remaining, "")

    def test_parse_commentary_then_final(self):
        """Test parsing commentary then final format."""
        text = "commentary User-visible preamble. assistantfinal The answer is 42."
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, "User-visible preamble.")
        self.assertEqual(events[1].event_type, "normal")
        self.assertEqual(events[1].content, "The answer is 42.")
        self.assertEqual(remaining, "")

    def test_parse_final_only(self):
        """Test parsing final-only format."""
        text = "assistantfinal The direct answer."
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, "The direct answer.")
        self.assertEqual(remaining, "")

    def test_parse_analysis_only(self):
        """Test parsing analysis-only format."""
        text = "analysis This is reasoning content."
        events, remaining = self.strategy.parse(text)

        # For analysis-only, streaming parse should keep header and emit with leading space
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[0].content, " This is reasoning content.")
        self.assertEqual(remaining, "analysis")

    def test_parse_incomplete_assistantfinal(self):
        """Test parsing with incomplete assistantfinal."""
        text = "analysis reasoning content assistantfin"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 0)
        self.assertEqual(remaining, text)  # Hold entire buffer

    def test_parse_partial_analysis_streaming(self):
        """Test streaming partial analysis content."""
        text = "analysis partial content"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[0].content, " partial content")  # Space preserved
        self.assertEqual(remaining, "analysis")  # Hold header

    def test_parse_case_insensitive(self):
        """Test case insensitive parsing."""
        text = "ANALYSIS reasoning ASSISTANTFINAL answer"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[1].event_type, "normal")

    def test_parse_plain_text_fallback(self):
        """Test parsing plain text without harmony markers."""
        text = "Just plain text without any markers."
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, "Just plain text without any markers.")
        self.assertEqual(remaining, "")

    def test_parse_analysis_no_space_after_header(self):
        """Test parsing analysis format without space after header (real gpt-oss output)."""
        text = "analysisThe user typed random strings. We should respond politely.assistantfinalIt looks like you're testing. How can I help?"
        events, remaining = self.strategy.parse(text)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(
            events[0].content,
            "The user typed random strings. We should respond politely.",
        )
        self.assertEqual(events[1].event_type, "normal")
        self.assertEqual(
            events[1].content, "It looks like you're testing. How can I help?"
        )


class TestHarmonyParser(CustomTestCase):
    def setUp(self):
        self.parser = HarmonyParser()

    def test_init(self):
        """Test HarmonyParser initialization."""
        self.assertIsNone(self.parser.strategy)
        self.assertEqual(self.parser._buffer, "")

    def test_strategy_selection_canonical(self):
        """Test automatic strategy selection for canonical format."""
        events = self.parser.parse("<|channel|>analysis<|message|>test<|end|>")

        self.assertIsInstance(self.parser.strategy, CanonicalStrategy)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "reasoning")

    def test_strategy_selection_text(self):
        """Test automatic strategy selection for text format."""
        events = self.parser.parse("analysis test content")

        self.assertIsInstance(self.parser.strategy, TextStrategy)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "reasoning")

    def test_strategy_selection_delayed(self):
        """Test strategy selection with insufficient initial content."""
        # First chunk doesn't have enough info
        events1 = self.parser.parse("some")
        self.assertEqual(len(events1), 0)
        self.assertIsNone(self.parser.strategy)

        # Second chunk triggers strategy selection
        events2 = self.parser.parse(" analysis content")
        self.assertIsInstance(self.parser.strategy, TextStrategy)
        self.assertEqual(len(events2), 1)

    def test_streaming_canonical_format(self):
        """Test streaming with canonical format."""
        chunks = [
            "<|channel|>analysis<|message|>",
            "reasoning content",
            "<|end|>",
            "<|start|>assistant<|channel|>final<|message|>",
            "final answer",
            "<|return|>",
        ]

        all_events = []
        for chunk in chunks:
            events = self.parser.parse(chunk)
            all_events.extend(events)

        self.assertEqual(len(all_events), 5)

        # Verify we get reasoning events
        reasoning_events = [e for e in all_events if e.event_type == "reasoning"]
        self.assertTrue(len(reasoning_events) > 0)

        # Verify we get normal events
        normal_events = [e for e in all_events if e.event_type == "normal"]
        self.assertTrue(len(normal_events) > 0)

        # Verify content is eventually parsed correctly
        combined_reasoning = "".join(e.content for e in reasoning_events)
        combined_normal = "".join(
            e.content
            for e in normal_events
            if e.content and "<|return|>" not in e.content
        )

        self.assertIn("reasoning content", combined_reasoning)
        self.assertIn("final answer", combined_normal)

    def test_streaming_text_format(self):
        """Test streaming with text format."""
        chunks = ["analysis reasoning", " content assistantfinal", " the answer"]

        all_events = []
        for chunk in chunks:
            events = self.parser.parse(chunk)
            all_events.extend(events)

        # Should have reasoning and normal events
        reasoning_events = [e for e in all_events if e.event_type == "reasoning"]
        normal_events = [e for e in all_events if e.event_type == "normal"]

        self.assertGreater(len(reasoning_events), 0)
        self.assertGreater(len(normal_events), 0)

    def test_streaming_commentary_filler(self):
        """Test that 'commentary' filler is filtered in streaming case."""
        # Test when commentary arrives as a separate chunk after <|call|>
        chunks = [
            "<|channel|>commentary to=functions.get_weather",
            "<|message|>",
            '{"location":"SF"}',
            "<|call|>",
            "comment",  # This arrives as separate chunk - should be filtered
            "ary",  # Continuation of the filler - should be filtered
            "<|channel|>commentary to=functions.get_temp",
            "<|message|>",
            '{"location":"NYC"}',
            "<|call|>",
            "comment",  # Another separate chunk - should be filtered
            "ary",  # Continuation of the filler - should be filtered
            "<|start|>assistant<|channel|>final",
            "<|message|>Done<|return|>",
        ]

        all_events = []
        for chunk in chunks:
            events = self.parser.parse(chunk)
            all_events.extend(events)

        # Count event types
        tool_events = [e for e in all_events if e.event_type == "tool_call"]
        normal_events = [e for e in all_events if e.event_type == "normal"]

        # Should have 2 tool calls and 1 final message
        self.assertEqual(len(tool_events), 2, "Should have 2 tool calls")
        self.assertEqual(
            len(normal_events), 1, "Should have 1 normal event (final message)"
        )

        # Verify no "commentary" in normal events
        for event in normal_events:
            self.assertNotEqual(
                event.content.strip().lower(),
                "commentary",
                "Commentary filler should not appear as normal content in streaming",
            )

        # Verify content
        self.assertEqual(tool_events[0].content, '{"location":"SF"}')
        self.assertEqual(tool_events[1].content, '{"location":"NYC"}')
        self.assertEqual(normal_events[0].content, "Done")

    def test_repetitive_tool_calls_with_commentary_filler(self):
        """Test handling of repetitive tool calls with 'commentary' filler text."""
        # This simulates malformed output with repeated tool calls and commentary filler
        text = (
            "<|channel|>analysis<|message|>Need to get weather<|end|>"
            '<|start|>assistant<|channel|>commentary to=functions.get_weather<|message|>{"city":"Boston"}<|call|>'
            "commentary"  # Filler that should be filtered
            '<|channel|>commentary to=functions.get_weather<|message|>{"city":"Boston"}<|call|>'
            "commentary"  # Another filler
            '<|channel|>commentary to=functions.get_weather<|message|>{"city":"Boston"}<|call|>'
            "<|channel|>analysis<|message|>Tool not responding<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Unable to fetch weather data<|return|>"
        )

        events = self.parser.parse(text)

        # Count event types
        reasoning_events = [e for e in events if e.event_type == "reasoning"]
        tool_events = [e for e in events if e.event_type == "tool_call"]
        normal_events = [e for e in events if e.event_type == "normal"]

        # Verify correct number of each type
        self.assertEqual(len(reasoning_events), 2, "Should have 2 reasoning events")
        self.assertEqual(len(tool_events), 3, "Should have 3 tool calls")
        self.assertEqual(
            len(normal_events), 1, "Should have 1 normal event (final message)"
        )

        # Verify no "commentary" filler in normal events
        for event in normal_events:
            self.assertNotEqual(
                event.content.strip().lower(),
                "commentary",
                "Commentary filler should not appear as normal content",
            )

        # Verify content is correct
        self.assertEqual(reasoning_events[0].content, "Need to get weather")
        self.assertEqual(reasoning_events[1].content, "Tool not responding")
        self.assertEqual(normal_events[0].content, "Unable to fetch weather data")


class TestIntegrationScenarios(CustomTestCase):
    """Integration tests for realistic Harmony parsing scenarios."""

    def test_complete_reasoning_flow(self):
        """Test complete reasoning flow from HARMONY_DOCS.md examples."""
        parser = HarmonyParser()

        text = (
            '<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>'
            "<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>"
        )

        events = parser.parse(text)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertIn("Simple arithmetic", events[0].content)
        self.assertEqual(events[1].event_type, "normal")
        self.assertEqual(events[1].content, "2 + 2 = 4.")

    def test_tool_call_sequence(self):
        """Test tool call sequence from HARMONY_DOCS.md examples."""
        parser = HarmonyParser()

        text = (
            "<|channel|>analysis<|message|>Need to use function get_weather.<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
            '{"location":"San Francisco"}<|call|>'
        )

        events = parser.parse(text)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[0].content, "Need to use function get_weather.")
        self.assertEqual(events[1].event_type, "tool_call")
        self.assertEqual(events[1].content, '{"location":"San Francisco"}')

    def test_preamble_sequence(self):
        """Test preamble sequence with multiple commentary blocks."""
        parser = HarmonyParser()

        text = (
            "<|channel|>analysis<|message|>Long chain of thought<|end|>"
            "<|start|>assistant<|channel|>commentary<|message|>**Action plan**: 1. Generate file 2. Start server<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.generate_file<|message|>"
            '{"template": "basic_html"}<|call|>'
        )

        events = parser.parse(text)

        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[1].event_type, "normal")
        self.assertIn("Action plan", events[1].content)
        self.assertEqual(events[2].event_type, "tool_call")

    def test_built_in_tool_call(self):
        """Test built-in tool call on analysis channel."""
        parser = HarmonyParser()

        text = '<|channel|>analysis to=browser.search<|message|>{"query": "SGLang"}<|call|>'

        events = parser.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "tool_call")
        self.assertEqual(events[0].content, '{"query": "SGLang"}')

    def test_tool_response_handling(self):
        """Test tool response message handling."""
        parser = HarmonyParser()

        text = '<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{"sunny": true, "temperature": 20}<|end|>'

        events = parser.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "normal")
        self.assertEqual(events[0].content, '{"sunny": true, "temperature": 20}')

    def test_text_fallback_formats(self):
        """Test various text fallback formats."""
        parser = HarmonyParser()

        # Test analysis then final
        events1 = parser.parse("analysis thinking assistantfinal answer")
        self.assertEqual(len([e for e in events1 if e.event_type == "reasoning"]), 1)
        self.assertEqual(len([e for e in events1 if e.event_type == "normal"]), 1)

        # Reset parser for next test
        parser = HarmonyParser()

        # Test final only
        events2 = parser.parse("assistantfinal direct answer")
        self.assertEqual(len(events2), 1)
        self.assertEqual(events2[0].event_type, "normal")

    def test_streaming_property_canonical(self):
        """Test streaming property: chunked parsing produces same semantic content as one-shot parsing."""
        full_text = (
            "<|channel|>analysis<|message|>reasoning content<|end|>"
            "<|start|>assistant<|channel|>final<|message|>final content"
        )

        # One-shot parsing
        parser1 = HarmonyParser()
        events_oneshot = parser1.parse(full_text)
        events_oneshot += parser1.parse("")

        # Chunked parsing
        parser2 = HarmonyParser()
        chunks = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "reasoning content",
            "<|end|>",
            "<|start|>assistant",
            "<|channel|>final",
            "<|message|>",
            "final ",
            "content",
        ]
        events_chunked = []
        for chunk in chunks:
            events_chunked.extend(parser2.parse(chunk))

        # Compare semantic content rather than exact event structure
        reasoning_oneshot = "".join(
            e.content for e in events_oneshot if e.event_type == "reasoning"
        )
        normal_oneshot = "".join(
            e.content for e in events_oneshot if e.event_type == "normal"
        )

        reasoning_chunked = "".join(
            e.content for e in events_chunked if e.event_type == "reasoning"
        )
        normal_chunked = "".join(
            e.content for e in events_chunked if e.event_type == "normal"
        )

        self.assertEqual(reasoning_chunked, reasoning_oneshot)
        self.assertEqual(normal_chunked, normal_oneshot)

    def test_streaming_property_text(self):
        """Test streaming property for text format."""
        full_text = "analysis reasoning content assistantfinal final answer"

        # One-shot parsing
        parser1 = HarmonyParser()
        events_oneshot = parser1.parse(full_text)

        # Chunked parsing
        parser2 = HarmonyParser()
        chunks = ["analysis reason", "ing content assistant", "final final answer"]
        events_chunked = []
        for chunk in chunks:
            events_chunked.extend(parser2.parse(chunk))

        # Combine content by type for comparison
        reasoning_oneshot = "".join(
            e.content for e in events_oneshot if e.event_type == "reasoning"
        )
        normal_oneshot = "".join(
            e.content for e in events_oneshot if e.event_type == "normal"
        )

        reasoning_chunked = "".join(
            e.content for e in events_chunked if e.event_type == "reasoning"
        )
        normal_chunked = "".join(
            e.content for e in events_chunked if e.event_type == "normal"
        )

        # Account for whitespace differences due to streaming - compare trimmed content
        self.assertEqual(reasoning_oneshot.strip(), reasoning_chunked.strip())
        self.assertEqual(normal_oneshot.strip(), normal_chunked.strip())


class TestEdgeCases(CustomTestCase):
    """Test edge cases and error conditions."""

    def test_malformed_channel_headers(self):
        """Test handling of malformed channel headers."""
        parser = HarmonyParser()

        # Unknown channel type
        text = "<|channel|>unknown<|message|>content<|end|>"
        events = parser.parse(text)

        # Should be held as incomplete since channel is unknown
        self.assertEqual(len(events), 0)

    def test_mixed_unknown_tokens(self):
        """Test handling of mixed unknown tokens."""
        parser = HarmonyParser()

        text = "text <|weird|> more text <|channel|>analysis<|message|>content<|end|>"
        events = parser.parse(text)

        # Should parse the valid parts
        reasoning_events = [e for e in events if e.event_type == "reasoning"]
        normal_events = [e for e in events if e.event_type == "normal"]

        self.assertEqual(len(reasoning_events), 1)
        self.assertGreater(len(normal_events), 0)

    def test_empty_input(self):
        """Test handling of empty input."""
        parser = HarmonyParser()
        events = parser.parse("")
        self.assertEqual(len(events), 0)

    def test_whitespace_preservation(self):
        """Test that whitespace is preserved correctly."""
        parser = HarmonyParser()

        text = "<|channel|>analysis<|message|>  content with spaces  <|end|>"
        events = parser.parse(text)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].content, "  content with spaces  ")

    def test_streaming_whitespace_preservation(self):
        """Test that streaming preserves whitespace between chunks."""
        parser = HarmonyParser()

        # Simulate streaming where space is at chunk boundary
        chunks = ["analysis The user typed ", '"wapppa". Not a question.']

        all_events = []
        for chunk in chunks:
            events = parser.parse(chunk)
            all_events.extend(events)

        # Combine all reasoning content
        reasoning_content = "".join(
            e.content for e in all_events if e.event_type == "reasoning"
        )

        # Should preserve the space before the quote
        self.assertIn('typed "wapppa"', reasoning_content)
        self.assertNotIn(
            'typed"wapppa"', reasoning_content
        )  # Should not be mashed together

    def test_consecutive_blocks_same_type(self):
        """Test consecutive blocks of the same type."""
        parser = HarmonyParser()

        text = (
            "<|channel|>analysis<|message|>first reasoning<|end|>"
            "<|channel|>analysis<|message|>second reasoning<|end|>"
        )
        events = parser.parse(text)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "reasoning")
        self.assertEqual(events[1].event_type, "reasoning")
        self.assertEqual(events[0].content, "first reasoning")
        self.assertEqual(events[1].content, "second reasoning")


if __name__ == "__main__":
    unittest.main()
