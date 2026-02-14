from sglang.srt.parser.harmony_parser import HarmonyParser


def test_split_channel_parsing():
    """Test parsing when <|channel|> marker is separated from type (realistic split)."""
    parser = HarmonyParser()

    # Chunk 1: Channel marker only
    events1 = parser.parse("<|channel|>")
    assert not events1

    # Chunk 2: Channel type and content
    events2 = parser.parse("analysis<|message|>Thinking")

    # Should detect reasoning
    assert len(events2) == 1
    assert events2[0].event_type == "reasoning"
    assert events2[0].content == "Thinking"


def test_incremental_parsing_no_duplication():
    """Test that incremental parsing does not duplicate content."""
    parser = HarmonyParser()

    # Initialize with analysis mode
    parser.parse("<|channel|>analysis<|message|>")

    # Chunk 1
    events1 = parser.parse("Thinking")
    assert len(events1) == 1
    assert events1[0].content == "Thinking"

    # Chunk 2
    events2 = parser.parse(" about")
    assert len(events2) == 1
    assert events2[0].content == " about"


def test_tool_call_marker_consumption():
    """Test that <|call|> marker is correctly identified and emits tool call."""
    parser = HarmonyParser()

    # Setup tool call start
    parser.parse("<|channel|>commentary to=tool<|constrain|>json<|message|>")

    # Content
    parser.parse('{"a":1}')

    # End with call
    events = parser.parse("<|call|>")

    assert len(events) == 1
    assert events[0].event_type == "tool_call"
    # Content includes buffered text.
    assert '{"a":1}' in events[0].content


def test_split_tool_call_tokens():
    """Test parsing when tool call TOKENS are split across chunks (atomic markers)."""
    parser = HarmonyParser()

    # Chunk 1: Channel marker
    parser.parse("<|channel|>")
    # Chunk 2: Channel content
    parser.parse("commentary to=tool")

    # Chunk 3: Constrain marker
    parser.parse("<|constrain|>")
    # Chunk 4: Constrain content
    parser.parse("json")

    # Chunk 5: Message marker
    parser.parse("<|message|>")

    # Chunk 6: Content
    parser.parse("{}")

    # Chunk 7: Call marker
    events = parser.parse("<|call|>")

    assert len(events) == 1
    assert events[0].event_type == "tool_call"
    assert "{}" in events[0].content
