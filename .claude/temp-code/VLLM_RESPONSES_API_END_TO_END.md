# vLLM Responses API - Complete End-to-End Flow Analysis

## Executive Summary

**For Harmony models (gpt-oss) in vLLM Responses API:**
- ❌ NO separate `ReasoningParser` class
- ❌ NO separate `ToolParser` class
- ✅ ONLY `HarmonyContext` + `parse_output_message()` routing

**Single parse operation extracts ALL channels:**
- `analysis` → Reasoning
- `commentary` → Tool calls
- `final` → Normal text

---

## Complete Request Flow

### Phase 1: Request Entry & Validation

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/openai/serving_responses.py`

**Lines 253-268**: Main entry point
```python
async def create_responses(
    self,
    request: ResponsesRequest,
    raw_request: Request | None = None,
) -> (
    AsyncGenerator[StreamingResponsesResponse, None]
    | ResponsesResponse
    | ErrorResponse
):
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        logger.error("Error with model %s", error_check_ret)
        return error_check_ret
    maybe_validation_error = self._validate_create_responses_input(request)
    if maybe_validation_error is not None:
        return maybe_validation_error
```

**Lines 230-251**: Validation for Harmony models
```python
def _validate_create_responses_input(
    self, request: ResponsesRequest
) -> ErrorResponse | None:
    # Key validation: logprobs not supported for Harmony
    if self.use_harmony and request.is_include_output_logprobs():
        return self.create_error_response(
            err_type="invalid_request_error",
            message="logprobs are not supported with gpt-oss models",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    # ... other validations
```

---

### Phase 2: Harmony Detection & Setup

**Lines 167-179**: Detection in `__init__`
```python
# Detect Harmony model
self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"

if self.use_harmony:
    logger.warning(
        "For gpt-oss, we ignore --enable-auto-tool-choice "
        "and always enable tool use."
    )
    # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
    # We need to add them to the stop token ids.
    if "stop_token_ids" not in self.default_sampling_params:
        self.default_sampling_params["stop_token_ids"] = []
    self.default_sampling_params["stop_token_ids"].extend(
        get_stop_tokens_for_assistant_actions()  # From harmony_utils
    )
```

**Key Function**: `get_stop_tokens_for_assistant_actions()`

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/harmony_utils.py`

**Lines 443-444**:
```python
def get_stop_tokens_for_assistant_actions() -> list[int]:
    return get_encoding().stop_tokens_for_assistant_actions()
```

---

### Phase 3: Request Preparation

**Lines 300-307**: Harmony-specific request creation
```python
if self.use_harmony:
    messages, request_prompts, engine_prompts = (
        self._make_request_with_harmony(request, prev_response)
    )
else:
    messages, request_prompts, engine_prompts = await self._make_request(
        request, prev_response, tokenizer
    )
```

**Lines 503-520**: `_make_request_with_harmony()`
```python
def _make_request_with_harmony(
    self,
    request: ResponsesRequest,
    prev_response: ResponsesResponse | None,
):
    if request.tool_choice != "auto":
        raise NotImplementedError(
            "Only 'auto' tool_choice is supported in response API with Harmony"
        )

    # Construct messages with Harmony format
    messages = self._construct_input_messages_with_harmony(request, prev_response)

    # Render to token IDs using Harmony encoding
    prompt_token_ids = render_for_completion(messages)
    engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_token_ids)

    # Add cache_salt if provided
    if request.cache_salt is not None:
        engine_prompt["cache_salt"] = request.cache_salt

    return messages, [prompt_token_ids], [engine_prompt]
```

**Lines 914-992**: `_construct_input_messages_with_harmony()` - Complex message construction
```python
def _construct_input_messages_with_harmony(
    self,
    request: ResponsesRequest,
    prev_response: ResponsesResponse | None,
) -> list[OpenAIHarmonyMessage]:
    messages: list[OpenAIHarmonyMessage] = []

    if prev_response is None:
        # New conversation
        tool_types = [tool.type for tool in request.tools]
        with_custom_tools = has_custom_tools(tool_types)

        # System message with reasoning effort, tool descriptions
        sys_msg = self._construct_harmony_system_input_message(
            request, with_custom_tools, tool_types
        )
        messages.append(sys_msg)

        # Developer message with custom tools
        if with_custom_tools:
            dev_msg = get_developer_message(
                instructions=request.instructions, tools=request.tools
            )
            messages.append(dev_msg)
    else:
        # Continue previous conversation
        prev_msgs = self.msg_store[prev_response.id]

        # Remove previous analysis messages if there's a new final message
        if len(prev_msgs) > 0:
            last_msg = prev_msgs[-1]
            assert isinstance(last_msg, OpenAIHarmonyMessage)
            if last_msg.channel == "final":
                # Find previous final message and remove analysis in between
                prev_final_msg_idx = -1
                for i in range(len(prev_msgs) - 2, -1, -1):
                    prev_msg_i = prev_msgs[i]
                    assert isinstance(prev_msg_i, OpenAIHarmonyMessage)
                    if prev_msg_i.channel == "final":
                        prev_final_msg_idx = i
                        break

                # Get messages from this turn
                recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                # Remove them from history
                del prev_msgs[prev_final_msg_idx + 1 :]
                # Add back non-analysis messages
                for msg in recent_turn_msgs:
                    assert isinstance(msg, OpenAIHarmonyMessage)
                    if msg.channel != "analysis":
                        prev_msgs.append(msg)

        messages.extend(prev_msgs)

    # Append new input
    if isinstance(request.input, str):
        messages.append(get_user_message(request.input))
    else:
        # Complex input with tool outputs
        if prev_response is not None:
            prev_outputs = copy(prev_response.output)
        else:
            prev_outputs = []
        for response_msg in request.input:
            messages.append(parse_response_input(response_msg, prev_outputs))
            # Track tool calls for matching with outputs
            if (
                isinstance(response_msg, dict)
                and response_msg.get("type") == "function_call"
            ):
                response_msg = ResponseFunctionToolCall.model_validate(response_msg)
            if isinstance(response_msg, ResponseFunctionToolCall):
                prev_outputs.append(response_msg)

    return messages
```

**Key Helper**: `render_for_completion()`

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/harmony_utils.py`

**Lines 282-287**:
```python
def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    return token_ids
```

**Lines 72-76**: Encoding singleton
```python
def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding
```

---

### Phase 4: Context Creation

**Lines 360-367**: Context instantiation
```python
context: ConversationContext
if self.use_harmony:
    if request.stream:
        context = StreamingHarmonyContext(messages, available_tools)
    else:
        context = HarmonyContext(messages, available_tools)
else:
    context = SimpleContext()
```

**Key Class**: `HarmonyContext`

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/context.py`

**Lines 153-179**: Class definition
```python
class HarmonyContext(ConversationContext):
    def __init__(
        self,
        messages: list,
        available_tools: list[str],
    ):
        self._messages = messages
        self.finish_reason: str | None = None
        self.available_tools = available_tools
        self._tool_sessions: dict[str, ClientSession | Tool] = {}
        self.called_tools: set[str] = set()

        # Create StreamableParser from openai-harmony
        self.parser = get_streamable_parser_for_assistant()

        self.num_init_messages = len(messages)
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.num_cached_tokens = 0
        self.num_reasoning_tokens = 0
        self.num_tool_output_tokens = 0

        # Turn tracking
        self.current_turn_metrics = TurnMetrics()
        self.all_turn_metrics: list[TurnMetrics] = []
        self.is_first_turn = True
        self.first_tok_of_message = True  # For streaming support
```

**Lines 447-448**: Parser creation helper
```python
def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)
```

---

### Phase 5: Tool Session Initialization

**Lines 552-554**: Initialize tool sessions (if needed)
```python
async with AsyncExitStack() as exit_stack:
    try:
        await self._initialize_tool_sessions(request, context, exit_stack)
```

**Lines 522-536**: Tool session setup
```python
async def _initialize_tool_sessions(
    self,
    request: ResponsesRequest,
    context: ConversationContext,
    exit_stack: AsyncExitStack,
):
    # Only initialize if request needs tools
    if len(request.tools) == 0:
        return

    mcp_tools = {
        tool.server_label: tool for tool in request.tools if tool.type == "mcp"
    }
    await context.init_tool_sessions(
        self.tool_server, exit_stack, request.request_id, mcp_tools
    )
```

**Context Tool Initialization**:

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/context.py`

**Lines 380-398**:
```python
async def init_tool_sessions(
    self,
    tool_server: ToolServer | None,
    exit_stack: AsyncExitStack,
    request_id: str,
    mcp_tools: dict[str, Mcp],
):
    if tool_server:
        for tool_name in self.available_tools:
            if tool_name not in self._tool_sessions:
                tool_type = _map_tool_name_to_tool_type(tool_name)
                headers = (
                    mcp_tools[tool_type].headers if tool_type in mcp_tools else None
                )
                tool_session = await exit_stack.enter_async_context(
                    tool_server.new_session(tool_name, request_id, headers)
                )
                self._tool_sessions[tool_name] = tool_session
                exit_stack.push_async_exit(self.cleanup_session)
```

---

### Phase 6: Token Generation & Processing

**Lines 555-556**: Generate tokens
```python
async for _ in result_generator:
    pass
```

This iterates through the generation result, which calls `context.append_output()` for each chunk.

**Key Method**: `append_output()`

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/context.py`

**Lines 185-207**: Token processing (non-streaming)
```python
def append_output(self, output: RequestOutput | list[Message]) -> None:
    if isinstance(output, RequestOutput):
        # Extract token IDs from output
        output_token_ids = output.outputs[0].token_ids

        # Reinitialize parser
        self.parser = get_streamable_parser_for_assistant()

        # Process each token through StreamableParser
        for token_id in output_token_ids:
            self.parser.process(token_id)  # ← Token-level processing!
            # Check if current token is part of reasoning content
            self._update_num_reasoning_tokens()

        # Update token usage statistics
        self._update_prefill_token_usage(output)
        self._update_decode_token_usage(output)

        # Append current turn to metrics
        self.all_turn_metrics.append(self.current_turn_metrics.copy())
        self.current_turn_metrics.reset()

        # Extract messages from parser
        output_msgs = self.parser.messages  # ← Messages with channels!

        # Save finish reason
        self.finish_reason = output.outputs[0].finish_reason
    else:
        # Tool output (direct Message format)
        output_msgs = output

    # Extend message list
    self._messages.extend(output_msgs)
```

**Lines 180-183**: Reasoning token tracking
```python
def _update_num_reasoning_tokens(self):
    # Count all analysis and commentary channels as reasoning tokens
    if self.parser.current_channel in {"analysis", "commentary"}:
        self.num_reasoning_tokens += 1
```

**Lines 209-271**: Token usage tracking
```python
def _update_prefill_token_usage(self, output: RequestOutput) -> None:
    """Update token usage for prefill phase (input prompt)"""
    if output.prompt_token_ids is not None:
        this_turn_input_tokens = len(output.prompt_token_ids)
    else:
        this_turn_input_tokens = 0

    # Update current turn input tokens
    self.current_turn_metrics.input_tokens = this_turn_input_tokens
    self.num_prompt_tokens += this_turn_input_tokens

    # Calculate tool tokens (after first turn)
    if self.is_first_turn:
        self.is_first_turn = False
    else:
        previous_turn = self.all_turn_metrics[-1]
        # tool tokens = current prefill - previous prefill - previous decode
        this_turn_tool_tokens = (
            self.current_turn_metrics.input_tokens
            - previous_turn.input_tokens
            - previous_turn.output_tokens
        )

        if this_turn_tool_tokens < 0:
            logger.error("Negative tool output tokens: %d", this_turn_tool_tokens)
            this_turn_tool_tokens = 0

        self.num_tool_output_tokens += this_turn_tool_tokens
        self.current_turn_metrics.tool_output_tokens = this_turn_tool_tokens

    # Update cached tokens
    num_cached_token = output.num_cached_tokens
    if num_cached_token is not None:
        self.num_cached_tokens += num_cached_token
        self.current_turn_metrics.cached_input_tokens = num_cached_token

def _update_decode_token_usage(self, output: RequestOutput) -> int:
    """Update token usage for decode phase (generated output)"""
    updated_output_token_count = 0
    if output.outputs:
        for completion_output in output.outputs:
            updated_output_token_count += len(completion_output.token_ids)
        self.num_output_tokens += updated_output_token_count
        self.current_turn_metrics.output_tokens += updated_output_token_count
    return updated_output_token_count
```

**Key Point**: `StreamableParser.process(token_id)` does all the heavy lifting:
- Parses tokens into structured messages
- Separates channels (analysis, commentary, final)
- Extracts recipients (functions.*, browser.*, python, etc.)
- Provides messages via `parser.messages`

---

### Phase 7: Output Item Extraction

**Lines 570-583**: Harmony path
```python
if self.use_harmony:
    assert isinstance(context, HarmonyContext)

    # Route messages to output items
    output = self._make_response_output_items_with_harmony(context)

    # Optionally include messages in response
    if request.enable_response_messages:
        input_messages = context.messages[: context.num_init_messages]
        output_messages = context.messages[context.num_init_messages :]

    num_tool_output_tokens = context.num_tool_output_tokens

    # Set status based on finish reason
    if len(output) > 0:
        if context.finish_reason == "length":
            status = "incomplete"
        elif context.finish_reason == "abort":
            status = "cancelled"
    else:
        status = "incomplete"
```

**Lines 819-831**: `_make_response_output_items_with_harmony()`
```python
def _make_response_output_items_with_harmony(
    self,
    context: HarmonyContext,
) -> list[ResponseOutputItem]:
    output_items: list[ResponseOutputItem] = []
    num_init_messages = context.num_init_messages

    # Iterate through NEW messages (skip initial prompt messages)
    for msg in context.messages[num_init_messages:]:
        output_items.extend(parse_output_message(msg))  # ← Route each message!

    # Handle generation stopped in the middle (incomplete)
    last_items = parse_remaining_state(context.parser)
    if last_items:
        output_items.extend(last_items)

    return output_items
```

**Key Point**: NO parsing here - just routing messages from `context.messages`!

---

### Phase 8: Message Routing - The Core "Parsing"

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/harmony_utils.py`

**Lines 290-398**: `parse_output_message()` - THE routing function

```python
def parse_output_message(message: Message) -> list[ResponseOutputItem]:
    """
    Parse a Harmony message into a list of output response items.
    Routes based on channel and recipient.
    """
    # Skip non-assistant messages (tool responses)
    if message.author.role != "assistant":
        return []

    output_items: list[ResponseOutputItem] = []
    recipient = message.recipient

    # ========== BROWSER TOOL CALLS (lines 302-330) ==========
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")

        content = message.content[0]
        browser_call = json.loads(content.text)  # Parse JSON arguments

        # Create action based on browser function
        if recipient == "browser.search":
            action = ActionSearch(
                query=f"cursor:{browser_call.get('query', '')}",
                type="search"
            )
        elif recipient == "browser.open":
            action = ActionOpenPage(
                url=f"cursor:{browser_call.get('url', '')}",
                type="open_page"
            )
        elif recipient == "browser.find":
            action = ActionFind(
                pattern=browser_call["pattern"],
                url=f"cursor:{browser_call.get('url', '')}",
                type="find",
            )
        else:
            raise ValueError(f"Unknown browser action: {recipient}")

        # Create web search item
        web_search_item = ResponseFunctionWebSearch(
            id=f"ws_{random_uuid()}",
            action=action,
            status="completed",
            type="web_search_call",
        )
        output_items.append(web_search_item)

    # ========== REASONING EXTRACTION (lines 331-344) ==========
    elif message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(
                        text=content.text,  # ← Direct text from Harmony
                        type="reasoning_text"
                    )
                ],
                status=None,  # NOTE: Only last output item has status
            )
            output_items.append(reasoning_item)

    # ========== TOOL CALL EXTRACTION (lines 345-377) ==========
    elif message.channel == "commentary":
        # User-defined function calls
        if recipient is not None and recipient.startswith("functions."):
            function_name = recipient.split(".")[-1]  # Extract function name

            for content in message.content:
                random_id = random_uuid()
                response_item = ResponseFunctionToolCall(
                    arguments=content.text,  # ← Already JSON from Harmony!
                    call_id=f"call_{random_id}",
                    type="function_call",
                    name=function_name,
                    id=f"fc_{random_id}",
                )
                output_items.append(response_item)

        # Built-in tool calls (python, browser, container) → Treat as reasoning
        elif recipient is not None and (
            recipient.startswith("python")
            or recipient.startswith("browser")
            or recipient.startswith("container")
        ):
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=content.text,
                            type="reasoning_text"
                        )
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
        else:
            raise ValueError(f"Unknown recipient: {recipient}")

    # ========== NORMAL TEXT EXTRACTION (lines 378-395) ==========
    elif message.channel == "final":
        contents = []
        for content in message.content:
            output_text = ResponseOutputText(
                text=content.text,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO (not supported for Harmony)
            )
            contents.append(output_text)

        text_item = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=contents,
            role=message.author.role,
            status="completed",
            type="message",
        )
        output_items.append(text_item)

    else:
        raise ValueError(f"Unknown channel: {message.channel}")

    return output_items
```

**Lines 401-440**: `parse_remaining_state()` - Handle incomplete generation

```python
def parse_remaining_state(parser: StreamableParser) -> list[ResponseOutputItem]:
    """
    Handle case where generation stopped mid-message.
    Extract partial content from parser's current state.
    """
    # No content in progress
    if not parser.current_content:
        return []

    # Not an assistant message
    if parser.current_role != Role.ASSISTANT:
        return []

    current_recipient = parser.current_recipient

    # Skip browser calls (already handled)
    if current_recipient is not None and current_recipient.startswith("browser."):
        return []

    # Incomplete reasoning
    if parser.current_channel == "analysis":
        reasoning_item = ResponseReasoningItem(
            id=f"rs_{random_uuid()}",
            summary=[],
            type="reasoning",
            content=[
                ResponseReasoningTextContent(
                    text=parser.current_content,  # ← Partial content
                    type="reasoning_text"
                )
            ],
            status=None,
        )
        return [reasoning_item]

    # Incomplete final message
    elif parser.current_channel == "final":
        output_text = ResponseOutputText(
            text=parser.current_content,  # ← Partial content
            annotations=[],
            type="output_text",
            logprobs=None,
        )
        text_item = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=[output_text],
            role="assistant",
            status="incomplete",  # ← Mark as incomplete!
            type="message",
        )
        return [text_item]

    return []
```

**Key Point**:
- Tool "parsing" is trivial: extract name from `recipient`, arguments from `content.text`
- Arguments are already JSON from Harmony - no complex parsing needed!
- Routing is just if/elif on `channel` and `recipient` fields

---

### Phase 9: Response Building

**Lines 609-632**: Usage calculation
```python
usage = ResponseUsage(
    input_tokens=num_prompt_tokens,
    output_tokens=num_generated_tokens,
    total_tokens=num_prompt_tokens + num_generated_tokens,
    input_tokens_details=InputTokensDetails(
        cached_tokens=num_cached_tokens,
        input_tokens_per_turn=[
            turn.input_tokens for turn in context.all_turn_metrics
        ],
        cached_tokens_per_turn=[
            turn.cached_input_tokens for turn in context.all_turn_metrics
        ],
    ),
    output_tokens_details=OutputTokensDetails(
        reasoning_tokens=num_reasoning_tokens,
        tool_output_tokens=num_tool_output_tokens,
        output_tokens_per_turn=[
            turn.output_tokens for turn in context.all_turn_metrics
        ],
        tool_output_tokens_per_turn=[
            turn.tool_output_tokens for turn in context.all_turn_metrics
        ],
    ),
)
```

**Lines 633-651**: Final response
```python
response = ResponsesResponse.from_request(
    request,
    sampling_params,
    input_messages=input_messages,
    output_messages=output_messages,
    model_name=model_name,
    created_time=created_time,
    output=output,  # ← Output items from parse_output_message()
    status=status,
    usage=usage,
)

if request.store:
    async with self.response_store_lock:
        stored_response = self.response_store.get(response.id)
        # Only update if not cancelled
        if stored_response is None or stored_response.status != "cancelled":
            self.response_store[response.id] = response

return response
```

---

## Streaming Flow

### Streaming Context

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/context.py`

**Lines 456-530**: `StreamingHarmonyContext`

```python
class StreamingHarmonyContext(HarmonyContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_output = None

        self.parser = get_streamable_parser_for_assistant()
        self.encoding = get_encoding()
        self.last_tok = None
        self.first_tok_of_message = True

    def append_output(self, output: RequestOutput | list[Message]) -> None:
        if isinstance(output, RequestOutput):
            # Only add prompt tokens once per message
            if self.first_tok_of_message:
                self._update_prefill_token_usage(output)

            # Reset flag if message finished
            self.first_tok_of_message = output.finished

            # Process tokens incrementally
            for tok in output.outputs[0].token_ids:
                self.parser.process(tok)

            self._update_decode_token_usage(output)

            # Update metrics when message complete
            if output.finished:
                self.all_turn_metrics.append(self.current_turn_metrics.copy())
                self.current_turn_metrics.reset()

            # Update reasoning token count
            self._update_num_reasoning_tokens()
            self.last_tok = tok

            # Extend messages incrementally as they're parsed
            if len(self._messages) - self.num_init_messages < len(self.parser.messages):
                self._messages.extend(
                    self.parser.messages[len(self._messages) - self.num_init_messages :]
                )
        else:
            # Tool output
            assert len(output) == 1
            msg = output[0]
            if msg.author.role == Role.TOOL and msg.recipient is None:
                msg.recipient = "assistant"

            # Render tool message to tokens
            toks = self.encoding.render(msg)
            for tok in toks:
                self.parser.process(tok)
            self.last_tok = toks[-1]

    def is_expecting_start(self) -> bool:
        """Check if parser is waiting for message start"""
        return self.parser.state == StreamState.EXPECT_START

    def is_assistant_action_turn(self) -> bool:
        """Check if last token was an assistant action token"""
        return self.last_tok in self.encoding.stop_tokens_for_assistant_actions()
```

### Streaming Event Generation

**File**: `/Users/simolin/sglang/.claude/vllm/vllm/entrypoints/openai/serving_responses.py`

**Lines 1445-1897**: `_process_harmony_streaming_events()`

This is a large function that generates streaming events. Key patterns:

**Lines 1467-1586**: Handle message transitions
```python
if ctx.is_expecting_start():
    current_output_index += 1
    sent_output_item_added = False

    if len(ctx.parser.messages) > 0:
        previous_item = ctx.parser.messages[-1]

        # Deal with completed tool call
        if previous_item.recipient is not None:
            if previous_item.recipient.startswith("functions."):
                function_name = previous_item.recipient[len("functions.") :]
                # Emit function call done events
                yield ResponseFunctionCallArgumentsDoneEvent(...)
                yield ResponseOutputItemDoneEvent(...)

        # Deal with completed reasoning
        elif previous_item.channel == "analysis":
            # Emit reasoning done events
            yield ResponseReasoningTextDoneEvent(...)
            yield ResponseOutputItemDoneEvent(...)

        # Deal with completed final message
        elif previous_item.channel == "final":
            # Emit text done events
            yield ResponseTextDoneEvent(...)
            yield ResponseOutputItemDoneEvent(...)
```

**Lines 1587-1637**: Stream final channel (normal text)
```python
if ctx.parser.last_content_delta:
    if (
        ctx.parser.current_channel == "final"
        and ctx.parser.current_recipient is None
    ):
        if not sent_output_item_added:
            # First delta: emit item added event
            yield ResponseOutputItemAddedEvent(
                item=ResponseOutputMessage(
                    id=current_item_id,
                    type="message",
                    role="assistant",
                    content=[],
                    status="in_progress",
                )
            )
            yield ResponseContentPartAddedEvent(...)

        # Emit text delta
        yield ResponseTextDeltaEvent(
            delta=ctx.parser.last_content_delta,
            ...
        )
```

**Lines 1638-1681**: Stream analysis channel (reasoning)
```python
elif (
    ctx.parser.current_channel == "analysis"
    and ctx.parser.current_recipient is None
):
    if not sent_output_item_added:
        # First delta: emit reasoning item added
        yield ResponseOutputItemAddedEvent(
            item=ResponseReasoningItem(
                type="reasoning",
                id=current_item_id,
                status="in_progress",
            )
        )
        yield ResponseReasoningPartAddedEvent(...)

    # Emit reasoning delta
    yield ResponseReasoningTextDeltaEvent(
        delta=ctx.parser.last_content_delta,
        ...
    )
```

**Lines 1862-1896**: Stream tool calls
```python
if (
    ctx.parser.current_channel == "commentary"
    and ctx.parser.current_recipient
    and ctx.parser.current_recipient.startswith("functions.")
):
    if is_first_function_call_delta is False:
        # First delta: emit tool call item added
        fc_name = ctx.parser.current_recipient[len("functions.") :]
        yield ResponseOutputItemAddedEvent(
            item=ResponseFunctionToolCall(
                name=fc_name,
                type="function_call",
                arguments="",
                status="in_progress",
            )
        )
    else:
        # Subsequent deltas: emit arguments delta
        yield ResponseFunctionCallArgumentsDeltaEvent(
            delta=ctx.parser.last_content_delta,
            ...
        )
```

**Key Point**: Streaming uses `parser.last_content_delta` to get incremental text updates per channel.

---

## Summary: Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Request Entry (serving_responses.py:253-268)            │
│    - Validate request                                       │
│    - Check Harmony model (self.use_harmony)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Message Construction (serving_responses.py:914-992)     │
│    - System message with reasoning effort                   │
│    - Developer message with tools                           │
│    - User message                                            │
│    - Previous conversation history                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Token Rendering (harmony_utils.py:282-287)              │
│    - render_for_completion(messages)                        │
│    - Conversation → Token IDs                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Context Creation (serving_responses.py:361-365)         │
│    - HarmonyContext(messages, available_tools)              │
│    - StreamableParser initialization                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Tool Session Init (serving_responses.py:552-554)        │
│    - Initialize browser/python/container sessions           │
│    - MCP tool setup                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Token Generation (Engine)                                │
│    - Backend generates token IDs                            │
│    - Returns RequestOutput with output_ids                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Token Processing (context.py:185-207)                   │
│    - context.append_output(output)                          │
│    - for token_id in output_token_ids:                      │
│    -     parser.process(token_id)  ← Harmony library!      │
│    - self.messages = parser.messages  ← Structured!        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Message Routing (serving_responses.py:819-831)          │
│    - for msg in context.messages:                           │
│    -     parse_output_message(msg)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Channel-Based Routing (harmony_utils.py:290-398)        │
│    - if channel == "analysis" → ReasoningItem               │
│    - if channel == "commentary" + "functions.*" → ToolCall  │
│    - if channel == "final" → OutputMessage                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. Response Building (serving_responses.py:633-651)       │
│     - ResponsesResponse with output items                   │
│     - Usage statistics                                      │
│     - Status (completed/incomplete/cancelled)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Insights

### NO Separate Parsers!

- ❌ No `ReasoningParser` class for Harmony
- ❌ No `ToolParser` class for Harmony
- ✅ Just `parse_output_message()` routing function

### Single Parse Operation

```python
# ALL channel extraction from ONE token pass:
for token_id in output_token_ids:
    parser.process(token_id)  # ← openai-harmony does the work

messages = parser.messages  # ← Already separated by channel!

# Then just route:
for msg in messages:
    if msg.channel == "analysis":
        → ReasoningItem
    elif msg.channel == "commentary":
        → ToolCall or ReasoningItem (built-in tools)
    elif msg.channel == "final":
        → OutputMessage
```

### Tool "Parsing" is Trivial

```python
# Extract function name from recipient
function_name = message.recipient.split(".")[-1]  # "functions.get_weather" → "get_weather"

# Arguments are already JSON from Harmony!
arguments = content.text  # No parsing needed!

# Create tool call
ResponseFunctionToolCall(
    name=function_name,
    arguments=arguments,  # Already JSON!
)
```

### Token vs Text

**Harmony Responses API**: Token-based processing
- Input: Token IDs from backend
- Processing: `StreamableParser.process(token_id)`
- Output: Structured messages with channels

**Non-Harmony Responses API**: Text-based processing
- Input: Text strings
- Processing: `ReasoningParser.extract_reasoning_content(text)`
- Output: Separated reasoning and normal text

---

## File Reference Summary

| File | Lines | Purpose |
|------|-------|---------|
| **serving_responses.py** | 167-179 | Harmony detection & setup |
| | 300-307 | Request creation routing |
| | 361-365 | Context creation |
| | 503-520 | Harmony request preparation |
| | 570-583 | Harmony output extraction |
| | 819-831 | Output item creation |
| | 914-992 | Input message construction |
| | 1445-1897 | Streaming event generation |
| **context.py** | 153-179 | HarmonyContext class |
| | 185-207 | Token processing (non-streaming) |
| | 209-297 | Token usage tracking |
| | 456-530 | StreamingHarmonyContext |
| **harmony_utils.py** | 72-76 | Encoding singleton |
| | 79-120 | System message construction |
| | 137-170 | Developer message construction |
| | 282-287 | Token rendering |
| | 290-398 | Message routing (THE parser!) |
| | 401-440 | Incomplete state handling |
| | 443-444 | Stop tokens |
| | 447-448 | Parser creation |

---

## Comparison with sglang

| Aspect | vLLM | sglang | Match? |
|--------|------|--------|--------|
| Detection | `model_type == "gpt_oss"` | Same | ✅ |
| Context | `HarmonyContext` | `HarmonyContext` | ✅ |
| Parser | `StreamableParser` | `StreamableParser` | ✅ |
| Processing | `parser.process(token_id)` | Same | ✅ |
| Messages | `parser.messages` | `parser.messages()` | ✅ |
| Routing | `parse_output_message()` | Same | ✅ |
| Incomplete | `parse_remaining_state()` | Same | ✅ |
| Separate Parsers | None | None | ✅ |

**Conclusion**: Nearly identical implementations! Same architecture, same patterns, same routing logic.
