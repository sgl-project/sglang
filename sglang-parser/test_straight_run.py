from parser_factory import create_parser
from test_framework import StreamingTestRunner
from test_fixtures import get_test_tools

# detector = create_parser(parser_source="default")
detector = create_parser(parser_source="legacy_v3")

runner = StreamingTestRunner(detector, tools=get_test_tools(), parser_mode="streaming")
result = runner.run_test(
    test_name="My Test",
    response_text="""Plain Text

<tool_call>
<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
</tool_call>
<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users</parameter>
<parameter=dry_run>True</parameter>
</function>
</tool_call>""",
    mode="token",
    verbose=True,
)

print()
print(result)
print()

if result.success and result.streaming_match:
    print("Test passed!")
else:
    print(f"Test failed: {result.error_message}")
    if result.streaming_errors:
        print(f"Streaming errors: {result.streaming_errors}")
