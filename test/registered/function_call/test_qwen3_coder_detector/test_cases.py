from test.registered.function_call.test_qwen3_coder_detector.test_fixtures import (
    get_test_tools,
)
from test.registered.function_call.test_qwen3_coder_detector.test_framework import (
    StreamingTestRunner,
)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


class TestCaseRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str = None):
        def decorator(func):
            test_name = name or func.__name__
            if test_name.startswith("test_"):
                test_name = test_name[5:]  # Remove "test_" prefix
            cls._registry[test_name] = func
            return func

        return decorator

    @classmethod
    def get_all(cls):
        return cls._registry.copy()

    @classmethod
    def get(cls, name: str):
        return cls._registry.get(name)

    @classmethod
    def get_names(cls):
        return list(cls._registry.keys())


test_case_registry = TestCaseRegistry()


class TestCase:
    def __init__(self, runner: StreamingTestRunner = None):
        self.runner = runner or StreamingTestRunner(tools=get_test_tools())

    def run(
        self,
        test_name: str,
        response_text: str,
        mode: str = "atomic_tags",
        expected: dict = None,
    ):
        """
        Run a single test case

        Args:
            test_name: Test name
            response_text: Response text
            mode: Streaming generation mode
            expected: Expected parse result, containing 'text' and 'tools' fields
        """
        return self.runner.run_test(
            test_name=test_name,
            response_text=response_text,
            mode=mode,
            expected=expected,
        )


@test_case_registry.register("plain_text")
def test_plain_text(runner: StreamingTestRunner = None):
    test_case = TestCase(runner)
    text = """This is plain text test"""

    expected = {
        "text": "This is plain text test",
        "tools": [],
    }

    return test_case.run("Plain Text", text, mode="atomic_tags", expected=expected)


@test_case_registry.register("single_tool_call")
def test_single_tool_call(runner: StreamingTestRunner = None):
    """Single tool call test"""
    test_case = TestCase(runner)
    text = """<function=get_current_weather>
<parameter=location>Boston</parameter>
<parameter=unit>celsius</parameter>
<parameter=days>3</parameter>
</function>"""
    expected = {
        "text": "",
        "tools": [
            {
                "name": "get_current_weather",
                "args": {"location": "Boston", "unit": "celsius", "days": 3},
            }
        ],
    }
    return test_case.run(
        "Standard Single Tool", text, mode="atomic_tags", expected=expected
    )


@test_case_registry.register("single_tool_call_with_text")
def test_single_tool_call_with_text(runner: StreamingTestRunner = None):
    """Single tool call test with text prefix"""
    test_case = TestCase(runner)
    text = """Text prefix + tool call test

<function=get_current_weather>
<parameter=location>Boston</parameter>
<parameter=unit>celsius</parameter>
<parameter=days>3</parameter>
</function>"""
    expected = {
        "text": "Text prefix + tool call test\n    \n",
        "tools": [
            {
                "name": "get_current_weather",
                "args": {"location": "Boston", "unit": "celsius", "days": 3},
            }
        ],
    }
    return test_case.run(
        "Standard Single Tool with Text", text, mode="atomic_tags", expected=expected
    )


@test_case_registry.register("double_tool_call")
def test_double_tool_call(runner: StreamingTestRunner = None):
    """Double tool call test"""
    test_case = TestCase(runner)
    text = """Text prefix + double tool call test

<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users</parameter>
<parameter=dry_run>True</parameter>
</function>
"""
    expected = {
        "text": "Text prefix + double tool call test\n\n",
        "tools": [
            {"name": "get_current_weather", "args": {"location": "New York"}},
            {
                "name": "sql_interpreter",
                "args": {"query": "SELECT * FROM users", "dry_run": True},
            },
        ],
    }
    return test_case.run(
        "Double Tool Call", text, mode="atomic_tags", expected=expected
    )


@test_case_registry.register("random_fragmentation")
def test_random_fragmentation_stress_test(runner: StreamingTestRunner = None):
    """Random fragmentation stress test"""
    test_case = TestCase(runner)
    text = """Text prefix + random fragmentation stress test

<function=get_current_weather>
<parameter=location>San Francisco</parameter>
</function>"""
    expected = {
        "text": "Text prefix + random fragmentation stress test\n\n",
        "tools": [
            {"name": "get_current_weather", "args": {"location": "San Francisco"}}
        ],
    }
    return test_case.run(
        "Random Fragmentation Stress Test", text, mode="atomic_tags", expected=expected
    )


@test_case_registry.register("implicit_parameter_close")
def test_implicit_parameter_close(runner: StreamingTestRunner = None):
    """Implicit parameter close test"""
    test_case = TestCase(runner)
    text = """Text prefix + implicit parameter close test

<function=get_current_weather>
<parameter=location>Paris
<parameter=unit>celsius
<parameter=days>3
</function>"""
    expected = {
        "text": "Text prefix + implicit parameter close test\n\n",
        "tools": [
            {
                "name": "get_current_weather",
                "args": {"location": "Paris", "unit": "celsius", "days": 3},
            }
        ],
    }
    return test_case.run(
        "Implicit Parameter Close", text, mode="atomic_tags", expected=expected
    )


@test_case_registry.register("complex_schema")
def test_complex_schema_toolcall(runner: StreamingTestRunner = None):
    """Complex schema tool call test"""
    test_case = TestCase(runner)
    text = """Text prefix + complex schema tool call test

<function=TodoWrite>
<parameter=todos>
[
 {"content": "Buy a new car", "status": 'pending'},
 {"content": "Buy a new house", "status": "completed"}
]
</parameter>
</function>

"""
    expected = {
        "text": "Text prefix + complex schema tool call test\n\n",
        "tools": [
            {
                "name": "TodoWrite",
                "args": {
                    "todos": [
                        {"content": "Buy a new car", "status": "pending"},
                        {"content": "Buy a new house", "status": "completed"},
                    ]
                },
            }
        ],
    }
    return test_case.run(
        "Complex Schema Tool Call", text, mode="atomic_tags", expected=expected
    )
