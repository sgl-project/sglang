import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*swig.*")
warnings.filterwarnings("ignore", message=".*SwigPyPacked.*")
warnings.filterwarnings("ignore", message=".*SwigPyObject.*")
warnings.filterwarnings("ignore", message=".*swigvarlink.*")

import pytest
from test_framework import StreamingTestRunner
from test_fixtures import get_test_tools
from test_cases import test_case_registry
from parser_factory import create_parser


def pytest_generate_tests(metafunc):
    if "test_name" in metafunc.fixturenames and "mode_type" in metafunc.fixturenames:
        parser_mode = metafunc.config.getoption("--parser-mode", default="both")

        test_names = test_case_registry.get_names()
        mode_types = []

        if parser_mode in ("both", "streaming"):
            mode_types.append("streaming")
        if parser_mode in ("both", "nonstream"):
            mode_types.append("nonstream")

        test_params = [(name, mode) for name in test_names for mode in mode_types]  # 组合
        test_ids = [f"{name}:{mode}" for name in test_names for mode in mode_types]  # ID

        metafunc.parametrize("test_name,mode_type", test_params, ids=test_ids)


@pytest.fixture(scope="module")
def test_runner(request):
    verbose = request.config.getoption("--verbose", default=False)
    parser_source = request.config.getoption("--parser-source", default="default")
    parser_mode = request.config.getoption("--parser-mode", default="both")

    # Create parser using factory
    try:
        detector = create_parser(parser_source)
    except Exception as e:
        pytest.fail(f"Failed to create parser from source '{parser_source}': {e}")

    runner = StreamingTestRunner(
        detector=detector,
        tools=get_test_tools(),
        parser_mode=parser_mode,
        verbose=verbose,
    )
    return runner


def test_parser(test_name, mode_type, test_runner, request):
    test_func = test_case_registry.get(test_name)
    result = test_func(runner=test_runner)

    if mode_type == "streaming":
        if result.streaming_match is None:
            pytest.fail(f"Streaming validation was not performed for '{test_name}'. This should not happen when mode_type is 'streaming'.")
        if result.streaming_match is False:
            error_msg = f"Streaming validation failed for '{test_name}': {', '.join(result.streaming_errors or [])}"
            if test_runner.verbose:
                print(f"\nStreaming errors: {result.streaming_errors}")
            pytest.fail(error_msg)
    elif mode_type == "nonstream":
        if result.non_streaming_match is None:
            pytest.fail(f"Non-streaming validation was not performed for '{test_name}'. This should not happen when mode_type is 'nonstream'.")
        if result.non_streaming_match is False:
            error_msg = f"Non-streaming validation failed for '{test_name}': {', '.join(result.non_streaming_errors or [])}"
            if test_runner.verbose:
                print(f"\nNon-streaming errors: {result.non_streaming_errors}")
            pytest.fail(error_msg)
