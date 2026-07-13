import unittest

from starlette.routing import Match

from sglang.srt.utils.common import _get_fastapi_request_path


class TestHTTPMetricsRoutePath(unittest.TestCase):
    def test_request_path_falls_back_when_matched_route_has_no_path(self):
        class IncludedRouterLike:
            def matches(self, scope):
                return Match.FULL, {}

        class App:
            routes = [IncludedRouterLike()]

        class URL:
            path = "/v1/loads"

        class Request:
            app = App()
            scope = {"path": "/v1/loads"}
            url = URL()

        path, is_handled_path = _get_fastapi_request_path(Request())

        self.assertEqual(path, "/v1/loads")
        self.assertFalse(is_handled_path)


if __name__ == "__main__":
    unittest.main()
