import json
import unittest
from sglang.test.test_utils import CustomTestCase
from sglang.srt.debug_utils import log_parser


class TestLogParser(CustomTestCase):
    def test_log_parser(self):
        lines = '''
(SGLangEngine pid=35555) [2025-10-31 03:45:20 TP0] Decode batch [51341], #running-req: 317, #token: 1094261, token usage: 0.67, cuda graph: True, gen throughput (token/s): 14806.57, #queue-req: 0,
(SGLangEngine pid=111711, ip=10.15.36.1) [2025-10-31 03:45:20 TP0] Decode batch [39913], #running-req: 78, #token: 432100, token usage: 0.27, cuda graph: True, gen throughput (token/s): 7269.16, #queue-req: 0,
[2025-11-03 14:31:10 DP6 TP6 EP6] Decode batch, #running-req: 251, #token: 2811200, token usage: 1.00, cuda graph: True, gen throughput (token/s): 2055.94, #queue-req: 655,
'''
        df = log_parser.parse(lines)

        print(df)
        print(df.write_json())

        expect_rows = ['TODO']
        self.assertEqual(df.to_dicts(), expect_rows)


if __name__ == "__main__":
    unittest.main()
