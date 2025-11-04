import json
import unittest

from sglang.srt.debug_utils import log_parser
from sglang.test.test_utils import CustomTestCase


class TestLogParser(CustomTestCase):
    def test_log_parser(self):
        lines = """
(SGLangEngine pid=35555) [2025-10-31 03:45:20 TP0] Decode batch [51341], #running-req: 317, #token: 1094261, token usage: 0.67, cuda graph: True, gen throughput (token/s): 14806.57, #queue-req: 0,
(SGLangEngine pid=111711, ip=10.15.36.1) [2025-10-31 03:45:20 TP0] Decode batch [39913], #running-req: 78, #token: 432100, token usage: 0.27, cuda graph: True, gen throughput (token/s): 7269.16, #queue-req: 0,
[2025-11-03 14:31:10 DP6 TP6 EP6] Decode batch, #running-req: 251, #token: 2811200, token usage: 1.00, cuda graph: True, gen throughput (token/s): 2055.94, #queue-req: 655,
"""
        expect_rows = json.loads(
            """[{"line":"(SGLangEngine pid=35555) [2025-10-31 03:45:20 TP0] Decode batch [51341], #running-req: 317, #token: 1094261, token usage: 0.67, cuda graph: True, gen throughput (token/s): 14806.57, #queue-req: 0,","1":"(SGLangEngine pid=35555)","pid":35555,"ip":null,"time":"2025-10-31 03:45:20","dp_rank":null,"tp_rank":0,"pp_rank":null,"8":" [51341]","num_running_req":317,"num_token":1094261,"token_usage":0.67,"gen_throughput":14806.57,"queue_req":0},{"line":"(SGLangEngine pid=111711, ip=10.15.36.1) [2025-10-31 03:45:20 TP0] Decode batch [39913], #running-req: 78, #token: 432100, token usage: 0.27, cuda graph: True, gen throughput (token/s): 7269.16, #queue-req: 0,","1":"(SGLangEngine pid=111711, ip=10.15.36.1)","pid":111711,"ip":"10.15.36.1","time":"2025-10-31 03:45:20","dp_rank":null,"tp_rank":0,"pp_rank":null,"8":" [39913]","num_running_req":78,"num_token":432100,"token_usage":0.27,"gen_throughput":7269.16,"queue_req":0}]"""
        )

        df = log_parser.parse(lines)
        print(df)
        print(df.write_json())

        assert len(df) == len(lines.strip().splitlines()), f"{len(df)=}"
        self.assertEqual(df.to_dicts(), expect_rows)


if __name__ == "__main__":
    unittest.main()
