import json
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.server_args import PortArgs, prepare_server_args
from sglang.test.test_utils import CustomTestCase


class TestPrepareServerArgs(CustomTestCase):
    def test_prepare_server_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "model_path",
                "--json-model-override-args",
                '{"rope_scaling": {"factor": 2.0, "rope_type": "linear"}}',
            ]
        )
        self.assertEqual(server_args.model_path, "model_path")
        self.assertEqual(
            json.loads(server_args.json_model_override_args),
            {"rope_scaling": {"factor": 2.0, "rope_type": "linear"}},
        )


class TestPortArgs(unittest.TestCase):
    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_standard_case(self, mock_temp_file, mock_is_port_available):

        mock_is_port_available.return_value = True
        mock_temp_file.return_value.name = "temp_file"

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = False

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.scheduler_input_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("ipc://"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_single_node_dp_attention(self, mock_is_port_available):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = None

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_dp_rank(self, mock_is_port_available):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = "192.168.1.1:25000"

        port_args = PortArgs.init_new(server_args, dp_rank=2)

        print(f"{port_args=}")
        self.assertTrue(port_args.scheduler_input_ipc_name.endswith(":25008"))

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_ipv4_address(self, mock_is_port_available):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://192.168.1.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv4_address(self, mock_is_port_available):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1"

        with self.assertRaises(AssertionError) as context:
            PortArgs.init_new(server_args)

        self.assertIn(
            "please provide --dist-init-addr as host:port", str(context.exception)
        )

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv4_address_invalid_port(
        self, mock_is_port_available
    ):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:abc"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_ipv6_address(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://[2001:db8::1]:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertTrue(
            port_args.detokenizer_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=False)
    def test_init_new_with_invalid_ipv6_address(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[invalid-ipv6]:25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid IPv6 address", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv6_address_missing_bracket(
        self, mock_is_port_available
    ):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1:25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid IPv6 address format", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_missing_port(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn(
            "a port must be specified in IPv6 address", str(context.exception)
        )

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_invalid_port(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:abcde"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("invalid port in IPv6 address", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_wrong_separator(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):

        mock_is_port_available.return_value = True

        server_args = MagicMock()
        server_args.port = 30000
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]#25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("expected ':' after ']'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
