from sglang.srt.server_args import ServerArgs


class SchedulerRecvSkipper:
    def __init__(self, server_args: ServerArgs):
        # Can be supported if needed, but may need e.g. `global_forward_mode`
        assert not server_args.enable_dp_attention

    def handle(self):
        TODO
