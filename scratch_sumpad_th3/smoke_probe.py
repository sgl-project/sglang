import torch

from scratch_sumpad_th3 import shape_probe


def main() -> None:
    shape_probe.build_module()

    shape_probe.note_step(
        rank=3,
        real_local_tokens=257,
        padded_local_tokens=2048,
        dp_pad_mode=1,
        dp_buffer_len=16384,
        global_max_tokens=257,
        global_sum_tokens=257,
    )
    hidden_states = torch.zeros(2048, 4096, device="cuda", dtype=torch.bfloat16)
    shape_probe.probe(0, hidden_states)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        with torch.cuda.graph(graph):
            shape_probe.probe(7, hidden_states)
    torch.cuda.current_stream().wait_stream(side_stream)

    shape_probe.note_step(
        rank=3,
        real_local_tokens=99,
        padded_local_tokens=2048,
        dp_pad_mode=2,
        dp_buffer_len=777,
        global_max_tokens=99,
        global_sum_tokens=99,
    )
    shape_probe.note_used_prefill_graph(True)
    graph.replay()
    torch.cuda.synchronize()

    print("PROBE_SMOKE_OK")


if __name__ == "__main__":
    main()
