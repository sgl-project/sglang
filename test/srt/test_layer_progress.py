import unittest

import torch

from sglang.srt.disaggregation.layer_progress import LayerProgress


class TestLayerProgressValidation(unittest.TestCase):
    def test_rejects_non_positive_layer_count(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            LayerProgress(0, torch.device("cuda"))

    def test_rejects_non_cuda_device(self):
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            LayerProgress(4, torch.device("cpu"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestLayerProgressCuda(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())

    def test_selected_ring_points(self):
        progress = LayerProgress(8, self.device, ring_layers=[1, 4, 7])
        generation = progress.start_forward()
        progress.record(0)
        progress.record(1)
        torch.cuda.synchronize()
        self.assertEqual(progress.completed_layers(generation), 2)
        self.assertEqual(progress.num_ring_points, 3)

    def test_rejects_stale_generation(self):
        progress = LayerProgress(4, self.device)
        old_generation = progress.start_forward()
        progress.record(3)
        torch.cuda.synchronize()
        self.assertEqual(progress.completed_layers(old_generation), 4)

        new_generation = progress.start_forward()
        torch.cuda.synchronize()
        self.assertEqual(progress.completed_layers(old_generation), 0)
        self.assertEqual(progress.completed_layers(new_generation), 0)

    def test_generation_wraps_before_int32_overflow(self):
        progress = LayerProgress(4, self.device)
        progress._generation = progress._max_generation
        generation = progress.start_forward()
        torch.cuda.synchronize()
        self.assertEqual(generation, 1)
        self.assertEqual(progress.completed_layers(generation), 0)

    def test_explicit_forward_stream_orders_generation_before_record(self):
        progress = LayerProgress(4, self.device, ring_layers=[3])
        stream = torch.cuda.Stream()
        generation = progress.start_forward(stream)
        with torch.cuda.stream(stream):
            progress.record(3)
        stream.synchronize()
        self.assertEqual(progress.completed_layers(generation), 4)

    def test_cuda_graph_replay_updates_each_generation(self):
        progress = LayerProgress(4, self.device, ring_layers=[1, 3])
        graph = torch.cuda.CUDAGraph()

        progress.start_forward()
        with torch.cuda.graph(graph):
            progress.record(1)
            progress.record(3)
        torch.cuda.synchronize()

        for _ in range(2):
            generation = progress.start_forward()
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(progress.completed_layers(generation), 4)


if __name__ == "__main__":
    unittest.main()
