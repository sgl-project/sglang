from torch import nn


class BaseCausalLM(nn.Module):
    def export_static_state(self):
        return dict(
            buffers=[
                (name, buffer.detach().clone()) for name, buffer in self.named_buffers()
            ]
        )

    def import_static_state(self, static_params):
        self_named_buffers = dict(self.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
