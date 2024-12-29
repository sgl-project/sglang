from torch import nn


class BaseCausalLM(nn.Module):
    def export_static_params(self):
        for name, param in self.named_parameters():
            TODO
        return TODO

    def import_static_params(self, static_params):
        TODO
