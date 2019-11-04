
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Scale(nn.Module):
    def __init__(self, size, bias=False):
        super(Scale, self).__init__()
        self.bias = bias
        self.gamma = nn.Parameter(torch.FloatTensor(size))
        if bias:
            self.beta = nn.Parameter(torch.FloatTensor(size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.gamma, 1.0)
        if self.bias:
            nn.init.constant(self.beta, 0)

    def forward(self, input):
        if len(input.shape) == 4:
            output = input * self.gamma.view(1, -1, 1, 1)
            if self.bias:
                output = output + self.beta.view(1, -1, 1, 1)
        elif len(input.shape) == 2:
            output = input * self.gamma.view(1, -1)
            if self.bias:
                output = output + self.beta.view(1, -1)
        else:
            assert(False)
        return output
