import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as LA

class QuantizationF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, islinear, scale):
        ctx.save_for_backward(input)
        ctx.islinear = islinear
        ctx.scale = scale
        if ctx.islinear: #linear
            if ctx.scale is None:
                return input
            else:
                input = (input / ctx.scale).round().clamp(-127, 127) * ctx.scale
                return input
        else: # relu
            if ctx.scale is None:
                return input.clamp(min=0.0)
            else:
                input = (input / ctx.scale).round().clamp(0, 255) * ctx.scale
                return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        if not ctx.islinear: # relu
            if ctx.scale is None:
                #grad_input[input.ge(6.0)] = 0
                grad_input[input.le(0.0)] = 0
            else:
                grad_input[input.ge(255*ctx.scale)] = 0
                grad_input[input.le(0.0)] = 0
        return grad_input, None, None


class Quantization(nn.Module):
    def __init__(self, islinear, scale=None):
        super(Quantization, self).__init__()
        self.islinear = islinear
        self.scale = scale

    def set_scale(self, scale):
        self.scale = scale

    def forward(self, x):
        return QuantizationF.apply(x, self.islinear, self.scale)

