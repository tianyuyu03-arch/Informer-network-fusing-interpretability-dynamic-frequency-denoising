import torch
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import math
class WaveletActivationFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input): # [128, 256, 64]
        ctx.save_for_backward(input)
        # 使用小波基函数进行前向传播
        output = (torch.cos(1.75 * input)) * (torch.exp(-(input**2)/2))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 使用小波基函数的导数进行反向传播
        grad_input = (-1) * (1.75 * torch.sin(1.75 * input) + input * torch.cos(1.75 * input)) * (torch.exp(-(input**2)/2))
        return grad_input * grad_output


if __name__ == '__main__':
    input = np.linspace(-np.pi*2, np.pi*2, 100)
    output = (np.cos(1.75 * input)) * (np.exp(-(input**2)/2))

    # 绘制函数曲线
    plt.plot(input, output)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Plot')
    plt.grid(True)
    plt.show()