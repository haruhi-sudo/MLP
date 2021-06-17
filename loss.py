import numpy as np

class MSE:
    '''
        均方误差，输入真实值，目标值
    '''
    def forward(self, output, target):
        return 0.5 * (output - target) ** 2 # 均方误差损失函数

    def backward(self, output, target):
        return output - target # 求导，反向传播时使用


class CE:
    '''
        交叉熵
    '''
    def forward(self, output, target):
        return - target * np.log(output + 1e-10) - (1 - target) * np.log(1 - output + 1e-10)

    def backward(self, output, target):
        return - target * 1. / (output + 1e-10) - (target - 1) * 1. / (1 - output + 1e-10)