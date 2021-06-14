import numpy as np

'''
    注意，不同于loss.py文件，backward不是直接对函数求导！！！
    而是用函数的输出，表示函数的导数
'''

class Sigmod:
    def forward(self, x):
        return 1.0 / (np.exp(-x) + 1.0)
    
    def backward(self, y):
        return y * (1-y) # sigmod函数的梯度可以用自身表示

class Relu:
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, y):
        return 1.0 * (y != 0)

class Linear:
    def forward(self, x):
        return x

    def backward(self, y):
        return np.ones(y.shape)   