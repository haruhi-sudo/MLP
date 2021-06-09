import numpy as np

class Sigmod:
    def forward(self, x):
        return 1.0 / (np.exp(-x) + 1.0)
    
    def backward(self, x):
        return x * (1-x) # sigmod函数的梯度可以用自身表示

class Relu:
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x):
        return 1.0 * (x > 0)

class Linear:
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones(x.shape)   