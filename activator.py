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

# class SoftMax:
#     def forward(self, x):
#         x = x - np.max(x)
#         return np.exp(x) / np.sum(np.exp(x))

#     def backward(self, y):
#         res = np.zeros(y.shape)

#         for i in range(y.shape[0]):
#             for j in range(y.shape[0]):
#                 if i == j:
#                     res[i] += (y[j] * (1 - y[j]))
#                 else:
#                     res[i] += (- y[i] * y[j])

#         return res