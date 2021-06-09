class MSE:
    '''
        均方误差，输入真实值，目标值
    '''
    def forward(self, real, target):
        return 0.5 * (real - target) ** 2 # 均方误差损失函数

    def backward(self, real, target):
        return real - target # 求导，反向传播时使用


class CE:
    def forward(self, t, y):
        return 0.5 * (t - y) ** 2

    def backward(self, t, y):
        return t - y