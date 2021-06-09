import numpy as np


class FCN:
    '''
        全连接层的一层，准确来说是两层之间的参数
    '''
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.input = np.zeros((input_size, 1))
        self.output = np.zeros((output_size, 1))
        
        # 全连接层激活函数
        self.activator = activator
        
        # 初始化
        std = np.sqrt(2. / (input_size + output_size))
        self.weights = np.random.normal(loc=0., scale=std, size=[output_size, input_size]) # 全连接层权重    
        self.bias =  np.random.normal(loc=0., scale=std, size=[output_size, 1]) # 全连接层偏置


    # 前向计算
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias
        self.output = self.activator.forward(self.output)


    # 反向传播计算梯度
    def backward(self, delta):
        '''
            输入这一层的误差，计算梯度和下一层的误差
            关于什么是误差delta，可参考报告
        '''
        self.delta = self.activator.backward(self.input) * np.dot(self.weights.T, delta) # 下一层的误差
        self.d_weights =  np.dot(delta, self.input.T) # 计算weights梯度
        self.d_bias = delta # 计算bias梯度
    

    def update(self, learning_rate):
        '''
            使用梯度下降算法更新权重
        '''
        self.weights += learning_rate * self.d_weights
        self.bias += learning_rate * self.d_bias

