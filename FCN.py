import numpy as np


class Sigmod:
    def forward(self, x):
        return 1.0 / (np.exp(-x) + 1.0)
    
    def backward(self, x):
        return x * (1-x) # sigmod函数的梯度可以用自身表示

class FCN:
    '''
        全连接层的一层，准确来说是两层之间的参数
    '''
    def __init__(self, input_size, output_size, activator):
        self.input = np.zeros((input_size, 1))
        self.output = np.zeros((output_size, 1))
        # 全连接层激活函数
        self.activator = activator
        # 全连接层权重
        self.weights = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 全连接层偏置
        self.bias =  np.zeros((output_size, 1))
    
    # 前向计算
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias
        self.output = self.activator.forward(self.output)

    # 反向传播更新
    def backward(self, delta):
        '''
            输入这一层的误差，计算梯度和下一层的误差
        '''
        self.delta = self.activator.backward(self.input) * np.dot(\
            self.weights.T, delta) # 下一层的误差
        self.d_weights =  np.dot(delta, self.input.T) # 计算weights梯度
        self.d_bias = delta # 计算bias梯度
    
    def update(self, learning_rate):
        '''
            使用梯度下降算法更新权重
        '''
        self.weights += learning_rate * self.d_weights
        self.bias += learning_rate * self.d_bias


class NetWork:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FCN(layers[i], layers[i+1], Sigmod()))

    def predict(self, predict_data):
        predict_result = predict_data
        for layer in self.layers:
            layer.forward(predict_result)
            predict_result = layer.output

        return predict_result


    def calc_gradient(self, label):
        # 输出层的误差
        delta = self.layers[-1].activator.backward(\
            self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta) # 将误差传递给隐藏层
            delta = layer.delta
        return delta

    def train(self, one_batch_data, one_batch_label, learning_rate, epoch):
        for i in range(epoch):
            for j in range(len(one_batch_data)):
                self.predict(one_batch_data[j])
                self.calc_gradient(one_batch_label[j])
                
                for layer in self.layers:
                    layer.update(learning_rate)

class DataLoader:
    def __init__(self, batch_size, dataset, labelset):
        self.batch_size = batch_size
        self.batch_data = []
        self.batch_label = []

        data_tmp = []
        label_tmp = []

        for i in range(len(dataset)):
            data_tmp.append(dataset[i])
            label_tmp.append(labelset[i])
            if i % batch_size == batch_size - 1:
                self.batch_data.append(data_tmp)
                self.batch_label.append(label_tmp)
                data_tmp = []
                label_tmp = []
        
        self.batch_data.append(data_tmp)
        self.batch_label.append(label_tmp)
