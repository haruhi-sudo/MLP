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

class FCN:
    '''
        全连接层的一层，准确来说是两层之间的参数
    '''
    def __init__(self, input_size, output_size, activator):
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


class NetWork:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(FCN(layers[i], layers[i+1], Sigmod()))

        self.layers.append(FCN(layers[i+1], layers[i+2], Sigmod()))

    def predict(self, predict_data):
        predict_result = predict_data
        for layer in self.layers:
            layer.forward(predict_result)
            predict_result = layer.output

        return predict_result


    def calc_gradient(self, label):
        # 输出层的误差
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta) # 将误差传递给隐藏层
            delta = layer.delta
        return delta

    def train(self, one_batch_data, one_batch_label, learning_rate):
        for j in range(len(one_batch_data)):
            self.predict(one_batch_data[j])
            self.calc_gradient(one_batch_label[j])
                
            for layer in self.layers:
                layer.update(learning_rate)

    def evaluate(self, train_data, train_label):
        test_result = []
        for i in range(len(train_data)):
            test_result.append([self.predict(train_data[i]), train_label[i]])
        print()
        return np.sum([0.5 * (x - y) ** 2 for (x, y) in test_result])


class DataLoader:
    '''
        随机梯度下降算法所需的batch
    '''
    def __init__(self, batch_size, dataset, labelset):
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
        
        if len(data_tmp) != 0:
            self.batch_data.append(data_tmp)
        
        if len(label_tmp) != 0:
            self.batch_label.append(label_tmp)
