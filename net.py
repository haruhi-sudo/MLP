import numpy as np
from fcn import FCN
from activator import Sigmod, Relu, Linear


class NetWork:
    def __init__(self, layers, activators, loss):
        self.layers = []
        self.loss = loss # 损失函数
        for i in range(len(layers) - 2):
            self.layers.append(FCN(layers[i], layers[i+1], activators[i]))

        self.layers.append(FCN(layers[i+1], layers[i+2], activators[i+1]))


    def predict(self, predict_data):
        predict_result = predict_data
        for layer in self.layers:
            layer.forward(predict_result)
            predict_result = layer.output

        return predict_result


    def calc_gradient(self, label):
        # 输出层的误差
        delta = - self.loss.backward(self.layers[-1].output, label)
        for layer in self.layers[::-1]:
            layer.backward(delta) # 将误差传递给隐藏层
            delta = layer.delta
        return delta


    def zero_gradient(self):
        for layer in self.layers[::-1]:
            # 记录上一次的梯度，用于计算动量
            layer.d_weights_old = layer.d_weights
            layer.d_bias_old = layer.d_bias

            layer.d_bias = 0
            layer.d_weights = 0
            

    def train(self, one_batch_data, one_batch_label, learning_rate, momentum):
        self.zero_gradient()
        # 合的导数等于导数的和
        for i in range(len(one_batch_data)):
        
            self.predict(one_batch_data[i])
            self.calc_gradient(one_batch_label[i])
                
        for layer in self.layers:
            layer.update(learning_rate, momentum)


    def evaluate(self, val_data, val_label):
        test_result = []
        for i in range(len(val_data)):
            test_result.append([self.predict(val_data[i]), val_label[i]])
        
        return np.sum([self.loss.forward(x, y) for (x, y) in test_result])


    def save_model(self, path):
        for i in range(len(self.layers)):
            np.savetxt(path + '/weights' + str(i) + '.txt', self.layers[i].weights)
            np.savetxt(path + '/bias' + str(i) + '.txt', self.layers[i].bias)


    def load_model(self, path):
        for i in range(len(self.layers)):
            self.layers[i].weights = np.loadtxt(path + '/weights' + str(i) + '.txt')\
                .reshape(self.layers[i].output_size, self.layers[i].input_size)
            self.layers[i].bias = np.loadtxt(path + '/bias' + str(i) + '.txt')\
                .reshape(self.layers[i].output_size, 1)


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

