import numpy as np
from net import NetWork, DataLoader
from loss import MSE

if __name__ == '__main__':
    train_data = []
    train_label = []
    with open('./data/sincos_train.txt','r') as f:
        for line in f.readlines():
            tmp = line[:-1].split(' ')
            train_data.append(np.array([[float(tmp[0])],[float(tmp[1])]]))
            train_label.append(np.array([[float(tmp[2])]]))

    dataLoader = DataLoader(batch_size=10, dataset=train_data, labelset=train_label)
    train_data = dataLoader.batch_data
    train_label = dataLoader.batch_label


    val_data = []
    val_label = []
    with open('./data/sincos_val.txt','r') as f:
        for line in f.readlines():
            tmp = line[:-1].split(' ')
            val_data.append(np.array([[float(tmp[0])],[float(tmp[1])]]))
            val_label.append(np.array([[float(tmp[2])]]))

    valDataLoader = DataLoader(batch_size=10, dataset=val_data, labelset=val_label)
    val_data = valDataLoader.batch_data
    val_label = valDataLoader.batch_label



    net = NetWork([2, 10, 1], MSE()) # 构建神经网络，输入层2，隐藏层300，输出层1
    net.load_model()
    epochs = 1000000
    learning_rate = 0.1
    momentum = 0.1
    loss = 0

    for epoch in range(1, epochs):
        for i in range(len(train_data)):
            net.train(one_batch_data=train_data[i], one_batch_label=train_label[i], learning_rate=learning_rate, momentum=momentum)
        
        for i in range(len(val_data)):
            loss = loss + net.evaluate(val_data=val_data[i], val_label=val_label[i])

        if(epoch % 100 == 0):
            print(loss/100)
            if(loss/100 < 0.01):
                break
            loss = 0
        
        if(epoch % 1000 == 0):
            net.save_model()

    net.save_model()

    with open('predict_res.txt', 'w') as f:
        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                predict_res = net.predict(train_data[i][j])
                f.write(str(predict_res) + '\n')