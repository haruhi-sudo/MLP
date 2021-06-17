import numpy as np
import time
from net import NetWork, DataLoader
from loss import MSE
from activator import Sigmod, Relu, Linear

# 使用DataLoader，将数据加载入batch
def load_data(path, batch_size=1):
    data = []
    label = []
    with open(path,'r') as f:
        for line in f.readlines():
            tmp = line[:-1].split(' ')
            data.append(np.array([[float(tmp[0])],[float(tmp[1])]]))
            label.append(np.array([[float(tmp[2])]]))

    dataLoader = DataLoader(batch_size=batch_size, dataset=data, labelset=label)
    return dataLoader.batch_data, dataLoader.batch_label


def calc_loss(data, label, net):
    loss = 0
    for i in range(len(data)):
        loss += net.evaluate(val_data=data[i], val_label=label[i])
    return loss


if __name__ == '__main__':
    train_path = './data/sincos_train.txt'
    val_path = './data/sincos_val.txt'
    
    train_data, train_label = load_data(train_path)
    val_data, val_label = load_data(val_path)


    net = NetWork(layers=[2, 10, 1], activators=[Sigmod(), Sigmod()],\
         loss=MSE()) # 创建神经网络，输入层2，隐藏层10，输出层1

    epochs = 10000

    trian_loss = 0
    val_loss = 0

    starttime = time.time()
    for epoch in range(1, epochs):
        for i in range(len(train_data)):
            net.train(one_batch_data=train_data[i], one_batch_label=train_label[i], learning_rate=0.1, momentum=0.)

        
        trian_loss += calc_loss(data=train_data, label=train_label, net=net)
        val_loss += calc_loss(data=val_data, label=val_label, net=net)

        # with open('loss.txt', 'a+') as f:
        #     f.write('train:{}    val:{} \n'.format(str(trian_loss/100), str(val_loss/100)))

        if(epoch % 100 == 0):
            print('train:{}    val:{}'.format(str(trian_loss/100), str(val_loss/100)))

            net.save_model('./model')
            if (val_loss/100 < 0.001):
                break
            trian_loss = 0
            val_loss = 0 

    endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime)
    
