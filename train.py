import numpy as np
from net import NetWork, DataLoader
from loss import MSE

# 使用DataLoader，将数据加载入batch
def load_data(path, batch_size=10):
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


    net = NetWork([2, 10, 1], MSE()) # 构建神经网络，输入层2，隐藏层10，输出层1

    epochs = 1000000

    trian_loss = 0
    val_loss = 0
    val_loss_old = 1e9

    for epoch in range(1, epochs):
        for i in range(len(train_data)):
            net.train(one_batch_data=train_data[i], one_batch_label=train_label[i], learning_rate=0.1, momentum=0.9)
        
        trian_loss += calc_loss(data=train_data, label=train_label, net=net)
        val_loss += calc_loss(data=val_data, label=val_label, net=net)

        if(epoch % 100 == 0):
            print('train:{}    val:{}'.format(str(trian_loss/100), str(val_loss/100)))
            if(val_loss_old <  val_loss and epoch > 1000):
                break
            net.save_model('./model')
            trian_loss = 0
            val_loss_old = val_loss
            val_loss = 0

            with open('predict_res.txt', 'w') as f:
                for i in range(len(val_data)):
                    for j in range(len(val_data[i])):
                        predict_res = net.predict(val_data[i][j])
                        f.write(str(predict_res) + '\n')    
