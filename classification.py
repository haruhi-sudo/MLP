import numpy as np
import PIL.Image as Image
import os
from net import NetWork, DataLoader
from loss import CE
from activator import Sigmod, Relu

# 控制图片大小
ex_height = 320 // 30
ex_width = 243 // 30
num_of_class = 15

def centralization(images, height = 320, width = 243):
    '''
        images是图片集合（矩阵集合）
        d是图片拉伸后的维度
        输出中心化后的矩阵central_vectors
    '''
    n = len(images) # n是图片个数
    d = height * width
    central_vectors = np.zeros([d, n])
    for i in range(n):
        images[i] = images[i].flatten()
        central_vectors[:, i] = images[i]

    row_mean = np.mean(central_vectors, axis=1).reshape(d, 1)
    central_vectors = central_vectors - row_mean
    return central_vectors


def load_images(path, height = 320, width = 243, batch_size = 1):
    files = os.listdir(path)
    ims = []
    labels = []
    for filename in files:

        im = Image.open(os.path.join(path, filename)).\
            resize((height, width), Image.ANTIALIAS).convert("L")
        
        ims.append(np.asarray(im, dtype=np.uint8))
        # one hot编码
        one_hot = np.zeros([num_of_class, 1])
        one_hot[int(filename[7:9]) - 1][0] = 1
        labels.append(one_hot) # 文件名的7到9位是标签
    
    res = []
    ims = centralization(ims, height, width)
    
    for i in range(ims.shape[1]):
        res.append(ims[:, i].reshape(height * width, 1))
    dataLoader = DataLoader(batch_size=batch_size, dataset=res, labelset=labels)
    return dataLoader.batch_data, dataLoader.batch_label



def calc_loss(data, label, net):
    loss = 0
    for i in range(len(data)):
        loss += net.evaluate(val_data=data[i], val_label=label[i])
    return loss


def accuracy(data, label, net):
    correct = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            predict_result = net.predict(data[i][j])
            if np.argmax(label[i][j]) == np.argmax(predict_result):
                correct += 1
    
    print('预测正确个数：{}'.format(correct))
    print(' ')
    return correct / len(predict_result)


if __name__ == '__main__':
    train_path = './yalefaces/train'
    val_path = './yalefaces/test'

    ap = 0
    k = 10
    train_id = 0

    train_images, train_labels = load_images(train_path + str(train_id), height=ex_height, width=ex_width)
    val_images, val_labels = load_images(val_path + str(train_id), height=ex_height, width=ex_width)
    
    net = NetWork(layers=[ex_height * ex_width, 10, 10, num_of_class], activators=[Relu(), Relu(), Sigmod()],\
        loss=CE()) # 创建神经网络，输入层2，隐藏层10，输出层1

    epochs = 100000

    trian_loss = 0
    val_loss = 0
    val_loss_old = 1e9


    for epoch in range(1, epochs):
        for i in range(len(train_images)):
            net.train(one_batch_data=train_images[i], one_batch_label=train_labels[i], learning_rate=1e-5, momentum=0.)
        
        trian_loss += calc_loss(data=train_images, label=train_labels, net=net)
        val_loss += calc_loss(data=val_images, label=val_labels, net=net)
        
        if(epoch % 100 == 0):
            print('train:{}    val:{}'.format(str(trian_loss/100), str(val_loss/100)))
            
            if(val_loss_old <  val_loss and epoch > 10000):
                break
            accuracy(data=val_images, label=val_labels, net=net)

            net.save_model('./model_images_2')

            trian_loss = 0
            val_loss_old = val_loss
            val_loss = 0