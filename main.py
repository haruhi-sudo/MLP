import numpy as np
from FCN import NetWork, DataLoader

train_data = []
train_label = []
with open('sincos.txt','r') as f:
    for line in f.readlines():
        tmp = line[:-1].split(' ')
        train_data.append(np.array([[float(tmp[0])],[float(tmp[1])]]))
        train_label.append(np.array([float(tmp[2])]))
        print(' ')

dataLoader = DataLoader(batch_size=10, dataset=train_data, labelset=train_label)
train_data = dataLoader.batch_data
train_label = dataLoader.batch_label

net = NetWork([2, 300, 1])

for epoch in range(100):
    for i in range(len(train_data)):
        net.train(one_batch_data=train_data[i], one_batch_label=train_label[i], learning_rate=0.1 ,epoch=1)


with open('predict_res.txt', 'w') as f:
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            predict_res = net.predict(train_data[i][j])
            f.write(str(predict_res) + '\n')