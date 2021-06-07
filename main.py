import numpy as np
from FCN import NetWork, DataLoader

train_data = []
train_label = []
with open('sincos_train.txt','r') as f:
    for line in f.readlines():
        tmp = line[:-1].split(' ')
        train_data.append(np.array([[float(tmp[0])],[float(tmp[1])]]))
        train_label.append(np.array([[float(tmp[2])]]))

dataLoader = DataLoader(batch_size=100, dataset=train_data, labelset=train_label)
train_data = dataLoader.batch_data
train_label = dataLoader.batch_label

net = NetWork([2, 30, 1])

# for epoch in range(100000):
#     loss = 0
#     for i in range(len(train_data)):
#         net.train(one_batch_data=train_data[i], one_batch_label=train_label[i], learning_rate=0.5)
#         for j in range(len(train_data[i])):
#             predict_res = net.predict(train_data[i][j])
#             loss = loss + np.sum(predict_res - train_label[i][j])
#     print(loss)
epochs = 10000000
learning_rate = 1 
loss = 0

for epoch in range(epochs):
    for i in range(len(train_data)):
        net.train(one_batch_data=train_data[i], one_batch_label=train_label[i], learning_rate=learning_rate/np.sqrt(epoch+1))
    
    for i in range(len(train_data)):
        loss = loss + net.evaluate(train_data=train_data[i], train_label=train_label[i])

    # if(epoch % 100 == 0):
    #     print(loss/100)
    #     if(loss/100 < 0.001):
    #         break
    #     loss = 0
    print(loss/10)
    loss = 0



with open('predict_res.txt', 'w') as f:
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            predict_res = net.predict(train_data[i][j])
            f.write(str(predict_res) + '\n')