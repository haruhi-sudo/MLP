import matplotlib.pyplot as plt #绘图用的模块  
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数  
import numpy as np  
from net import NetWork, DataLoader
from loss import MSE
from activator import Sigmod, Relu, Linear

def fun(x,y,a=5,b=3,c=5,d=3):  
    return a*np.sin(b*x) + c* np.cos(d*y) - 10


if __name__ == '__main__':
    _min = -10
    _range = 20


    net = NetWork(layers=[2, 10, 1], activators=[Sigmod(),Sigmod()],\
         loss=MSE()) # 创建神经网络，输入层2，隐藏层10，输出层1
    
    net.load_model('./model_5_3')
    

    X=np.arange(-1,1,0.1)  
    Y=np.arange(-1,1,0.1)
    X,Y=np.meshgrid(X,Y)
    Z2=fun(X,Y)

    Z = np.zeros([20, 20])
    
    # 由归一化的数据恢复原始数值
    for i in range(20):
        for j in range(20):
            Z[i][j] = net.predict(np.array([[X[i][j]],[Y[i][j]]]))[0][0] * _range + _min

    fig1=plt.figure()
    ax=Axes3D(fig1)

    
    plt.title("This is main title")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)#用取样点(x,y,z)去构建曲面  
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    ax.set_xlabel('x label', color='r')  
    ax.set_ylabel('y label', color='g')  
    ax.set_zlabel('z label', color='b')
    plt.show() 

