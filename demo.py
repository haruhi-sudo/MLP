import matplotlib.pyplot as plt #绘图用的模块  
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数  
import numpy as np  
from net import NetWork, DataLoader
from loss import MSE

def fun(x,y):  
    return 10*np.sin(2*x) + 10 * np.sin(2*y)  

if __name__ == '__main__':

    net = NetWork([2, 10, 10, 1], MSE()) # 构建神经网络，输入层2，隐藏层10，输出层1
    net.load_model('./model_3layer')
    
    X=np.arange(-1,1,0.1)  
    Y=np.arange(-1,1,0.1)
    X,Y=np.meshgrid(X,Y)  
    Z = np.zeros([20, 20])
    for i in range(20):
        for j in range(20):
            Z[i][j] = net.predict(np.array([[X[i][j]],[Y[i][j]]]))[0][0] * 40 - 20

    fig1=plt.figure()#创建一个绘图对象  
    ax=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)  

    Z2=fun(X,Y)#用取样点横纵坐标去求取样点Z坐标  
    plt.title("This is main title")#总标题  
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)#用取样点(x,y,z)去构建曲面  
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=plt.cm.coolwarm)#用取样点(x,y,z)去构建曲面 
    ax.set_xlabel('x label', color='r')  
    ax.set_ylabel('y label', color='g')  
    ax.set_zlabel('z label', color='b')#给三个坐标轴注明  
    plt.show()#显示模块中的所有绘图对象  