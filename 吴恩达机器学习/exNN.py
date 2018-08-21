# -*- coding:utf-8 -*-
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.array([1,0.05,0.1]).reshape(-1,1) # 列向量
y = np.array([0.01, 0.99]).reshape(-1,1) # 列向量
theta1 = np.array([[0.35,0.15,0.2],[0.35,0.25,0.3]]).T
theta2 = np.array([[0.6,0.4,0.45],[0.6,0.5,0.55]]).T

# 前向传播
# 输入层-->隐含层
z1 = np.dot(theta1.T, x)
a = sigmoid(z1)
a1, a2 = a[0], a[1]
a = np.insert(a, 0, values = 1, axis = 0)
# 隐含层--> 输出层
z2 = np.dot(theta2.T, a)
h = sigmoid(z2)
h1, h2 = h[0], h[1]

# 反向传播
# 代价函数（总误差）
J = -(np.dot(y.T, np.log(h))+np.dot((1-y).T, np.log(1-h)))
eta = 0.5
# 输出层-->隐含层
delta_h = (h-y)*h*(1-h)
theta2_tmp = theta2[1:] - eta * delta_h.T * a[1:]
# 隐含层-->输入层
delta_a = theta2[1:] @ delta_h * a[1:] * (1-a[1:])
theta1_tmp = theta1[1:] - eta * delta_a.T * x[1:]
# 最后更新权值
theta1[1:] = theta1_tmp
theta2[1:] = theta2_tmp