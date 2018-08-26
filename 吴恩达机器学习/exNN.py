# -*- coding:utf-8 -*-
import numpy as np

class NeuralNetwork:
    def __init__(self, x, y, theta1, theta2, eta):
        self.x = x
        self.y = y
        self.theta1 = theta1
        self.theta2 = theta2
        self.eta = eta
        self.a = self.hiddenLayer(self.theta1)
        self.h = self.output(self.theta1, self.theta2)
        self.delta_h = self.h-self.y
        self.delta_a = self.theta2[1:] @ self.delta_h * self.a[1:] * (1-self.a[1:])
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def hiddenLayer(self, theta1):
        a = self.sigmoid(theta1.T @ self.x)
        a = np.insert(a, 0, values = 1, axis = 0)
        return a
    
    def output(self, theta1, theta2):
        h = self.sigmoid(theta2.T @ self.hiddenLayer(theta1))
        return h
    
    def costFunJ(self, theta1, theta2):
        h = self.output(theta1, theta2)
        return -(self.y.T @ np.log(h) + (1-self.y).T @ np.log(1-h))
    
    def updateTheta(self):
        self.theta1[1:] = self.theta1[1:] - self.eta * self.delta_a.T * self.x[1:]
        self.theta2[1:] = self.theta2[1:] - self.eta * self.delta_h.T * self.a[1:]
        return
    
    @property
    def gradChick(self):
        dVec1 = self.delta_a.T * self.x[1:]
        dVec1 = dVec1.reshape(-1,1)
        dVec2 = self.delta_h.T * self.a[1:]
        dVec2 = dVec2.reshape(-1,1)
        dVec = np.concatenate([dVec1, dVec2])
        epsilon = 0.0001
        grad1 = np.zeros(theta1[1:].shape)
        grad2 = np.zeros(theta2[1:].shape)
        for i in range(2):
            for j in range(2):
                thetaPlus1 = self.theta1.copy()
                thetaPlus1[i+1,j] =  thetaPlus1[i+1,j] + epsilon
                thetaMinus1 = self.theta1.copy()
                thetaMinus1[i+1,j] =  thetaMinus1[i+1,j] - epsilon         
                grad1[i][j] = (self.costFunJ(thetaPlus1,theta2)-self.costFunJ(thetaMinus1, theta2))/(2*epsilon)
                thetaPlus2 = self.theta2.copy()
                thetaPlus2[i+1,j] =  thetaPlus2[i+1,j] + epsilon
                thetaMinus2 = self.theta2.copy()
                thetaMinus2[i+1,j] =  thetaMinus2[i+1,j] - epsilon         
                grad2[i][j] = (self.costFunJ(theta1,thetaPlus2)-self.costFunJ(theta1,thetaMinus2))/(2*epsilon)
        grad1 = grad1.reshape(-1,1)
        grad2 = grad2.reshape(-1,1)
        grad = np.concatenate([grad1, grad2])
        if np.sum(np.abs(dVec - grad)) <= 0.0001:
            return True
        else:
            return False
        
if __name__=='__main__':
    x = np.array([1,0.05,0.1]).reshape(-1,1) # 列向量
    y = np.array([0.01, 0.99]).reshape(-1,1) # 列向量
    theta1 = np.array([[0.35,0.15,0.2],[0.35,0.25,0.3]]).T
    theta2 = np.array([[0.6,0.4,0.45],[0.6,0.5,0.55]]).T
    eta = 0.5
    NN = NeuralNetwork(x, y, theta1, theta2, eta)
    if NN.gradChick:
        NN.updateTheta()
        print('theta1=', NN.theta1)
        print('theta2=', NN.theta2)
    else:
        print('请检查反向传播的实现是否正确')