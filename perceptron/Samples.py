import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from scipy.spatial import ConvexHull

class Samples:
    def __init__(self, num, min_x, max_x, min_y, max_y):
        self.num = num
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.k = round(uniform(-10, 10))
        self.b = round(uniform(min_y, max_y))
        
        #x为横坐标，y为纵坐标，lab为标签，points为所有点坐标(x,y)的集合
        x, y, lab = [], [], []
        points = []
        for i in range(self.num):
            x.append(round(uniform(self.min_x, self.max_x), 2))
            y.append(round(uniform(self.min_y, self.max_y), 2))
            lab0 = self.__lab(x[-1], y[-1], self.k, self.b)
            if not lab0:
                x[-1] = x[-1] + (self.max_x - self.min_x)/self.num
                lab0 = self.__lab(x[-1], y[-1], self.k, self.b)
            lab.append(lab0)
            points.append([x[-1], y[-1]])
                
        self.x = np.array(x)
        self.y = np.array(y)
        self.points = np.array(points)
        self.lab = np.array(lab).reshape(self.num, 1)

        points1, points2=[], []
        for i in range(self.num):
            if self.lab[i] > 0:
                points1.append([x[i], y[i]])
            else:
                points2.append([x[i], y[i]])
        self.points1 = np.array(points1)
        self.points2 = np.array(points2)
        
        
    #若(x,y)在y=k*x+b上方返回1，在下方返回-1，在直线上返回0
    def __lab(self, x, y, k, b):
        if np.abs(y - k * x - b) <= 0.001:
            return 0
        return np.sign(y - k * x - b)

    def show(self):
        plt.plot(self.points1[:,0], self.points1[:,1], 'ro')
        plt.plot(self.points2[:,0], self.points2[:,1], 'o')
        #t = np.linspace(-5, 5, 10)
        #plt.plot(t, self.k * t + self.b, 'black')

        hull1 = ConvexHull(self.points1)
        hull2 = ConvexHull(self.points2)

        plt.plot(self.points1[hull1.vertices,0], self.points1[hull1.vertices,1], 'g--', lw=2)
        plt.plot(self.points2[hull2.vertices,0], self.points2[hull2.vertices,1], 'y--', lw=2)
        plt.show()