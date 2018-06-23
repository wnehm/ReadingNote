import numpy as np 

class Perceptron:
    def __init__(self, points, lables):
        self.points = points
        self.lables = lables
        self.num = len(points)
        
        a = np.zeros((self.num, 1), dtype=int)
        eta = 1
        b = 0
        while not (self.__lab(a, b) == self.lables).all():
            for i in range(self.num):
                if self.lables[i] * (np.sum(a * self.lables * self.points * self.points[i]) + b) <= 0:
                    a[i] = a[i] + eta
                    b = b + eta * self.lables[i]
        self.b = b            
        self.w =np.sum(a * self.lables * self.points, axis=0)
        
    def __lab(self, a, b):
        return np.sign(np.sum(np.sum(a * self.lables * self.points, axis=0) * self.points, axis=1) + b).reshape(self.num,1)
        
    def classify(self, point):
        return np.sign(np.sum(self.w * point) + self.b)