import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np

X, y = [], []
filename = r'ex2/ex2data1.txt'
with open(filename) as f:
    for line in f:
        tmp = list(map(float, line.strip('\n').split(',')))
        X.append([tmp[0], tmp[1]])
        y.append([tmp[2]])
X, y = np.array(X), np.array(y)
m, n = X.shape # m样本数，n特征数

# 逻辑回归模型
def h(X, theta):
    # X 是样本特征 m*(n+1) np.array
    # theta 是权值向量 
    return 1 / (1 + np.exp(-X.dot(theta))) # numpy 中矩阵乘法为 dot()

# 标准化
def featureNormalize(X):
    X_mean = X.mean(0) # 均值
    X_std = X.std(0) # 标准差
    X_norm = (X - X_mean) / X_std
    return X_mean, X_std, X_norm

# 代价函数
def computeCost(theta, X, y):
    # y 样本输出
    J = -1 / m * (np.dot(y.T, np.log(h(X, theta))) + np.dot((1 - y).T, np.log(1 - h(X, theta))))
    return J

# 梯度
def gradFunc(theta, X, y):
    grad = 1 / m * np.dot(X.T, (h(X, theta) - y))
    return grad


# 梯度下降
def gradientDescent(X, y, theta, alpha, num_iters):
    # alpha 更新率
    # num_iters 迭代次数
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - alpha * gradFunc(theta, X, y)
        J_history[i] = computeCost(theta, X, y) #保存每一步的代价函数
    return theta, J_history

def prediction(x, theta):
    if type(x) == 'list':
        x = np.array(x)
    return 1 / (1 + x.dot(theta))

theta = np.zeros((n + 1, 1))
X_mean, X_std, X_norm = featureNormalize(X)
X_norm = np.column_stack((np.ones((m, 1)), X_norm))
alpha =0.1
num_iters = 500
# theta, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)
# plt.plot(range(num_iters), J_history)
Result = op.minimize(fun = computeCost, x0 = theta, args = (X_norm, y), jac = gradFunc)
theta = Result.x

index0 = np.argwhere(y == 0)[:,0]
index1 = np.argwhere(y == 1)[:,0]

t1 = np.linspace(30, 100, 100)
t1_norm = (t1 -X_mean[0]) / X_std[0]
t2 = np.linspace(30, 100, 100)
t2_norm = (t2 -X_mean[1]) / X_std[1]

pre = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        x = np.column_stack((np.ones((1, 1)), t1_norm[i], t2_norm[j]))
        pre[i][j] = prediction(x, theta)

fig = plt.figure()
ax = plt.axes()
ax.contour(t1_norm, t2_norm, pre, levels = 0)
ax.plot(X_norm[index0, 1], X_norm[index0, 2] , 'o')
ax.plot(X_norm[index1, 1], X_norm[index1, 2] , 'x')

ax.set_xlabel('X1')
ax.set_ylabel('X2')