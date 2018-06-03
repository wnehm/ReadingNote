import numpy as np
import time
from mnist import MNIST
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle 
from knn import knn, decision
from Process import *


t1 = time.time()
mndata = MNIST('E:\\dataset\mnist')
imdata, imlab = mndata.load_training()
tsdata, tslab = mndata.load_testing()

imdata_ar = np.array(imdata)
imlab_ar = np.array(imlab)
tsdata_ar = np.array(tsdata)

k = 7
p = 2
is_right = []
results = []
max_steps = 100
process_bar = ShowProcess(max_steps) # 1.在循环前定义类的实体， max_steps是总的步数 
for i in range(10):
    dist_list, point_list, labels_list = knn(tsdata_ar[i], imdata_ar, k, p, imlab_ar)
    reslab = decision(labels_list)
    results.append(reslab)
    if reslab == tslab[i]:
        is_right.append(1)
    else:
        is_right.append(0)
    process_bar.show_process()      # 2.显示当前进度

'''
f = open('results_knn.txt', 'w')
f.write(str(results))
f.close

f = open('is_right_knn.txt', 'w')
f.write(str(is_right))
f.close
'''

error = is_right.count(0) / len(is_right)
process_bar.close('done')            # 3.处理结束后显示消息  
t2 = time.time()
print('{0}近邻下错误率为：{1}'.format(k, error))
print('运行时间为:', t2-t1)

'''
data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2],
    [5, 6], [3, 6], [7, 8], [3, 4], [1, 1], [3, 4.3]]
data = np.array(data)
x = np.array([3, 5])
dist_list, point_list, lab_list = knn(x, data, 3, 2)
dist_min = min(dist_list)
dist_max = max(dist_list)
print('与样本点的距离分别是:', dist_list)
print('最近邻点分别是:', point_list)
print('最小距离是:', dist_min)

fig = plt.figure()
plt.plot(data[:, 0], data[:, 1], 'o')
plt.plot(x[0], x[1],'ro')
ax = fig.add_subplot(111)
cir1 = Circle((x[0], x[1]), dist_max, alpha = 0.4)
ax.add_patch(cir1)
plt.show()
'''