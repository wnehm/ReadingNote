import time
from mnist import MNIST
import numpy as np
from knn_kd import kd_build, nn_search
from Process import *

t1 = time.time()

mndata = MNIST('E:\\dataset\mnist')
imdata, imlab = mndata.load_training()
tsdata, tslab = mndata.load_testing()

imdata_ar = np.array(imdata)
imlab_ar = np.array(imlab)
tsdata_ar = np.array(tsdata)

kdtree = kd_build(imdata_ar)

is_right = []
results = []
max_steps = 100
process_bar = ShowProcess(max_steps) # 1.在循环前定义类的实体， max_steps是总的步数 
for i in range(100):
    point = tsdata_ar[i]
    nn = nn_search(point, kdtree)
    res = imlab[imdata.index(nn[1].tolist())]
    results.append(res)
    if res == tslab[i]:
        is_right.append(1)
    else:
        is_right.append(0)
    process_bar.show_process()      # 2.显示当前进度

error = is_right.count(0) / len(is_right)
process_bar.close('done')            # 3.处理结束后显示消息  
t2 = time.time()
print('最近邻下错误率为：',error)
print('运行时间为:', t2-t1)

'''
f = open('results_nn0530.txt', 'w')
f.write(str(results))
f.close

f = open('is_right_nn0530.txt', 'w')
f.write(str(is_right))
f.close
'''