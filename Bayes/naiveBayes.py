import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt 
from Process import *
import time

# 计时器，计算运行时间
t1 = time.time() 

# 导入mnist数据集
mndata = MNIST('E:\\dataset\mnist')
imdata, imlab = mndata.load_training()
tsdata, tslab = mndata.load_testing()

# 转化为np.array()格式
imdata_ar = np.array(imdata)
imlab_ar = np.array(imlab)
tsdata_ar = np.array(tsdata)

# 对样本二值化
imdata2=[]
for i in range(len(imdata)):
    imdata2.append([])
    for j in imdata[i]:
        if j > 50:
            imdata2[i].append(1)
        else:
            imdata2[i].append(0)

# 将二值化后的样本转化为np.array()格式
imdata2_ar = np.array(imdata2)


# 统计各个数字的样本数量
N = len(imdata) 
count = []
for i in range(10):
    count.append(imlab.count(i))
count = list(map(int, count))

# 将样本按 0-9 排序,并转化为list结构
imdata2_list = imdata2_ar[imlab_ar.argsort()].tolist()
imlab_list = imlab_ar[imlab_ar.argsort()].tolist()

split_list = [] # 分割列表
imdata2_split = [] # 分割后的样本
split_list.append(count[0]) 
imdata2_split.append([imdata2_list[:split_list[0]]])
for i in range(1,10):
    split_list.append(split_list[-1] + count[i]) # 分割列表split_list[i]=sum(count[:i])
    imdata2_split.append([imdata2_list[split_list[i-1]:split_list[i]]]) # 分割后的样本imdata_split[i][0]为数字i的所有样本集合

# 统计所有样本中，各个维数的值为0的数量和1的数量，并求条件概率
# 由于数据维数太高（28*28）直接计算条件概率，在连乘后会出现概率都为0，
# 因此对条件概率扩大10000倍，并取整
Px0y, Px1y = [], []
for i in range(10):
    Px0y.append([])
    Px1y.append([])
    for j in range(28*28):
        count0 = list(map(lambda x: x[j], imdata2_split[i][0])).count(0)
        count1 = count[i]-count0
        Px0y[i].append(int(10000*(count0+1) / (count[i]+2)))
        Px1y[i].append(int(10000*(count1+1) / (count[i]+2)))

# 测试样本二值化
tsdata2 = []
for i in range(len(tsdata)):
    tsdata2.append([])
    for j in tsdata[i]:
        if j > 50:
            tsdata2[i].append(1)
        else:
            tsdata2[i].append(0)

# 进度条初始化
max_steps = len(tsdata2)
process_bar = ShowProcess(max_steps)

# 贝叶斯判别，并统计正确率
right = []
for k in range(len(tsdata2)):
    X = tsdata[k]
    result = []
    for i in range(10):
        mul = 1
        for j in range(len(X)):
            if X[j] == 0:
                mul = mul * Px0y[i][j]
            else:
                mul = mul * Px1y[i][j]
        result.append(mul*count[i])
        
    if result.index(max(result)) == tslab[k]:
        right.append(1)
    else:
        right.append(0)
    
    process_bar.show_process()     
process_bar.close('done') 

print('正确率为：',right.count(1) / len(right))
t2 = time.time()
