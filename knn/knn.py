import numpy as np


def distance(point, data, p):
    if p < 1:
        raise ValueError('p >= 1')
    if p == float('inf'):
        return np.amax(np.abs(data - point))
    elif p == 1:
        return np.sum(np.abs(data - point), axis=1)
    else:
        return np.power(np.sum(np.abs(data - point) ** p, axis=1), 1/p)

def knn(point, data, k, p, labels=[]):
    '''
    输入：point: 输入实例点，np.array()格式；data: 数据集，np.array()格式；
        labels：数据集data的标签，np.array()格式；k: 近邻点数量；p: Lp距离量度。
    输出：dist_list：k个最小距离；point_list：k个近邻点；labels_list：对应标签；
        均为np.array()格式。
    '''
    dist_all = distance(point, data, p)
    sort_list = np.argsort(dist_all)
    dist_list = dist_all[sort_list][:k]
    point_list = data[sort_list][:k]
    labels_list = None
    if len(labels):
        labels_list = labels[sort_list][:k]   
    return dist_list, point_list, labels_list


def decision(labels_list):
    count = np.bincount(labels_list)
    return np.argmax(count)
