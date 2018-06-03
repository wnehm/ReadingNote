import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from mpl_toolkits.mplot3d import Axes3D, axes3d

#kd树节点
class Node:
    '''
    树的节点对象
    属性：
        point: 当前节点值，
        lchild: 左子节点，
        rchild: 右子节点，
        dim: 当前节点分割维度
    '''
    def __init__(self, point, lchild, rchild, dim):
        self.point = point
        self.lchild = lchild
        self.rchild = rchild
        self.dim = dim

#不再使用，求方差可采用np.var()
def get_variance(data):
    '''
    功能：求一组数据每一列的方差
    输入：np.array格式的数据组
    输出：1 * n的数组，n 为原数据的列数
    '''
    if len(data) == 0:
        return 0
    mean_data = np.mean(data, axis=0)
    variance_data = np.sum((data - mean_data) ** 2, axis=0) / len(data)
    return variance_data

def kd_build(data):
    '''
    功能：建立kd树
    输入：包含所有样本点的数据组，要求为np.array格式，
        每行对应一个样本，每列对应一个维度
    输出：Node对象
    '''
    if not len(data):
        point = None
        lchild = None
        rchild = None
        split_dim = None
    elif len(data) == 1:
        point = data[0]
        lchild = None
        rchild = None
        split_dim = None
    elif data.size == len(data):
        point = data
        lchild = None
        rchild = None
        split_dim = None
    else:
        #variance_data = get_variance(data)  #求各个维度的方差
        variance_data = np.var(data, axis=0)
        split_dim = np.argsort(variance_data)[-1]  #取方差最大的维度作为分割维度
        data = data[data[:,split_dim].argsort()]  #按分割维度对数据排序
        point = data[len(data) // 2]  #取排序后的中位数作为当前节点
        ldata = data[:len(data) // 2]  #当前节点左边为左区域
        rdata = data[len(data) // 2 + 1:]  #当前节点右边为右区域
        if ldata.size == 0:
            lchild = None
        else: 
            lchild = kd_build(ldata)
        if rdata.size == 0:
            rchild = None
        else:
            rchild = kd_build(rdata)
    return Node(point, lchild, rchild, split_dim)


def get_distance(point1, point2, p=2):
    '''
    功能：求两点之间的Lp距离，Lp距离定义如下：
    Lp(x1, x2)=sum(|x1-x2|^p)^(1/p), for p>=1
    '''
    if p < 1:
        raise ValueError('p >= 1')
    if p == float('inf'):
        return np.amax(np.abs(point1 - point2))
    elif p == 1:
        return np.sum(np.abs(point1 - point2))
    else:
        return np.power(np.sum(np.abs(point1 - point2) ** p), 1/p)


def find_leaf(point, kdtree):
    '''
    功能：搜索point所在区域的叶节点，同时保存搜索路径;
    输入：point: 目标点，为 1*n 的 np.array; 
        kdtree: kd树;
        route: 递归的记录搜索路径，默认为空列表，0为左，1为右;
    输出：[叶节点, 路径];
    例子：
        point = np.array([3, 4.5])
        data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
        kdtree = kd_build(data)  
        [leaf, route] = find_leaf(point, kdtree)
        [leaf.point, route]
        [array[(4,7)], [0, 1]] 
    '''
    route = []
    while kdtree.lchild and kdtree.rchild:
        if point[kdtree.dim] <= kdtree.point[kdtree.dim]:
            kdtree = kdtree.lchild
            route.append(0)
        else:
            kdtree = kdtree.rchild
            route.append(1)
    else:
        return kdtree, route


def find_node(kdtree, route):
    '''
    功能：根据路径找节点
    输入：kdtree: kd树;
        route: 路径, 路径由0, 1组成的列表，0代表左1代表右;
    输出：路径的终点;
    例子：
        point = np.array([3, 4.5])
        data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
        kdtree = kd_build(data)
        node = find_node(kdtree, [0, 1])
        node.point
        array([4,7])
    '''
    for i in range(len(route)):
        if route[i] == 0:
            try:
                kdtree = kdtree.lchild
            except AttributeError:
                print(route)
        else:
            kdtree = kdtree.rchild
    return kdtree


def nn_search(point, kdtree, p=2):
    '''
    功能：求点在 kd 树中 最近邻点;
    输入：point: 目标点为 1*n 的 np.array; 
        kdtree: kd树对象;
        p: Lp 距离度量参数, 默认为2, 即欧式距离;
    输出：最近距离，最近邻点;
    '''
    leaf, route = find_leaf(point, kdtree)
    nearest_distance = get_distance(point, leaf.point, p)
    nn = leaf.point
    for i in range(len(route)):
        parent = find_node(kdtree, route[:-(i+1)])
        if route[-(i+1)] and parent.lchild:
            brother = parent.lchild
        elif not route[-(i+1)] and parent.rchild:
            brother = parent.rchild
        else:
            brother = None
        current_distance = get_distance(point, parent.point, p)
        if current_distance < nearest_distance:
            nearest_distance = current_distance
            nn = parent.point
        if brother:
            dist = np.abs(parent.point[parent.dim] - point[parent.dim])
            if nearest_distance > dist:  #交割
                current_distance, current_point = nn_search(point, brother, p)
                if current_distance < nearest_distance:
                    nearest_distance = current_distance
                    nn = current_point
    return nearest_distance, nn

def kd_search(point, kdtree, k=1, p=2):
    '''
    功能：求点在 kd 树中 k 个近邻点;
    输入：point: 目标点为 1*n 的 np.array; 
        kdtree: kd树对象;
        k: 近邻点数 k, 默认为1, 即最近邻;
        p: Lp 距离度量参数, 默认为2, 即欧式距离;
    输出：[[距离1， 点1], [距离2， 点2],...,[距离k， 点k]];
    '''
    leaf, route = find_leaf(point, kdtree)
    nearest_distance = get_distance(point, leaf.point, p)
    point_list = [] #建立空堆，用于保存距离及k近邻点，python只有小顶堆，采用取负的形式实现“大顶堆”
    heapq.heappush(point_list, [-nearest_distance, list(leaf.point)])
    for i in range(len(route)): #按照搜索路径由下向上迭代寻找近邻点
        parent = find_node(kdtree, route[:-(i+1)])
        if route[-(i+1)] and parent.lchild: #若当前节点为其父节点的右节点，且其父节点的左子节点存在，则兄弟节点为其父节点的左子节点
            brother = parent.lchild
        elif not route[-(i+1)] and parent.rchild: #若当前节点为其父节点的左节点，且其父节点的右子节点存在，则兄弟节点为其父节点的右子节点
            brother = parent.rchild
        else: #否则当前节点无兄弟节点
            brother = None
        current_distance = get_distance(point, parent.point, p)
        if len(point_list) < k: #成对的取前k(k为偶数时是k+1)个父节点和兄弟节点存入堆中
            heapq.heappush(point_list, [-current_distance, list(parent.point)])
            if brother:
                current_distance = get_distance(point, brother.point, p)
                heapq.heappush(point_list, [-current_distance, list(brother.point)])
        else:
            heapq.heappushpop(point_list, [-current_distance, list(parent.point)])
            if brother:
                dist = np.abs(parent.point[parent.dim] - point[parent.dim]) #当前点到其父节点的距离
                if -point_list[0][0] > dist: #堆中最大距离大于当前点到其父节点的距离（球体相交）
                    temp_list = kd_search(point, brother, k, p)  #递归地在子空间进行近邻搜索
                    for i in range(len(temp_list)): #将堆中的距离变为正
                        temp_list[i][0] = -temp_list[i][0]
                    for i in temp_list:
                        heapq.heappushpop(point_list, i)  #更新point_list
    for i in range(len(point_list)): #将堆中的距离变为正
        point_list[i][0] = -point_list[i][0]
    return heapq.nsmallest(min(k, len(point_list)), point_list)

def decision(labels_list):
    count = np.bincount(labels_list)
    return np.argmax(count)

'''
from mpl_toolkits.mplot3d import Axes3D
data3 = [[2, 3, 2], [5, 4, 5], [9, 6, 1], [4, 7, 3], [8, 1, 6], [7, 2, 5],
    [5, 6, 4], [3, 6, 8], [7, 8, 2], [3, 4, 5], [1, 1, 6], [3, 4, 9]]
data3 = np.array(data3)
point3 = np.array([3, 5, 4.5])
kdtree3 = kd_build(data3)
nd3, nn3 = nn_search(point3, kdtree3, 2)
print('最近距离', nd3)
print('最近邻点', nn3)

x, y, z = data3[:, 0], data3[:, 1], data3[:,2]
ax3 = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
ax3.scatter(x, y, z, c='b')  # 绘制数据点
ax3.scatter(point3[0], point3[1], point3[2], c='r')  # 绘制数据点

# center and radius
radius = point3
radius = nd3
# data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = radius * np.outer(np.cos(u), np.sin(v)) + point3[0]
y = radius * np.outer(np.sin(u), np.sin(v)) + point3[1]
z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + point3[2]

ax3.plot_wireframe(x, y, z,  rstride=12, cstride=12, color='y')

ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_zlim(0, 10)
plt.show()

'''