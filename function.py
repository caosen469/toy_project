import numpy as np

def poi_cbg_transform(x):
    if not(np.isnan(x)):
        return str(int(x))
    else:
        return ""


# 查看adjacency_flow的尺寸
def func1():
    row = cbg_adjacency.shape[0]
    for i in range(row):
        a = cbg_adjacency[i, :]
        non_zero = len(np.where(a!=0)[0])
        if non_zero == 0:
            print(i)

# 查看连通子图的节点数目
[len(c) for c in sorted(nx.connected_components(queen_graph), key=len, reverse=True)]

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def func2():
    row, col = adjacency_flow.shape
    for i in range(row):
        for j in range(col):
            if adjacency_flow[i,j] !=0:
                print(adjacency_flow[i,j])