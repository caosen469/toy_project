import pandas as pd
import numpy as np
import networkx as nx
import torch
import re
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

pattern_df = pd.read_csv(r"D:\Project\GNN_resilience\result\pattern.csv")
cbg_df = pd.read_csv(r"D:\Project\GNN_resilience\result\cbg.csv")
adjacency_flow = np.loadtxt(r"D:\Project\GNN_resilience\result\adjacency_flow.csv",delimiter=',')
adjacency_distance = np.loadtxt(r"D:\Project\GNN_resilience\result\adjacency_distance.csv",delimiter=',')

# 加载graph
queen_graph = nx.read_gpickle(r"D:\Project\GNN_resilience\data\big_graph\big_graph.gpickle")

# 计算degree属性
x_degree = torch.tensor(np.array(list(dict(queen_graph.degree).values()))).reshape((-1,1))
# 计算位置、人口等属性
x_feature_list = []
for idx in zip(list(queen_graph.nodes)): 
    idx = idx[0]
    # 获取对应行数据
    row = cbg_df[cbg_df.index==idx]
    row_population = row["POP2012"].values[0]
    # 获取经纬度
    row_xy = re.findall("\d+\.?\d*", row["centroid"].values[0])
    row_latitude = float(row_xy[1])
    row_longitude = float(row_xy[0])
    value_list = {"population": row_population, "latitude": row_latitude, "longitude": row_longitude}
    value_list = [row_population, row_latitude, row_longitude]
    x_feature_list.append(value_list)

x_feature_tensor = torch.tensor(np.array(x_feature_list))

del x_feature_list

# 读取网络结构
data = dgl.from_networkx(queen_graph)
# 加入节点数据
data.ndata['feature'] = torch.cat((x_feature_tensor, x_degree), dim=1)
# 加入连接数据