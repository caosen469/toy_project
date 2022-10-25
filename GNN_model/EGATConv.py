# 构建一个2层的GNN模型
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.data import CiteseerGraphDataset
import numpy as np
import dgl
import networkx as nx
import glob
import matplotlib.pyplot as plt
import random
import os
import dgl.function as fn
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = "cuda:0"

# 自己写一个rankloss
class RankLoss(nn.Module):
    """计算rankloss"""
    def __init__(self):
        ...

    # pred 是预测的值，gt是ground truth
    def foward(slef, pred, gt):
        
        # 把prediction中每个元素转换成第几大
        a = pred[train_mask]

# in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads

class EGAT(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hid_node_feats, hid_edge_feats, out_node_feats, out_edge_feats, num_heads=3):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.EGATConv(in_node_feats, in_edge_feats, hid_node_feats, hid_edge_feats, num_heads)
        self.conv2 = dglnn.EGATConv(hid_node_feats, hid_edge_feats, out_node_feats, out_edge_feats, 1)


    def forward(self, graph, node_feats, edge_feats):
        # 输入是节点的特征
        new_node_feats, new_edge_feats = self.conv1(graph, node_feats.float(), edge_feats.float())
        new_node_feats = torch.mean(new_node_feats, dim=1)
        new_edge_feats = torch.mean(new_edge_feats, dim=1)
        new_node_feats, new_edge_feats = self.conv2(graph, new_node_feats, new_edge_feats)
        return new_node_feats
    
# 评估模型
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(10, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_u = torch.squeeze(h_u)
        h_v = edges.dst['h']
        h_v = torch.squeeze(h_v)

        node_feature = torch.cat([h_u, h_v], 1)
        score = self.W(node_feature)
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

## 建立模型
class Model(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hid_node_feats, hid_edge_feats, out_node_feats, out_edge_feats, num_heads=3):
        super().__init__()
        self.egat = EGAT(in_node_feats, in_edge_feats, hid_node_feats, hid_edge_feats, out_node_feats, out_edge_feats)
        self.pred = MLPPredictor(out_node_feats, 1)
    def forward(self, g, node_feats, edge_feats):
        h = self.egat(g, node_feats, edge_feats)
        return self.pred(g, h)
################################################################################################
################################################################################################
data_save_path = r"D:\Project\GNN_resilience\data\training_data"
graph_dataset = glob.glob(r"D:\Project\GNN_resilience\data\training_data\*.gpickle")

# 开始遍历图进行训练
random.shuffle(graph_dataset)
training_set = graph_dataset[:int(len(graph_dataset)*2/3)]
test_set = graph_dataset[int(len(graph_dataset)*2/3):]
losses =[]
# 遍历训练图
# for training_sample in training_set:
for training_sample in training_set:
    G = nx.read_gpickle(training_sample)
    G = nx.Graph.to_directed(G)
    graph = dgl.from_networkx(G, node_attrs = ["population", "latitude", "longitude"], edge_attrs=["weight", "mobility_flow", "rank_label", "value_label"], device=device)


    ## 给DGL的图创建feature
    node_features = torch.concat((graph.ndata["population"].reshape((-1,1)),graph.ndata["latitude"].reshape((-1,1)),graph.ndata["longitude"].reshape((-1,1))),1)

    graph.ndata['feature'] = node_features.to(device)

    edge_features = torch.concat((graph.edata["weight"].reshape((-1,1)), graph.edata["mobility_flow"].reshape((-1,1))),1)
    graph.edata["feature"] = edge_features.to(device)

    graph.edata['label'] = graph.edata["value_label"].to(device)

    graph.edata['train_mask'] = torch.zeros(graph.edata["value_label"].shape[0], dtype=torch.bool, device=device).bernoulli(0.6)


    node_features = graph.ndata['feature']
    edge_features = graph.edata["feature"] 
    edge_label = graph.edata['label']
    train_mask = graph.edata['train_mask']
    model = Model(3, 2, 10, 10, 5, 5, 3).cuda()
    opt = torch.optim.Adam(model.parameters())
    
    criterion = nn.MSELoss(reduction="mean")
    for epoch in range(800):
        pred = model(graph, node_features, edge_features)
        # loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
        loss = torch.sqrt(criterion(pred[train_mask].to(torch.float32), edge_label[train_mask].to(torch.float32)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(loss.item())
    print("#################################################")

# plt.plot(losses)
# plt.xlabel("training step")
# plt.ylabel("RMSE loss")
# plt.show()

# # 测试一下model
# test_graph = test_set[0]
# G = nx.read_gpickle(training_sample)
# G = nx.Graph.to_directed(G)
# graph = dgl.from_networkx(G, node_attrs = ["population", "latitude", "longitude"], edge_attrs=["weight", "mobility_flow", "rank_label", "value_label"], device=device)

# ## 给DGL的图创建feature
# node_features = torch.concat((graph.ndata["population"].reshape((-1,1)),graph.ndata["latitude"].reshape((-1,1)),graph.ndata["longitude"].reshape((-1,1))),1)

# graph.ndata['feature'] = node_features.to(device)
# edge_features = torch.concat((graph.edata["weight"].reshape((-1,1)), graph.edata["mobility_flow"].reshape((-1,1))),1)
# graph.edata["feature"] = edge_features.to(device)
# graph.edata['label'] = graph.edata["value_label"].to(device)
# node_features = graph.ndata['feature']
# edge_label = graph.edata['label']
# pred = model(graph, node_features)

# pred_result = torch.argsort(pred, dim=0)
# gt_result = torch.argsort(graph.edata['label'],dim=0)
# gt_result = gt_result.reshape((-1,1))
# c= torch.cat((pred_result, gt_result), dim=1)