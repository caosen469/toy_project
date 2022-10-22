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

device = "cuda:0"

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
    
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

import dgl.function as fn
class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
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
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, 1)
    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)
################################################################################################
################################################################################################
data_save_path = r"D:\Project\GNN_resilience\data\training_data"
training_data_list = glob.glob(r"D:\Project\GNN_resilience\data\training_data\*.gpickle")
# for each in training_data_list:
file_name = training_data_list[0]
G = nx.read_gpickle(file_name)
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
edge_label = graph.edata['label']
train_mask = graph.edata['train_mask']
model = Model(3, 20, 1).cuda()
opt = torch.optim.Adam(model.parameters())
losses =[]
for epoch in range(10000):
    pred = model(graph, node_features)
    loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    print(loss.item())

plt.plot(losses)
plt.xlabel("training step")
plt.ylabel("MSE loss")