import numpy as np
import networkx as nx
import ast
import matplotlib.pyplot as plt


def all_graph_distance(sub_mobility_flow, sub_queen, sub_distance, sub_graph_nodes):
    row, col = sub_mobility_flow.shape
    graph_total_distance = 0
    for i in range(row):
        for j in range(i+1, col):
            # 每个连接上人的flow
            if sub_mobility_flow[i,j]!=0:
                flow_volume = sub_mobility_flow[i,j]
                # 获得node编号，1590这种
                start_node = list(sub_queen.nodes)[i]
                end_node = list(sub_queen.nodes)[j]
                flow_path = nx.shortest_path(sub_queen, start_node, end_node)
            
                node_index =[sub_graph_nodes.index(int(each)) for each in flow_path] 
                
                shortest_path_distance = [sub_distance[node_index[idx], node_index[idx+1]] for idx in range(len(node_index)-1)]

                graph_total_distance += sum(shortest_path_distance)
    return graph_total_distance  


sub_queen = nx.read_gexf(r"D:\Project\GNN_resilience\code\scratch\debug_flow_distance\sub_queen.gexf")
sub_queen_copy_edges = open(r"D:\Project\GNN_resilience\code\scratch\debug_flow_distance\nodes.txt", 'r').readlines()[0]
sub_queen_copy_edges = eval(sub_queen_copy_edges)
# sub_queen = open(r"D:\Project\GNN_resilience\code\scratch\debug_flow_distance\nodes.txt", 'r').readlines()[0]
# sub_queen = eval(sub_queen)
sub_mobility_flow = np.load(r"D:\Project\GNN_resilience\code\scratch\debug_flow_distance\sub_mobility_flow.npy")
sub_graph_nodes = open(r"D:\Project\GNN_resilience\code\scratch\debug_flow_distance\sub_graph_nodes.txt", 'r').readlines()[0]
sub_graph_nodes = eval(sub_graph_nodes)
idx = 22

sub_distance = np.load(r"D:\Project\GNN_resilience\code\scratch\debug_flow_distance\sub_distance.npy")


post_attack_distance = []
sub_queen_copy = sub_queen.copy()

initial_distance = all_graph_distance(sub_mobility_flow, sub_queen_copy, sub_distance, sub_graph_nodes)
# 删除一条边
# sub_queen_copy_edges = list(sub_queen_copy.edges).copy()
sub_queen_copy_edges = open(r"D:\Project\GNN_resilience\code\scratch\debug_flow_distance\sub_queen_copy_edges.txt", 'r').readlines()[0]
sub_queen_copy_edges = eval(sub_queen_copy_edges)
sub_queens = []
sub_queens_adjacency = []

for idx, each in enumerate(sub_queen_copy_edges):
    if idx > 22: break
    sub_queen_copy.remove_edge(str(each[0]), str(each[1]))
    # 计算连通子图个数小于2则停止
    if nx.number_connected_components(sub_queen_copy) >1:
        break
    post_attack_distance.append(all_graph_distance(sub_mobility_flow, sub_queen_copy, sub_distance, sub_graph_nodes))
    if len(post_attack_distance)>=2:
        post_attack_distance[-1] < post_attack_distance[-2]
    sub_queens.append(sub_queen_copy.copy())
    sub_queens_adjacency.append(np.array(nx.adjacency_matrix(sub_queens[-1]).todense()))
    
post_attack_distance

# 计算所右边的左端路径

def shortest_path_dict(sub_mobility_flow, sub_queen, sub_distance, sub_graph_nodes):
    row, col = sub_mobility_flow.shape
    graph_total_distance = 0
    result_dict = {}
    for i in range(row):
        for j in range(i+1, col):
            # 每个连接上人的flow
            if sub_mobility_flow[i,j]!=0:
                flow_volume = sub_mobility_flow[i,j]
                # 获得node编号，1590这种
                start_node = list(sub_queen.nodes)[i]
                end_node = list(sub_queen.nodes)[j]
                flow_path = nx.shortest_path(sub_queen, start_node, end_node)

                result_dict[(start_node, end_node)]=flow_path

               
    return result_dict 
a = (sub_mobility_flow, sub_queens[-2], sub_distance, sub_graph_nodes)
a = (sub_mobility_flow, sub_queens[-1], sub_distance, sub_graph_nodes)