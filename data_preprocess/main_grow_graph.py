from copy import deepcopy
import geopandas as gp 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import function
# import pysal as ps
from libpysal import weights
import os
import networkx as nx
import math
import glob
import geopy.distance
import shapely
import ast
from scipy.spatial.distance import cdist
import random
import re
import csv

####################################################################################################################################

def poi_cbg_transform(x):
    if not(np.isnan(x)):
        return str(int(x))
    else:
        return ""

def get_latitude(x, poi_df):
    try:
        poi_df = poi_df[poi_df["placekey"]==x.placekey]
        latitude = poi_df.iloc[0,:]["latitude"]
        return latitude
    except Exception:
        ...


# os.environ['PROJ_LIB']=r'D:\Users\19688\Anaconda2\envs\geopands2\Lib\site-packages\osgeo\data\proj'
#%% 读取目标county的CBG数据

# cbg_df = gp.read_file(r"D:\Project\GNN_resilience\data\texas_cbg\cbg_1\Export_Output.shp") # 读取整个德克萨斯州的CBG的shapefile

## cbg_df["FIPS"] 是十二位的cbg fips code，可以用来筛选出位于Harris County的 cbg，数据格式是字符串

# # Harris County FIPS: 48201
county_FIPS = '48201'

# # 筛选出harris county的cbg
# cbg_df = cbg_df[cbg_df["FIPS"].str.startswith(county_FIPS)]
# cbg_df.reset_index(drop=True)

# # 保存目标county cbg数据
# cbg_df.to_file(r'D:\Project\GNN_resilience\data\harris_cbg\harris.shp', driver='ESRI Shapefile')

#%% 构建CBG网络

# 读取目标county cbg的shapefile
cbg_df = gp.read_file(r'D:\Project\GNN_resilience\data\harris_cbg\harris.shp')

## 将cbg中的multipolygon转化为polygon

# 读取cbg 中所有geometry的类型
set([type(each) for each in cbg_df["geometry"]])
# 查看不是polygon的行
cbg_df[cbg_df["geometry"].map(lambda x: type(x)==shapely.geometry.multipolygon.MultiPolygon)]

# 对于是multipolygon的行，只保留面积最大的那个
def multipoly(x):
    if type(x) !=shapely.geometry.multipolygon.MultiPolygon:
        return x
    else:
        a = list(x)
        area = [each.area for each in a]
        max_idx = np.argmax(area)
        return a[max_idx]

cbg_df["geometry"] = cbg_df["geometry"].map(lambda x: multipoly(x))

# 基于cbg_df["geometry"]计算出来对应polygon的质心
cbg_df["centroid"] = cbg_df["geometry"].map(lambda x: x.centroid)

## 构建Queeen 邻接矩阵
queen = weights.Queen.from_dataframe(cbg_df)
# 构建queen邻接图
queen_graph = queen.to_networkx()

# 消除孤岛
island_node = 758
connected_nodes = [68,275,475,962,1610,344,84,856]
for node in connected_nodes:
    queen_graph.add_edge(island_node, node)


centroids = np.array([[each.xy[0][0], each.xy[1][0]] for each in cbg_df["centroid"]])
##  添加边权
for e in queen_graph.edges():
    # 获取第一个cbg和第二个cbg得centroid计算距离
    cbg1 = cbg_df.iloc[e[0]]["centroid"]
    cbg2 = cbg_df.iloc[e[1]]["centroid"]
    queen_graph[e[0]][e[1]]["weight"] = geopy.distance.geodesic((cbg1.xy[1][0], cbg1.xy[0][0]), (cbg2.xy[1][0], cbg2.xy[0][0])).km

cbg_adjacency = np.array(nx.adjacency_matrix(queen_graph).todense())

#  可视化
positions = dict(zip(queen_graph.nodes, centroids))
# ax = cbg_df.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
# ax.axis("off")
# nx.draw(queen_graph, positions, ax=ax, node_size=5, node_color="r")
# plt.show()

# %% poi数据预处理

# ## 读取poi parttern数据
# # pattern_df = pd.read_csv(r"D:\software\清华云盘\download\mobility\03\21\patterns-part1.csv", nrows=100)

# ##
# pattern_df = pd.read_csv(r"D:\software\清华云盘\download\mobility\county_poi.csv")
# pattern_df = pd.read_csv(r"D:\Project\GNN_resilience\data\county_poi.csv")


# pattern_df["poi_cbg"] = pattern_df["poi_cbg"].map(lambda x: poi_cbg_transform(x))

# # pattern_df["poi_cbg"]是poi所在的CBG，pattern_df["visitor_home_cbgs"]是来访者poi

# # 留下目标county中的poi
# pattern_df = pattern_df[pattern_df["poi_cbg"].str.startswith(county_FIPS)]
# pattern_df.reset_index(inplace=True)

# ## 留下county中grocery store类型的poi

# # 先将pattern的naics_code转化为字符串
# pattern_df["naics_code"] = pattern_df["naics_code"].map(lambda x:poi_cbg_transform(x))

# # 留下naics开头是4451的poi
# pattern_df = pattern_df[pattern_df["naics_code"].str.startswith('4451')]

# # 获得每个poi的经纬度
# file_list = glob.glob(r'D:\software\清华云盘\download\mobility\**\**\*.csv')

# # 在pattern_df中添加经纬度两列
# # pattern_df["latitude"] = pd.Series()
# # pattern_df["longitude"] = pd.Series()

# # # 提取出core_poi文件
# # file_list_of_core_poi = [each for each in file_list if "core" in each]

# # for each in file_list_of_core_poi:
# #     poi_df = pd.read_csv(each)
# #     poi_df = poi_df[["placekey", "latitude", "longitude"]]

# #     # 使用apply函数，为pattern_df查找poi_df中的经纬度
# #     pattern_df = pattern_df.fillna(poi_df)

# #     if pattern_df["latitude"].isna().sum() == 0:
# #         break
# # del poi_df   

# #%% 给pattern_df家上服务区域，给cbg_df加入被服务的poi

# ## 根据 poi所在cbg和相邻cbg构建服务关系

# # 筛选出日期小一点的poi
# pattern_df = pattern_df[pattern_df["start_day"]=='2021-01-25']

# 获取目标poi所在cbg的k跳邻居
def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output

def hop_neighbor(x, queen_graph, cbg_df):
    network_idx = cbg_df[cbg_df["FIPS"]==x].index[0]
    return get_neigbors(queen_graph, network_idx, depth=2)


# pattern_df["served_cbg"] = pattern_df["poi_cbg"].map(lambda x: hop_neighbor(x, queen_graph, cbg_df))

# # 将所有的service合并成一个列表
# pattern_df["served_cbg"] = pattern_df["served_cbg"].map(lambda x: [each for key in x.keys() for each in x[key]])

# # 遍历poi 所有served cbg，给cbg加入 served_poi属性
# cbg_df["served_poi"] = cbg_df.apply(lambda x: [], axis=1)

# for idx in range(pattern_df.shape[0]): # 遍历每一行
#     placekey = pattern_df.iloc[idx]["placekey"]
#     cbg = pattern_df.iloc[idx]["served_cbg"]

#     for each in cbg: # each 应该是一个cbg在networt里的编号
#         FIPS = cbg_df.iloc[each][-1].append(placekey)
        
# ## 根据 poi所在cbg和距离计算服务关系

# # def distance_neighbors(x, cbg_df):
# #     # x是poi所在的cbg编号
# #     poi_lat = x["latitude"]
# #     poi_long = x["longitude"]

# #     # 计算这个poi和所有的cbg的距离


# # pattern_df["served_cbg_distance"] = pattern_df["poi_cbg"].apply( distance_neighbors(x, cbg_df), axis=1)

# ## 之前写的
# def long_lat_2_distance(x, poi_lat, poi_long):
#     # 获得cbg的lat
#     # print("poi_lat is", poi_lat)
#     # print("poi_long is", poi_long)
#     cbg_lat = x.xy[1][0]
#     cbg_long = x.xy[0][0]
#     # print("cbg_lat is", cbg_lat)
#     # print("cbg_long is", cbg_long)

#     # return geopy.distance.geodesic((poi_long, poi_lat), (cbg_long, cbg_lat)).km
#     return geopy.distance.geodesic((poi_lat, poi_long), (cbg_lat,cbg_long)).km
#     # print("####################")
#     # cdist(np.array([poi_lat, poi_long]).reshape((1,-1)), np.array([cbg_lat, cbg_long]).reshape((1,-1)), metric="euclidean")
#     # print("####################")

# # x是一个poi点
# def poi_2_cbg(x, cbg_df):
#     cbg_df2 = cbg_df.copy()
#     poi_lat = x["latitude"]
#     poi_long = x["longitude"]
#     # 计算每个cbg跟poi点的关系
#     cbg_df2["poi_distance"] = cbg_df2["centroid"].map(lambda x: long_lat_2_distance(x, poi_lat, poi_long))
#     # 筛选出poi_distance小于10km的行
#     cbg_df2 = cbg_df2[cbg_df2["poi_distance"] <= 10]
#     return list(cbg_df2.index)

# # 计算poi的服务关系
# pattern_df["served_cbg_distance"] = pattern_df.apply(lambda x: poi_2_cbg(x, cbg_df), axis=1)

# cbg_df["served_poi_distance"] = cbg_df.apply(lambda x: [], axis=1)

# for idx in range(pattern_df.shape[0]): # 遍历每一行
#     placekey = pattern_df.iloc[idx]["placekey"]
#     cbg = pattern_df.iloc[idx]["served_cbg"]

#     for each in cbg: # each 应该是一个cbg在networt里的编号
#         FIPS = cbg_df.iloc[each][-1].append(placekey)
        

# #%% Flow assign

# ## 基于POI数据，计算出flow 数目
# # pattern_df["visitor_home_cbgs"] = pd.Series()
# # file_list_of_pattern = [each for each in file_list if "pattern" in each]

# # for each in file_list_of_pattern:
# #     pattern = pd.read_csv(each)
# #     pattern = pattern[["placekey", "visitor_home_cbgs"]]

# #     # 使用apply函数，为pattern_df查找poi_df中的经纬度
# #     pattern_df = pattern_df.fillna(pattern)

# #     # if pattern_df["visitor_home_cbgs"].isna().sum() == 0:
# #     #     break
# # del pattern  

# # pattern_df["visitor_home_cbgs"]是一个字典，key是cbg编号，value是visit数目
# adjacency_flow = np.zeros(cbg_adjacency.shape)

# for i in range(pattern_df.shape[0]):
#     visitor_cbg_dict = pattern_df.iloc[i]["visitor_home_cbgs"]
#     visitor_cbg_dict = ast.literal_eval(visitor_cbg_dict)
#     end_cbg = pattern_df.iloc[i]["poi_cbg"]
#     end_node = cbg_df[cbg_df["FIPS"]==end_cbg].index[0]

#     # visitor也来自于county内的
#     keys = [key for key in visitor_cbg_dict.keys() if "48201" in key]

#     # key 是一个FIPS
#     for key in keys:
#         value = visitor_cbg_dict[key] # 人数
#         start_node = cbg_df[cbg_df["FIPS"]==key].index[0]

#         # 求取两个点之间的路径
#         shortest_path = nx.dijkstra_path(queen_graph, start_node, end_node)
        
#         end_nodes = shortest_path[1:]

#         for end_node in end_nodes:
#             adjacency_flow[start_node, end_node] = adjacency_flow[start_node, end_node] + value

# # 两个方向加到一起
# row,col = adjacency_flow.shape

# for i in range(row):
#     for j in range(i,col):
#         adjacency_flow[i,j] += adjacency_flow[j,i]

# for i in range(row):
#     for j in range(0,i+1):
#         adjacency_flow[j,i] = adjacency_flow[i,j]

# adjacency_distance = cbg_adjacency.copy()
# row, col = adjacency_distance.shape
# for i in range(row):
#     for j in range(i+1, col):
#         if adjacency_distance[i, j]!=0:
#             adjacency_distance[i, j] = geopy.distance.geodesic((centroids[i,1], centroids[i,0]),(centroids[j,1], centroids[j,0])).km

# for i in range(row):
#     for j in range(0,i+1):
#         adjacency_distance[j,i] = adjacency_distance[i,j]
# adjacency travel distance 计算cbg间的活动距离

## 保存与处理数据
# pattern_df.to_csv(r"D:\Project\GNN_resilience\result\pattern.csv")
# cbg_df.to_csv(r"D:\Project\GNN_resilience\result\cbg.csv")
# np.savetxt(r"D:\Project\GNN_resilience\result\adjacency_flow.csv", adjacency_flow,delimiter=',')
# np.savetxt(r"D:\Project\GNN_resilience\result\adjacency_distance.csv", adjacency_distance,delimiter=',')

pattern_df = pd.read_csv(r"D:\Project\GNN_resilience\result\pattern.csv")
cbg_df = pd.read_csv(r"D:\Project\GNN_resilience\result\cbg.csv")
adjacency_flow = np.loadtxt(r"D:\Project\GNN_resilience\result\adjacency_flow.csv",delimiter=',')
adjacency_distance = np.loadtxt(r"D:\Project\GNN_resilience\result\adjacency_distance.csv",delimiter=',')

####################################################################################################################################
#%% 从大的网络里采样得到导出子图induced graph
random.seed(12345)
def all_graph_distance(sub_mobility_flow, sub_queen, sub_distance, sub_graph_nodes):
    row, col = sub_mobility_flow.shape
    graph_total_distance = 0
    for i in range(row):
        for j in range(i+1, col):
            # 每个连接上人的flow
            if sub_mobility_flow[i,j]!=0:
                flow_volume = sub_mobility_flow[i,j]
                start_node = list(sub_queen.nodes)[i]
                end_node = list(sub_queen.nodes)[j]
                # flow_path = nx.shortest_path(sub_queen, start_node, end_node)
                # node_index =[sub_graph_nodes.index(each) for each in flow_path] 
                # shortest_path_distance = [sub_distance[node_index[idx], node_index[idx+1]] for idx in range(len(node_index)-1)]
                graph_total_distance += nx.shortest_path_length(sub_queen, start_node, end_node, weight="weight") * flow_volume
    return graph_total_distance  

####################################################################################################################################
# 产生第一个节点
# for graph_add_number in range(100):
node_kick_off = random.randint(0,len(queen_graph.nodes))
node_number=50
def get_node_neighbors(node_kick_off, queen_graph, node_number):
    total_node_number = 0
    sample_nodes = []

    hop = get_neigbors(queen_graph, node_kick_off, depth=1)[1]
    sample_nodes = sample_nodes + hop

    while total_node_number <= node_number:
        random.shuffle(hop)
        hop = [get_neigbors(queen_graph, each, depth=1)[1] for each in hop]        
        hop = [each2 for each1 in hop for each2 in each1]

        sample_nodes = sample_nodes + hop
        total_node_number = len(sample_nodes)
        
    return list(set(sample_nodes))

# 获取网络节点
sub_graph_nodes = get_node_neighbors(node_kick_off, queen_graph, node_number)    

# 找到一个新节点
for source_node in sub_graph_nodes:
    # 与该source_node相连的节点
    connected_node_of_source_node = list(nx.neighbors(queen_graph, source_node))
    # 判断有没有节点在新的图里
    for each in connected_node_of_source_node:
        # 没有的话添加进去，得到新图索引，跳出循环
        if not(each in sub_graph_nodes):
            sub_graph_nodes.append(each) 
            break
    break

# 获取子图
sub_queen = queen_graph.subgraph(sub_graph_nodes).copy()

# 获取子图对应的cbg数据
sub_cbg_df = cbg_df.iloc[sub_graph_nodes, :]
# sub_cbg_df = cbg_df.iloc[list(sub_queen.nodes), :]

# 获取子图对应的poi数据
def sub_poi(x, cbg_df, sub_cbg_df):
    node_index = cbg_df[cbg_df["FIPS"]==x].index[0]
    if node_index in sub_cbg_df.index:
        return True
    else: 
        return False 

sub_pattern_df = pattern_df[pattern_df["poi_cbg"].map(lambda x:sub_poi(x, cbg_df, sub_cbg_df))]  

# 获取子图对应mobility flow
sub_mobility_flow = adjacency_flow[sub_graph_nodes,:]
sub_mobility_flow = sub_mobility_flow[:, sub_graph_nodes]
# sub_mobility_flow = adjacency_flow[list(sub_queen.nodes),:]
# sub_mobility_flow = sub_mobility_flow[:, list(sub_queen.nodes)]

# 获取子图对应的距离矩阵
sub_distance = adjacency_distance[sub_graph_nodes,:]
sub_distance = sub_distance[:,sub_graph_nodes]
# sub_distance = adjacency_distance[list(sub_queen.nodes),:]
# sub_distance = sub_distance[:,list(sub_queen.nodes)]

####################################################################################################################################
# %% 攻击网络

## 拆除连接，导致travel距离上升  

## 加入循环

# 构造一个结果字典，键是连接，值是均值
disntance_importance_dict = {key: 0 for key in list(sub_queen.edges)}


stability_list = []
# 计算距离，给一个现在的flow + 邻接矩阵 + 位移距离
for monte_times in range(500):
    sub_queen_copy = sub_queen.copy()
    initial_distance = all_graph_distance(sub_mobility_flow, sub_queen_copy, sub_distance, sub_graph_nodes)

    ## 删除subqueen里的edge
    post_attack_distance = []
    # 删除一条边
    sub_queen_copy_edges = list(sub_queen_copy.edges).copy()
    random.shuffle(sub_queen_copy_edges)
    for idx, each in enumerate(sub_queen_copy_edges):
        # 删除一条边
        sub_queen_copy.remove_edge(each[0], each[1])
        # 计算连通子图个数小于2则停止
        if nx.number_connected_components(sub_queen_copy) >1:
            break
        # 计算删除边之后全图的travel时间
        post_attack_distance.append(all_graph_distance(sub_mobility_flow, sub_queen_copy, sub_distance, sub_graph_nodes))
        if len(post_attack_distance)>=2:
            post_attack_distance[-1] < post_attack_distance[-2]

    # 攻击完毕
    # post_attack_distance = [each/initial_distance for each in post_attack_distance]
    
    # 计算攻击一次后全图的变化
    # 这是第二个元素到最后一个元素
    attack_result_list = [post_attack_distance[idx] - post_attack_distance[idx-1] for idx in range(1, len(post_attack_distance))]
    # 第一个元素
    first_element = [post_attack_distance[0]-initial_distance]
    # 形成列表
    attack_result_list = first_element + attack_result_list

    # 形成字典 边：对应的距离增加
    attack_result_dict = {sub_queen_copy_edges[idx]:attack_result_list[idx] for idx in range(len(post_attack_distance))}

    # # 相比于前一次攻击，网络的全图距离增加的百分比
    # attack_result_dict = {list(attack_result_dict.keys())[idx]:((attack_result_dict[list(attack_result_dict.keys())[idx]])-(attack_result_dict[list(attack_result_dict.keys())[idx-1]])/(attack_result_dict[list(attack_result_dict.keys())[idx-1]]))for idx in range(1, len(attack_result_dict))}

    for key in attack_result_dict.keys():
        disntance_importance_dict[key] += attack_result_dict[key]
    
    stability_list.append(attack_result_dict)

for key in disntance_importance_dict.keys():
    disntance_importance_dict[key] /= 500

graph_number = 10
file =  open("D:\\Project\\GNN_resilience\\data\\sample_stability\\sample_times" + str(graph_number) + ".txt", "w")
file.writelines(str(stability_list))
file.close()

# 排序，找到key_player
key_player_cbg_edge = sorted(disntance_importance_dict, key=disntance_importance_dict.get, reverse=True)

    ####################################################################################################################################
    # ## 保存图的数据
    # data_save_path = r"D:\Project\GNN_resilience\data\training_data"

    # # networkx 添加节点属性
    # # 字典键是对应的节点编号，值是一系列属性
    # node_value_dict = {}
    # for idx in zip(list(sub_queen.nodes)): 
    #     idx = idx[0]
    #     # 获取对应行数据
    #     row = sub_cbg_df[sub_cbg_df.index==idx]
    #     row_population = row["POP2012"].values[0]
    #     # 获取经纬度
    #     row_xy = re.findall("\d+\.?\d*", row["centroid"].values[0])
    #     row_latitude = float(row_xy[1])
    #     row_longitude = float(row_xy[0])
    #     value_dict = {"population": row_population, "latitude": row_latitude, "longitude": row_longitude}
    #     node_value_dict[idx] = value_dict

    # nx.set_node_attributes(sub_queen, node_value_dict)

    # # 存储边的数据
    # edge_value_dict = {}
    # for idx in zip(list(sub_queen.edges)):
    #     idx = idx[0]
    #     start_node = sub_graph_nodes.index(idx[0])
    #     end_node = sub_graph_nodes.index(idx[1])
    #     flow = sub_mobility_flow[start_node, end_node]
    #     value_dict = {'mobility_flow': flow}
    #     edge_value_dict[idx] = value_dict
    # nx.set_edge_attributes(sub_queen, edge_value_dict)

    # # 存储边的lable
    # # 排序值
    # edge_rank_label_dict = {each:{"rank_label":idx+1} for idx, each in enumerate(key_player_cbg_edge)}
    # nx.set_edge_attributes(sub_queen, edge_rank_label_dict)
    # # 绝对值
    # abs_value = {key:{"value_label":disntance_importance_dict[key]} for key in disntance_importance_dict.keys()}
    # nx.set_edge_attributes(sub_queen, abs_value)
    # # 保存图
    # save_path = r"D:\Project\GNN_resilience\data\training_data"
    # file_path = save_path + "\\" +str(len(os.listdir(save_path))) + ".gpickle"
    # print(file_path)
    # nx.write_gpickle(sub_queen, file_path)

    # ## 画图
    # pos = nx.spring_layout(sub_queen)
    # important_edges = key_player_cbg_edge[:3]
    # path_edges = important_edges

    # # 画普通图
    # plt.figure(figsize=(20,10))
    # nx.draw_networkx_nodes(sub_queen, pos, nodelist=sub_queen.nodes)
    # nx.draw_networkx_edges(sub_queen, pos, edgelist=set(sub_queen.edges)-set(path_edges), connectionstyle='arc3, rad = 0.3')

    # # 画关键图
    # nx.draw_networkx_edges(sub_queen,pos,edgelist=path_edges,edge_color='r', width=3, connectionstyle='arc3, rad = 0.3')

    # # 画边权
    # edge_labels = nx.get_edge_attributes(sub_queen, 'mobility_flow')
    # nx.draw_networkx_edge_labels(sub_queen, pos, edge_labels)

    # plt.savefig("D:\\Project\\GNN_resilience\\data\\training_data_fig\\" + str(len(os.listdir(save_path)))+".png")
    # plt.close()

# 保存图的结构
# 保存节点属性
# 保存边的属性
# 保存边的属性

####################################################################################################################################
# 可视化
# nx.draw(sub_queen)
# nx.draw_networkx_edges(key_player_cbg_edge[0],edge_color='r',)

# 加入到词典里

# 转化成前后值变化



## 随即攻击一百次，每次只要删除这个边了，就计算导致图的上升趋势，如果断在他这了，记1000，最后单独统计1000的次数

####################################################################################################################################
# # %% 攻击POI节点

# # sub_pattern_df记录了poi信息
# # sub_cbg_df 记录了sub_cbg_df信息

# # sub_pattern["served_cbg", "served_cbg_distance"]转换成列表
# sub_pattern_df.loc[:,"served_cbg"] = sub_pattern_df["served_cbg"].map(lambda x: eval(x))
# sub_pattern_df.loc[:,"served_cbg_distance"] = sub_pattern_df["served_cbg_distance"].map(lambda x: eval(x))

# # 消除掉sub_pattern_df中，不在子图范围内的服务cbg
# def outsider_removal(x, sub_cbg_index):
#     return [each for each in x if each in sub_cbg_index]
# # 在区域内的cbg
# sub_cbg_index = list(sub_cbg_df.index)
# sub_pattern_df["served_cbg_distance"] = sub_pattern_df["served_cbg_distance"].map(lambda x: outsider_removal(x, sub_cbg_index))

# sub_pattern_df["served_cbg"] = sub_pattern_df["served_cbg"].map(lambda x: outsider_removal(x, sub_cbg_index))

# # 消除掉sub_cbg_df中不在子网范围内的poi (依据poi所在的cbg)
# sub_cbg_df["served_poi"] = sub_cbg_df["served_poi"].map(lambda x: eval(x))
# sub_cbg_df["served_poi_distance"] = sub_cbg_df["served_poi_distance"].map(lambda x: eval(x))

# def poi_removal(x, pattern_df, sub_cbg_index, cbg_df):
#     cbg_index = [pattern_df[pattern_df["placekey"]==each]["poi_cbg"].values[0] for each in x]
#     cbg_index = [cbg_df[cbg_df["FIPS"]==each].index.values[0] for each in cbg_index]
    
#     # poi_cbg_pair = [[cbg_index[idx], x[idx]]for idx in range(len(x))]

#     poi_final = [x[idx] for idx in range(len(x)) if cbg_index[idx] in sub_cbg_index]

#     return poi_final

# sub_cbg_df["served_poi_distance"] = sub_cbg_df["served_poi_distance"].map(lambda x: poi_removal(x, pattern_df, sub_cbg_index, cbg_df))
# sub_cbg_df["served_poi"] = sub_cbg_df["served_poi"].map(lambda x: poi_removal(x, pattern_df, sub_cbg_index, cbg_df))



# ## 开始攻击节点
# def poi_attack(x, poi_delete):
#     if not(poi_delete in x):
#         return x
#     else:
#         a = x.copy()
#         a.remove(poi_delete)
#         return a

# total_un_served_population = 0

# poi_list = list(sub_pattern_df["placekey"])
# total_population = sub_cbg_df["POP2012"].sum()
# # 构造一个结果字典，键是placekey，值是重要性
# poi_important_dict = {key:0 for key in poi_list}
# sub_cbg_df["served_poi_distance2"] = sub_cbg_df["served_poi_distance"]

# for monte_times in range(500):
#     sub_cbg_df_copy=sub_cbg_df.copy()
#     random.shuffle(poi_list)

#     # 删除一个poi，考虑served_poi_distance
#     total_un_served_populations = []
#     for poi_delete in poi_list:
#         # 删除一个poi，更新poi服务关系
#         sub_cbg_df_copy["served_poi_distance"] = sub_cbg_df_copy["served_poi_distance"].map(lambda x:poi_attack(x, poi_delete))

#         # 得到cbg还有多少poi服务
#         sub_cbg_df_copy["remained_service"] = sub_cbg_df_copy["served_poi_distance"].map(lambda x: len(x))

#         # 统计所有cbg剩余poi服务等于0的行
#         unserved_poi = sub_cbg_df_copy[sub_cbg_df_copy["remained_service"]==0]
#         total_un_served_population += unserved_poi["POP2012"].sum()
#         # 记录数据
#         total_un_served_populations.append([poi_delete, total_un_served_population])
#         # 删除所有cbg剩余poi服务等于0的行
#         sub_cbg_df_copy = sub_cbg_df_copy[sub_cbg_df_copy["remained_service"]>0]

#     # 依次删除poi的结果
#     poi_delete_result = [(each[1]/total_population) for each in total_un_served_populations]
#     poi_delete_result = [poi_delete_result[idx+1]-poi_delete_result[idx] for idx in range(len(poi_delete_result)-1)]

#     # 该次攻击结果
#     poi_attack_result = {poi_list[idx+1]:poi_delete_result[idx] for idx in range(len(poi_list)-1)}

#     # 加到总结果里
#     for key in poi_attack_result.keys():
#         poi_important_dict[key] += poi_attack_result[key]

# for key in poi_important_dict.keys():
#     poi_important_dict[key] /=500

# key_player_poi = sorted(poi_important_dict, key=poi_important_dict.get, reverse=True)

# plt.plot([each[1]/total_population for each in total_un_served_populations])
# plt.xlabel("# of delted poi")
# plt.ylabel(" unserviced population/total_pop")
