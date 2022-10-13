import geopandas as gp 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import function
# import pysal as ps
from libpysal import weights
import os
import networkx as nx
import math
import glob
import geopy.distance
import shapely

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

# ## cbg_df["FIPS"] 是十二位的cbg fips code，可以用来筛选出位于Harris County的 cbg，数据格式是字符串

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
cbg_adjacency = np.array(nx.adjacency_matrix(queen_graph).todense())

#  可视化
centroids = np.array([[each.xy[0][0], each.xy[1][0]] for each in cbg_df["centroid"]])
positions = dict(zip(queen_graph.nodes, centroids))
ax = cbg_df.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
ax.axis("off")
nx.draw(queen_graph, positions, ax=ax, node_size=5, node_color="r")
plt.show()

# %% poi数据预处理

## 读取poi parttern数据
# pattern_df = pd.read_csv(r"D:\software\清华云盘\download\mobility\03\21\patterns-part1.csv", nrows=100)

pattern_df = pd.read_csv(r"D:\software\清华云盘\download\mobility\pattern_time_limit.csv")



pattern_df["poi_cbg"] = pattern_df["poi_cbg"].map(lambda x: poi_cbg_transform(x))

# pattern_df["poi_cbg"]是poi所在的CBG，pattern_df["visitor_home_cbgs"]是来访者poi

# 留下目标county中的poi
pattern_df = pattern_df[pattern_df["poi_cbg"].str.startswith(county_FIPS)]
pattern_df.reset_index(inplace=True)

## 留下county中grocery store类型的poi

# 先将pattern的naics_code转化为字符串
pattern_df["naics_code"] = pattern_df["naics_code"].map(lambda x:poi_cbg_transform(x))

# 留下naics开头是4451的poi
pattern_df = pattern_df[pattern_df["naics_code"].str.startswith('4451')]

# 获得每个poi的经纬度
file_list = glob.glob(r'D:\software\清华云盘\download\mobility\**\**\*.csv')

# 在pattern_df中添加经纬度两列
pattern_df["latitude"] = pd.Series()
pattern_df["longitude"] = pd.Series()

# 提取出core_poi文件
file_list_of_core_poi = [each for each in file_list if "core" in each]

for each in file_list_of_core_poi:
    poi_df = pd.read_csv(each)
    poi_df = poi_df[["placekey", "latitude", "longitude"]]

    # 使用apply函数，为pattern_df查找poi_df中的经纬度
    pattern_df = pattern_df.fillna(poi_df)

    if pattern_df["latitude"].isna().sum() == 0:
        break
del poi_df   

#%% 给pattern_df家上服务区域，给cbg_df加入被服务的poi

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

# # x是一个poi点
# def poi_2_cbg(x, cbg_df):
#     cbg_df2 = cbg_df.copy()
#     lat = x["latitude"]
#     long = x["longitude"]
#     # 计算每个cbg跟poi点的关系
#     cbg_df2["poi_distance"] = cbg_df2["centroid"].map(lambda x: long_lat_2_distance(x, lat, long))
#     # 筛选出poi_distance小于2km的行
#     cbg_df2 = cbg_df2[cbg_df2["poi_distance"] <= 10]
#     print(cbg_df2)
#     return list(cbg_df2["FIPS"])

## 计算poi的服务关系
# pattern_df["served_cbg"] = pattern_df.apply(lambda x: poi_2_cbg(x, cbg_df), axis=1)

## 根据 poi所在cbg和相邻cbg构建服务关系

# 筛选出日期小一点的poi
pattern_df = pattern_df[pattern_df["start_day"]=='2021-02-08']

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


pattern_df["served_cbg"] = pattern_df["poi_cbg"].map(lambda x: hop_neighbor(x, queen_graph, cbg_df))

# 将所有的service合并成一个列表
pattern_df["served_cbg"] = pattern_df["served_cbg"].map(lambda x: [each for key in x.keys() for each in x[key]])

# 遍历poi 所有served cbg，给cbg加入 served_poi属性
cbg_df["served_poi"] = cbg_df.apply(lambda x: [], axis=1)

for idx in range(pattern_df.shape[0]): # 遍历每一行
    placekey = pattern_df.iloc[idx]["placekey"]
    cbg = pattern_df.iloc[idx]["served_cbg"]

    for each in cbg: # each 应该是一个cbg在networt里的编号
        FIPS = cbg_df.iloc[each][-1].append(placekey)
        

## cbg得到的poi服务关系

#%% 

# # 读取德克萨斯州数据

# # 批量获取路径
# file_list = glob.glob(r'D:\software\清华云盘\download\mobility\**\**\*.csv')

# # 提取出core_poi文件
# file_list_of_core_poi = [each for each in file_list if "core" in each]

# file_list_of_pattern = [each for each in file_list if "pattern" in each]

