import geopandas as gp 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import function
# import pysal as ps
from libpysal import weights
import os
import networkx as nx
# os.environ['PROJ_LIB']=r'D:\Users\19688\Anaconda2\envs\geopands2\Lib\site-packages\osgeo\data\proj'
#%% 读取目标county的CBG数据

# cbg_df = gp.read_file(r"D:\Project\GNN_resilience\data\texas_cbg\cbg_1\Export_Output.shp") # 读取整个德克萨斯州的CBG的shapefile

# ## cbg_df["FIPS"] 是十二位的cbg fips code，可以用来筛选出位于Harris County的 cbg，数据格式是字符串

# # Harris County FIPS: 48201
# county_FIPS = '48201'

# # 筛选出harris county的cbg
# cbg_df = cbg_df[cbg_df["FIPS"].str.startswith(county_FIPS)]
# cbg_df.reset_index(drop=True)

# # 保存目标county cbg数据
# cbg_df.to_file(r'D:\Project\GNN_resilience\data\harris_cbg\harris.shp', driver='ESRI Shapefile')

#%% 构建CBG网络

# 读取目标county cbg的shapefile
cbg_df = gp.read_file(r'D:\Project\GNN_resilience\data\harris_cbg\harris.shp')

# 基于cbg_df["geometry"]计算出来对应polygon的质心
cbg_df["centroid"] = cbg_df["geometry"].map(lambda x: x.centroid)

## 构建Queeen 邻接矩阵
queen = weights.Queen.from_dataframe(cbg_df)
# 构建queen邻接图
queen_graph = queen.to_networkx()

#  可视化
centroids = np.array([[each.xy[0][0], each.xy[1][0]] for each in cbg_df["centroid"]])
positions = dict(zip(queen_graph.nodes, centroids))
ax = cbg_df.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
ax.axis("off")
nx.draw(queen_graph, positions, ax=ax, node_size=5, node_color="r")
plt.show()

#%% 

# # 读取德克萨斯州数据

# # 批量获取路径
# file_list = glob.glob(r'D:\software\清华云盘\download\mobility\**\**\*.csv')

# # 提取出core_poi文件
# file_list_of_core_poi = [each for each in file_list if "core" in each]

# file_list_of_pattern = [each for each in file_list if "pattern" in each]
