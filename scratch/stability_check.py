import pandas as pd
import numpy as np 
import ast
import os
import glob
import matplotlib.pyplot as plt

# 获取采样数据
file_path = glob.glob(r"D:\Project\GNN_resilience\data\sample_stability\*.txt")

for each in file_path:
    stability_list_file = open(each)
    stability_list = stability_list_file.readlines()
    stability_list = stability_list[0]
    # 长为500的list，每个元素是以此网络攻击的结果
    stability_list = eval(stability_list)

    # 获取所有出现过的key
    edge_list = [list(each.keys()) for each in stability_list]
    edge_list = [each2 for each1 in edge_list for each2 in each1]
    edge_list = list(set(edge_list))

    # 提取一个遍所有的抽样结果
    def edge_sample_result(key, stability_list):
        result = []
        for each in stability_list:
            if key in each.keys():
               result.append(each[key])
        return result 


    edge_sample_result = {key:edge_sample_result(key, stability_list) for key in edge_list}

    # 获取所有
    stability_list_file.close()
    
for key in list(edge_sample_result.keys())[:30]:
    plt.figure()
    plt.ylabel("resilience under one sample")
    plt.xlabel("sample times") 
    plt.plot(edge_sample_result[key])
