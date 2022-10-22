import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import ast
import datetime
import time
# import function
import glob
# import zipfile 
# import rarfile
# import gzip
import tqdm
import math

#%%

# 批量获取路径
file_list = glob.glob(r'D:\software\清华云盘\download\mobility\**\**\*.csv')

# 提取出core_poi文件
file_list_of_core_poi = [each for each in file_list if "core" in each]

file_list_of_pattern = [each for each in file_list if "pattern" in each]

# 构造naics-placekey 词典
# nacis_placekey_dict = function.nacis_key_dict(file_list_of_core_poi)
#%%
# 读取所有的pandas
df_poi1 = pd.read_csv(file_list_of_core_poi[0])
df_poi1 = df_poi1[df_poi1["region"].map(lambda x: x == "TX")]
df_poi1 = df_poi1[["placekey", "naics_code", "latitude", "longitude"]]
for file_path in tqdm.tqdm(file_list_of_core_poi[1:]):
    df_poi2 = pd.read_csv(file_path)
    df_poi2 = df_poi2[df_poi2["region"].map(lambda x: x == "TX")]
    df_poi2 = df_poi2[["placekey", "naics_code", "latitude", "longitude"]]
    
    df_poi1 = pd.concat([df_poi1, df_poi2], axis=0, ignore_index=0)

df_poi1 = df_poi1.drop_duplicates(subset=["placekey"])
df_poi1 = df_poi1.reset_index(drop=True)
df_poi1.to_csv(r"D:\software\清华云盘\download\mobility\poi.csv")

# df1 = pd.read_csv(r"D:\software\清华云盘\download\mobility\poi.csv")

#%%

def get_day_date_format(x):
    x = x.split("T")[0]
    date = datetime.datetime.strptime(x, "%Y-%m-%d")
    return date

#%%

# 批量获取路径
file_list = glob.glob(r'D:\software\清华云盘\download\mobility\**\**\*.csv')

# 提取出core_poi文件
file_list_of_core_poi = [each for each in file_list if "core" in each]

file_list_of_pattern = [each for each in file_list if "pattern" in each]

df1 = pd.read_csv(r"D:\software\清华云盘\download\mobility\poi.csv")

#%%
# def date_range_judege(x, start_day=datetime.datetime(2021,1,13), end_day = datetime.datetime(2021,1,17)):
#     if (x.start_day>=start_day) and (x.start_day<end_day):
#         return True

#     if (x.end_day>start_day) and (x.end_day<=end_day):
#         return True
    
#     return False

def date_range_judege(x, start_day=datetime.datetime(2021,1,25), end_day = datetime.datetime(2021,1,17)):
    if (x.start_day==start_day):
        return True
    return False

df_pattern1 = pd.read_csv(file_list_of_pattern[0])
# df_pattern1 = df_pattern1[["placekey", "safegraph_place_id", "city" , "region", "iso_country_code", "safegraph_brand_ids", "brands", 'date_range_start', 'date_range_end', 'raw_visit_counts', 'raw_visitor_counts', 'visits_by_day', 'visits_by_each_hour', "poi_cbg" ,"visitor_daytime_cbgs", "distance_from_home", "visitor_home_cbgs"]]

df_pattern1 = df_pattern1[["placekey", "safegraph_place_id", "city" , "region", "safegraph_brand_ids", "brands", 'date_range_start', 'date_range_end', 'visits_by_day', 'visits_by_each_hour', "poi_cbg" ,"visitor_daytime_cbgs", "distance_from_home", "visitor_home_cbgs"]]

def poi_cbg_transform(x):
    if not(np.isnan(x)):
        return str(int(x))
    else:
        return ""

df_pattern1 = df_pattern1[df_pattern1["region"].map(lambda x: x == "TX")] # 提取德克萨斯地区

df_pattern1["poi_cbg"] = df_pattern1["poi_cbg"].map(lambda x: poi_cbg_transform(x))

# 只留存目标county的poi
county_FIPS = '48201'
df_pattern1 =df_pattern1[df_pattern1["poi_cbg"].str.startswith(county_FIPS)]

# 转换时间
df_pattern1["start_day"] = df_pattern1["date_range_start"].map(lambda x: get_day_date_format(x))
df_pattern1["end_day"] = df_pattern1["date_range_end"].map(lambda x:get_day_date_format(x))
df_pattern1["time_indicator"] = df_pattern1.apply(date_range_judege, axis=1)
df_pattern1 = df_pattern1[df_pattern1["time_indicator"]==True]

for df_pattern_path in tqdm.tqdm(file_list_of_pattern[1:]):
    df = pd.read_csv(df_pattern_path)
    df = df[["placekey", "safegraph_place_id", "city" , "region", "safegraph_brand_ids", "brands", 'date_range_start', 'date_range_end', 'visits_by_day', 'visits_by_each_hour', "poi_cbg" ,"visitor_daytime_cbgs", "distance_from_home", "visitor_home_cbgs"]]
    df = df[df["region"].map(lambda x: x == "TX")] # 提取德克萨斯地区
    
    df["poi_cbg"] = df["poi_cbg"].map(lambda x: poi_cbg_transform(x))

    df =df[df["poi_cbg"].str.startswith(county_FIPS)]

    df["visits_by_day"] = df["visits_by_day"].map(lambda x: eval(x))
    df["start_day"] = df["date_range_start"].map(lambda x: get_day_date_format(x))
    df["end_day"] = df["date_range_end"].map(lambda x:get_day_date_format(x))
    df["time_indicator"] = df.apply(date_range_judege, axis=1)
    df = df[df["time_indicator"]==True]
    
    df_pattern1 = pd.concat([df_pattern1, df], axis=0)
    print(df_pattern1.shape)

# 加入naics code数据
df_pattern1

df_pattern1 = df_pattern1.merge(df1, how="left", on="placekey")
df_pattern1 = df_pattern1[df_pattern1["naics_code"].map(lambda x: not math.isnan(x))]

# 使用fill试试
# df1 = df1.set_index("placekey")

# b = df_pattern1.copy()
# b["naics_code"] = pd.Series()
# b["latitude"] = pd.Series()
# b["longitude"] = pd.Series()
# b = b.set_index("placekey")

# b = b.fillna(df1)
# b = b.reset_index()
df_pattern1.to_csv("county_poi.csv")

# %%
