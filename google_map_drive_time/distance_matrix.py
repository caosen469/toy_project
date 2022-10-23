import pandas as pd
import numpy as np
import math
import selenium
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import re
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import tqdm
################################################################################################
################################################################################################
#%%
pattern_df = pd.read_csv(r"D:\Project\GNN_resilience\result\pattern.csv")
cbg_df = pd.read_csv(r"D:\Project\GNN_resilience\result\cbg.csv")

def get_latitude(x):
    a = re.findall("\d+\.?\d*", x)
    return float(a[1])
def get_longitude(x):
    a = re.findall("\d+\.?\d*", x)
    return -1*float(a[0])

cbg_df["latitude"] = cbg_df["centroid"].map(lambda x:get_latitude(x))
cbg_df["longitude"] = cbg_df["centroid"].map(lambda x:get_longitude(x))


# 形成位置矩阵
travel_time = np.zeros((cbg_df.shape[0],cbg_df.shape[0]))


################################################################################################
################################################################################################
# 谷歌数据
http = "https://www.google.com/maps/place/29%C2%B043'11.8%22N+95%C2%B017'49.4%22W/@29.7033924,-95.3879111,11.38z/data=!4m5!3m4!1s0x0:0x5124d9f437e56683!8m2!3d29.7199493!4d-95.2970657"
driver = webdriver.Firefox()
driver.get(http)

# 点击direction
direction_button = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[4]/div[1]/button/span/img")
direction_button.click()
# 点击汽车键
driver_tab = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[2]/div/div/div/div[2]/button/img")
driver_tab.click()

first_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[1]/div[2]/div[1]/div/input")
first_input.clear()

second_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[2]/div[2]/div[1]/div/input")
second_input.clear()

################################################################################################
################################################################################################
results = {}
row, col = travel_time.shape

count = 0
for i in tqdm.tqdm(range(2, row)):
    first_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[1]/div[2]/div[1]/div/input")
    first_input.send_keys(str(cbg_df.iloc[i]["latitude"])+", "+ str(cbg_df.iloc[i]["longitude"]))

    try:
        locator = (By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[1]/div[2]/div[1]/div/input")
        text = str(cbg_df.iloc[i]["latitude"])+", "+ str(cbg_df.iloc[i]["longitude"])
        WebDriverWait(driver, 30).until(
        EC.text_to_be_present_in_element_value(locator,text)) #This is a dummy element
    finally:
        a=text

    # 点击确认键
    firtst_input_search = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[1]/div[2]/button[1]")
    firtst_input_search.click()

    time.sleep(1)
    for j in range(i+1, col):

    ## 开始循环


        # 确认

        # 在第二个输入框输入数据
        second_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[2]/div[2]/div[1]/div/input")
        second_input.send_keys(str(cbg_df.iloc[j]["latitude"])+", "+ str(cbg_df.iloc[j]["longitude"]))

        try:
            locator = (By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[2]/div[2]/div[1]/div/input")
            text = str(cbg_df.iloc[j]["latitude"])+", "+ str(cbg_df.iloc[j]["longitude"])
            WebDriverWait(driver, 30).until(
            EC.text_to_be_present_in_element_value(locator,text)) #This is a dummy element
        finally:
            a=text

        # 点击确认键
        second_input_search = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[2]/div[2]/button[1]")
        second_input_search.click()

        # 查看页面是否有变化 小车标
        locator = '/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[4]/div[1]/div[1]/div[1]/div[1]/div[2]'
        # # locator = "#section-directions-trip-travel-mode-0"
        # try:
        #     WebDriverWait(driver, 300).until(EC.visibility_of_element_located((By.XPATH, locator)))
        # finally:
        #     a=1
        time.sleep(1.5)
        one_travel_time = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[4]/div[1]/div[1]/div[1]/div[1]/div[1]/span[1]").text
        
        one_travel_time = float(re.findall("\d+\.?\d*", one_travel_time)[0])

        # 删除输入


        second_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[2]/div[2]/div[1]/div/input")
        second_input.clear()
        results[(i,j)] = one_travel_time
        travel_time[i,j] = one_travel_time
    first_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[3]/div[1]/div[1]/div[2]/div[1]/div/input")
    first_input.clear()
# 在输入框中点击第一个位置

# %%