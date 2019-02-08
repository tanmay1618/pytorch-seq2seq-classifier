# coding: utf-8
directory_in_str = "/home/tanmay/Documents/cv/spotmentor/machine-learning-assessment/data/docs"
import pandas as pd
import os
import json

dict_data = []
for file in os.listdir(directory_in_str):
    filename = file
    if filename.endswith(".json"): 
        full_path = os.path.join(directory_in_str, filename)
        with open(full_path) as f:
            data = json.load(f)
            temp_data = {}
            temp_data["description"]=data["jd_information"]["description"]
            temp_data["id"] = data["_id"]
            dict_data.append(temp_data)
        continue
    else:
        continue
    
df = pd.DataFrame.from_dict(dict_data, orient='columns')
df_dep = pd.read_csv('document_departments.csv')
df_dep.columns=["id","department"]
full_table= df.merge(df_dep,on='id',how='left')
#full_table.to_csv('../dataset.csv')
