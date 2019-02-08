import pandas as pd
import json
import os
import numpy as np
import torch
import unicodedata
import re

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def embeddedTensorFromSentence(sentence,device,word_emb,N_word):
	desc_tokens = sentence.split(" ")
	emb_tokens = []
	#print(len(desc_tokens))
	for token in desc_tokens:
		val = word_emb.get(token, np.zeros(N_word, dtype=np.float32))
		emb_tokens.append(torch.tensor(val,dtype = torch.float ,device=device).view(1,1,N_word))
	#return emb_tokens
	return emb_tokens

def check_exists(dep,classes):
	group1 = ["Ticketing"]
	group2 = ["Sales"]
	group3 = ["Digital Marketing","Customer service","Marketing"]
	group4 = [x for x in classes[4:]]
	if( dep in group1):
		return "Ticketing"
	elif(dep in group2):
		return "Sales"
	elif(dep in group3):
		return "group3"
	else:
		return "group4"

def load_data(device):
	directory_in_str = "/home/tanmay/Documents/cv/spotmentor/machine-learning-assessment/data/docs"
	dict_data = []
	for file in os.listdir(directory_in_str):
		filename = file
		if filename.endswith(".json"): 
			full_path = os.path.join(directory_in_str, filename)
			with open(full_path) as f:
		    		data = json.load(f)
		    		temp_data = {}
		    		temp_data["description"]=data["jd_information"]["description"]
		    		temp_data["id"] = int(data["_id"])
		    		dict_data.append(temp_data)
	   
	df = pd.DataFrame.from_dict(dict_data, orient='columns')
	df_dep = pd.read_csv('../data/document_departments.csv')
	df_dep.columns=["id","department"]
	classes_ = df_dep["department"].unique()
	df_dep["department_new"] = df_dep["department"].apply(check_exists,classes=classes_)
	full_table= df.merge(df_dep,on='id',how='left')
	#import pdb;pdb.set_trace();
	#import pdb;pdb.set_trace();
	classes_ = df_dep["department_new"].unique()
	te = df_dep["department_new"].value_counts()
	print(te)
	dc = te.to_dict()
	no = full_table.shape[0]
	weight_list = []
	for it in classes_:
		weight_list.append(1/dc[it])
	weight_tensor = torch.tensor(weight_list,dtype = torch.float ,device=device)
	return full_table, classes_, weight_tensor

def load_word_emb(file_name):
	script_dir = os.path.dirname(__file__)
	abs_file_path = os.path.join(script_dir, file_name)
	print(('Loading word embedding from %s'%file_name))
	ret = {}
	with open(abs_file_path) as inf:
		for idx, line in enumerate(inf):
			if (idx >= 10000):
				break
			info = line.strip().split(' ')
			if info[0].lower() not in ret:
				ret[info[0]] = np.array([float(x) for x in info[1:]])
	return ret

