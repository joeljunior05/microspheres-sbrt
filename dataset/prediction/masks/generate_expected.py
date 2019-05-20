import os  
from os.path import isfile, join  
import json

import pandas as pd
import numpy as np

def remove_excess(csv_df, padding=10, size=480):  
    return csv_df[(csv_df.y > padding) & (csv_df.x > padding) & (csv_df.x < 480-padding) & (csv_df.y < 480 - padding)] 

def to_dict(filename):
    res = {}
    res['filename'] = filename
    df_csv = pd.read_csv(filename) 
    df_csv.r = df_csv.r.astype(int)   
    df_csv = remove_excess(df_csv, padding=10) 
    r = df_csv.r

    hist = r.groupby(r).count() 
    res["histogram"] = hist.to_dict()
    res["count"] = r.count()
    res["spheres"] = list(zip(df_csv.x, df_csv.y, df_csv.r)) 
    return res

def default(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def save_json(rows):
    with open("expected.json", 'w') as outfile:
        json.dump(rows, outfile, default=default)

mypath = "."

onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]  

onlyfiles = filter(lambda x: ".csv" in x, onlyfiles)    

save_json(list(map(to_dict, onlyfiles)))
