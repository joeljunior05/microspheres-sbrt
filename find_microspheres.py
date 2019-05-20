import os
from os.path import isfile, join
import math
import csv
import json
import numpy as np
import pandas as pd
import cv2

from utils.granul import run_granul

expected_result = None
with open('dataset/prediction/masks/expected.json') as f:
    expected_result = json.load(f)

inputs_folder = "dataset/prediction/inputs"
outputs_folder = "dataset/prediction/outputs_granul"

def remove_excess(csv_df, padding=14, size=480):  
    return csv_df[(csv_df.y > padding) & (csv_df.x > padding) & (csv_df.x < 480-padding) & (csv_df.y < 480 - padding)] 

def generate_count(filename): 
    res = {} 
    res["filename"]=filename 
    csv_df = pd.read_csv(join(outputs_folder, filename))
    csv_df.x = csv_df.x.astype(int)
    csv_df.y = csv_df.y.astype(int)
    
    csv_df.w = csv_df.w.astype(int)
    
    csv_df = remove_excess(csv_df, padding=10)
    result = csv_df.w.astype(int) 
    
    hist = result.groupby(result).count() 
    res["histogram"] = hist.to_dict() 
    res["count"] = result.count()
    res["spheres"] = list(zip(csv_df.x, csv_df.y, csv_df.w))     
    return res 

def find_acerto(actual, expected, limit=0):
    r = expected[2]/8 #25% ratio
    diff_r = abs(actual[2]/2 - expected[2]/2)
    
    p = 0#15
    
    x1 = actual[0] - p
    x2 = expected[0]
    
    y1 = actual[1] - p
    y2 = expected[1]
    
    dist = math.hypot(x2 - x1, y2 - y1)
    
    return dist <= r and diff_r <= r

def calc_T(actual_spheres, expected_spheres):
    acertos = 0
    limit = 0
    p = 0#15
    white1 = np.zeros((480, 480, 3))
    white2 = np.zeros((480, 480, 3))
    white3 = np.zeros((480, 480, 3))
    pad = 4
    
    expected = list(expected_spheres)
    
    for a in actual_spheres:
        antes = len(expected)
        for idx, e in enumerate(expected):
            white1[e[0] - pad: e[0] + pad, e[1]-pad: e[1]+pad] = [0, 0, 255]
            if find_acerto(a, e, limit):
                acertos += 1
                del expected[idx]
                break
            
            limit -= 1
            
        depois = len(expected)
        
        if antes != depois:
            white3[a[0] - pad - p: a[0] + pad -p, a[1]-pad-p: a[1]+pad-p] = [0, 255, 0]
            
    VP = acertos
    FP = len(actual_spheres) - acertos
    FN = len(expected)
    
    T = VP / (VP + FP + FN)
    
    sum_w = white1+white2+white3
    sum_w[(sum_w[:, :, 0] == 0) &
          (sum_w[:, :, 1] == 0) &
          (sum_w[:, :, 2] == 0) ] = [255, 255, 255]
    
    return T, sum_w, [VP, FP, FN]

def compare_actual_expected(actuals, expecteds):
    res = {}
    for j in actuals:
        for k in expecteds:
            if k['filename'] in j['filename']:
                res[k['filename']] = {"expected_value" : int(k['count']),
                                      "diff_count" : int(j['count']) - int(k['count']),
                                     "actual_value": int(j['count']),
                                     "percentage" : 1 - abs(int(j['count']) - int(k['count'])) / int(k['count']),
                                     "actual_sphere": j['spheres'],
                                     "expected_sphere": k['spheres'],}
    general = 0     
    ex_sum = 0
    ac_sum = 0
    T_media = 0
    
    VP = 0
    FP = 0
    FN = 0
    print("imagem & real & VP & FP & FN & T \\\\")
    for idx, k in enumerate(res.keys()):
        res[k]["T"], white, acc_err = calc_T(res[k]["actual_sphere"], res[k]["expected_sphere"])
        VP += acc_err[0]
        FP += acc_err[1]
        FN += acc_err[2]
        print("{} & {} & {} & {} & {} & {} \\\\".format(idx+1, res[k]["expected_value"], acc_err[0], acc_err[1], acc_err[2], res[k]["T"]))
        T_media += res[k]["T"] 
        general += res[k]["percentage"]
        ex_sum += len(res[k]["expected_sphere"])
        ac_sum += len(res[k]["actual_sphere"])
        
    print("ACC: {}".format(general / len(res.keys())))
    print("T: {}".format(T_media / len(res.keys())))
    print("EX: {} AC: {} ".format(ex_sum, ac_sum))
    print("VP: {} FP: {} FN: {}".format(VP, FP, FN))
    return res

run_granul(inputs_folder, outputs_folder)

onlyfiles = [f for f in os.listdir(outputs_folder) if isfile(join(outputs_folder, f))]
onlyfiles = filter(lambda x: ".csv" in x, onlyfiles)
actual_result = list(map(generate_count, onlyfiles))
comparation = compare_actual_expected(actual_result, expected_result)