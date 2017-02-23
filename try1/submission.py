'''
Create submission

@author: botpi
'''
import tensorflow as tf
import numpy as np
import epinn31 as model
import params
import os
import pandas as pd
import scipy.io

print "begin"
path = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage1_100_100_200_test/"
files = [path + file for file in os.listdir(path)]
features_file = "data/resp_18000"
sub_file = "data/stage1_submission_6.csv"
    
r = []
r.append(["id", "cancer"])

features = scipy.io.loadmat(features_file)
parameters = params.get()
preds = model.eval_conv(files, parameters, features)

a = []
for pred in preds:
    name, prob = pred
    d = {}
    d["id"] = name.split("/")[-1:][0].split("_")[0]
    d["cancer"] = prob[0][1]
    a.append(d)

df = pd.DataFrame(a)
df = df.groupby("id").mean()
print df
df.to_csv(sub_file)

print "end"