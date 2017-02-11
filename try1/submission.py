'''
Create submission

Created on Nov 9, 2016

246 550 29
positives 243 totals 796
AUC 1.0

218 1809 31
positives 181 totals 2027
AUC 0.994771301495

253 1905 5
positives 272 totals 2158
AUC 0.99501312336
gran total 4982

data: pos, neg, nan 246 550 29
positives 243 totals 796
AUC 1.0

data: pos, neg, nan 218 1809 31
positives 181 totals 2027
AUC 0.994771301495

data: pos, neg, nan 253 1905 5
positives 272 totals 2158
AUC 0.99501312336

gran total 4982


@author: botpi
'''
import tensorflow as tf
import numpy as np
from epinn31 import *
import scipy.io
from apiepi import *
from params import param

print "begin"
group = "train"
group = "test"
sub_file = "submission_conv_48.csv"
    
r = []
r.append(["File", "Class"])

for i in range(3):
    patient = i+1
    features = scipy.io.loadmat("resp_%s_new" % patient)
    #images, labels, names = read_images("%s %s_new" % (group, ii))
    images, labels, names = read_images("%s_%s_new" % (group, patient))
    parameters = param(patient)
    
    prob = eval_conv(images, parameters, features)
    
    p = 0
    for i in xrange(len(names)):
#         if prob[i][1] > 0.999:
#             r.append([names[i] + ".mat", 1])
#         elif prob[i][1] < 0.001:
#             r.append([names[i] + ".mat", 0])
#         else:
#             r.append([names[i] + ".mat", prob[i][1] ])

        r.append([names[i] + ".mat", prob[i][1] ])
        if prob[i][1]>0.5:
            p += 1

    print "positives", p, "totals", len(names)
    if group == "train":
        print "AUC", auc(labels, prob)

print "gran total", len(r)
np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s")

print "end"