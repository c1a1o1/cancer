'''
Train Convulotional


'''
import tensorflow as tf
import numpy as np
#from epinn31 import *
import scipy.io
import epinn24 as model
import params
import apiepi as api
import os

print "begin"
parameters = params.get()
training_epochs = 20

path = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage1_100_100_200/"
files = [path + file for file in os.listdir(path)]
x, y, name = model.tf_queue(files)

features, prob, acc, cost = model.train_tf(x, y, parameters, training_epochs=training_epochs)

print "Accuracy:", "epochs", "learning rate", "cv1 size", "cv2 size", "cv1 channels", "cv2channels", "hidden", "img resize", "dropout"
print acc, training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"], parameters["cv2_channels"], parameters["hidden"], parameters["img_resize"], parameters["dropout"]
print "AUC", auc(labels, prob), "Cost", cost, "patient", patient, "con todo"

scipy.io.savemat("resp_%s_new" % patient, features, do_compression=True)    
print "end"