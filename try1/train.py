'''
Train Convulotional

Epoch: 13916 cost= 1.412284732
Epoch: 13923 cost= 0.278997272

'''
import tensorflow as tf
import numpy as np
import scipy.io
import epinn24 as model
import params
import apiepi as api
import os

print "begin"
parameters = params.get()
training_epochs = 40000

path = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage1_100_100_200/"
files = [path + file for file in os.listdir(path)]

features, prob, acc, cost = model.train_tf(files, parameters, training_epochs=training_epochs)

print ("Accuracy:", "epochs", "learning rate", "cv1 size", "cv2 size", "cv1 channels", "cv2channels",
		"hidden", "img height", "img_width", "dropout")
print (acc, training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], 
	  parameters["cv1_channels"], parameters["cv2_channels"], parameters["hidden"], parameters["img_height"],
	  parameters["img_width"], parameters["dropout"])
print "Cost:", cost

scipy.io.savemat("resp_1", features, do_compression=True)    
print "end"