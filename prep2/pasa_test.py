import pandas as pd 
import os

path = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/"
group = "stage1_100_100_200"
path_orig = path + group
path_dest =  path_orig + "_test"


samples = pd.read_csv(path + "/stage1_sample_submission.csv")
#print samples

for sample in samples.values:
	print sample
	name = sample[0] + "_1.tfrecords"
	os.rename(path_orig + "/" + name, path_dest + "/" + name)
	name = sample[0] + "_2.tfrecords"
	os.rename(path_orig + "/" + name, path_dest + "/" + name)

