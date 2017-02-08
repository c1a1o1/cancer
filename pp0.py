import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import glob

#p = sns.color_palette()

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage1/"
directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage 1 samples/"


os.listdir(directory)

ns = []
for d in os.listdir(directory):
	ns.append(len(os.listdir(directory + d)))
	print("Patient '{}' has {} scans".format(d, len(os.listdir(directory + d))))

print('----')
print max(ns), min(ns), len(ns)
print('Total patients {} Total DCM files {}'.format(len(os.listdir(directory)), 
                                                      len(glob.glob(directory + '*/*.dcm'))))
s=0
for y in ns:
	if y == 162:
		s +=1
print "160:", s


plt.hist(ns, bins=5000)
plt.show()