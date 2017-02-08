# read slices from files .mat (vec, val and proj ) and plot

import numpy as np
import matplotlib.pyplot as plt
import dicom
import os
import scipy.io

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples"
group = "stage1"
path = directory + '/' + group 
patient = "00cba091fa4ad62cc3200a657aeb957e"

patients = [s for s in os.listdir(path + '/')]
#a = {}
#a["patients"] = patients
#scipy.io.savemat('patients', a, do_compression=True)
np.savetxt('patients.csv', patients, delimiter=',', fmt="%s")