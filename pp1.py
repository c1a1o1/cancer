from pydicom import dicomio

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage 1 samples/"
patient = "0a0c32c9e08cc2ea76a71649de56be6d"
ds = dicomio.read_file(directory + patient + "/0a67f9edb4915467ac16a565955898d3.dcm")  # plan dataset
print "name:", ds.PatientName
print "dir setup:", ds.dir("setup")    # get a list of tags with "setup" somewhere in the name
print ds
#print ds.PatientSetupSequence[0]
#ds.PatientSetupSequence[0].PatientPosition = "HFP"
#ds.save_as("rtplan2.dcm")