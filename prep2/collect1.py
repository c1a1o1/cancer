import os

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/"

f1 = os.listdir(directory + "stage1_100")
f2 = os.listdir(directory + "stage1")

f3 = {x.split('.')[0] for x in f1}
f4 = {x.split('.')[0] for x in f2}

print f3 - f4
print f4 - f3
