import os


directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/"

f1 = set(os.listdir(directory + "stage1_100"))
f2 = set(os.listdir(directory + "roma/stage1_100"))

f3 = f2 -f1

print len(f2 - f1)
print len(f2), len(f1)
print f3, len(f3)
z()
for f in f3:
	os.rename(directory + "roma/stage1_100/" + f, directory + "stage1_100/" + f)