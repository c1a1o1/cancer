# plot patient scans

import dicom
import matplotlib.pyplot as plt
import glob

def get_slice_location(dcm):
    return float(dcm[0x0020, 0x1041].value)

# Returns a list of images for that patient_id, in ascending order of Slice Location
def load_patient(patient_id):
	directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage 1 samples/"
	files = glob.glob(directory + '{}/*.dcm'.format(patient_id))
	imgs = {}
	for f in files:
		dcm = dicom.read_file(f)
		img = dcm.pixel_array
		img[img == -2000] = 0
		sl = get_slice_location(dcm)
		imgs[sl] = img
    
	# Not a very elegant way to do this
	sorted_imgs = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]
	return sorted_imgs

pat = load_patient('0a38e7597ca26f9374f8ea2770ba870d')

f, plots = plt.subplots(11, 10, sharex='all', sharey='all', figsize=(10, 11))
# matplotlib is drunk
#plt.title('Sorted Slices of Patient 0a38e7597ca26f9374f8ea2770ba870d - No cancer')
for i in range(110):
    plots[i // 10, i % 10].axis('off')
    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)

plt.show()