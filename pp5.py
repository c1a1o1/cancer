# plot patient scans

import dicom
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np

import base64
from IPython.display import HTML

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

# This function takes in a single frame from the DICOM and returns a single frame in RGB format.
def normalise(img):
    normed = (img / 14).astype(np.uint8) # Magic number, scaling to create int between 0 and 255
    img2 = np.zeros([img.shape[0],img.shape[1], 3], dtype=np.uint8)
    for i in range(3):
        img2[:, :, i] = normed
    return img2

def animate(pat, gifname):
    # Based on @Zombie's code
    fig = plt.figure()
    anim = plt.imshow(pat[0], cmap=plt.cm.bone)
    def update(i):
        anim.set_array(pat[i])
        return anim,
    
    a = animation.FuncAnimation(fig, update, frames=range(len(pat)), interval=50, blit=True)
    a.save(gifname, writer='imagemagick')


print "begin" 
#pat = load_patient('0a38e7597ca26f9374f8ea2770ba870d') # no cancer
pat = load_patient('0acbebb8d463b4b9ca88cf38431aac69') # cancer
npat = [normalise(p) for p in pat]   
animate(pat, 'test.gif')

import webbrowser
webbrowser.open('test.gif')

print "end"
