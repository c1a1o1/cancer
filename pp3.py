# plot one patient scan

import dicom
import matplotlib.pyplot as plt

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage 1 samples/"
patient = "4ec5ef19b52ec06a819181e404d37038"

dcm = directory + '0a38e7597ca26f9374f8ea2770ba870d/4ec5ef19b52ec06a819181e404d37038.dcm'
print('Filename: {}'.format(dcm))
dcm = dicom.read_file(dcm)
print dcm

img = dcm.pixel_array
img[img == -2000] = 0

plt.axis('off')
plt.imshow(img)
plt.show()

plt.axis('off')
plt.imshow(-img) # Invert colors with -
plt.show()