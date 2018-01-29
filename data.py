import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import rasterio

image_path = "D:\\Attack On Footprints\\2_Ortho_RGB\\"
segment_path = "D:\\Attack On Footprints\\5_Labels_for_participants\\"

image_tif = rasterio.open(image_path + "top_potsdam_2_10_RGB.tif")
segmented_tif = rasterio.open(segment_path + "top_potsdam_2_10_label.tif")

img_shape = image_tif.shape

print(img_shape)

image = segmented = np.ones((*img_shape, 3))
image[:, :, 0] = image_tif.read(1)
image[:, :, 1] = image_tif.read(2)
image[:, :, 2] = image_tif.read(3)
image = image.astype('uint8')

segmented[:, :, 0] = segmented_tif.read(1)
segmented[:, :, 1] = segmented_tif.read(2)
segmented[:, :, 2] = segmented_tif.read(3)
segmented = segmented.astype('uint8')

plt.imshow(image)
plt.imshow(segmented, alpha = 0.5)
