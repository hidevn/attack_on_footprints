import numpy as np
import glob, os
import rasterio
# @params: batch: a four-tensor which contain the minibatch
# @return: the mean and std of each channel
def get_mean_stds(batch):
    nb_channels = batch.shape[3]
    channel_data = np.reshape(np.transpose(batch, [3, 0, 1, 2]), (nb_channels, -1))
    means = np.mean(channel_data, axis = 1)
    stds = np.std(channel_data, axis = 1)
    return means, stds

def list_tiff_files(location):
    os.chdir(location)
    return glob.glob("*.tif")

def read_single_image(path, num_bands = 3):
    with rasterio.open(path) as f:
        img_shape = f.shape
        image = np.ones((*img_shape, num_bands))
        image[:, :, 0] = f.read(1)
        image[:, :, 1] = f.read(2)
        image[:, :, 2] = f.read(3)
        image = image.astype('uint8')
        return image

# @params:
# window: [[ymin, ymax], [xmin, xmax]]
def read_window(path, window, num_bands = 3):
    image = read_single_image(path, num_bands)
    print('reading window from +' + path + ':')
    print(window)
    return image[window[0][0]:window[0][1], window[1][0]:window[1][1], 0:num_bands]

def get_image_size(path, num_bands = 3):
    image = read_single_image(path, num_bands)
    return image.shape