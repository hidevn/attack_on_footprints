import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import image as img
import rasterio
import utils
import model

def normalize(batch):
        mean, std = utils.get_mean_stds(batch)
        batch = batch - mean[np.newaxis, np.newaxis, np.newaxis, :]
        batch = batch/std[np.newaxis, np.newaxis, np.newaxis, :]
        return batch

def predict(path, x_start = 0, y_start = 0, m = None):
    img = utils.read_single_image(path)
    target_size = 256
    if m == None:
        m = model.fcn_resnet((256, 256, 3), 6)
        m.load_weights('./model.h5')
    nb_rows, nb_cols, _ = np.shape(img)
    # img = scipy.misc.imresize(img, (512, 512, 3))
    plt.imshow(img)
    '''
    result = np.zeros((nb_rows, nb_cols, 6))
    for row_begin in range(0, nb_rows, target_size):
        for col_begin in range(0, nb_cols, target_size):
            row_end = row_begin + target_size
            col_end = col_begin + target_size
            if row_end <= nb_rows and col_end <= nb_cols:
                window = [[row_begin, row_end], [col_begin, col_end]]
                i = utils.read_img_window(img, window)
                i = np.expand_dims(i, axis = 0)
                i = normalize(i)
                result[window[0][0]:window[0][1], window[1][0]:window[1][1], :] = m.predict(i, 1)[0]
                result = np.argmax(result, axis = 2)
    '''
    window = [[x_start, x_start+256], [y_start, y_start+256]]
    i = utils.read_img_window(img, window)
    plt.imshow(i)
    i = np.expand_dims(i, axis = 0)
    i = normalize(i)
    result = m.predict(i, 1)[0]
    # result = np.argmax(result, axis = 2)
    plt.figure(2)
    plt.imshow(result[:, :, 1])
    return result[:, :, 1]

    
                
    