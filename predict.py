import numpy as np
from matplotlib import pyplot as plt
import utils
import model

def normalize(batch):
        mean, std = utils.get_mean_stds(batch)
        batch = batch - mean[np.newaxis, np.newaxis, np.newaxis, :]
        batch = batch/std[np.newaxis, np.newaxis, np.newaxis, :]
        return batch

def predict(path, m = None, threshold = None):
    i = utils.read_single_image(path)
    if m == None:
        m = model.fcn_resnet((256, 256, 3), 1)
        m.load_weights('./weights.45-0.32.h5')
    nb_rows, nb_cols, _ = np.shape(i)
    plt.imshow(i)
    i = np.expand_dims(i, axis = 0)
    i = normalize(i)
    result = m.predict(i, 1)[0]
    plt.figure(2)
    if threshold == None:
        plt.imshow(result[:, :, 0])
    else:
        plt.imshow(result[:, :, 0] > threshold)
    return result[:, :, 0]
