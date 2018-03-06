import numpy as np
import utils
from matplotlib import pyplot as plt
import time


class PotsdamDataGenerator:
    def __init__(self, target_size, batch_size, train_ratio=0.8):
        self.image_dir = "D:\\Attack On Footprints\\2_Ortho_RGB\\"
        self.label_dir = "D:\\Attack On Footprints\\5_Labels_for_participants\\"
        self.train_ratio = train_ratio
        training_indexs = [(2, 10), (3, 10), (3, 11), (3, 12), (4, 11), (4, 12), (5, 10),
            (5, 12), (6, 10), (6, 11), (6, 12), (6, 8), (6, 9), (7, 11),
            (7, 12), (7, 7), (7, 9), (2, 11), (2, 12), (4, 10), (5, 11),
            (6, 7), (7, 10), (7, 8)]
        test_indexs = [(2, 13), (2, 14), (3, 13), (3, 14), (4, 13), (4, 14), (4, 15),
            (5, 13), (5, 14), (5, 15), (6, 13), (6, 14), (6, 15), (7, 13)]

        nb_train_inds = int(round(train_ratio * len(training_indexs)))
        self.training_imgs = training_indexs[0:nb_train_inds]
        self.val_imgs = training_indexs[nb_train_inds:]
        self.training_label = training_indexs[0:nb_train_inds]
        self.val_label = training_indexs[nb_train_inds:]
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_bands = 3
        
    def img_format(self, tup):
        return "top_potsdam_%d_%d_RGB.tif"%(tup[0], tup[1])
    def label_format(self, tup):
        return "top_potsdam_%d_%d_label.tif"%(tup[0], tup[1])
    
    def image_generator(self, data_type):
        folder_path = self.image_dir
        if data_type == 'train':
            image_names = self.training_imgs
        elif data_type == 'validation':
            image_names = self.val_imgs
        else:
            return
        target_size = self.target_size
        while True:
            for tup in image_names:
                f = self.img_format(tup)
                # nb_rows, nb_cols, _ = utils.get_image_size(folder_path + f, self.num_bands)
                nb_rows, nb_cols, _ = (256, 256, 3)
                c_img = utils.read_single_image(folder_path + f, 3)
                for row_begin in range(0, nb_rows, target_size[0]):
                    for col_begin in range(0, nb_cols, target_size[1]):
                        row_end = row_begin + target_size[0]
                        col_end = col_begin + target_size[1]
                        if row_end <= nb_rows and col_end <= nb_cols:
                            window = [[row_begin, row_end], [col_begin, col_end]]
                            img = utils.read_img_window(c_img, window, self.num_bands)
                            yield img, self.get_label_from_tup(tup, window)
    
    def random_image_generator(self, data_type):
        folder_path = self.image_dir
        if data_type == 'train':
            image_names = self.training_imgs
        elif data_type == 'validation':
            image_names = self.val_imgs
        else:
            return
        target_size = self.target_size
        nb_files = len(image_names)
        target_size = self.target_size
        while True:
            ran = np.random.randint(0, nb_files)
            tup = image_names[ran]
            f = self.img_format(tup)
            start = time.time()
            # print('Reading image: ' + str(ran))
            # c_img = utils.read_single_image(folder_path + f, self.num_bands)
            # c_label = utils.read_single_image(self.label_dir + self.label_format(tup), self.num_bands)
            # nb_rows, nb_cols, _ = utils.get_image_size(folder_path + f, self.num_bands)
            nb_rows, nb_cols, _ = (256, 256, 3)
            row_begin = np.random.randint(0, nb_rows - target_size[0] + 1)
            col_begin = np.random.randint(0, nb_cols - target_size[1] + 1)
            row_end = row_begin + target_size[0]
            col_end = col_begin + target_size[1]
            window = ((row_begin, row_end), (col_begin, col_end))
            img = utils.read_window(folder_path + f, window, self.num_bands)
            label = utils.read_window(self.label_dir + self.label_format(tup), window, self.num_bands)
            one_hot_label = self.label_encoding(label)
            # print('Finished')
            end = time.time()
            print('Read image: ' + str(ran) + ', time: ' + str(end-start))
            yield img, one_hot_label
            # c_img = c_label = None
                        
    def get_label_from_tup(self, tup, window):
        f = self.label_format(tup)
        folder_path = self.label_dir
        label = utils.read_window(folder_path + f, window, self.num_bands)
        label = self.label_encoding(label)
        return label
                        
    def batch_generator(self, data_type):
        batch_size = self.batch_size
        img_gen = self.image_generator(data_type)
        while True:
            batch = []
            batch_label = []
            try:
                for i in range(0, batch_size):
                    start = time.time()
                    img, label = next(img_gen)
                    batch.append(img)
                    batch_label.append(label)
            except StopIteration:
                if len(batch) > 0:
                    yield np.array(batch), np.array(batch_label)
                raise StopIteration
            end = time.time()
            print('time: ' + str(end - start))
            yield np.array(batch), np.array(batch_label)
    
    def transform_generator(self, data_type):
        gen = self.batch_generator(data_type)
        
        def transform(x):
            batch, batch_label = x
            batch = batch.astype(np.float32)
            nb_rotations = np.random.randint(0, 4)
            batch = np.transpose(batch, [1, 2, 3, 0])
            batch = np.rot90(batch, nb_rotations)
            batch = np.transpose(batch, [3, 0, 1, 2])
            batch_label = np.transpose(batch_label, [1, 2, 3, 0])
            batch_label = np.rot90(batch_label, nb_rotations)
            batch_label = np.transpose(batch_label, [3, 0, 1, 2])
            if np.random.uniform() > 0.5:
                batch = np.flip(batch, axis=1)
                batch_label = np.flip(batch_label, axis=1)
            if np.random.uniform() > 0.5:
                batch = np.flip(batch, axis=2)
                batch_label = np.flip(batch_label, axis=2) 
            batch = self.normalize(batch)
            return batch, batch_label
            
        return map(transform, gen)
                
    def label_encoding(self, label_file):
        impervious_surfaces = [255, 255, 255]
        building = [0, 0, 255]
        low_vegetation = [0, 255, 255]
        tree = [0, 255, 0]
        car = [255, 255, 0]
        background = [255, 0, 0]
        # label_tensor = np.zeros((*label_file.shape[0:2], 6))
        label_tensor = np.zeros((*label_file.shape[0:2], 1))
        nb_rows, nb_cols, _ = label_file.shape
        '''
        for i in range(0, nb_rows):
            for j in range(0, nb_cols):
                pixel_rgb = label_file[i, j, :]
                if np.array_equal(pixel_rgb, impervious_surfaces):
                    label_tensor[i, j, 0] = 1
                elif np.array_equal(pixel_rgb, building):
                    label_tensor[i, j, 1] = 1
                elif np.array_equal(pixel_rgb, low_vegetation):
                    label_tensor[i, j, 2] = 1
                elif np.array_equal(pixel_rgb, tree):
                    label_tensor[i, j, 3] = 1
                elif np.array_equal(pixel_rgb, car):
                    label_tensor[i, j, 4] = 1
                elif np.array_equal(pixel_rgb, background):
                    label_tensor[i, j, 5] = 1
        '''
        for i in range(0, nb_rows):
            for j in range(0, nb_cols):
                pixel_rgb = label_file[i, j, :]
                if np.array_equal(pixel_rgb, building):
                    label_tensor[i, j, 0] = 1
        return label_tensor
    
    def normalize(self, batch):
        mean, std = utils.get_mean_stds(batch)
        batch = batch - mean[np.newaxis, np.newaxis, np.newaxis, :]
        batch = batch/std[np.newaxis, np.newaxis, np.newaxis, :]
        return batch
    