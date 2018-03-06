import numpy as np
import utils
import os


class DataGenerator:
    def __init__(self, target_size, batch_size, train_ratio=0.8):
        self.train_dir = "Data\\trainlala\\"
        self.val_dir = "Data\\valala\\"
        self.train_label = "Data\\trainlabel\\"
        self.val_label = "Data\\valabel\\"
        self.train_ratio = train_ratio
        self.train_files = os.listdir(self.train_dir)
        self.val_files = os.listdir(self.val_dir)
        self.count_train = len(self.train_files)
        self.count_val = len(self.val_files)
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_bands = 3
        
    def get_label_name(self, img_name):
        return img_name[:-4] + '_mask.TIF'
    
    def image_generator(self, data_type):
        if data_type == 'train':
            i_dir = self.train_dir
            l_dir = self.train_label
            num_img = self.count_train
            i_list = self.train_files
        elif data_type == 'validation':
            i_dir = self.val_dir
            l_dir = self.val_label
            num_img = self.count_val
            i_list = self.val_files
        else:
            return
        while True:
            f = i_list[np.random.randint(0, num_img)]
            c_img = utils.read_single_image(i_dir + f, 3)
            c_label = utils.read_label(l_dir + self.get_label_name(f))
            yield c_img, c_label
                        
    def batch_generator(self, data_type):
        batch_size = self.batch_size
        img_gen = self.image_generator(data_type)
        while True:
            batch = []
            batch_label = []
            try:
                for i in range(0, batch_size):
                    img, label = next(img_gen)
                    batch.append(img)
                    batch_label.append(label)
            except StopIteration:
                if len(batch) > 0:
                    yield np.array(batch), np.array(batch_label)
                raise StopIteration
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
    
    def normalize(self, batch):
        mean, std = utils.get_mean_stds(batch)
        batch = batch - mean[np.newaxis, np.newaxis, np.newaxis, :]
        batch = batch/std[np.newaxis, np.newaxis, np.newaxis, :]
        return batch
    