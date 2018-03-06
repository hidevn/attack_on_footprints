import data_gen_n
import model
import train

generator = data_gen_n.DataGenerator((256, 256), 8)
options = {"lr_schedule": [(0, 1e-5), (80, 1e-6)],
           "epochs": 5,
           "init_lr": 1e-5,
           "steps_per_epoch": 32,
           "validation_steps": 4
           }

train.train_model('D:\\Attack On Footprints\\', model.fcn_resnet((256, 256, 3), 1), options, generator)