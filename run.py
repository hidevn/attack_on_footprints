import data_gen
import model
import train

generator = data_gen.PotsdamDataGenerator((256, 256), 8)
options = {"lr_schedule": [(0, 1e-5), (80, 1e-6)],
           "epochs": 100,
           "init_lr": 1e-5,
           "steps_per_epoch": 512,
           "validation_steps": 128
           }

train.train_model('D:\\Attack On Footprints\\', model.fcn_resnet((256, 256, 3), 6), options, generator)