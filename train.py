from keras.callbacks import (Callback, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau)
from keras.optimizers import Adam
from os.path import join



def make_callbacks(run_path, options):
    model_checkpoint = ModelCheckpoint(filepath = join(run_path, 'model.h5'), save_best_only = True, save_weights_only = True)
    reduce_lr = ReduceLROnPlateau(verbose = 1, epsilon = 0.001, patience = 10)
    def get_lr(epoch):
        for epoch_thresh, lr in options['lr_schedule']:
            if epoch >= epoch_thresh:
                curr_lr = lr
            else:
                break
        return curr_lr
    lr_sche = LearningRateScheduler(get_lr)
    callbacks = [model_checkpoint, reduce_lr, lr_sche]
    return callbacks

def train_model(run_path, model, options, generator):
    #train_gen = generator.transform_generator('train')
    #validation_gen = generator.transform_generator('validation')
    
    train_gen = generator.batch_generator('train')
    validation_gen = generator.batch_generator('validation')
    model.compile(Adam(lr = options['init_lr']), 'categorical_crossentropy', metrics = ['accuracy'])
    
    callback = make_callbacks(run_path, options)
    
    model.fit_generator(train_gen,
                        steps_per_epoch = options['steps_per_epoch'],
                        epochs = options['epochs'],
                        validation_data = validation_gen,
                        validation_steps = options['validation_steps'],
                        callbacks = callback)