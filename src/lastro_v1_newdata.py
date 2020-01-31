#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.metrics import roc_curve
from helpers import build_generator_dataframe, get_file_id
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import re
import pandas as pd
import os
import sys
import pickle
import configparser
from data_generator_function import TiffImageDataGenerator
from build_dataset import preprocess_band
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from threadsafe_iter import threadsafe_iter #Tensorflow2.0 has a bug concerning this, fixed in the repo though.
import datetime
## Fix errors about not finding Conv
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Helper classes and functions
class ResumeHistory(tf.keras.callbacks.History):
    """Callback to read a pickled history dict and append new training progress to it.

    params: history_path (str): The path to the history pickle.
    """
    def __init__(self, history_path):
        self.history_path = history_path
        self.use_history_file_flag = os.path.isfile(
            history_path) and (os.path.getsize(history_path) > 0)
        self.check_history_exists(history_path)
        if self.use_history_file_flag:
            self.previous_epoch = len(list(self.history_old.values())[0])
            print('Successfully loaded the existing history.')
            self.total_epoch = list(range(1, self.previous_epoch+1))
            self.complete_history = self.history_old
            print('Found %i total epochs saved.' % self.previous_epoch)
        else:
            self.previous_epoch =0 
            self.total_epoch = []
            self.complete_history = {}
        super(ResumeHistory, self).__init__()

    def check_history_exists(self, history_path):
        if self.use_history_file_flag:
            with open(history_path, 'rb') as file_pi:
                self.history_old = pickle.load(file_pi)
        else:
            self.history_old={}
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.use_history_file_flag:
            total_epoch = epoch + self.previous_epoch
        else:
            total_epoch = epoch
        self.total_epoch.append(total_epoch)
        for k, v in logs.items():
            self.complete_history.setdefault(k, []).append(v)
        with open(self.history_path, 'wb') as file_pi:
            pickle.dump(self.complete_history, file_pi)


def change_learning_rate(model, learning_rate='same'):
    """Manually change the learning rate of a model based on value set
    in config file.

    param: model (Keras model): Keras model in which to change the learning rate.
    param: learning_rate (float or str): If float, defines the new learning rate, 
            if \'same\' keeps the previous learning rate."""

    try:
        learning_rate = float(learning_rate)
        print('Changing learnig rate from %e to %e.' %
            (K.get_value(model.optimizer.lr), learning_rate))
        K.set_value(model.optimizer.lr, learning_rate)
    except ValueError:
        if learning_rate == 'same':
            pass
        else:
            raise NotImplementedError(
                    'learning_rate should be float or \'same\'')
def build_lastro_model(kernel_size_1, kernel_size_2, pool_size, input_shape, dropout_kind):
    """ Build LASTRO's CNN model.

    Builds Keras model for LASTRO's CNN in 
    Schaefer, C., Geiger, M., Kuntzer, T., & Kneib, J.-P. (2017). 
    Deep Convolutional Neural Networks as strong gravitational lens detectors. 
    https://doi.org/10.1051/0004-6361/201731201

    param: kernel_size_1 (int): Kernel size for first convolutional layer.
    param: kernel_size_2 (int): Kernel size for subsequent convolutional layers.
    param: pool_size (int): Width of the square pool window.
    param: input_shape (tuple of int): Input image size.
    param: dropout_kind (function): Either keras' Dropout or SpatialDropout2D
    returns: Keras' sequential model"""

    return Sequential([
            Conv2D(16, kernel_size_1, padding='same', activation='relu',
                input_shape=input_shape),
            Conv2D(16, kernel_size_2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
            BatchNormalization(axis=-1),
            Conv2D(32, kernel_size_2, padding='same', activation='relu'),
            Conv2D(32, kernel_size_2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
            BatchNormalization(axis=-1),
            Conv2D(64, kernel_size_2, padding='same', activation='relu'),
            Conv2D(64, kernel_size_2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
            BatchNormalization(axis=-1),
            dropout_kind(0.5),
            Conv2D(128, kernel_size_2, padding='same', activation='relu'),
            dropout_kind(0.4),
            Conv2D(128, kernel_size_2, padding='same', activation='relu'),
            BatchNormalization(axis=-1),
            dropout_kind(0.3),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.2),
            Dense(1024, activation='relu'),
            Dropout(0.2),
            Dense(1024, activation='relu'),
            BatchNormalization(axis=-1),
            Dense(1, activation='sigmoid')
        ])
def main():
    """Fit the model.

    If checkpoint does not exist in checkpoints or in the results directory, a new model is created and fitted
    according to parameters set in the config file. If a checkpoint or end model (in results dir) is found, it is loaded
    and the training is resumed according to values defined in the config file. By default, a checkpoint is preferred
    over an end model since it is assumed to be more recent (in case of manual stopping of training)."""

    if len(sys.argv) != 2:
        config_file = 'config_lesta_df.ini'
    else:
        config_file = sys.argv[1]
    if not os.path.isfile(config_file):
        sys.exit('ERROR:\tThe config file %s was not found.' % config_file)

    config = configparser.ConfigParser()
    config.read(config_file)
    print("\nConfiguration file:\n")
    for section in config.sections():
        print("Section: %s" % section)
        for options in config.options(section):
            print("  %s = %s" % (options, config.get(section, options)))
    if not bool(config['general'].getboolean('use_gpu')):
        sys.stdout.write('\nNot using GPU.\n')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Paths
    WORKDIR = config['general']['workdir']
    sys.stdout.write('Project directory: %s\n' % WORKDIR)
    SRC = os.path.join(WORKDIR, 'src')
    DATA = os.path.join(WORKDIR, 'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN_MULTIBAND = config['general']['train_multiband']
    TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')
    # Catalog 
    lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col=0)
    dataframe_for_generator = pd.read_csv(os.path.join(DATA, 'catalog/newdata_catalog.csv')) 
    # Extract data proportions for loss weighting
    n_lens_clean = len(lens_df[lens_df['is_lens'] == True])
    n_nolens_clean = len(lens_df[lens_df['is_lens'] == False])
    equal_class_coeff = np.array([n_lens_clean/n_nolens_clean, 1]) 
    natural_class_coeff = np.array([1000 * n_lens_clean/n_nolens_clean, 1])
    # Training parameters
    batch_size = config['trainparams'].getint('batch_size')
    epochs = config['trainparams'].getint('epochs')
    data_bias = config['trainparams']['data_bias']
    test_fraction = config['trainparams'].getfloat('test_fraction')
    augment_train_data = bool(int(config['trainparams']['augment_train_data']))
    kernel_size_1 = int(config['trainparams']['kernel_size_1'])
    kernel_size_2 = int(config['trainparams']['kernel_size_2'])
    dropout_config = config['trainparams']['dropout_kind'] #Import dropout and check values are valid.
    if dropout_config == 'dropout':
        dropout_kind = Dropout
    elif dropout_config == 'spatialdropout':
        dropout_kind = SpatialDropout2D
    else:
        raise NotImplementedError(
            'dropout_kind must be \'dropout\' or \'spatialdropout\'\nPlease check config file.')
    pool_size = int(config['trainparams']['pool_size'])

    
    bands = [config['bands'].getboolean('VIS0'),False, False, False]
    print("The bands are: ", bands)
    binary = False #bool(int(config['general']['binary']))
    ratio = float(config['trainparams']['lens_nolens_ratio'])
    # Split catalog in train and test (validation) sets. We used fixed state 42.
    train_df, val_df = train_test_split(
        dataframe_for_generator, test_size=test_fraction, random_state=42)
    total_train = len(train_df)
    total_val = len(val_df)
    print("The number of objects in the whole training sample is: ", total_train)
    print("The number of objects in the whole validation sample is: ", total_val)
    print("The test fraction is: ", test_fraction)
    if config['trainparams']['subsample_train'] == 'total': #Import subsample size and check values are as expected.
        subsample_train = total_train
        subsample_val = total_val
    else:
        try:
            subsample_train = int(config['trainparams']['subsample_train'])
            subsample_val = int(subsample_train*test_fraction/(1.-test_fraction))
        except:
            raise ValueError('subsample_train should be \'total\' or int.')
    print("The number of objects in the training subsample is: ", subsample_train)
    print("The number of objects in the validation subsample is: ", subsample_val)
    train_steps_per_epoch = int(subsample_train//batch_size)
    val_steps_per_epoch = int(subsample_val//batch_size)
    print("The number of training steps is: ", train_steps_per_epoch)
    print("The number of validation steps is: ", val_steps_per_epoch)

    # Create TiffImageDataGenerator objects to inherit random transformations from Keras' class.
    image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                                rotation_range=0,
                                                fill_mode='wrap',
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                preprocessing_function=preprocess_band,
                                                data_format='channels_last',
                                                dtype='float32')
    image_data_gen_val = TiffImageDataGenerator(dtype='float32', preprocessing_function=preprocess_band)
    # Create Generator objects from the initialized TiffImageDataGenerators.
    # To train
    train_data_gen = image_data_gen_train.prop_image_generator_dataframe(train_df,
                                                                        directory='',
                                                                        x_col='filename',
                                                                        y_col='label',
                                                                        batch_size=batch_size,
                                                                        validation=not(
                                                                            augment_train_data),
                                                                        ratio=ratio,
                                                                        bands=bands,
                                                                        binary=binary)
    # To validate
    val_data_gen = image_data_gen_val.prop_image_generator_dataframe(val_df,
                                                                    directory='',
                                                                    x_col='filename',
                                                                    y_col='label',
                                                                    batch_size=batch_size,
                                                                    validation=True,
                                                                    ratio=ratio, bands=bands,
                                                                    binary=binary)
    # To predict/evaluate                                                            
    roc_val_data_gen = image_data_gen_val.prop_image_generator_dataframe(val_df,
                                                                        directory='',
                                                                        x_col='filename',
                                                                        y_col='label',
                                                                        batch_size=batch_size,
                                                                        validation=True,
                                                                        ratio=ratio,
                                                                        bands=bands,
                                                                        binary=binary)
    # To safely obtain image size
    temp_data_gen = image_data_gen_train.image_generator_dataframe(train_df,
                                                                directory='',
                                                                x_col='filename',
                                                                y_col='label',
                                                                batch_size=1,
                                                                validation=True,
                                                                bands=bands,
                                                                binary=binary)

    # Obtain image size
    image, _ = next(temp_data_gen)
    input_shape = image[0].shape
    # Define correct bias to initialize (use if not forcing generator to load equal proportions of data)
    output_bias = tf.keras.initializers.Constant(
        np.log(n_lens_clean/n_nolens_clean))

    # Path to save checkpoints
    model_type = 'lastro_cnn'
    save_dir = os.path.join(RESULTS, 'checkpoints/%s/'%model_type)
    model_name = '%s_Tr%i_Te%i_bs%i_ep%.03d_aug%i_VIS%i_NIR%i%i%i_DB%s_ratio%.01f_ks%i%i_ps%i_%s_%s.h5' % (model_type,
                                                                                                        subsample_train,
                                                                                                        subsample_val,
                                                                                                        batch_size,
                                                                                                        epochs,
                                                                                                        int(
                                                                                                            augment_train_data),
                                                                                                        bands[0],
                                                                                                        bands[1],
                                                                                                        bands[2],
                                                                                                        bands[3],
                                                                                                        data_bias,
                                                                                                        ratio,
                                                                                                        kernel_size_1,
                                                                                                        kernel_size_2,
                                                                                                        pool_size,
                                                                                                        dropout_kind.__name__,
                                                                                                        'newdata')
    # Create path of checkpoints if necessary
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Checkpoint file path
    checkpoint_filepath = os.path.join(save_dir, model_name)
    # Final model file path
    end_model_name = os.path.join(RESULTS, model_name)
    print("The model name is: ", model_name)
    history_path = os.path.join(RESULTS, model_name.replace('h5', 'history'))

    # Callbacks
    # Checkpoint callback to save every epoch for better resuming training
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                    save_best_only=False,
                                                    verbose=1,
                                                    monitor='val_acc',
                                                    save_freq='epoch')
    # Checkpoint best model
    cp_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath.replace('.h5', '_BEST.h5'),
                                                        save_best_only=True,
                                                        verbose=1,
                                                        monitor='val_acc')
    # Early stopping callback (currently using a very high patience to avoid it triggering)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                #min_delta=0.1,
                                                patience=30,
                                                verbose=1,
                                                mode='auto',
                                                baseline=None,
                                                restore_best_weights=True)
    # Learning rate reducer callback.
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                    cooldown=0,
                                                    patience=20,
                                                    min_lr=0.5e-6,
                                                    monitor='val_acc',
                                                    verbose=1,
                                                    mode='auto')
    # Callback to save log to csv. It is probably better to use this than save-resume history.
    logger_callback = tf.keras.callbacks.CSVLogger(
        checkpoint_filepath.replace('.h5', '.log'), separator=',', append=True)
    # Callback to resume history if history_path exists.
    history_callback = ResumeHistory(history_path)
    # Callback to use Tensorboard
    log_dir=os.path.join(RESULTS, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

    # Define metrics for the model.
    metrics = [keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='acc'),
            keras.metrics.AUC(name='auc')]
    # If there are no checkpoints or final models saved, compile a new one.          
    if not os.path.isfile(checkpoint_filepath) and not os.path.isfile(end_model_name):
        model = build_lastro_model(kernel_size_1, kernel_size_2, pool_size, input_shape, dropout_kind)
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=metrics)
    elif not os.path.isfile(checkpoint_filepath) and os.path.isfile(end_model_name):
        print('Loading existing model from result.')
        model = tf.keras.models.load_model(end_model_name)
        epochs = int(config['trainparams']['new_epochs'])
        learning_rate = config['trainparams']['learning_rate']
        change_learning_rate(model, learning_rate)
    elif os.path.isfile(checkpoint_filepath):
        print('Loading existing model from checkpoint.')
        model = tf.keras.models.load_model(checkpoint_filepath)
        epochs = int(config['trainparams']['new_epochs'])
        learning_rate = config['trainparams']['learning_rate']
        change_learning_rate(model, learning_rate)
    model.summary()
    # Define class weights for unevenly distributed (biased) dataset.
    if data_bias == 'natural':
        sys.stdout.write(
            'Using natural data bias: 1000x more non lenses than lenses.\n')
        class_coeff = natural_class_coeff
    elif data_bias == 'none':
        sys.stdout.write(
            'Using no data bias (simulate equal proportion among classes).\n')
        class_coeff = equal_class_coeff
    elif data_bias == 'raw':
        sys.stdout.write('Using the raw bias (no weights applied).\n')
        class_coeff = [1, 1]
    else:
        raise NotImplementedError('data_bias must be either natural, none or raw.')
    class_weights = {0: class_coeff[0], 1: class_coeff[1]}
    sys.stdout.write('Using loss weights: %s\n' % class_weights)

    # Fit the model and save the history callback.
    # Use multiprocessing True when using > 1 workers. (Seems to cause problems)
    # Expect tf update to use threadsafe_iter class.
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=subsample_train//batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=subsample_val//batch_size,
        callbacks=[cp_callback, es_callback, lr_reducer,
                cp_best_callback, history_callback, logger_callback, tensorboard_callback],
        class_weight=class_weights,
    #    use_multiprocessing=True,
        verbose=1,
    #    workers=16
    )
    model.save(end_model_name)
    # If training finishes normally (Is not stopped by user), save final model.
    # Save complete history if the training was resumed.
    if history_callback.use_history_file_flag:
        with open(history_path, 'wb') as file_pi:
            pickle.dump(history_callback.complete_history, file_pi)
    else:
        with open(history_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    # Score trained model.
    scores = model.evaluate_generator(
        val_data_gen, verbose=2, steps=val_steps_per_epoch)
    images_val, labels_true = next(roc_val_data_gen)
    labels_score = model.predict(images_val, batch_size=batch_size, verbose=2)
    fpr, tpr, thresholds = roc_curve(np.ravel(labels_true), np.ravel(labels_score)) 
    auc = history.history['val_auc'][-1]
    acc = history.history['val_acc'][-1]
    # Save TPR and FPR metrics to plot ROC.
    np.savetxt(os.path.join(RESULTS, model_name.replace(
        'h5', 'FPRvsTPR.dat')), np.array([fpr, tpr]).T, header = 'auc=%.3f\nacc=%.3f'%(auc, acc))
if __name__=='__main__':
    main()
