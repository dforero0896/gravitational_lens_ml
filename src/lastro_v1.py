#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import pickle
import configparser
from data_generator_function import TiffImageDataGenerator
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re

if len(sys.argv) != 2:
    config_file = 'config.ini'
else:
    config_file = sys.argv[1]
if not os.path.isfile(config_file):
    sys.exit('ERROR:\tThe config file %s was not found.'%config_file)

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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, SpatialDropout2D
from sklearn.metrics import roc_curve 
from tensorflow import keras
from helpers import build_generator_dataframe, get_file_id
        
###### Paths
WORKDIR = config['general']['workdir']    
sys.stdout.write('Project directory: %s\n'%WORKDIR)
SRC = os.path.join(WORKDIR, 'src')
DATA = os.path.join(WORKDIR, 'data')
RESULTS = os.path.join(WORKDIR, 'results')
TRAIN_MULTIBAND = config['general']['train_multiband']
TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')

lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col = 0)
dataframe_for_generator = build_generator_dataframe(lens_df, TRAIN_MULTIBAND)
# Extract data proportions for loss weighting
n_lens_clean = len(lens_df[lens_df['is_lens'] == True])
n_nolens_clean = len(lens_df[lens_df['is_lens'] == False])
equal_class_coeff = np.array([n_lens_clean/n_nolens_clean,1])
natural_class_coeff = np.array([1000 * n_lens_clean/n_nolens_clean,1])
# Training parameters
batch_size = config['trainparams'].getint('batch_size')
epochs = config['trainparams'].getint('epochs')
IMG_HEIGHT = 200
IMG_WIDTH = 200
data_bias = config['trainparams']['data_bias']
test_fraction = config['trainparams'].getfloat('test_fraction')
augment_train_data = bool(int(config['trainparams']['augment_train_data']))
kernel_size_1 = 8
kernel_size_2 = 6
dropout_kind=Dropout
train_df, val_df = train_test_split(dataframe_for_generator, test_size=test_fraction, random_state=42)
total_train = len(train_df)
total_val = len(val_df)
print("The number of objects in the whole training sample is: ", total_train)
print("The number of objects in the whole validation sample is: ", total_val)
print("The test fraction is: ", test_fraction)
image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          zca_epsilon=1e-06,
                                          rotation_range=0,
                                          width_shift_range=0.0,
                                          height_shift_range=0.0,
#                                         brightness_range=0,
                                          shear_range=0.0,
#                                         zoom_range=(0.9, 1.1),
                                          channel_shift_range=0.0,
                                          fill_mode='wrap',
                                          cval=0.0,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          rescale=None,
                                          preprocessing_function=None,
                                          data_format='channels_last',
                                          validation_split=0.2,
                                          dtype='float32')
image_data_gen_val = TiffImageDataGenerator(dtype='float32')
bands = [config['bands'].getboolean('VIS0'), 
    config['bands'].getboolean('NIR1'),
    config['bands'].getboolean('NIR2'),
    config['bands'].getboolean('NIR3')]
print("The bands are: ", bands)

ratio = float(config['trainparams']['lens_nolens_ratio'])
train_data_gen = image_data_gen_train.prop_image_generator_dataframe(train_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                  y_col='labels', 
				  batch_size = batch_size, 
				  validation=not(augment_train_data),
				  ratio=ratio,
				  bands = bands, binary = False)
val_data_gen = image_data_gen_val.prop_image_generator_dataframe(val_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                  y_col='labels', 
				  batch_size = batch_size, 
				  validation=True, 
				  ratio = ratio, bands = bands, 
				  binary = False)
roc_val_data_gen = image_data_gen_val.prop_image_generator_dataframe(val_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                  y_col='labels', 
                                  batch_size=batch_size, 
				  validation=True, 
				  ratio=ratio,
                                  bands=bands,
				  binary=False)

temp_data_gen = image_data_gen_train.image_generator_dataframe(train_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                  y_col='labels', 
				  batch_size=batch_size, 
				  validation=True,
                                  bands=bands,
				  binary=False)

image, _ = next(temp_data_gen)
input_shape = image[0].shape
# Define correct bias to initialize
output_bias = tf.keras.initializers.Constant(np.log(n_lens_clean/n_nolens_clean))

# Define metrics for the model.
metrics = [keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='acc'),
      keras.metrics.AUC(name='auc')]


model = Sequential([
    Conv2D(16, kernel_size_1, padding='same', activation='relu', 
           input_shape=input_shape),
    Conv2D(16, kernel_size_2, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(axis=-1),
    Conv2D(32, kernel_size_2, padding='same', activation='relu'),
    Conv2D(32, kernel_size_2, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(axis=-1),
    Conv2D(64, kernel_size_2, padding='same', activation='relu'),
    Conv2D(64, kernel_size_2, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(axis=-1),
    Dropout(0.2),
    Conv2D(128, kernel_size_2, padding='same', activation='relu'),
    Dropout(0.2),
    Conv2D(128, kernel_size_2, padding='same', activation='relu'),
    BatchNormalization(axis=-1),
    Dropout(0.2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(1024, activation='relu'),
    BatchNormalization(axis=-1),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=metrics)
model.summary()

save_dir = os.path.join(RESULTS, 'checkpoints/lastro_cnn/')
model_type = 'lastro_cnn'
model_name = '%s_Tr%i_Te%i_bs%i_ep%.03d_aug%i_VIS%i_NIR%i%i%i_DB%s_ratio%.01f_ks%i%i_%s.h5' % (model_type,
                                                                        total_train,
                                                                        total_val,
                                                                        batch_size,
                                                                        epochs,
                                                                        int(augment_train_data),
                                                                        bands[0], 
                                                                        bands[1],
                                                                        bands[2],
                                                                        bands[3],
                                                                        data_bias,
                                                                        ratio,
	kernel_size_1, 
	kernel_size_2,
	dropout_kind.__name__)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
print("The model name is: ", model_name)
checkpoint_dir = os.path.dirname(filepath)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                 save_best_only=True,
                                                 verbose=1,
						 monitor='val_acc')

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', 
					       min_delta=1, 
					       patience=10, 
					       verbose=0, 
				               mode='auto', 
					       baseline=None, 
					       restore_best_weights=True)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
		                                  cooldown=0,
                  		                  patience=5,
		                                  min_lr=0.5e-6,
						  monitor='val_acc')
# Define class weights for unevenly distributed (biased) dataset.
if data_bias == 'natural':
    sys.stdout.write('Using natural data bias: 1000x more non lenses than lenses.\n')
    class_coeff = natural_class_coeff
elif data_bias == 'none':
    sys.stdout.write('Using no data bias (simulate equal proportion among classes).\n')
    class_coeff = equal_class_coeff
elif data_bias == 'raw':
    sys.stdout.write('Using the raw bias (no weights applied).\n')
    class_coeff = [1,1]
else:
    raise NotImplementedError('data_bias must be either natural or none.')
class_weights = {0:class_coeff[0], 1:class_coeff[1]}
sys.stdout.write('Using weights: %s\n'%class_weights)

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train//batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val//batch_size,
    callbacks = [cp_callback, es_callback, lr_reducer],
    class_weight = class_weights,
#   use_multiprocessing=True,
    verbose=2
)

model.save(os.path.join(RESULTS,model_name))
with open(os.path.join(RESULTS,model_name.replace('h5', 'history')), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
# Score trained model.
scores = model.evaluate_generator(val_data_gen, verbose=2, steps=total_val//batch_size)
images_val, labels_true = next(roc_val_data_gen)
labels_score = model.predict(images_val, batch_size=batch_size, verbose=2)
fpr, tpr, thresholds = roc_curve(np.ravel(labels_true), np.ravel(labels_score))
print(fpr)
print(tpr)


print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

