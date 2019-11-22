#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import pickle
if len(sys.argv) != 3:
        sys.exit('ERROR:\tPlease provide the path of the project directory.\nUSAGE:\t%s USE_GPU PROJECT_DIR\n'%sys.argv[0])
if not bool(int(sys.argv[1])):
    sys.stdout.write('Not using GPU.\n')
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from data_generator_function import TiffImageDataGenerator
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow import keras
from helpers import build_generator_dataframe, get_file_id
WORKDIR=os.path.abspath(sys.argv[2])
sys.stdout.write('Project directory: %s\n'%WORKDIR)
SRC = os.path.join(WORKDIR, 'src')
DATA = os.path.join(WORKDIR,'data')
RESULTS = os.path.join(WORKDIR, 'results')
TRAIN_MULTIBAND = os.path.join(DATA, 'train_multiband')
TRAIN_MULTIBAND_AUGMENT = os.path.join(DATA, 'train_multiband_augment')
TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')

lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col = 0)
dataframe_for_generator = build_generator_dataframe(lens_df, TRAIN_MULTIBAND)
# Extract data proportions for loss weighting
n_lens_clean = len(lens_df[lens_df['is_lens'] == True])
n_nolens_clean = len(lens_df[lens_df['is_lens'] == False])
equal_class_coeff = np.array([n_lens_clean/n_nolens_clean,1])
natural_class_coeff = np.array([1000 * n_lens_clean/n_nolens_clean,1])

batch_size = 32 
epochs = 1
IMG_HEIGHT = 200
IMG_WIDTH = 200
data_bias = 'none'

train_df, val_df = train_test_split(dataframe_for_generator, test_size=0.1, random_state=42)
total_train = len(train_df)
total_val = len(val_df)
image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          zca_epsilon=1e-06,
                                          rotation_range=20,
                                          width_shift_range=0.0,
                                          height_shift_range=0.0,
                                          brightness_range=(0.8, 1.1),
                                          shear_range=0.0,
                                          zoom_range=(0.9, 1.1),
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

train_data_gen = image_data_gen_train.image_generator_dataframe(train_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                 y_col='labels', batch_size = batch_size, validation=False)
val_data_gen = image_data_gen_val.image_generator_dataframe(val_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                 y_col='labels', batch_size = batch_size, validation=True)
# Define correct bias to initialize
output_bias = tf.keras.initializers.Constant(np.log(n_lens_clean/n_nolens_clean))

# Define metrics for the model.
metrics = [keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.AUC(name='auc')]


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,4)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid', 
    bias_initializer = output_bias)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=metrics)
model.summary()

checkpoint_path = os.path.join(RESULTS, 'checkpoints/simple_cnn/simple_cnn.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Define class weights for unevenly distributed (biased) dataset.
if data_bias == 'natural':
    sys.stdout.write('Using natural data bias: 1000x more non lenses than lenses.\n')
    class_coeff = natural_class_coeff
elif data_bias == 'none':
    sys.stdout.write('Using no data bias (simulate equal proportion among classes).\n')
    class_coeff = equal_class_coeff
elif data_bias == 'raw':
    sys.stdout.write('Using the raw bias (no weights applied).\n')
    class_coeff = 1.
else:
    raise NotImplementedError('data_bias must be either natural or none.')
class_weights = {0:class_coeff[0], 1:class_coeff[1]}
sys.stdout.write('Using weights: %s'%class_weights)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=10,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=5,
    callbacks = [cp_callback, es_callback],
    class_weight = class_weights
)

model.save(os.path.join(RESULTS,'simple_cnn.h5'))
with open(os.path.join(RESULTS,'simple_cnn_history'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



