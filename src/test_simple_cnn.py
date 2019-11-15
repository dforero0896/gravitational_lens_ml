#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
if len(sys.argv) != 3:
        sys.exit('ERROR:\tPlease provide the path of the project directory.\nUSAGE:\t%s USE_GPU PROJECT_DIR\n'%sys.argv[0])
if not bool(int(sys.argv[1])):
    sys.stdout.write('Not using GPU.')
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

WORKDIR=os.path.abspath(sys.argv[2])
sys.stdout.write('Project directory: %s\n'%WORKDIR)
SRC = os.path.join(WORKDIR, 'src')
DATA = os.path.join(WORKDIR,'data')
RESULTS = os.path.join(WORKDIR, 'results')
TRAIN_MULTIBAND = os.path.join(DATA, 'train_multiband')
TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')

def get_file_id(filename, delimiters = '_|\.|-'):
    id_ = [int(s) for s in re.split(delimiters, filename) if s.isdigit()][0]
    return id_
lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col = 0)
local_test_files = os.listdir(TRAIN_MULTIBAND)
local_test_id = [
    get_file_id(filename)
    for filename in local_test_files
]
local_test_df = pd.DataFrame()
local_test_df['filenames'] = local_test_files
local_test_df['labels'] = lens_df.loc[local_test_id, 'is_lens'].values.astype(int)

batch_size = 1 
epochs = 15
IMG_HEIGHT = 200
IMG_WIDTH = 200

train_df, val_df = train_test_split(local_test_df, test_size=0.1, random_state=42)
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
                                          fill_mode='nearest',
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
val_data_gen = image_data_gen_val.image_generator_dataframe(train_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                 y_col='labels', batch_size = batch_size, validation=True)

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
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

checkpoint_path = os.path.join(RESULTS, 'checkpoints/simple_cnn/simple_cnn.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=10000,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=10000,
    callbacks = [cp_callback]
)

model.save(os.path.join(RESULTS,'simple_cnn.h5'))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(os,path,join(RESULTS, 'plots/results.png'), dpi = 200)

