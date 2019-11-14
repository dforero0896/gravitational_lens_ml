#!/usr/bin/env python
from data_generator_function import TiffImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

WORKDIR='/home/daniel/gdrive/EPFL/2019-2020/MachineLearning/Project/gravitational_lens_ml'
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
train_df, val_df = train_test_split(local_test_df, test_size=0.1, random_state=42)
total_train = len(train_df)
total_val = len(val_df)
train_data_gen = TiffImageDataGenerator(featurewise_center=False,
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


fig, ax = plt.subplots(1,4, figsize = (10, 2.5))
for a in ax.ravel():
    img, label = next(train_data_gen.image_generator_dataframe(train_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                 y_col='labels', batch_size = 1, validation=True))
    a.imshow(img[0][:,:,2])
    a.set_xlabel(label[0])
plt.show()