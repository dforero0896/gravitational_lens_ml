#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
if len(sys.argv) != 4:
        sys.exit('ERROR:\tPlease provide the path of the project directory.\nUSAGE:\t%s PROJECT_DIR PARALLEL? OVERWITE\n'%sys.argv[0])
from data_generator_function import TiffImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from helpers import *
from data_generator_function import TiffImageDataGenerator
parallel = bool(int(sys.argv[2]))
overwrite = bool(int(sys.argv[3]))
if parallel:
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()   # Size of communicator
    rank = MPI.COMM_WORLD.Get_rank()   # Ranks in communicator
    name = MPI.Get_processor_name()    # Node where this MPI process runs
    sys.stdout.write('Using %i tasks. This is number %i\n'%(size, rank))

WORKDIR=os.path.abspath(sys.argv[1])
sys.stdout.write('Project directory: %s\n'%WORKDIR)
SRC = os.path.join(WORKDIR, 'src')
DATA = os.path.join(WORKDIR,'data')
RESULTS = os.path.join(WORKDIR, 'results')
TRAIN_MULTIBAND = os.path.join(DATA, 'train_multiband')
TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')
TRAIN_MULTIBAND_AUGMENT = os.path.join(DATA, 'train_multiband_augment')
if not os.path.isdir(TRAIN_MULTIBAND_AUGMENT):
    os.mkdir(TRAIN_MULTIBAND_AUGMENT)
lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col = 0)
local_test_df = build_generator_dataframe(lens_df, TRAIN_MULTIBAND)
no_lens_df = local_test_df[local_test_df['labels'] == 0]
indices = range(len(lens_df[lens_df['is_lens'] == False]), len(lens_df[lens_df['is_lens'] == True]))
if parallel:
    new_indices = np.array_split(indices, size)
    indices = new_indices[rank]
augment_nolens = TiffImageDataGenerator(featurewise_center=False,
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
                                          zoom_range=(0.9, 1),
                                          channel_shift_range=0.0,
                                          fill_mode='wrap',
                                          cval=0.0,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          rescale=None,
                                          preprocessing_function=None,
                                          data_format='channels_last',
                                          dtype='float32')
augment_nolens_gen = augment_nolens.image_generator_dataframe(no_lens_df,
                                  directory=TRAIN_MULTIBAND,
                                  x_col='filenames',
                                 y_col='labels', batch_size = 1, validation=False) 
for i in indices:
    outname = os.path.join(TRAIN_MULTIBAND_AUGMENT, 'image_%i_augment.tiff'%i)
    if os.path.isfile(outname) and not overwrite:
        continue
    random_mod_nolens, label = next(augment_nolens_gen)
    print(random_mod_nolens[0].shape)
    tifffile.imwrite(outname, random_mod_nolens[0])
    
if parallel:
    MPI.Finalize()

    
