#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tifffile
import os
from astropy.io import fits

class TiffImageDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(TiffImageDataGenerator, self).__init__(*args, **kwargs)

    def get_input(self, path, binary=False):
        """Get input data from disk.
        
        Define how input data is loaded from the disk. 
		param: path (str): The path to the image file (tiff or npy, see bin.)
        param: binary (bool): If True, expects a `.npy` binary. If False, loads a `.tiff` file.
                Defaults to False.
		returns: img (ndarray): The image as a 3D array of size (HEIGHT, WIDTH, CHANNELS)"""
        if binary:
            img = np.load(path)
        else:
#            img = tifffile.imread(path)
            with fits.open(path, ignore_missing_end=True) as tab:
                img = tab[0].data
        return img

    def image_generator_dataframe(self,
                                  dataframe,
                                  directory='',
                                  x_col='filename',
                                  y_col='class',
                                  batch_size=64,
                                  validation=False,
                                  bands=[True, True, True, True],
                                  binary=True,
                                  get_ids=False,
                                  seed = 42,
                                  id_col='ID'):
        """Loads tiff image data by batches and automatically applies transformations.

        param: dataframe (pandas.DataFrame): Dataframe containing columns 'filename' and 'class'.
        param: directory (str): Path to the directory containing filenames in column 'filename'.
        param: x_col (str): Column name of the column containing image filenames. Defaults to 'filenames'.
        param: y_col (str): Column name of the column containing image classes. Defaults to 'class'.
        param: batch_size (int): Number of images to load at a time. Defaults to 64.
        param: validation (bool): Whether or not the generator is used for validation. If True, no transformations are applied.
                                    Defaults to False.
        param: bands (list of bool): Boolean mask of channels to use. Defaults to [True, True, True, True] (use all channels).
        param: binary (bool): If True, loads images from `.npy` binaries. Else, loads `.tiff` files. Defaults to True.
        param: get_ids (bool): If True, the generator yields batch_x, batch_y, batch_ID (for testing purposes).
        param: id_col (str): Column name of the ID column. Only used if get_ids=True.
        param: seed (int): Set random state for generator.
        yields: batch_x, batch_y [, batch_id]
        """
        files = dataframe[x_col].values
        np.random.seed(seed)
        while True:
            # Select files (paths/indices) for the batch
            batch_paths = np.random.choice(a=files,
                                           size=batch_size,
                                           replace=False)
            batch_input = []
            batch_output = []
            if get_ids: batch_id = []
            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input = self.get_input(os.path.join(directory,input_path), binary=binary)
                if len(input.shape)==3: input = input[:,:,bands]
                else:input=input[:,:,None]
                output = dataframe[dataframe[x_col] ==
                                   input_path][y_col].values[0]
                if self.preprocessing_function:
                    input = self.preprocessing_function(input)
                if not validation:
                    input = self.random_transform(input)
                batch_input += [input]
                batch_output += [output]
                if get_ids: batch_id += [dataframe[dataframe[x_col] ==
                                   input_path][id_col].values[0]]
            # Return a tuple of (input,output) to feed the network
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)
            if get_ids:
                batch_id=np.array(batch_id)
                yield (batch_x, batch_y, batch_id)
            else:
                yield (batch_x, batch_y)

    def prop_image_generator_dataframe(self,
                                       dataframe,
                                       directory='',
                                       x_col='filename',
                                       y_col='class',
                                       batch_size=64,
                                       validation=False,
                                       bands=[True, True, True, True],
                                       binary=True,
                                       ratio=0.5,
                                       get_ids=False,
                                       seed = 42,
                                       id_col='ID'):
        """Loads tiff image data by batches and automatically applies transformations. Forces the
        proportion of positive/negative to be ratio.

        param: dataframe (pandas.DataFrame): Dataframe containing columns 'filename' and 'class'.
        param: directory (str): Path to the directory containing filenames in column 'filename'.
        param: x_col (str): Column name of the column containing image filenames. Defaults to 'filenames'.
        param: y_col (str): Column name of the column containing image classes. Defaults to 'class'.
        param: batch_size (int): Number of images to load at a time. Defaults to 64.
        param: validation (bool): Whether or not the generator is used for validation. If True, no transformations are applied.
                                    Defaults to False.
        param: bands (list of bool): Boolean mask of channels to use. Defaults to [True, True, True, True] (use all channels).
        param: bin (bool): If True, loads images from `.npy` binaries. Else, loads `.tiff` files. Defaults to False.
        param: ratio (float): Ratio positive/negative to force in each batch.
        param: get_ids (bool): If True, the generator yields batch_x, batch_y, batch_ID (for testing purposes).
        param: id_col (str): Column name of the ID column. Only used if get_ids=True.
        param: seed (int): Set random state for generator.
        yields: batch_x, batch_y [, batch_id]
        """
        lens_df = dataframe[dataframe[y_col] == 1]
        nonlens_df = dataframe[dataframe[y_col] == 0]
        lens_size = int(ratio * batch_size)
        nonlens_size = batch_size - lens_size
        np.random.seed(seed)
        while True:
            # Select files (paths/indices) for the batch
            batch_paths_lens = np.random.choice(a=lens_df[x_col].values,
                                                size=lens_size,
                                                replace=False)
            batch_paths_nonlens = np.random.choice(a=nonlens_df[x_col].values,
                                                   size=nonlens_size, replace=False)
            batch_paths = np.concatenate(
                (batch_paths_lens, batch_paths_nonlens)).reshape(
                    (lens_size + nonlens_size))
            batch_input = []
            batch_output = []
            if get_ids: batch_id = []
            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input = self.get_input(os.path.join(directory,input_path), binary=binary)
                if len(input.shape)==3: input = input[:,:,bands]
                else:input=input[:,:,None]
                output = dataframe[dataframe[x_col] ==
                                   input_path][y_col].values[0]
                if self.preprocessing_function:
                    input = self.preprocessing_function(input)
                if not validation:
                    input = self.random_transform(input)
                batch_input += [input]
                batch_output += [output]
                if get_ids: batch_id += [dataframe[dataframe[x_col] ==
                                   input_path][id_col].values[0]]
            # Return a tuple of (input,output) to feed the network
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            if get_ids:
                batch_id=np.array(batch_id)
                yield (batch_x, batch_y, batch_id)
            else:
                yield (batch_x, batch_y)

    def generator_from_directory(self, directory, id_logger, bands = [True, True, True, True], batch_size = 10, binary = True):    
        files = (f for f in os.listdir(directory))
        while True:
            batch_input = []
            for i in range(batch_size):
                fn = next(files)
                input = self.get_input(os.path.join(directory,fn), binary=binary)
                if len(input.shape)==3: input = input[:,:,bands]
                else:input=input[:,:,None]
                batch_input += [input]
                id_logger += [fn]
            batch_x = np.array(batch_input)
            yield batch_x
