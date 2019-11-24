#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
#if __name__ == '__main__':
#	print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tifffile
import os

class TiffImageDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(TiffImageDataGenerator, self).__init__(*args, **kwargs)
    def get_input(self, path):
        """Get imput data from disk.
        
        Define how input data is loaded from the disk. 
		param: path (str): The path to the image file (tiff)
		returns: img (ndarray): The image as a 3D array of size (HEIGHT, WIDTH, CHANNELS)"""
        
        img = tifffile.imread(path)
        return img

    def image_generator_dataframe(self,dataframe,
                              directory='',
                              x_col='filename',
                              y_col='class',
                              batch_size=64,
                              validation=False, bands = [True, True, True, True]):
        """Loads tiff image data by batches and automatically applies transformations.

        param: dataframe (pandas.DataFrame): Dataframe containing columns 'filename' and 'class'.
        param: directory (str): Path to the directory containing filenames in column 'filename'.
        param: x_col (str): Column name of the column containing image filenames. Defaults to 'filenames'.
        param: y_col (str): Column name of the column containing image classes. Defaults to 'class'.
        param: batch_size (int): Number of images to load at a time. Defaults to 64.
        param: validation (bool): Whether or not the generator is used for validation. If True, no transformations are applied.
                                    Defaults to False.
        param: bands (list of bool): Boolean mask of channels to use. Defaults to [True, True, True, True] (use all channels).
        yields: batch_x, batch_y
        """
        files = dataframe[x_col].values
        while True:
            # Select files (paths/indices) for the batch
            batch_paths = np.random.choice(a=files, size=batch_size, replace=False)
            batch_input = []
            batch_output = []

            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input = self.get_input(os.path.join(directory, input_path))[:,:,bands]
                output = dataframe[dataframe[x_col] == input_path][y_col].values[0]
                if self.preprocessing_function:
                    input = self.preprocessing_function(input)
                if not validation:
                    input = self.random_transform(input)
                batch_input += [input]
                batch_output += [output]
            # Return a tuple of (input,output) to feed the network
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield (batch_x, batch_y)
    
    def prop_image_generator_dataframe(self,
                                       dataframe,
                                       directory='',
                                       x_col='filename',
                                       y_col='class',
                                       batch_size=64,
                                       validation=False,
                                       bands=[True, True, True, True],
                                       ratio=0.5):
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
        param: ratio (float): Ratio positive/negative to force in each batch.
        yields: batch_x, batch_y
        """
        lens_df = dataframe[dataframe[y_col] == 1]
        nonlens_df = dataframe[dataframe[y_col] == 0]
        lens_size = int(ratio * batch_size)
        nonlens_size = batch_size - lens_size
        while True:
            # Select files (paths/indices) for the batch
            batch_paths_lens = np.random.choice(a=lens_df[x_col].values,
                                                size=lens_size, replace=False)
            batch_paths_nonlens = np.random.choice(
                a=nonlens_df[x_col].values, size=nonlens_size)
            batch_paths = np.concatenate(
                (batch_paths_lens, batch_paths_nonlens)).reshape(
                    (lens_size + nonlens_size))
            batch_input = []
            batch_output = []

            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input = self.get_input(os.path.join(
                    directory, input_path))[:, :, bands]
                output = dataframe[dataframe[x_col] ==
                                   input_path][y_col].values[0]
                if self.preprocessing_function:
                    input = self.preprocessing_function(input)
                if not validation:
                    input = self.random_transform(input)
                batch_input += [input]
                batch_output += [output]
            # Return a tuple of (input,output) to feed the network
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield (batch_x, batch_y)
