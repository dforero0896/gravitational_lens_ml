#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
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
                              validation=False):
        """Loads tiff image data by batches and automatically applies transformations.

        param: dataframe (pandas.DataFrame): Dataframe containing columns 'filename' and 'class'.
        param: directory (str): Path to the directory containing filenames in column 'filename'.
        param: x_col (str): Column name of the column containing image filenames. Defaults to 'filenames'.
        param: y_col (str): Column name of the column containing image classes. Defaults to 'class'.
        param: batch_size (int): Number of images to load at a time. Defaults to 64.
        param: validation (bool): Whether or not the generator is used for validation. If True, no transformations are applied.
                                    Defaults to False.
        yields: batch_x, batch_y
        """
        files = dataframe[x_col].values
        while True:
            # Select files (paths/indices) for the batch
            batch_paths = np.random.choice(a=files, size=batch_size)
            batch_input = []
            batch_output = []

            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input = self.get_input(os.path.join(directory, input_path))
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
