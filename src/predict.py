#!/usr/bin/env python
from helpers import build_generator_dataframe, get_file_id
from data_generator_function import TiffImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import re
import configparser
import pickle
import numpy as np


def main():
    if len(sys.argv) == 2:
        config_file = 'config_lesta_df.ini'
        model_name = sys.argv[1]
    elif len(sys.argv) == 3:
        config_file = sys.argv[1]
        model_name = sys.argv[2]
    else:
        sys.exit(
            'ERROR:\tUnexpected number of arguments.\nUSAGE:\t%s [CONFIG_FILE] MODEL_FILENAME'
            % sys.argv[0])
    if not os.path.isfile(config_file):
        sys.exit('ERROR:\tThe config file %s was not found.' % config_file)
    if not os.path.isfile(model_name):
        sys.exit('ERROR:\tThe model file %s was not found.' % model_name)
    
    # Import configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    # Extract parameters from model name
    if 'train_multiband_bin' in model_name:
        datadir = 'train_multiband_bin'
    elif 'train_multiband_noclip_bin' in model_name:
        datadir = 'train_multiband_noclip_bin'
    else:
        datadir = 'train_multiband_noclip_bin'

    # Extract bands from filename
    bands = []
    if 'VIS0' in model_name:
        bands.append(False)
    elif 'VIS1' in model_name:
        bands.append(True)
    if 'NIR000' in model_name:
        [bands.append(False) for i in range(3)]
    elif 'NIR111' in model_name:
        [bands.append(True) for i in range(3)]
    bands = list(np.array(bands).reshape(-1))
    print("The bands are: ", bands)
    # Extract split ratio from filename
    for param in model_name.split('_'):
        if 'ratio' in param:
            ratio = float(param.replace('ratio', ''))

    # Paths
    WORKDIR = config['general']['workdir']
    sys.stdout.write('Project directory: %s\n' % WORKDIR)
    DATA = os.path.join(WORKDIR, 'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN_MULTIBAND = os.path.join(DATA, datadir)
    TEST_MULTIBAND = TRAIN_MULTIBAND.replace('train', 'test')
    image_catalog = pd.read_csv(os.path.join(
        DATA, 'catalog/image_catalog2.0train.csv'),
                                comment='#',
                                index_col=0)
    print('The shape of the image catalog: ' + str(image_catalog.shape) + "\n")

    lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'),
                          index_col=0)
    dataframe_for_generator = build_generator_dataframe(
        lens_df, TEST_MULTIBAND)
    print(dataframe_for_generator['filenames'])
    # Split the TRAIN_MULTIBAND set into train and validation sets. Set test_size below!
    train_df, val_df = train_test_split(
        dataframe_for_generator,
        test_size=config['trainparams'].getfloat('test_fraction'),
        random_state=42)
    total_train = len(train_df)
    total_val = len(val_df)
    print("The number of objects in the whole training sample is: ",
          total_train)
    print("The number of objects in the whole validation sample is: ",
          total_val)
    test_fraction = float(config["trainparams"]["test_fraction"])
    print("The test fraction is: ", test_fraction)
    if config['trainparams']['subsample_train'] == 'total':
        subsample_train = total_train
        subsample_val = total_val
    else:
        try:
            subsample_train = int(config['trainparams']['subsample_train'])
            subsample_val = int(subsample_train * test_fraction /
                                (1. - test_fraction))
        except:
            raise ValueError('subsample_train should be \'total\' or int.')

    print("The number of objects in the training subsample is: ",
          subsample_train)
    print("The number of objects in the validation subsample is: ",
          subsample_val)
    # Create Tiff Image Data Generator objects for train and validation
    image_data_gen_val = TiffImageDataGenerator(dtype='float32')
    
    # Create generators for Images and Labels
    test_data_gen = image_data_gen_val.prop_image_generator_dataframe(
        val_df,
        directory=TRAIN_MULTIBAND,
        x_col='filenames',
        y_col='labels',
        batch_size=subsample_val,
        validation=True,
        ratio=ratio,
        bands=bands,
        binary=True)

    # Obtain model from the saving directory
    model_name_base = os.path.basename(model_name)
    model = tf.keras.models.load_model(model_name)
    model.summary()
    history_path = model_name.replace('h5', 'history')
 
    images_val, labels_true = next(test_data_gen)
    print(labels_true)
    predictions = model.predict(images_val,
                                 batch_size=1,
                                 verbose=2,
                                 workers=16,
                                 use_multiprocessing=True)
    print(predictions)
if __name__ == '__main__':
    main()
