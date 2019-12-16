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
    # Avoid using GPU to evaluate models.
    sys.stdout.write('\nNot using GPU.\n')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
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

    image_catalog = pd.read_csv(os.path.join(
        DATA, 'catalog/image_catalog2.0train.csv'),
                                comment='#',
                                index_col=0)
    print('The shape of the image catalog: ' + str(image_catalog.shape) + "\n")

    lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'),
                          index_col=0)
    dataframe_for_generator = build_generator_dataframe(
        lens_df, TRAIN_MULTIBAND)
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

    augment_train_data = bool(int(config['trainparams']['augment_train_data']))
    # Create Tiff Image Data Generator objects for train and validation
    image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                                  rotation_range=0,
                                                  fill_mode='wrap',
                                                  horizontal_flip=True,
                                                  vertical_flip=True,
                                                  preprocessing_function=None,
                                                  data_format='channels_last',
                                                  dtype='float32')
    image_data_gen_val = TiffImageDataGenerator(dtype='float32')
    
    # Create generators for Images and Labels
    roc_val_data_gen = image_data_gen_val.prop_image_generator_dataframe(
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
    # Checkpoints dir
    save_dir = os.path.join(RESULTS, 'checkpoints/lastro_cnn/')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name_base)

    # Plots
    # History
    if os.path.isfile(history_path):
        with open(history_path, 'rb') as file_pi:
            history = pickle.load(file_pi)
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.plot(
            range(len(history['loss'])),
            history['val_loss'],
            label='Validation loss',
            #               marker='o',
            c='b',
            lw=3)
        ax1.plot(
            range(len(history['loss'])),
            history['loss'],
            label='Training loss',
            #               marker='o',
            c='r',
            lw=3)
        ax2.set_ylim([0.5, 1])
        ax2.plot(
            range(len(history['loss'])),
            history['val_acc'],
            label='Validation accuracy',
            #               marker='^',
            c='b',
            ls='--',
            fillstyle='none',
            lw=3)
        ax2.plot(
            range(len(history['loss'])),
            history['acc'],
            label='Training accuracy',
            #               marker='^',
            c='r',
            ls='--',
            fillstyle='none',
            lw=3)
        ax1.set_xlabel('Epoch')
        ax1.legend(loc=(-0.1, 1))
        ax2.legend(loc=(0.9, 1))
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        plt.gcf()
        plt.savefig(os.path.join(
            RESULTS, 'plots/' +
            os.path.basename(history_path).replace('.history', '.png')),
                    dpi=200)

    # Roc curve
    images_val, labels_true = next(roc_val_data_gen)
    print(labels_true)
    labels_score = model.predict(images_val,
                                 batch_size=1,
                                 verbose=2,
                                 workers=16,
                                 use_multiprocessing=True)
    fpr, tpr, thresholds = roc_curve(np.ravel(labels_true),
                                     np.ravel(labels_score))
    scores = model.evaluate(images_val,
                            labels_true,
                            batch_size=True,
                            verbose=1,
                            workers=16,
                            use_multiprocessing=True)
    scores_dict = {
        metric: value
        for metric, value in zip(model.metrics_names, scores)
    }
    print(scores)
    print(model.metrics_names)
    acc = scores_dict['acc']
    auc = scores_dict['auc']
    np.savetxt(os.path.join(RESULTS,
                            model_name_base.replace('h5', 'FPRvsTPR.dat')),
               np.array([fpr, tpr]).T,
               header='auc=%.3f\nacc=%.3f' % (auc, acc))
    plt.figure(2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1])
    plt.legend()

    plt.plot(fpr,
             tpr,
             label='Validation\nAUC=%.3f\nACC=%.3f' % (auc, acc),
             lw=3)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], lw=3)
    plt.legend()
    plt.savefig(os.path.join(
        RESULTS, 'plots/ROCsklearn_' +
        os.path.basename(model_name).replace('.h5', '.png')),
                dpi=200)


if __name__ == '__main__':
    main()
