import os   
import sys
import re
import configparser
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from data_generator_function import TiffImageDataGenerator
#from helpers import build_generator_dataframe, get_file_id
tf.debugging.set_log_device_placement(False)

def build_generator_dataframe(id_label_df, directory):
    files = os.listdir(directory)
    files_new = []
    
    ids = id_label_df["ID"][:]
    extension = os.path.splitext(files[0])[1]
    for id_ in ids:
        pathfile = directory + "/image_" + str(id_) + "_multiband" + extension
        files_new.append(pathfile)
        
    df = pd.DataFrame()
    df['filenames'] = files_new
    df['labels'] = id_label_df.loc[:, 'is_lens'].values.astype(int)
    df['ID'] = ids
    return df

def main():
    if len(sys.argv) == 1:
        config_file = 'config_resnet.ini'
        print('branza')
    elif len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        sys.exit('ERROR:\tUnexpected number of arguments.\nUSAGE:\t%s [CONFIG_FILE]'% sys.argv[0])
    if not os.path.isfile(config_file):
        sys.exit('ERROR:\tThe config file %s was not found.' % config_file)
    
    # Avoid using GPU to evaluate models.
    sys.stdout.write('\nNot using GPU.\n')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Import configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    
    print("\nConfiguration file:\n")
    for section in config.sections():
        print("Section: %s" % section)
        for options in config.options(section):
            print("  %s = %s" % (options, config.get(section, options)))
            
    ###### Paths
    WORKDIR = config['general']['workdir']    
    sys.stdout.write('Project directory: %s\n'%WORKDIR)
    DATA = os.path.join(WORKDIR, 'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN_MULTIBAND = config['general']['train_multiband']
    
    image_catalog = pd.read_csv(os.path.join(DATA, 'catalog/image_catalog2.0train.csv'), comment='#', index_col=0)
    print('The shape of the image catalog: ' + str(image_catalog.shape) + "\n")  

    # Training parameters
    batch_size = config['trainparams'].getint('batch_size')  # orig paper trained all networks with batch_size=128
    epochs = config['trainparams'].getint('epochs')
    n = config['trainparams'].getint('n')
    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = config['trainparams'].getint('resnetversion')
    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    # Model name, depth and version
    
    # This is ok if we use weighted losses.
    lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'))
    dataframe_for_generator = build_generator_dataframe(lens_df, TRAIN_MULTIBAND)
   
    ###### Split the TRAIN_MULTIBAND set into train and validation sets. Set test_size below!
    train_df, val_df = train_test_split(dataframe_for_generator, test_size=config['trainparams'].getfloat('test_fraction'), random_state=42)
    total_train = len(train_df)
    total_val = len(val_df)
    print("The number of objects in the whole training sample is: ", total_train)
    print("The number of objects in the whole validation sample is: ", total_val)
    test_fraction = float(config["trainparams"]["test_fraction"])
    print("The test fraction is: ", test_fraction)
    if config['trainparams']['subsample_train'] == 'total':
        subsample_train = total_train
        subsample_val = total_val
    else:
        try:
            subsample_train = int(config['trainparams']['subsample_train'])
            subsample_val = int(subsample_train*test_fraction/(1.-test_fraction))
        except:
            raise ValueError('subsample_train should be \'total\' or int.')
    
    print("The number of objects in the training subsample is: ", subsample_train)
    print("The number of objects in the validation subsample is: ", subsample_val)
    
    augment_train_data = bool(int(config['trainparams']['augment_train_data']))
    ###### Create Tiff Image Data Generator objects for train and validation
    image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                          rotation_range=0,
                                          fill_mode='wrap',
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          rescale=None,
                                          preprocessing_function=None,
                                          data_format='channels_last',
                                          dtype='float32')

    image_data_gen_val = TiffImageDataGenerator(dtype='float32')
 
    bands = [config['bands'].getboolean('VIS0'), 
        config['bands'].getboolean('NIR1'),
        config['bands'].getboolean('NIR2'),
        config['bands'].getboolean('NIR3')]
    print("The bands are: ", bands)
    ###### Create generators for Images and Labels
    ratio = 0.5

    roc_val_data_gen = image_data_gen_val.prop_image_generator_dataframe(val_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=subsample_val, validation=True, ratio=ratio,
                                bands=bands)
    
    ###### Obtain model from the saving directory
    model_type = 'RN%dv%d' % (depth, version)
    
    model_name = '%s_Tr%i_Te%i_bs%i_ep%.03d_aug%i_VIS%i_NIR%i%i%i_DB%s_ratio%.01f_dropout_CORRECT.h5' % (model_type,
                                                                        subsample_train,
                                                                        subsample_val,
                                                                        batch_size,
                                                                        epochs,
                                                                        int(augment_train_data),
                                                                        config['bands'].getint('VIS0'), 
                                                                        config['bands'].getint('NIR1'),
                                                                        config['bands'].getint('NIR2'),
                                                                        config['bands'].getint('NIR3'),
                                                                        config['trainparams']['data_bias'],
                                                                        ratio)
    
    if not os.path.isfile(os.path.join(RESULTS, model_name)):
        sys.exit('ERROR:\tThe model file %s was not found.' % model_name)
    
    model = tf.keras.models.load_model(os.path.join(RESULTS, model_name))
    model.summary()
    print(model_name)
    
    history_path = os.path.join(RESULTS, model_name.replace('h5', 'history'))
    
    # Plots
    # History
    if os.path.isfile(history_path):
        with open(history_path, 'rb') as file_pi:
            history = pickle.load(file_pi)

        # Plots
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.plot(range(len(history['loss'])),
                history['val_loss'],
                label='Validation loss',
                marker='o',
                c='b')
        ax1.plot(range(len(history['loss'])),
                history['loss'],
                label='Training loss',
                marker='o',
                c='r')
        ax1.set_ylim([0,1])
        ax2.plot(range(len(history['loss'])),
                history['val_acc'],
                label='Validation accuracy',
                marker='o',
                c='b',
                ls='--',
                fillstyle='none')
        ax2.plot(range(len(history['loss'])),
                history['acc'],
                label='Training accuracy',
                marker='o',
                c='r',
                ls='--',
                fillstyle='none')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc=(-0.1, 1))
        ax2.legend(loc=(0.9, 1))
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        plt.gcf()
        plt.savefig(os.path.join(RESULTS, 'plots/' + os.path.basename(history_path).replace('.history', '.png')), dpi=200)
        

    # Roc curve 
    images_val, labels_true = next(roc_val_data_gen)
    print(labels_true)
    labels_score = model.predict(images_val, batch_size=subsample_val, verbose=2)
    fpr, tpr, thresholds = roc_curve(np.ravel(labels_true), np.ravel(labels_score))
    print(fpr)
    print(tpr)

    # Score
    scores = model.evaluate(images_val, labels_true, verbose=2)
    
    scores_dict = { 
        metric: value 
        for metric, value in zip(model.metrics_names, scores)
    }

    print(scores)
    print(model.metrics_names)
    
    acc = scores_dict['acc']
    auc = scores_dict['auc']
    np.savetxt(os.path.join(RESULTS, model_name.replace('h5', 'FPRvsTPR.dat')), np.array([fpr, tpr]).T,
               header='auc=%.3f\nacc=%.3f' % (auc, acc))
    
    plt.figure(2)
    plt.plot(fpr, tpr, 'or', label='Validation\nAUC=%.3f\nACC=%.3f' % (auc, acc))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1])
    plt.legend()
    plt.savefig(os.path.join(RESULTS, 'plots/ROCsklearn_' + os.path.basename(history_path).replace('.history', '.png')),
                dpi=200)
    
if __name__ == '__main__':
    main()
