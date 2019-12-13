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
tf.debugging.set_log_device_placement(False)

def get_file_id(filename, delimiters='_|\\.|-'):
    id_ = [int(s) for s in re.split(delimiters, filename) if s.isdigit()][0]
    return id_

def build_generator_dataframe(id_label_df, directory):
    files = os.listdir(directory)
    ids = [
        get_file_id(filename)
        for filename in files
    ]
    df = pd.DataFrame()
    df['filenames'] = files
    df['labels'] = id_label_df.loc[ids, 'is_lens'].values.astype(int)
    df['ID'] = ids
    return df

def main():
    if len(sys.argv) == 2:
        config_file = 'config_lesta_df.ini'
        model_name = sys.argv[1]
    elif  len(sys.argv) == 3:
        config_file = sys.argv[1]
        model_name = sys.argv[2]
    else:
        sys.exit('ERROR:\tUnexpected number of arguments.\nUSAGE:\t%s [CONFIG_FILE] MODEL_FILENAME'%sys.argv[0])
    if not os.path.isfile(config_file):
        sys.exit('ERROR:\tThe config file %s was not found.'%config_file)
    if not os.path.isfile(model_name):
        sys.exit('ERROR:\tThe model file %s was not found.'%model_name)
    
    sys.stdout.write('\nNot using GPU.\n')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    

    if len(tf.config.experimental.list_physical_devices('GPU')):
        print('GPU found. Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
        print(tf.config.experimental.list_physical_devices('GPU'))
        if(len(tf.config.experimental.list_logical_devices('GPU'))):
            print('Logical GPU found. Num logical GPUs Available: ', len(tf.config.experimental.list_logical_devices('GPU')))
            print(tf.config.experimental.list_logical_devices('GPU'))
    else:
        print("No GPU found")

    if len(tf.config.experimental.list_physical_devices('CPU')):
        print('Physical CPU found. Num Physical CPUs Available: ', len(tf.config.experimental.list_physical_devices('CPU')))
        print(tf.config.experimental.list_physical_devices('CPU'))
        if(len(tf.config.experimental.list_logical_devices('CPU'))):
            print('Logical CPU found. Num logical CPUs Available: ', len(tf.config.experimental.list_logical_devices('CPU')))
            print(tf.config.experimental.list_logical_devices('CPU'))
    else:
        print("No CPU found")
    config = configparser.ConfigParser()
    config.read(config_file)
    if 'train_multiband_bin' in model_name: datadir = 'train_multiband_bin'
    elif 'train_multiband_noclip_bin' in model_name: datadir = 'train_multiband_noclip_bin'
    else: datadir = 'train_multiband_noclip_bin'
    print("\nConfiguration file:\n")
    for section in config.sections():
        print("Section: %s" % section)
        for options in config.options(section):
            print("  %s = %s" % (options, config.get(section, options)))
            
    ###### Paths
    WORKDIR = config['general']['workdir']    
    #WORKDIR = os.path.abspath(sys.argv[2])
    sys.stdout.write('Project directory: %s\n'%WORKDIR)
    #SRC = os.path.join(WORKDIR, 'src')
    DATA = os.path.join(WORKDIR, 'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN_MULTIBAND = os.path.join(DATA, datadir)
    #TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')

    image_catalog = pd.read_csv(os.path.join(DATA, 'catalog/image_catalog2.0train.csv'), comment='#', index_col=0)
    print('The shape of the image catalog: ' + str(image_catalog.shape) + "\n")  

    # Training parameters
    batch_size = config['trainparams'].getint('batch_size')  # orig paper trained all networks with batch_size=128
    epochs = config['trainparams'].getint('epochs')
    num_classes = 1
    data_bias = 'none'
    # Model parameter
    
    # This is ok if we use weighted losses.
    lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col=0)
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
    train_steps_per_epoch = int(subsample_train//batch_size)
    val_steps_per_epoch = int(subsample_val//batch_size)
    print("The number of training steps is: ", train_steps_per_epoch)
    print("The number of validation steps is: ", val_steps_per_epoch)
    
    augment_train_data = bool(int(config['trainparams']['augment_train_data']))
    ###### Create Tiff Image Data Generator objects for train and validation
    image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          zca_epsilon=1e-06,
                                          rotation_range=0,
                                          width_shift_range=0.0,
                                          height_shift_range=0.0,
#                                         brightness_range=0,
                                          shear_range=0.0,
#                                         zoom_range=(0.9, 1.1),
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
    bands = []
    if 'VIS0' in model_name: bands.append(False)
    elif 'VIS1' in model_name: bands.append(True)
    if 'NIR000' in model_name: [bands.append(False) for i in range(3)]
    elif 'NIR111' in model_name: [bands.append(True) for i in range(3)]
    bands = list(np.array(bands).reshape(-1))
    print("The bands are: ", bands)
    ###### Create generators for Images and Labels
    ratio = 0.5
    roc_val_data_gen = image_data_gen_val.prop_image_generator_dataframe(val_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=subsample_val, validation=True, ratio=ratio,
                                bands=bands, binary=True)
    
    ###### Obtain model from the saving directory
    model_name_base = os.path.basename(model_name)
    model = tf.keras.models.load_model(model_name)
    model.summary()
    history_path =  model_name.replace('h5', 'history')

    ## Checkpoints
    save_dir = os.path.join(RESULTS, 'checkpoints/lastro_cnn/')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name_base)

    ### Plots
    ## History
    if os.path.isfile(history_path):
        with open(history_path, 'rb') as file_pi:
            history = pickle.load(file_pi)
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.plot(range(len(history['loss'])),
                history['val_loss'],
                label='Validation loss',
 #               marker='o',
                c='b',
		lw=3)
        ax1.plot(range(len(history['loss'])),
                history['loss'],
                label='Training loss',
 #               marker='o',
                c='r',
		lw=3)
        ax2.set_ylim([0.5,1])
        ax2.plot(range(len(history['loss'])),
                history['val_acc'],
                label='Validation accuracy',
 #               marker='^',
                c='b',
                ls='--',
                fillstyle='none',
		lw=3)
        ax2.plot(range(len(history['loss'])),
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
        plt.savefig(os.path.join(RESULTS, 'plots/' + os.path.basename(history_path).replace('.history', '.png')),
                    dpi=200)
        
    ##Roc curve 
    images_val, labels_true = next(roc_val_data_gen)
    print(labels_true)
    labels_score = model.predict(images_val, batch_size=1, verbose=2)
    fpr, tpr, thresholds = roc_curve(np.ravel(labels_true), np.ravel(labels_score))
    auc = np.trapz(tpr, fpr)
    labels_score = (labels_score > 0.5).astype(int)
    acc = np.mean((labels_true==labels_score))
    print(acc)
    np.savetxt(os.path.join(RESULTS, model_name_base.replace('h5', 'FPRvsTPR.dat')), np.array([fpr, tpr]).T, header = 'acc = %.3f'%acc)
    plt.figure(2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1])
    plt.legend()

    plt.plot(fpr, tpr, label='Validation\nAUC=%.3f\nACC=%.3f'%(auc, acc) ,lw =3)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], lw=3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS, 'plots/ROCsklearn_' + os.path.basename(model_name).replace('.h5', '.png')),
                dpi=200)
    

if __name__ == '__main__':
    main()
